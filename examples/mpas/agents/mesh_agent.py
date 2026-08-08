# SPDX-License-Identifier: Apache-2.0

"""MeshAgent - Manages mesh generation and partitioning.

This agent handles the complete mesh lifecycle for MPAS forecasts:
- Downloads and builds Metis graph partitioning library
- Downloads and builds MPAS-Tools (hex_projection, grid_rotate)
- Installs MPAS-Limited-Area for regional mesh extraction
- Downloads precomputed global mesh files
- Generates regional meshes via hex_projection + grid_rotate
- Generates regional meshes via MPAS-Limited-Area
- Partitions meshes for MPI parallel execution
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from parsl.app.errors import BashExitFailure

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task, python_task


@chiltepin_agent()
class MeshAgent:
    """Agent for managing mesh generation and partitioning.

    Provides a top-level ``generate_mesh()`` action that routes to the
    correct generation path based on stored configuration.  Individual
    utility actions remain callable for advanced use.
    """

    def __init__(
        self,
        work_dir: str,
        mesh_config: Dict[str, Any],
        metis_version: str = "5.2.1",
        mpas_tools_version: str = "2.0.0",
        limited_area_version: str = "v2.2",
    ):
        """Initialize MeshAgent.

        Parameters
        ----------
        work_dir : str
            Directory where mesh tools will be installed and where mesh will be created
        mesh_config: Dict[str, Any]
            Mesh generation configuration (type, resolution, method, params, ranks)
        metis_version : str, optional
            Metis version to install, by default "5.2.1"
        mpas_tools_version : str, optional
            MPAS-Tools version to clone, by default "2.0.0"
        limited_area_version : str, optional
            MPAS-Limited-Area version to clone, by default "v2.2"
        """
        self.work_dir = Path(work_dir)
        self.mesh_config = dict(mesh_config)
        self.metis_version = metis_version
        self.mpas_tools_version = mpas_tools_version
        self.limited_area_version = limited_area_version
        self.log_dir = self.work_dir / "logs"

        # Metis state
        self.metis_downloaded = False
        self.metis_built = False
        self.metis_source_dir: Optional[Path] = None
        self.gpmetis_path: Optional[Path] = None

        # MPAS-Tools state (hex_projection + grid_rotate)
        self.mesh_tools_installed = False
        self.mpas_tools_dir: Optional[Path] = None
        self.hex_projection_path: Optional[Path] = None
        self.grid_rotate_path: Optional[Path] = None

        # MPAS-Limited-Area state
        self.limited_area_installed = False
        self.limited_area_source_dir: Optional[Path] = None
        self.create_region_path: Optional[Path] = None

        # Mesh data
        self.global_mesh_downloaded = False
        self.mesh_data_url = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes"
        self.mesh_data_dir = self.work_dir / "mesh_data"
        # Precomputed resolutions available for download from UCAR
        self.resolution_cells = {
            "480km": 2562,
            "384km": 4002,
            "240km": 10242,
            "120km": 40962,
            "60km": 163842,
            "48km": 256002,
            "30km": 655362,
            "24km": 1024002,
            "15km": 2621442,
            "12km": 4096002,
            "10km": 5898242,
            "7.5km": 10485762,
            "5km": 23592962,
            "4km": 36864002,
            "3.75km": 41943042,
            "3km": 65536002,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_resolution_km(resolution: str) -> float:
        """Convert resolution string like '120km' or '7.5km' to float km."""
        return float(resolution.lower().replace("km", ""))

    @staticmethod
    def _write_project_hexes_namelist(
        work_dir: Path, cell_spacing_km: float,
        extent_x_km: float, extent_y_km: float,
        center_lat: float, center_lon: float,
    ) -> None:
        """Write namelist.projections for hex_projection tool."""
        work_dir.mkdir(parents=True, exist_ok=True)
        namelist = work_dir / "namelist.projections"
        namelist.write_text(
            f"&mesh\n"
            f"  cell_spacing_km = {cell_spacing_km}\n"
            f"  mesh_length_x_km = {extent_x_km}\n"
            f"  mesh_length_y_km = {extent_y_km}\n"
            f"  earth_radius_km = 6371.229\n"
            f"/\n"
            f"&projection\n"
            f'  projection_type = "lambert_conformal"\n'
            f"/\n"
            f"&lambert_conformal\n"
            f"  reference_longitude_degrees = {center_lon}\n"
            f"  standard_longitude_degrees = {center_lon}\n"
            f"  reference_latitude_degrees = {center_lat}\n"
            f"  standard_parallel_1_degrees = {center_lat}\n"
            f"  standard_parallel_2_degrees = {center_lat}\n"
            f"/\n"
        )


    @staticmethod
    def _write_rotate_namelist(
        work_dir: Path,
        center_lat: float, center_lon: float,
        rotation_degrees: float,
        original_lat: Optional[float] = None,
        original_lon: Optional[float] = None,
    ) -> None:
        """Write namelist.input for grid_rotate tool."""
        work_dir.mkdir(parents=True, exist_ok=True)
        namelist = work_dir / "namelist.input"
        orig_lat = center_lat if original_lat is None else original_lat
        orig_lon = center_lon if original_lon is None else original_lon
        namelist.write_text(
            f"&input\n"
            f"   config_original_latitude_degrees = {orig_lat}\n"
            f"   config_original_longitude_degrees = {orig_lon}\n"
            f"\n"
            f"   config_new_latitude_degrees = {center_lat}\n"
            f"   config_new_longitude_degrees = {center_lon}\n"
            f"   config_birdseye_rotation_counter_clockwise_degrees = "
            f"{rotation_degrees}\n"
            f"/\n"
        )

    @staticmethod
    def _write_create_region_spec(
        work_dir: Path, create_region_config: Dict[str, Any], name: str,
    ) -> Path:
        """Write a region specification file for MPAS-Limited-Area create_region.

        Supports polygon, circle, ellipse, and channel region types.
        The region type is determined by which key is present in create_region_config.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        if "polygon" in create_region_config:
            poly = create_region_config["polygon"]
            spec_file = work_dir / f"{name}.custom.pts"
            lines = [
                f"Name: {name}",
                "Type: custom",
                f"Point: {poly['point']}",
            ]
            for vertex in poly.get("vertices", []):
                lines.append(str(vertex))
        elif "circle" in create_region_config:
            circ = create_region_config["circle"]
            spec_file = work_dir / f"{name}.circle.pts"
            lines = [
                f"Name: {name}",
                "Type: circle",
                f"Point: {circ['point']}",
                f"radius: {circ['radius']}",
            ]
        elif "ellipse" in create_region_config:
            ell = create_region_config["ellipse"]
            spec_file = work_dir / f"{name}.ellipse.pts"
            lines = [
                f"Name: {name}",
                "Type: ellipse",
                f"Point: {ell['point']}",
                f"Semi-major-axis: {ell['semi-major-axis']}",
                f"Semi-minor-axis: {ell['semi-minor-axis']}",
                f"Orientation-angle: {ell.get('orientation-angle', 0)}",
            ]
        elif "channel" in create_region_config:
            chan = create_region_config["channel"]
            spec_file = work_dir / f"{name}.channel.pts"
            lines = [
                f"Name: {name}",
                "Type: channel",
                f"Upper-lat: {chan['upper-lat']}",
                f"Lower-lat: {chan['lower-lat']}",
            ]
        else:
            raise ValueError(
                f"No recognized region type in limited_area config. "
                f"Expected one of: polygon, circle, ellipse, channel"
            )

        spec_file.write_text("\n".join(lines) + "\n")
        return spec_file

    # -------------------------------------------------------------------------
    # Private bash tasks — Metis
    # -------------------------------------------------------------------------

    @bash_task
    def _download_metis(self) -> str:
        """Download Metis and GKlib source code from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/metis/build/{self.metis_version}
            mkdir -p {self.work_dir}/metis/build/{self.metis_version}
            cd {self.work_dir}/metis/build/{self.metis_version}
            # Clone GKlib tools needed by metis
            git clone https://github.com/KarypisLab/GKlib.git
            # Fetch and untar metis tarball
            wget -T 30 -t 3 https://github.com/KarypisLab/METIS/archive/refs/tags/v{self.metis_version}.tar.gz
            tar -xzf v{self.metis_version}.tar.gz
            echo "Completed download at $(date)"
            """
        )

    @bash_task
    def _build_metis(self) -> str:
        """Build GKlib and Metis, install to prefix directory."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/metis/build/{self.metis_version}/GKlib
            make config prefix={self.work_dir}/metis/{self.metis_version}
            make install
            cd {self.work_dir}/metis/build/{self.metis_version}/METIS-{self.metis_version}
            make config prefix={self.work_dir}/metis/{self.metis_version}
            make install
            # rm -rf {self.work_dir}/metis/build
            echo "Completed build at $(date)"
            """
        )

    @bash_task
    def _partition_mesh(self, mesh_path: str, num_ranks: int) -> str:
        """Run gpmetis to partition a mesh file."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started partition at $(date)"
            echo "Executing on $(hostname)"
            echo "Partitioning {mesh_path} into {num_ranks} parts"
            {self.gpmetis_path} -minconn -contig -niter=200 {mesh_path} {num_ranks}
            echo "Completed partition at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Tools (hex_projection + grid_rotate)
    # -------------------------------------------------------------------------

    @bash_task
    def _clone_mpas_tools(self) -> str:
        """Clone MPAS-Tools repository from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS-Tools clone at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/MPAS-Tools
            mkdir -p {self.work_dir}
            cd {self.work_dir}
            git clone https://github.com/MPAS-Dev/MPAS-Tools.git
            cd MPAS-Tools
            git checkout {self.mpas_tools_version}
            echo "Completed MPAS-Tools clone at $(date)"
            """
        )

    @bash_task
    def _build_hex_projection(self) -> str:
        """Build hex_projection (project_hexes) from MPAS-Tools."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started hex_projection build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/MPAS-Tools/mesh_tools/hex_projection
            make clean || true
            make
            echo "Completed hex_projection build at $(date)"
            """
        )

    @bash_task
    def _build_grid_rotate(self) -> str:
        """Build grid_rotate from MPAS-Tools."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started grid_rotate build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/MPAS-Tools/mesh_tools/grid_rotate
            make clean || true
            make
            echo "Completed grid_rotate build at $(date)"
            """
        )

    @bash_task
    def _project_hexes(self, work_dir: str) -> str:
        """Run project_hexes in the given working directory."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started hex_projection at $(date)"
            echo "Executing on $(hostname)"
            cd {work_dir}
            {self.hex_projection_path}
            echo "Completed hex_projection at $(date)"
            """
        )

    @bash_task
    def _grid_rotate(self, work_dir: str, input_file: str, output_file: str) -> str:
        """Run grid_rotate to reposition a mesh."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started grid_rotate at $(date)"
            echo "Executing on $(hostname)"
            cd {work_dir}
            {self.grid_rotate_path} {input_file} {output_file}
            echo "Completed grid_rotate at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @bash_task
    def _install_limited_area(self) -> str:
        """Install MPAS-Limited-Area by cloning from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS-Limited-Area install at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/MPAS-Limited-Area
            mkdir -p {self.work_dir}
            cd {self.work_dir}
            git clone https://github.com/MPAS-Dev/MPAS-Limited-Area.git
            cd MPAS-Limited-Area
            git checkout {self.limited_area_version}
            echo "Completed MPAS-Limited-Area install at $(date)"
            """
        )

    @bash_task
    def _create_region(
        self, resolution: str, region_spec: str, output_dir: str,
    ) -> str:
        """Run create_region to cut a regional mesh from a global mesh."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started create_region at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}
            {self.create_region_path} {region_spec} \
                {self.mesh_data_dir}/x1.{self.resolution_cells[resolution]}.static.nc
            echo "Completed create_region at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private python tasks — Mesh operations
    # -------------------------------------------------------------------------

    @python_task
    def _download_global_mesh(self, resolution: str) -> None:
        """Download precomputed global mesh files for specified resolution."""
        import tarfile
        import urllib.request

        mesh_data_dir = Path(self.mesh_data_dir)
        mesh_data_dir.mkdir(parents=True, exist_ok=True)

        cells = self.resolution_cells[resolution]
        filename = f"x1.{cells}_static.tar.gz"
        url = f"{self.mesh_data_url}/{filename}"
        tarball = mesh_data_dir / filename

        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(path=mesh_data_dir)
        tarball.unlink()

    @python_task
    def _plot_mpas_mesh_png(self, file_path: str, resolution: Optional[str] = None) -> str:
        """Plot an MPAS mesh to a PNG image and return the output path."""
        import geoviews as gv
        import holoviews as hv
        import numpy as np
        from PIL import Image
        import uxarray as ux

        mesh_path = Path(file_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Could not find mesh file at: {mesh_path}")

        hv.extension("matplotlib")
        ux_ds = ux.open_dataset(
            str(mesh_path),
            str(mesh_path),
            grid_kwargs={"decode_times": False},
            decode_times=False,
        )
        has_boundary_mask = "bdyMaskCell" in ux_ds.data_vars

        if has_boundary_mask:
            mesh_plot = ux_ds["bdyMaskCell"].plot.polygons(
                cmap="Set1",
                colorbar=True,
                clabel="Boundary Mask Layer",
            )
        else:
            mesh_plot = ux_ds.uxgrid.plot.edges()

        # Keep map overlay best-effort so mesh plotting remains robust.
        try:
            coastlines = gv.feature.coastline.opts(edgecolor="black", linewidth=1.0)
            borders = gv.feature.borders.opts(edgecolor="dimgray", linewidth=0.8)
            states = gv.feature.states.opts(edgecolor="gray", linewidth=0.6)
            grid_lines = gv.feature.grid.opts(
                edgecolor="gray", linestyle="--", linewidth=0.5,
            )
            final_layout = mesh_plot * coastlines * borders * states * grid_lines
        except Exception:
            final_layout = mesh_plot

        lon = ux_ds.uxgrid.face_lon.values
        lat = ux_ds.uxgrid.face_lat.values
        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        pad_frac = 0.05
        lon_pad = max((lon_max - lon_min) * pad_frac, 0.1)
        lat_pad = max((lat_max - lat_min) * pad_frac, 0.1)
        final_layout = final_layout.opts(
            xlim=(lon_min - lon_pad, lon_max + lon_pad),
            ylim=(lat_min - lat_pad, lat_max + lat_pad),
        )

        plot_name = mesh_path.stem.removesuffix(".static")
        # Render at presentation-friendly resolution.
        title = f"MPAS Mesh: {plot_name}"
        if resolution:
            title = f"{title} ({resolution})"
        final_layout = final_layout.opts(fig_inches=9, title=title)

        output_path = mesh_path.with_name(f"{plot_name}.png")
        hv.save(final_layout, str(output_path), fmt="png", dpi=200)

        # Clamp final image size because backend/aspect settings can produce
        # unexpectedly large pixel dimensions.
        max_width_px = 2200
        max_height_px = 1400
        with Image.open(output_path) as img:
            width, height = img.size
            scale = min(max_width_px / width, max_height_px / height, 1.0)
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                resized.save(output_path)

        return str(output_path)


    # -------------------------------------------------------------------------
    # Public agent actions — Metis
    # -------------------------------------------------------------------------

    @agent_action
    async def install_metis(self) -> None:
        """Download and build Metis in one step with Parsl pipelining."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        download_future = self._download_metis(
            executor=["service"],
            stdout=str(self.log_dir / "metis_download.stdout"),
            stderr=str(self.log_dir / "metis_download.stderr"),
        )
        build_future = self._build_metis(
            executor=["compute"],
            stdout=str(self.log_dir / "metis_build.stdout"),
            stderr=str(self.log_dir / "metis_build.stderr"),
            inputs=[download_future],
        )

        try:
            await asyncio.wrap_future(download_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_download.stderr"
            )
        self.metis_source_dir = self.work_dir / "metis" / self.metis_version
        self.metis_downloaded = True

        try:
            await asyncio.wrap_future(build_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_build.stderr"
            )
        self.gpmetis_path = self.metis_source_dir / "bin" / "gpmetis"
        self.metis_built = True

    @agent_action
    async def partition_mesh(
        self, mesh_path: str, num_ranks: Union[int, List[int]],
    ) -> Union[str, Dict[int, str]]:
        """Partition MPAS mesh graph file for MPI execution.

        Parameters
        ----------
        mesh_path : str
            Path to graph.info file
        num_ranks : int or list[int]
            One rank count or multiple rank counts to partition for

        Returns
        -------
        str or dict
            If int input, returns one partition file path.
            If list input, returns {rank: partition_file_path}.
        """
        if not self.metis_built:
            raise RuntimeError(
                "Must call install_metis() or install() before partition_mesh()"
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)

        is_single = isinstance(num_ranks, int)
        ranks = [num_ranks] if is_single else sorted(set(num_ranks))

        for rank in ranks:
            if rank <= 0:
                raise ValueError(f"Rank counts must be > 0, got {rank}")

        futures = [
            self._partition_mesh(
                mesh_path,
                rank,
                executor=["compute"],
                stdout=str(self.log_dir / f"partition_mesh.{rank}.stdout"),
                stderr=str(self.log_dir / f"partition_mesh.{rank}.stderr"),
            )
            for rank in ranks
        ]

        results = await asyncio.gather(
            *[asyncio.wrap_future(future) for future in futures],
            return_exceptions=True,
        )
        for rank, result in zip(ranks, results):
            if isinstance(result, BashExitFailure):
                raise RuntimeError(
                    f"Metis partition failed for {rank} ranks "
                    f"(exit {result.exitcode}), "
                    f"see {self.log_dir}/partition_mesh.{rank}.stderr"
                )
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"Metis partition failed for {rank} ranks: {result}"
                )

        partition_paths = {rank: f"{mesh_path}.part.{rank}" for rank in ranks}
        if is_single:
            return partition_paths[ranks[0]]
        return partition_paths

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Tools
    # -------------------------------------------------------------------------

    @agent_action
    async def install_mpas_tools(self) -> None:
        """Clone MPAS-Tools and build hex_projection + grid_rotate."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        clone_future = self._clone_mpas_tools(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_tools_clone.stdout"),
            stderr=str(self.log_dir / "mpas_tools_clone.stderr"),
        )
        hex_future = self._build_hex_projection(
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection_build.stdout"),
            stderr=str(self.log_dir / "hex_projection_build.stderr"),
            inputs=[clone_future],
        )
        rotate_future = self._build_grid_rotate(
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate_build.stdout"),
            stderr=str(self.log_dir / "grid_rotate_build.stderr"),
            inputs=[clone_future],
        )

        try:
            await asyncio.wrap_future(clone_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Tools clone failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_tools_clone.stderr"
            )
        self.mpas_tools_dir = self.work_dir / "MPAS-Tools"

        try:
            await asyncio.wrap_future(hex_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection_build.stderr"
            )
        self.hex_projection_path = (
            self.mpas_tools_dir / "mesh_tools" / "hex_projection" / "project_hexes"
        )

        try:
            await asyncio.wrap_future(rotate_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate_build.stderr"
            )
        self.grid_rotate_path = (
            self.mpas_tools_dir / "mesh_tools" / "grid_rotate" / "grid_rotate"
        )

        self.mesh_tools_installed = True

    @agent_action
    async def project_hexes(
        self,
        resolution: str,
        project_hexes_config: Dict[str, Any],
        mesh_name: str,
    ) -> Dict[str, str]:
        """Generate a regional mesh using project_hexes (+ optional rotate).

        Parameters
        ----------
        resolution : str
            Cell spacing (e.g., "15km", "3km")
        project_hexes_config : dict
            project_hexes configuration block
        mesh_name : str
            Name used for working directory under mesh_data

        Returns
        -------
        dict
            {"mesh": path, "graph": path}
        """
        if not self.mesh_tools_installed:
            raise RuntimeError(
                "Must call install_mpas_tools() or install() before "
                "project_hexes()"
            )

        work_dir = self.mesh_data_dir / mesh_name
        work_dir.mkdir(parents=True, exist_ok=True)

        self._write_project_hexes_namelist(
            work_dir,
            self._parse_resolution_km(resolution),
            float(project_hexes_config["extent_x_km"]),
            float(project_hexes_config["extent_y_km"]),
            float(project_hexes_config["center_lat"]),
            float(project_hexes_config["center_lon"]),
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        future = self._project_hexes(
            str(work_dir),
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection.stdout"),
            stderr=str(self.log_dir / "hex_projection.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection.stderr"
            )

        mesh_path = str(work_dir / "mpas_hex_mesh.nc")
        rotation = float(project_hexes_config.get("rotation_degrees", 0.0))
        if rotation != 0.0:
            hex_mesh_path = mesh_path
            mesh_path = await self.grid_rotate(
                mesh_path,
                float(project_hexes_config["center_lat"]),
                float(project_hexes_config["center_lon"]),
                rotation,
            )
            Path(hex_mesh_path).unlink(missing_ok=True)

        # Rename outputs to use the mesh_name convention
        final_mesh_path = work_dir / f"{mesh_name}.nc"
        final_graph_path = work_dir / f"{mesh_name}.graph.info"
        Path(mesh_path).rename(final_mesh_path)
        (work_dir / "graph.info").rename(final_graph_path)

        return {
            "mesh": str(final_mesh_path),
            "graph": str(final_graph_path),
        }

    @agent_action
    async def grid_rotate(
        self,
        input_mesh: str,
        center_lat: float,
        center_lon: float,
        rotation_degrees: float = 0.0,
        original_lat: Optional[float] = None,
        original_lon: Optional[float] = None,
    ) -> str:
        """Rotate/reposition a mesh using grid_rotate.

        Parameters
        ----------
        input_mesh : str
            Path to input mesh NetCDF file
        center_lat : float
            Target center latitude (degrees)
        center_lon : float
            Target center longitude (degrees)
        rotation_degrees : float, optional
            Counter-clockwise rotation (degrees), by default 0.0
        original_lat : float, optional
            Original mesh center latitude. Defaults to center_lat.
        original_lon : float, optional
            Original mesh center longitude. Defaults to center_lon.

        Returns
        -------
        str
            Path to rotated mesh file
        """
        if not self.mesh_tools_installed:
            raise RuntimeError(
                "Must call install_mpas_tools() or install() before "
                "grid_rotate()"
            )

        work_dir = Path(input_mesh).parent
        self._write_rotate_namelist(
            work_dir,
            center_lat,
            center_lon,
            rotation_degrees,
            original_lat,
            original_lon,
        )

        output_file = str(work_dir / "rotated_mesh.nc")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._grid_rotate(
            str(work_dir),
            input_mesh,
            output_file,
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate.stdout"),
            stderr=str(self.log_dir / "grid_rotate.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate.stderr"
            )

        return output_file

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @agent_action
    async def install_limited_area(self) -> None:
        """Install MPAS-Limited-Area by cloning from GitHub."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._install_limited_area(
            executor=["service"],
            stdout=str(self.log_dir / "limited_area_install.stdout"),
            stderr=str(self.log_dir / "limited_area_install.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Limited-Area install failed (exit {e.exitcode}), "
                f"see {self.log_dir}/limited_area_install.stderr"
            )

        self.limited_area_source_dir = self.work_dir / "MPAS-Limited-Area"
        self.create_region_path = self.limited_area_source_dir / "create_region"
        self.limited_area_installed = True

    @agent_action
    async def create_region(
        self,
        resolution: str,
        create_region_config: Dict[str, Any],
        mesh_name: str,
    ) -> Dict[str, str]:
        """Run create_region to cut a regional mesh from a global mesh.

        Writes the region specification file from create_region_config,
        then invokes the create_region utility.

        Parameters
        ----------
        resolution : str
            Resolution of the parent global mesh (e.g., "120km")
        create_region_config : dict
            Region config containing one of: polygon, circle, ellipse, channel
        mesh_name : str
            Name used for the spec file and output directory

        Returns
        -------
        dict
            {"static": path, "graph": path}
        """
        if not self.limited_area_installed:
            raise RuntimeError(
                "Must call install_limited_area() or install() before "
                "create_region()"
            )
        if not self.global_mesh_downloaded:
            raise RuntimeError(
                "Must call download_global_mesh() before "
                "create_region()"
            )

        output_dir = self.mesh_data_dir / mesh_name
        spec_file = self._write_create_region_spec(
            output_dir, create_region_config, mesh_name,
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._create_region(
            resolution,
            str(spec_file),
            str(output_dir),
            executor=["compute"],
            stdout=str(self.log_dir / "create_region.stdout"),
            stderr=str(self.log_dir / "create_region.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"create_region failed (exit {e.exitcode}), "
                f"see {self.log_dir}/create_region.stderr"
            )

        return {
            "static": str(output_dir / f"{mesh_name}.static.nc"),
            "graph": str(output_dir / f"{mesh_name}.graph.info"),
        }


    # -------------------------------------------------------------------------
    # Public agent actions — Combined install
    # -------------------------------------------------------------------------

    @agent_action
    async def install(self) -> None:
        """Install all mesh tools: Metis, MPAS-Tools, and Limited-Area.

        Runs installations concurrently where possible.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metis: download then build (pipelined)
        metis_download_future = self._download_metis(
            executor=["service"],
            stdout=str(self.log_dir / "metis_download.stdout"),
            stderr=str(self.log_dir / "metis_download.stderr"),
        )
        metis_build_future = self._build_metis(
            executor=["compute"],
            stdout=str(self.log_dir / "metis_build.stdout"),
            stderr=str(self.log_dir / "metis_build.stderr"),
            inputs=[metis_download_future],
        )

        # MPAS-Tools: clone then build hex_projection + grid_rotate
        clone_mpas_tools_future = self._clone_mpas_tools(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_tools_clone.stdout"),
            stderr=str(self.log_dir / "mpas_tools_clone.stderr"),
        )
        hex_future = self._build_hex_projection(
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection_build.stdout"),
            stderr=str(self.log_dir / "hex_projection_build.stderr"),
            inputs=[clone_mpas_tools_future],
        )
        rotate_future = self._build_grid_rotate(
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate_build.stdout"),
            stderr=str(self.log_dir / "grid_rotate_build.stderr"),
            inputs=[clone_mpas_tools_future],
        )

        # MPAS-Limited-Area (just a clone, no build)
        limited_area_future = self._install_limited_area(
            executor=["service"],
            stdout=str(self.log_dir / "limited_area_install.stdout"),
            stderr=str(self.log_dir / "limited_area_install.stderr"),
        )

        # Await Metis
        try:
            await asyncio.wrap_future(metis_download_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_download.stderr"
            )
        self.metis_source_dir = self.work_dir / "metis" / self.metis_version
        self.metis_downloaded = True

        try:
            await asyncio.wrap_future(metis_build_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_build.stderr"
            )
        self.gpmetis_path = self.metis_source_dir / "bin" / "gpmetis"
        self.metis_built = True

        # Await MPAS-Tools
        try:
            await asyncio.wrap_future(clone_mpas_tools_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Tools clone failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_tools_clone.stderr"
            )
        self.mpas_tools_dir = self.work_dir / "MPAS-Tools"

        try:
            await asyncio.wrap_future(hex_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection_build.stderr"
            )
        self.hex_projection_path = (
            self.mpas_tools_dir / "mesh_tools" / "hex_projection" / "project_hexes"
        )

        try:
            await asyncio.wrap_future(rotate_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate_build.stderr"
            )
        self.grid_rotate_path = (
            self.mpas_tools_dir / "mesh_tools" / "grid_rotate" / "grid_rotate"
        )
        self.mesh_tools_installed = True

        # Await MPAS-Limited-Area
        try:
            await asyncio.wrap_future(limited_area_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Limited-Area install failed (exit {e.exitcode}), "
                f"see {self.log_dir}/limited_area_install.stderr"
            )
        self.limited_area_source_dir = self.work_dir / "MPAS-Limited-Area"
        self.create_region_path = self.limited_area_source_dir / "create_region"
        self.limited_area_installed = True

    # -------------------------------------------------------------------------
    # Public agent actions — Mesh operations
    # -------------------------------------------------------------------------

    @agent_action
    async def download_global_mesh(self, resolution: str) -> Dict[str, str]:
        """Download precomputed global mesh files for specified resolution.

        Parameters
        ----------
        resolution : str
            Mesh resolution (e.g., "120km", "60km")

        Returns
        -------
        dict
            {"static": path, "graph": path}
        """
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' not available for download. "
                f"Available: {list(self.resolution_cells.keys())}"
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_data_dir.mkdir(parents=True, exist_ok=True)

        future = self._download_global_mesh(
            resolution,
            executor=["service"],
        )
        try:
            await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(
                f"Global mesh download failed: {e}"
            )

        self.global_mesh_downloaded = True
        cells = self.resolution_cells[resolution]
        return {
            "static": str(self.mesh_data_dir / f"x1.{cells}.static.nc"),
            "graph": str(self.mesh_data_dir / f"x1.{cells}.graph.info"),
        }


    @agent_action
    async def plot_mesh(
        self,
        mesh_file: str,
        output_format: str = "png",
        resolution: Optional[str] = None,
    ) -> str:
        """Generate a plot of the mesh file and return the image path.

        Parameters
        ----------
        mesh_file : str
            Path to the mesh NetCDF file to plot
        output_format : str, optional
            Output image format, by default "png" (only PNG is supported)
        resolution : str, optional
            Resolution label for the plot title (e.g., "120km")

        Returns
        -------
        str
            Path to the generated plot image
        """
        if output_format.lower() != "png":
            raise ValueError("Only PNG output is supported")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._plot_mpas_mesh_png(
            mesh_file,
            resolution,
            executor=["compute"],
        )
        try:
            return await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(f"Mesh plotting failed: {e}") from e

    # -------------------------------------------------------------------------
    # Public agent actions — Top-level orchestrator
    # -------------------------------------------------------------------------

    @agent_action
    async def update_config(self, mesh_config: Dict[str, Any]) -> None:
        """Replace the current mesh configuration."""
        self.mesh_config = dict(mesh_config)

    @agent_action
    async def ensure_tools_installed(self) -> None:
        """Install only the tools required by the current mesh configuration.

        Can be called early to overlap tool installation with other agents.
        Also called automatically by ``generate_mesh()`` as a safety net.
        """
        mesh_config = self.mesh_config
        regional = mesh_config.get("regional")
        install_tasks = []

        needs_partition = bool(
            mesh_config.get("init_ranks") or mesh_config.get("forecast_ranks")
        )
        if needs_partition and not self.metis_built:
            install_tasks.append(self.install_metis())

        if regional is not None:
            if "project_hexes" in regional:
                if not self.mesh_tools_installed:
                    install_tasks.append(self.install_mpas_tools())
            elif "create_region" in regional:
                if not self.limited_area_installed:
                    install_tasks.append(self.install_limited_area())

        if install_tasks:
            await asyncio.gather(*install_tasks)


    @agent_action
    async def generate_mesh(self) -> Dict[str, Any]:
        """Generate a mesh using the stored configuration.

        Routes to the correct generation path and partitions the result.
        Required tools are installed automatically on demand.

        Returns
        -------
        dict
            - mesh: path to final mesh file (in mesh_name subdirectory)
            - graph: path to graph.info file (in mesh_name subdirectory)
            - plot: path to PNG mesh plot (if successfully generated)
            - plot_error: error message if plotting failed
            - partitions: dict of {num_ranks: partition_file_path}
        """
        mesh_config = self.mesh_config
        resolution = mesh_config.get("resolution", "120km")
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")
        init_ranks = mesh_config.get("init_ranks")
        forecast_ranks = mesh_config.get("forecast_ranks")
        regional = mesh_config.get("regional")

        await self.ensure_tools_installed()

        result: Dict[str, Any] = {"partitions": {}}

        if regional is None:
            # Global mesh — download precomputed from UCAR
            if resolution not in self.resolution_cells:
                raise ValueError(
                    f"Resolution '{resolution}' not supported. "
                    f"Available: {list(self.resolution_cells.keys())}"
                )
            paths = await self.download_global_mesh(resolution)

            # Symlink into mesh_name subdir for consistent naming
            mesh_dir = self.mesh_data_dir / mesh_name
            mesh_dir.mkdir(parents=True, exist_ok=True)
            static_link = mesh_dir / f"{mesh_name}.static.nc"
            graph_link = mesh_dir / f"{mesh_name}.graph.info"
            if not static_link.exists():
                static_link.symlink_to(Path(paths["static"]).resolve())
            if not graph_link.exists():
                graph_link.symlink_to(Path(paths["graph"]).resolve())
            result["mesh"] = str(static_link)
            result["graph"] = str(graph_link)

        elif "project_hexes" in regional:
            project_hexes_config = regional["project_hexes"]
            hex_result = await self.project_hexes(
                resolution,
                project_hexes_config,
                mesh_name,
            )
            result["mesh"] = hex_result["mesh"]
            result["graph"] = hex_result["graph"]

        elif "create_region" in regional:
            create_region_config = regional["create_region"]
            if not self.global_mesh_downloaded:
                await self.download_global_mesh(resolution)
            create_region_result = await self.create_region(
                resolution, create_region_config, mesh_name,
            )
            result["mesh"] = create_region_result["static"]
            result["graph"] = create_region_result["graph"]

        else:
            raise ValueError(
                "Unrecognized regional config. Expected 'project_hexes' "
                "or 'create_region' key under 'regional'."
            )


        # Partition for MPI if graph file available and ranks specified
        if result.get("graph"):
            ranks_list = []
            if init_ranks and init_ranks > 1:
                ranks_list.append(init_ranks)
            if forecast_ranks and forecast_ranks > 1:
                ranks_list.append(forecast_ranks)
            if ranks_list:
                partition_paths = await self.partition_mesh(
                    result["graph"], ranks_list,
                )
                result["partitions"].update(partition_paths)

        if result.get("mesh"):
            try:
                result["plot"] = await self.plot_mesh(
                    result["mesh"],
                    "png",
                    resolution,
                )
            except Exception as e:
                result["plot_error"] = str(e)

        return result
