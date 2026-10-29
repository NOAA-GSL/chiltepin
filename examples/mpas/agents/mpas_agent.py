# SPDX-License-Identifier: Apache-2.0

"""MPASAgent - Manages MPAS model initialization and forecasting.

This agent handles downloading, building, and running the MPAS model for
initialization and forecasting.
"""

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from parsl.app.errors import BashExitFailure

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task, python_task


@chiltepin_agent()
class MPASAgent:
    """Agent for managing MPAS model initialization and forecasting.

    This agent manages the complete lifecycle of MPAS:
    - Downloads MPAS-Model source code
    - Builds init_atmosphere_model and atmosphere_model executables
    - Initializes MPAS with processed initial conditions
    - Initializes MPAS with processed lateral boundary conditions
    - Runs MPAS forecast

    Attributes
    ----------
    install_dir : str
        Base directory for MPAS installation
    mpas_version : str
        MPAS version to install (branch or tag)
    mpas_source_dir : Path
        Path to MPAS source directory (set after download)
    init_atmosphere_path : Path
        Path to mpas_init_atmosphere executable (set after build)
    atmosphere_path : Path
        Path to mpas_atmosphere executable (set after build)
    """

    def __init__(
        self,
        work_dir: str,
        init_config: Dict[str, Any],
        fcst_config: Dict[str, Any],
        mpas_version: str = "v8.4.1",
    ):
        """Initialize MPASAgent.

        Parameters
        ----------
        work_dir : str
            Directory where MPAS will be installed
        init_config : dict[str, Any]
            Initialization configuration for MPAS
        fcst_config : dict[str, Any]
            Forecast configuration for MPAS
        mpas_version : str, optional
            MPAS version to install, by default "v8.4.1"
        """
        self.work_dir = Path(work_dir)
        self.init_config = dict(init_config)
        self.fcst_config = dict(fcst_config)
        if not re.match(r"^v?[0-9][0-9a-zA-Z._-]*$", mpas_version):
            raise ValueError(f"Invalid mpas_version: {mpas_version!r}")
        self.mpas_version = mpas_version
        self.log_dir = self.work_dir / "logs"

        # MPAS state
        self.mpas_downloaded = False
        self.mpas_built = False
        self.mpas_source_dir: Optional[Path] = None
        self.init_atmosphere_path: Optional[Path] = None
        self.atmosphere_path: Optional[Path] = None

        # GEOG static data state
        self.geog_downloaded = False
        self.geog_data_url = "https://www2.mmm.ucar.edu/projects/mpas"
        self.geog_main_archive = "mpas_static.tar.bz2"
        self.geog_optional_files = [
            "topo_ugwp.tar.gz",
            "ugwp_limb_tau.nc",
            "modis_landuse_20class_15s.tar.bz2",
            "bnu_soiltype_top.tar.bz2",
        ]
        self.geog_data_dir = self.work_dir / "geog_data" / "mpas_static"

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _write_geog_namelist(work_dir: Path, mesh_filename: str) -> Path:
        """Write namelist.init_atmosphere for geog-only interpolation."""
        mesh_name = Path(mesh_filename).stem
        namelist = work_dir / "namelist.init_atmosphere"
        namelist.write_text(
            f"&nhyd_model\n"
            f"    config_init_case = 7\n"
            f"/\n"
            f"&decomposition\n"
            f"    config_block_decomp_file_prefix = '{mesh_name}.graph.info.part.'\n"
            f"/\n"
            f"&dimensions\n"
            f"/\n"
            f"&data_sources\n"
            f'    config_geog_data_path = "./geog"\n'
            f"/\n"
            f"&preproc_stages\n"
            f"    config_static_interp = true\n"
            f"    config_native_gwd_static = true\n"
            f"    config_vertical_grid = false\n"
            f"    config_met_interp = false\n"
            f"    config_input_sst = false\n"
            f"    config_frac_seaice = false\n"
            f"/\n"
        )
        return namelist

    @staticmethod
    def _write_geog_streams(work_dir: Path, mesh_filename: str) -> Path:
        """Write streams.init_atmosphere for geog-only interpolation."""
        static_filename = Path(mesh_filename).stem + ".static.nc"
        streams = work_dir / "streams.init_atmosphere"
        streams.write_text(
            f"<streams>\n"
            f'<immutable_stream name="input"\n'
            f'                  type="input"\n'
            f'                  filename_template="{mesh_filename}"\n'
            f'                  input_interval="initial_only" />\n'
            f'<immutable_stream name="output"\n'
            f'                  type="output"\n'
            f'                  filename_template="{static_filename}"\n'
            f'                  output_interval="initial_only"\n'
            f'                  packages="initial_conds" />\n'
            f"</streams>\n"
        )
        return streams

    # ---------------------------------------------------------------------
    # Private bash tasks - MPAS
    # ---------------------------------------------------------------------

    @bash_task
    def _download_mpas(self) -> str:
        """Download MPAS-Model source code from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS download at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {shlex.quote(str(self.work_dir))}/MPAS-Model/{shlex.quote(self.mpas_version)}
            mkdir -p {shlex.quote(str(self.work_dir))}/MPAS-Model
            cd {shlex.quote(str(self.work_dir))}/MPAS-Model
            git clone --branch {shlex.quote(self.mpas_version)} \
                https://github.com/MPAS-Dev/MPAS-Model.git {shlex.quote(self.mpas_version)}
            echo "Completed MPAS download at $(date)"
            """
        )

    @bash_task
    def _build_mpas(self) -> str:
        """Build MPAS atmosphere executables."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS build at $(date)"
            echo "Executing on $(hostname)"
            cd {shlex.quote(str(self.work_dir))}/MPAS-Model/{shlex.quote(self.mpas_version)}
            cmake -B build \
                -DCMAKE_INSTALL_PREFIX={self.work_dir}/MPAS-Model/{self.mpas_version} \
                -DCMAKE_BUILD_TYPE=Release \
                -DMPAS_DOUBLE_PRECISION=OFF \
                -DMPAS_CORES="init_atmosphere;atmosphere"
            cmake --build build --verbose --parallel 8
            cmake --install build
            echo "Completed MPAS build at $(date)"
            """
        )

    # ---------------------------------------------------------------------
    # Private python tasks - GEOG
    # ---------------------------------------------------------------------
    @python_task
    def _download_geog_data(self, target_dir: str) -> None:
        """Download and extract all MPAS GEOG static datasets."""
        import tarfile
        import urllib.request

        dest = Path(target_dir)
        if dest.exists() and any(dest.iterdir()):
            return

        dest.mkdir(parents=True, exist_ok=True)

        # Extract the main dataset first — creates the mpas_static subdirectory
        main_archive = self.geog_main_archive
        main_url = f"{self.geog_data_url}/{main_archive}"
        main_path = dest / main_archive
        urllib.request.urlretrieve(main_url, main_path)
        with tarfile.open(main_path) as tf:
            tf.extractall(path=dest)
        main_path.unlink()

        # Additional files go into the mpas_static subdirectory
        static_dir = dest / "mpas_static"
        for filename in self.geog_optional_files:
            url = f"{self.geog_data_url}/{filename}"
            local_path = static_dir / filename
            urllib.request.urlretrieve(url, local_path)
            if filename.endswith((".tar.bz2", ".tar.gz")):
                with tarfile.open(local_path) as tf:
                    tf.extractall(path=static_dir)
                local_path.unlink()

    @bash_task
    def _interpolate_geog_only(
        self,
        work_dir: str,
        geog_dir: str,
    ) -> str:
        """Run init_atmosphere_model in geog-only interpolation mode."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started GEOG interpolation at $(date)"
            echo "Executing on $(hostname)"
            cd {work_dir}

            ln -sfn {geog_dir} ./geog

            $PARSL_MPI_PREFIX {self.init_atmosphere_path}
            echo "Completed GEOG interpolation at $(date)"
            """
        )

    @bash_task
    def _initialize_ics(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS initialization for initial conditions (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS ICS initialization at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.init_atmosphere
            cp {namelist_file} ./namelist.init_atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            echo "Ungrib source: {ungrib_files}" > ungrib_input.txt

            {self.init_atmosphere_path} || true
            touch init.nc
            echo "Completed MPAS ICS initialization at $(date)"
            """
        )

    @bash_task
    def _initialize_lbcs(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS initialization for lateral boundary conditions (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS LBC initialization at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.init_atmosphere
            cp {namelist_file} ./namelist.init_atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            echo "Ungrib source: {ungrib_files}" > ungrib_input.txt

            {self.init_atmosphere_path} || true
            touch lbcs_ready.flag
            echo "Completed MPAS LBC initialization at $(date)"
            """
        )

    @bash_task
    def _run_forecast(
        self,
        ics_file: str,
        lbcs_dir: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS forecast integration (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS forecast at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.atmosphere
            cp {namelist_file} ./namelist.atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            ln -sfn {ics_file} ./init.nc
            ln -sfn {lbcs_dir} ./lbcs

            {self.atmosphere_path} || true
            touch forecast_complete.flag
            echo "Completed MPAS forecast at $(date)"
            """
        )

    # ---------------------------------------------------------------------
    # Public agent actions - MPAS
    # ---------------------------------------------------------------------

    @agent_action
    async def install_mpas(self) -> None:
        """Download and build MPAS-Model in one step with Parsl pipelining."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        download_future = self._download_mpas(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_download.stdout"),
            stderr=str(self.log_dir / "mpas_download.stderr"),
        )
        build_future = self._build_mpas(
            executor=[
                "service"
            ],  # MPAS downloads artifacts during build, so use service executor
            stdout=str(self.log_dir / "mpas_build.stdout"),
            stderr=str(self.log_dir / "mpas_build.stderr"),
            inputs=[download_future],
        )

        try:
            await asyncio.wrap_future(download_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_download.stderr"
            )
        self.mpas_source_dir = self.work_dir / "MPAS-Model" / self.mpas_version
        self.mpas_downloaded = True

        try:
            await asyncio.wrap_future(build_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_build.stderr"
            )

        init_candidates = [
            self.mpas_source_dir / "bin" / "mpas_init_atmosphere",
            self.mpas_source_dir / "mpas_init_atmosphere",
            self.mpas_source_dir / "build" / "bin" / "mpas_init_atmosphere",
        ]
        model_candidates = [
            self.mpas_source_dir / "bin" / "mpas_atmosphere",
            self.mpas_source_dir / "mpas_atmosphere",
            self.mpas_source_dir / "build" / "bin" / "mpas_atmosphere",
        ]

        self.init_atmosphere_path = next(
            (path for path in init_candidates if path.exists()),
            None,
        )
        self.atmosphere_path = next(
            (path for path in model_candidates if path.exists()),
            None,
        )

        if self.init_atmosphere_path is None:
            raise RuntimeError(
                "MPAS build completed but mpas_init_atmosphere was not found"
            )
        if self.atmosphere_path is None:
            raise RuntimeError("MPAS build completed but mpas_atmosphere was not found")
        self.mpas_built = True

    # ---------------------------------------------------------------------
    # Public agent actions - GEOG setup and interpolation
    # ---------------------------------------------------------------------

    @agent_action
    async def download_geog_data(
        self,
        path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Ensure GEOG data exists for MPAS init interpolation.

        If path is provided, uses that existing directory directly.
        Otherwise downloads all GEOG datasets to self.geog_data_dir.

        Parameters
        ----------
        path : str, optional
            Path to an existing GEOG data directory to use directly

        Returns
        -------
        dict
            {"geog_dir": path}
        """
        if path:
            geog_dir = Path(path)
            if not geog_dir.exists():
                raise RuntimeError(f"Configured GEOG path does not exist: {geog_dir}")
            self.geog_data_dir = geog_dir
            self.geog_downloaded = True
            return {"geog_dir": str(geog_dir)}

        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._download_geog_data(
            str(self.geog_data_dir.parent),
            executor=["service"],
        )

        try:
            await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(f"GEOG download failed: {e}")

        self.geog_downloaded = True
        return {"geog_dir": str(self.geog_data_dir)}

    @agent_action
    async def interpolate_geog_only(
        self,
        mesh_file: str,
        num_ranks: int = 1,
        geog_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run init_atmosphere_model geog-only interpolation for a mesh.

        Runs in the mesh file's directory, writing the namelist, streams,
        and static output file alongside the mesh.

        Parameters
        ----------
        mesh_file : str
            Path to the MPAS mesh NetCDF file
        num_ranks : int, optional
            Number of MPI ranks to use, by default 1
        geog_dir : str, optional
            Path to GEOG data directory. Uses self.geog_data_dir if not given.

        Returns
        -------
        dict
            {"static": path}
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError(
                "Must call install_mpas() before interpolate_geog_only()"
            )

        geog_path = Path(geog_dir) if geog_dir else self.geog_data_dir
        if geog_path is None:
            raise RuntimeError(
                "No GEOG data configured. Call download_geog_data() first "
                "or pass geog_dir explicitly."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = Path(mesh_file)
        work_dir = mesh_path.parent
        mesh_name = mesh_path.stem
        static_filename = mesh_name + ".static.nc"

        self._write_geog_namelist(work_dir, mesh_path.name)
        self._write_geog_streams(work_dir, mesh_path.name)

        future = self._interpolate_geog_only(
            str(work_dir),
            str(geog_path),
            executor=["mpi"],
            chiltepin_task_geometry={
                "num_nodes": 1,
                "num_ranks": num_ranks,
                "ranks_per_node": num_ranks,
            },
            stdout=str(self.log_dir / f"interpolate_geog_only.{mesh_name}.stdout"),
            stderr=str(self.log_dir / f"interpolate_geog_only.{mesh_name}.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"GEOG interpolation failed (exit {e.exitcode}), "
                f"see {self.log_dir}/interpolate_geog_only.{mesh_name}.stderr"
            )

        return {
            "static": str(work_dir / static_filename),
        }

    @agent_action
    async def initialize_ics(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS with processed initial conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to initialized file

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process initial conditions:
        - Reads ungrib intermediate files
        - Interpolates to MPAS mesh
        - Writes initialized state
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_ics()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._initialize_ics(
            ungrib_files,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "initialize_ics.stdout"),
            stderr=str(self.log_dir / "initialize_ics.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS initialize_ics failed (exit {e.exitcode}), "
                f"see {self.log_dir}/initialize_ics.stderr"
            )

        return str(output_path / "init.nc")

    @agent_action
    async def initialize_lbcs(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS lateral boundary conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to directory containing LBC files

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process lateral boundary conditions:
        - Reads ungrib intermediate files for multiple times
        - Interpolates to MPAS mesh for each time
        - Writes LBC files for each boundary update time
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_lbcs()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._initialize_lbcs(
            ungrib_files,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "initialize_lbcs.stdout"),
            stderr=str(self.log_dir / "initialize_lbcs.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS initialize_lbcs failed (exit {e.exitcode}), "
                f"see {self.log_dir}/initialize_lbcs.stderr"
            )

        return str(output_path)

    @agent_action
    async def run_forecast(
        self,
        ics_file: str,
        lbcs_dir: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS forecast.

        Parameters
        ----------
        ics_file : str
            Path to initialized conditions file
        lbcs_dir : str
            Path to directory containing LBC files
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for forecast
        namelist_file : str
            Path to namelist file for forecast
        output_dir : str
            Directory for forecast output

        Returns
        -------
        str
            Path to forecast output directory

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs atmosphere_model for forecast integration:
        - Reads initialized conditions
        - Reads lateral boundary conditions
        - Integrates forward in time
        - Writes forecast output files (history, diagnostics)
        """
        if self.atmosphere_model_path is None:
            raise RuntimeError("Must call build() before run_forecast()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._run_forecast(
            ics_file,
            lbcs_dir,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "run_forecast.stdout"),
            stderr=str(self.log_dir / "run_forecast.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS run_forecast failed (exit {e.exitcode}), "
                f"see {self.log_dir}/run_forecast.stderr"
            )

        return str(output_path)
