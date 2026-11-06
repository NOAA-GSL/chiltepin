# SPDX-License-Identifier: Apache-2.0

"""Generate MPAS meshes."""

import asyncio
from pathlib import Path
from typing import Any, Dict

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import python_task


class _MeshBehavior:
    """Mesh behavior for generating MPAS meshes.

    Provides a top-level ``generate_mesh()`` action that downloads the requested
    global mesh and creates named links to its files.
    """

    def __init__(
        self,
        work_dir: str,
    ):
        """Initialize the mesh behavior.

        Parameters
        ----------
        work_dir : str
            Directory where mesh tools will be installed and logs will be written
        """
        self.work_dir = Path(work_dir)
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Mesh data
        self.mesh_data_url = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes"
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
    def _resolve_mesh_data_dir(mesh_data_dir: str) -> Path:
        """Resolve and create the mesh data directory on the agent's shared filesystem."""
        resolved_dir = Path(mesh_data_dir).expanduser().resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        return resolved_dir

    # -------------------------------------------------------------------------
    # Private python tasks — Mesh operations
    # -------------------------------------------------------------------------

    @python_task
    def _download_global_mesh(self, resolution: str, mesh_data_dir: str) -> None:
        """Reuse nonempty regular mesh files, or download and validate both."""
        import shutil
        import tarfile
        import tempfile
        import urllib.request
        from pathlib import Path

        directory = Path(mesh_data_dir)
        directory.mkdir(parents=True, exist_ok=True)

        cells = self.resolution_cells[resolution]
        filename = f"x1.{cells}_static.tar.gz"
        url = f"{self.mesh_data_url.rstrip('/')}/{filename}"
        expected = (
            f"x1.{cells}.static.nc",
            f"x1.{cells}.graph.info",
        )

        paths = [directory / name for name in expected]
        if all(
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0
            for path in paths
        ):
            return

        # Stage the download in a temporary directory until validated
        with tempfile.TemporaryDirectory(
            prefix=f".x1.{cells}-", dir=directory
        ) as temporary_dir:
            staging = Path(temporary_dir)
            tarball = staging / filename
            extracted = staging / "extracted"
            extracted.mkdir()

            with (
                urllib.request.urlopen(url, timeout=60) as response,
                tarball.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            with tarfile.open(tarball, "r:gz") as archive:
                archive.extractall(path=extracted, filter="data")

            for name in expected:
                path = extracted / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(
                        f"Mesh archive {filename!r} must contain "
                        f"a nonempty regular file {name!r}"
                    )

            for name in expected:
                (extracted / name).replace(directory / name)

    # -------------------------------------------------------------------------
    # Public agent actions — Mesh operations
    # -------------------------------------------------------------------------

    @agent_action
    async def download_global_mesh(
        self,
        resolution: str,
        mesh_data_dir: str,
    ) -> Dict[str, str]:
        """Download precomputed global mesh files for specified resolution.

        Parameters
        ----------
        resolution : str
            Mesh resolution (e.g., "120km", "60km")
        mesh_data_dir : str
            Shared directory for downloaded files. Relative paths and ``~`` are
            resolved on the agent before submitting the download task.

        Returns
        -------
        dict
            {"static": path, "graph": path}, with absolute paths.

        Raises
        ------
        ValueError
            If the resolution is unsupported.
        RuntimeError
            If the download task fails. The original exception is retained
            as the cause.
        """
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' not available for download. "
                f"Available: {list(self.resolution_cells.keys())}"
            )

        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)

        future = self._download_global_mesh(
            resolution,
            str(request_mesh_data_dir),
            executor=["service"],
        )
        try:
            await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError("Global mesh download failed") from e

        cells = self.resolution_cells[resolution]
        return {
            "static": str(request_mesh_data_dir / f"x1.{cells}.static.nc"),
            "graph": str(request_mesh_data_dir / f"x1.{cells}.graph.info"),
        }

    @agent_action
    async def generate_mesh(
        self,
        mesh_config: Dict[str, Any],
        mesh_data_dir: str,
    ) -> Dict[str, Any]:
        """Generate requested mesh.

        Parameters
        ----------
        mesh_config : dict
            Optional ``resolution`` (default ``"120km"``) and ``name``
            (default ``"mesh_<resolution>"``).
        mesh_data_dir : str
            Directory where this request's mesh files are written

        Returns
        -------
        dict
            - mesh: path to final mesh file (in mesh_name subdirectory)
            - graph: path to graph.info file (in mesh_name subdirectory)
        """
        resolution = mesh_config.get("resolution", "120km")
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")
        if (
            not isinstance(mesh_name, str)
            or mesh_name in ("", ".", "..")
            or Path(mesh_name).name != mesh_name
        ):
            raise ValueError("Mesh name must be a nonempty directory name, not a path")

        result: Dict[str, Any] = {}

        # Global mesh — download precomputed from UCAR
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' not supported. "
                f"Available: {list(self.resolution_cells.keys())}"
            )
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)
        paths = await self.download_global_mesh(resolution, str(request_mesh_data_dir))

        # Symlink into mesh_name subdir for consistent naming
        mesh_dir = request_mesh_data_dir / mesh_name
        mesh_dir.mkdir(parents=True, exist_ok=True)
        static_link = mesh_dir / f"{mesh_name}.static.nc"
        graph_link = mesh_dir / f"{mesh_name}.graph.info"
        static_link.unlink(missing_ok=True)
        static_link.symlink_to(Path(paths["static"]).resolve())
        graph_link.unlink(missing_ok=True)
        graph_link.symlink_to(Path(paths["graph"]).resolve())
        result["mesh"] = str(static_link)
        result["graph"] = str(graph_link)

        return result


@chiltepin_agent()
class MeshAgent(_MeshBehavior):
    """Deployable agent for downloading and naming precomputed global meshes.

    Parameters
    ----------
    work_dir : str
        Directory where mesh tools will be installed and agent logs will be written
    """
