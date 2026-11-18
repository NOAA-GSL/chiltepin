# SPDX-License-Identifier: Apache-2.0

"""Generate MPAS meshes."""

import asyncio
import logging
import math
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import parsl
from parsl.app.errors import BashExitFailure
from parsl.errors import NoDataFlowKernelError

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task, python_task

logger = logging.getLogger(__name__)

# A version such as "5.2.1". The download script adds the "v" that names the tag.
_VERSION_RE = re.compile(r"[0-9][0-9a-zA-Z._-]*")

# Names of the downloaded global mesh files, which a mesh name must not reuse
_SHARED_MESH_FILE_RE = re.compile(r"x1\.[0-9]+\.(static\.nc|graph\.info)")


# Most rank counts one partition_mesh call may request, since each one is a task
_MAX_PARTITION_RANKS = 16

# A resolution such as "7.5km", for meshes that are not downloaded
_RESOLUTION_KM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?km")

# Keys a mesh request may contain
_MESH_CONFIG_KEYS = frozenset(
    {"resolution", "name", "init_ranks", "forecast_ranks", "regional"}
)

# Methods a regional request may name, and the keys of a project_hexes block
_REGIONAL_METHODS = ("project_hexes", "create_region")
_PROJECT_HEXES_REQUIRED = ("center_lat", "center_lon", "extent_x_km", "extent_y_km")
_PROJECT_HEXES_KEYS = frozenset(_PROJECT_HEXES_REQUIRED + ("rotation_degrees",))


class _MeshBehavior:
    """Mesh behavior for generating MPAS meshes.

    Provides a top-level ``generate_mesh()`` action that downloads the requested
    global mesh, creates named links to its files, and optionally partitions its
    graph for MPI execution. Tools a request needs, such as Metis for
    partitioning, are installed on demand.
    """

    def __init__(
        self,
        work_dir: str,
        metis_version: str = "5.2.1",
    ):
        """Initialize the mesh behavior.

        Parameters
        ----------
        work_dir : str
            Directory where mesh tools will be installed and logs will be written
        metis_version : str, optional
            Metis version to install, by default "5.2.1"

        Raises
        ------
        ValueError
            If ``metis_version`` is not a plain version string.
        """
        if not isinstance(metis_version, str) or not _VERSION_RE.fullmatch(
            metis_version
        ):
            raise ValueError(f"Invalid metis_version: {metis_version!r}")

        # Tasks run elsewhere with their own working directory, so use an absolute path
        self.work_dir = Path(os.path.abspath(Path(work_dir).expanduser()))
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metis state
        self.metis_version = metis_version
        self.metis_built = False
        self.gpmetis_path: Optional[Path] = None

        # Serializes tool installs, so concurrent requests that need a tool wait
        # for one install instead of each starting their own
        self._install_lock = asyncio.Lock()

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

    def __getstate__(self) -> Dict[str, Any]:
        """Return picklable state for task submission.

        Task methods send ``self`` to workers. A lock with waiting requests cannot
        be pickled, and workers do not need it.
        """
        state = self.__dict__.copy()
        del state["_install_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state on a worker, with a fresh lock."""
        self.__dict__.update(state)
        self._install_lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _q(value: Any) -> str:
        """Shell-quote a value for safe interpolation into bash scripts."""
        return shlex.quote(str(value))

    @staticmethod
    def _resolve_mesh_data_dir(mesh_data_dir: str) -> Path:
        """Resolve and create the mesh data directory on the agent's shared filesystem."""
        resolved_dir = Path(mesh_data_dir).expanduser().resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        return resolved_dir

    @staticmethod
    def _remove_mesh_dir(mesh_dir: Path) -> None:
        """Remove a mesh directory and everything in it after a failed request.

        The request created the directory, so everything in it is its own. The
        links inside are removed without following them, so the shared downloaded
        mesh files they point to are left alone. A failure is logged and never
        raised, so cleanup cannot hide the error that caused it.
        """
        shutil.rmtree(mesh_dir, ignore_errors=True)
        if mesh_dir.exists() or mesh_dir.is_symlink():
            logger.warning(
                "Could not remove %s; remove it before requesting this mesh again",
                mesh_dir,
            )

    def _validate_mesh_config(self, mesh_config: Dict[str, Any]) -> None:
        """Check a mesh request configuration, as passed to ``generate_mesh()``.

        Every action that takes a mesh configuration calls this first, so
        invalid requests fail before any resource check, install or download.
        New keys and rules for them belong here.

        Raises
        ------
        ValueError
            If ``mesh_config`` is not a dict, has an unknown key, or has an
            invalid ``resolution``, ``name``, ``init_ranks``, ``forecast_ranks``
            or ``regional`` block. A name must not be that of a downloaded global
            mesh file.
        """
        if not isinstance(mesh_config, dict):
            raise ValueError(
                f"mesh_config must be a dict, got {type(mesh_config).__name__}"
            )
        unknown = sorted(str(k) for k in mesh_config if k not in _MESH_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown mesh_config keys: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(_MESH_CONFIG_KEYS))}"
            )
        regional = mesh_config.get("regional")
        if regional is not None:
            self._validate_regional(regional)
        resolution = mesh_config.get("resolution", "120km")
        if regional is None:
            # A global mesh is downloaded, so only the available ones can be used
            if (
                not isinstance(resolution, str)
                or resolution not in self.resolution_cells
            ):
                raise ValueError(
                    f"Resolution {resolution!r} not supported. "
                    f"Available: {list(self.resolution_cells.keys())}"
                )
        else:
            self._parse_resolution_km(resolution)
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")
        if (
            not isinstance(mesh_name, str)
            or mesh_name in ("", ".", "..")
            or Path(mesh_name).name != mesh_name
        ):
            raise ValueError("Mesh name must be a valid file name and not a path")
        if _SHARED_MESH_FILE_RE.fullmatch(mesh_name):
            raise ValueError(
                f"Mesh name {mesh_name!r} is the name of a downloaded global mesh file"
            )
        for key in ("init_ranks", "forecast_ranks"):
            value = mesh_config.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")

    @staticmethod
    def _parse_resolution_km(resolution: Any) -> float:
        """Return a resolution such as ``"7.5km"`` as a number of kilometers.

        Raises
        ------
        ValueError
            If it is not a positive, finite number of kilometers written as
            digits followed by ``km``.
        """
        if isinstance(resolution, str) and _RESOLUTION_KM_RE.fullmatch(resolution):
            kilometers = float(resolution[:-2])
            if math.isfinite(kilometers) and kilometers > 0:
                return kilometers
        raise ValueError(
            f"Resolution {resolution!r} is not valid. Use a positive number of "
            f"kilometers such as '15km' or '7.5km'"
        )

    @staticmethod
    def _number(config: Dict[str, Any], key: str, where: str) -> float:
        """Return a required or optional value as a finite number.

        Booleans and strings are rejected, since a mesh request must give real
        numbers.
        """
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{where}.{key} must be a finite number, got {value!r}")
        return float(value)

    @classmethod
    def _validate_regional(cls, regional: Any) -> None:
        """Check the ``regional`` block of a mesh request.

        Raises
        ------
        ValueError
            If the block is not a dict holding exactly one known method, names
            ``create_region``, which is not supported yet, or has an invalid
            ``project_hexes`` block.
        """
        if not isinstance(regional, dict):
            raise ValueError(f"regional must be a dict, got {type(regional).__name__}")
        unknown = sorted(str(k) for k in regional if k not in _REGIONAL_METHODS)
        if unknown:
            raise ValueError(
                f"Unknown regional keys: {', '.join(unknown)}. "
                f"Allowed: {', '.join(_REGIONAL_METHODS)}"
            )
        if len(regional) != 1:
            raise ValueError(
                "regional must contain exactly one of "
                + " or ".join(repr(method) for method in _REGIONAL_METHODS)
            )
        if "create_region" in regional:
            raise ValueError("regional 'create_region' is not supported yet")
        cls._validate_project_hexes(regional["project_hexes"])

    @classmethod
    def _validate_project_hexes(cls, config: Any) -> None:
        """Check a ``regional.project_hexes`` block.

        Raises
        ------
        ValueError
            If it is not a dict, is missing a required key, has an unknown key,
            or has a value that is not a finite number in its allowed range.
        """
        where = "regional.project_hexes"
        if not isinstance(config, dict):
            raise ValueError(f"{where} must be a dict, got {type(config).__name__}")
        unknown = sorted(str(k) for k in config if k not in _PROJECT_HEXES_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown {where} keys: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(_PROJECT_HEXES_KEYS))}"
            )
        missing = [key for key in _PROJECT_HEXES_REQUIRED if key not in config]
        if missing:
            raise ValueError(f"{where} is missing required keys: {', '.join(missing)}")
        if not -90 <= cls._number(config, "center_lat", where) <= 90:
            raise ValueError(f"{where}.center_lat must be between -90 and 90")
        if not -180 <= cls._number(config, "center_lon", where) <= 360:
            raise ValueError(f"{where}.center_lon must be between -180 and 360")
        for key in ("extent_x_km", "extent_y_km"):
            if cls._number(config, key, where) <= 0:
                raise ValueError(f"{where}.{key} must be greater than 0")
        if "rotation_degrees" in config:
            cls._number(config, "rotation_degrees", where)

    @staticmethod
    def _partition_ranks(mesh_config: Dict[str, Any]) -> List[int]:
        """Return the distinct rank counts above 1 requested by a mesh config.

        A single rank needs no partition. The config must already have been
        checked by ``_validate_mesh_config()``.
        """
        ranks: List[int] = []
        for key in ("init_ranks", "forecast_ranks"):
            value = mesh_config.get(key)
            if value is not None and value > 1 and value not in ranks:
                ranks.append(value)
        return ranks

    @staticmethod
    def _configured_resources() -> Optional[set]:
        """Return the names of the resources the workflow defines.

        Returns None if no workflow is loaded in this process, in which case
        the resources cannot be checked and the check is skipped.
        """
        try:
            return set(parsl.dfk().executors)
        except NoDataFlowKernelError:
            logger.warning("No workflow is loaded, so resources are not checked")
            return None

    def _require_resources(self, action: str, *needed: str) -> None:
        """Fail before any work starts if a needed resource is not defined.

        Checking first avoids starting a download, or leaving files behind, for
        work that cannot finish.

        Raises
        ------
        RuntimeError
            If the workflow does not define one of the resources.
        """
        configured = self._configured_resources()
        if configured is None:
            return
        missing = [name for name in needed if name not in configured]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            defined = sorted(name for name in configured if not name.startswith("_"))
            raise RuntimeError(
                f"{action} needs the resource {names}, which the agent's workflow "
                f"configuration does not define. Defined resources: {defined}"
            )

    @property
    def _gpmetis(self) -> Path:
        """Path of the gpmetis executable."""
        return self._metis_prefix / "bin" / "gpmetis"

    def _metis_usable(self) -> bool:
        """Whether a finished Metis install with a runnable gpmetis exists."""
        gpmetis = self._gpmetis
        return (
            (self._metis_prefix / ".installed").is_file()
            and gpmetis.is_file()
            # Some filesystems report every file as executable to os.access
            and bool(gpmetis.stat().st_mode & 0o111)
            and os.access(gpmetis, os.X_OK)
        )

    @property
    def _metis_prefix(self) -> Path:
        """Directory Metis is installed into."""
        return self.work_dir / "metis" / self.metis_version

    # -------------------------------------------------------------------------
    # Private bash tasks — Metis
    # -------------------------------------------------------------------------

    @bash_task
    def _download_metis(self) -> str:
        """Download Metis and GKlib source code from GitHub."""
        import textwrap

        build = f"{self.work_dir}/metis/build/{self.metis_version}"
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            build={self._q(build)}
            rm -rf "$build"
            mkdir -p "$build"
            cd "$build"
            # Clone GKlib tools needed by metis, giving up if the transfer stalls
            git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone https://github.com/KarypisLab/GKlib.git
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

        build = f"{self.work_dir}/metis/build/{self.metis_version}"
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started build at $(date)"
            echo "Executing on $(hostname)"
            build={self._q(build)}
            prefix={self._q(self._metis_prefix)}
            # Start from an empty prefix so a failed earlier install leaves nothing
            rm -rf "$prefix"
            cd "$build/GKlib"
            make config prefix="$prefix"
            make install
            cd "$build/METIS-{self.metis_version}"
            make config prefix="$prefix"
            make install
            # The sources are no longer needed. A failed build keeps them to debug.
            cd /
            # Failing to remove the sources, for example from busy files on a
            # network filesystem, must not fail a finished build
            rm -rf "$build" || true
            # Mark the install complete. The agent only uses a marked install.
            touch "$prefix/.installed"
            echo "Completed build at $(date)"
            """
        )

    @bash_task
    def _partition_mesh(self, mesh_path: str, num_ranks: int) -> str:
        """Run gpmetis to partition a mesh graph file."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started partition at $(date)"
            echo "Executing on $(hostname)"
            printf 'Partitioning %s into %s parts\\n' {self._q(mesh_path)} {int(num_ranks)}
            {self._q(self.gpmetis_path)} -minconn -contig -niter=200 {self._q(mesh_path)} {int(num_ranks)}
            echo "Completed partition at $(date)"
            """
        )

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
    # Public agent actions — Metis
    # -------------------------------------------------------------------------

    @agent_action
    async def install_metis(self) -> None:
        """Download and build Metis, unless it is already installed.

        Metis counts as installed once a build has finished in ``work_dir``, so a
        restarted agent reuses it. Requests that need Metis while it is being
        installed wait for that install. If it fails, each waiting request tries
        again in turn. Called automatically when a request needs Metis, but can
        be called earlier to have it ready.

        Raises
        ------
        RuntimeError
            If the ``service`` or ``compute`` resource is not defined, or the
            download or build fails. The message names the log file.
        """
        async with self._install_lock:
            # Check the install on disk each time, so a removed or damaged
            # install is rebuilt instead of failing every later partition
            if not self._metis_usable():
                self.metis_built = False
                self._require_resources("Installing Metis", "service", "compute")
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
                    ) from e
                except Exception as e:
                    raise RuntimeError(f"Metis download failed: {e}") from e
                try:
                    await asyncio.wrap_future(build_future)
                except BashExitFailure as e:
                    raise RuntimeError(
                        f"Metis build failed (exit {e.exitcode}), "
                        f"see {self.log_dir}/metis_build.stderr"
                    ) from e
                except Exception as e:
                    raise RuntimeError(f"Metis build failed: {e}") from e
                if not self._metis_usable():
                    raise RuntimeError(
                        f"Metis build finished but {self._gpmetis} is not usable, "
                        f"see {self.log_dir}/metis_build.stderr"
                    )
            self.gpmetis_path = self._gpmetis
            self.metis_built = True

    @agent_action
    async def partition_mesh(
        self,
        mesh_path: str,
        num_ranks: Union[int, List[int]],
    ) -> Union[str, Dict[int, str]]:
        """Partition an MPAS graph file for MPI execution with Metis.

        Installs Metis first if it is not already installed.

        Parameters
        ----------
        mesh_path : str
            Path to the graph.info file. A relative path or one starting with
            ``~`` is made absolute on the agent. Partition files are written
            next to it.
        num_ranks : int or list[int]
            One rank count, or several rank counts to partition for

        Returns
        -------
        str or dict
            A single partition file path if ``num_ranks`` is an int, or
            ``{rank: partition_file_path}`` if it is a list. The paths are
            absolute.

        Raises
        ------
        ValueError
            If ``mesh_path`` is empty, ``num_ranks`` is empty, any rank
            count is not an integer greater than 1, or more than 16 distinct
            rank counts are requested.
        RuntimeError
            If the ``compute`` resource is not defined, Metis cannot be
            installed, or a partition fails. The message names the log file.
        """
        if not isinstance(mesh_path, str) or not mesh_path:
            raise ValueError("mesh_path must be a non-empty string")
        # Tasks run elsewhere, and a path starting with "-" would look like an option
        mesh_path = os.path.abspath(os.path.expanduser(mesh_path))
        is_single = not isinstance(num_ranks, (list, tuple, set))
        requested = [num_ranks] if is_single else list(num_ranks)
        if not requested:
            raise ValueError("num_ranks must not be empty")
        for rank in requested:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 2:
                raise ValueError(
                    f"Rank counts must be integers greater than 1, got {rank!r}"
                )
        ranks = sorted(set(requested))
        if len(ranks) > _MAX_PARTITION_RANKS:
            raise ValueError(
                f"At most {_MAX_PARTITION_RANKS} distinct rank counts can be "
                f"partitioned in one call, got {len(ranks)}"
            )

        self._require_resources("Partitioning", "compute")
        await self.install_metis()

        # Each call gets its own logs, since concurrent calls can use the same ranks
        call_id = uuid.uuid4().hex[:8]
        logs = {
            rank: self.log_dir / f"partition_mesh.{rank}.{call_id}" for rank in ranks
        }
        futures = [
            self._partition_mesh(
                mesh_path,
                rank,
                executor=["compute"],
                stdout=f"{logs[rank]}.stdout",
                stderr=f"{logs[rank]}.stderr",
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
                    f"(exit {result.exitcode}), see {logs[rank]}.stderr"
                ) from result
            if isinstance(result, BaseException):
                raise RuntimeError(
                    f"Metis partition failed for {rank} ranks: {result}"
                ) from result

        partition_paths = {rank: f"{mesh_path}.part.{rank}" for rank in ranks}
        if is_single:
            return partition_paths[ranks[0]]
        return partition_paths

    @agent_action
    async def ensure_tools_installed(self, mesh_config: Dict[str, Any]) -> None:
        """Install the tools a mesh request needs.

        Can be called early to overlap tool installation with other work. Also
        called by ``generate_mesh()`` before it downloads the mesh, so a failed
        install does not waste a download.

        Parameters
        ----------
        mesh_config : dict
            Mesh request configuration, as passed to ``generate_mesh()``.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        self._validate_mesh_config(mesh_config)
        if self._partition_ranks(mesh_config):
            await self.install_metis()

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
        if not isinstance(resolution, str) or resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution {resolution!r} not available for download. "
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

        Tools the request needs, such as Metis for partitioning, are installed
        first if they are not already installed.

        Parameters
        ----------
        mesh_config : dict
            Optional ``resolution`` (default ``"120km"``) and ``name``
            (default ``"mesh_<resolution>"``). Optional ``init_ranks`` and
            ``forecast_ranks`` are positive integers giving MPI rank counts to
            partition the mesh graph for. A count of 1 needs no partition.
        mesh_data_dir : str
            Directory where this request's mesh files are written

        Returns
        -------
        dict
            - static: path to static mesh file (in mesh_name subdirectory)
            - graph: path to graph.info file (in mesh_name subdirectory)
            - partitions: dict of {num_ranks: partition_file_path} (in mesh_name
              subdirectory), present only if a partition was requested

        Raises
        ------
        FileExistsError
            If the named mesh directory already exists.
        ValueError
            If ``mesh_config`` is not a dict, or the name, resolution, or a rank
            count is invalid. A name must not be that of a downloaded global mesh
            file such as ``x1.2562.graph.info``.
        RuntimeError
            If a resource the request needs (``service``, plus ``compute`` for
            partitioning) is not defined, or a download, tool install, or
            partition fails.
        """
        self._validate_mesh_config(mesh_config)
        if mesh_config.get("regional") is not None:
            # Removed once regional meshes are generated
            raise RuntimeError("Regional meshes cannot be generated yet")
        ranks = self._partition_ranks(mesh_config)
        resolution = mesh_config.get("resolution", "120km")
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")

        result: Dict[str, Any] = {}

        # Global mesh — download precomputed from UCAR
        # Check resources before creating anything, so a request that cannot
        # finish downloads nothing and leaves nothing behind
        needed = ["service"] + (["compute"] if ranks else [])
        self._require_resources("This mesh request", *needed)
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)
        mesh_dir = request_mesh_data_dir / mesh_name
        mesh_dir.mkdir()
        complete = False
        try:
            # Install tools before downloading so a failure wastes no download
            await self.ensure_tools_installed(mesh_config)
            paths = await self.download_global_mesh(
                resolution, str(request_mesh_data_dir)
            )
            for key, suffix in (("static", "static.nc"), ("graph", "graph.info")):
                link = mesh_dir / f"{mesh_name}.{suffix}"
                link.symlink_to(Path(paths[key]).resolve())
                result[key] = str(link)
            if ranks:
                result["partitions"] = await self.partition_mesh(result["graph"], ranks)
            complete = True
            return result
        finally:
            # Remove the failed mesh completely so the name can be used again
            if not complete:
                self._remove_mesh_dir(mesh_dir)


@chiltepin_agent()
class MeshAgent(_MeshBehavior):
    """Deployable agent for downloading, naming and partitioning MPAS meshes.

    Parameters
    ----------
    work_dir : str
        Directory where mesh tools will be installed and agent logs will be written
    metis_version : str, optional
        Metis version to install for partitioning, by default "5.2.1"
    """
