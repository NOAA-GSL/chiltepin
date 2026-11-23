# SPDX-License-Identifier: Apache-2.0

"""Generate MPAS meshes."""

import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import parsl
from parsl.app.errors import BashExitFailure
from parsl.errors import NoDataFlowKernelError

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.agents.mpas._geo_lookup import BOUNDARY_LAYERS
from chiltepin.agents.mpas._plot_mesh import (
    EXIT_MISSING_MAP,
    EXIT_VIEW_OF_REGIONAL_MESH,
    MAP_LAYERS,
    REQUIRED_PARTS,
    layer_complete,
    layer_path,
)
from chiltepin.agents.mpas._region_shapes import (
    lambert_apex_km,
    project_hexes_reach_km,
    standard_parallel,
)
from chiltepin.tasks import bash_task, python_task

logger = logging.getLogger(__name__)

# A version such as "5.2.1". The download script adds the "v" that names the tag.
_VERSION_RE = re.compile(r"[0-9][0-9a-zA-Z._-]*")

# Names of the downloaded global mesh files, which a mesh name must not reuse
_SHARED_MESH_FILE_RE = re.compile(r"x1\.[0-9]+\.(static\.nc|graph\.info)")


# Longest prompt a prompt action takes, in characters
_MAX_PROMPT_CHARS = 10_000
# Largest margin around a prompt's region, in km
_MAX_BUFFER_KM = 5000.0
# Margin around a prompt's region unless another is given, in km
_DEFAULT_BUFFER_KM = 50.0
# A reasoning effort litellm passes to a model, such as "none" or "low"
_REASONING_EFFORT_RE = re.compile(r"[a-z]{1,16}")
# A contact for Nominatim: an email address, without spaces or control characters
_CONTACT_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
# Most rank counts one partition_mesh call may request, since each one is a task
_MAX_PARTITION_RANKS = 16

# A resolution such as "7.5km", for meshes that are not downloaded
_RESOLUTION_KM_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?km")

# Longest mesh name in bytes. Files are named after the mesh, with a suffix such
# as ".graph.info.part.<ranks>", and a file name can be at most 255 bytes long.
_MAX_MESH_NAME_BYTES = 200

# Suffix of the record a finished mesh keeps of the request that made it,
# written last, so a mesh without one is not finished
_MESH_RECORD_SUFFIX = ".mesh.json"
# Keys a mesh request may contain
_MESH_CONFIG_KEYS = frozenset(
    {"resolution", "name", "init_ranks", "forecast_ranks", "regional", "plot"}
)
# Keys a mesh request made from a prompt may contain. The prompt gives the
# resolution and the region, and a name unless one is given.
_PROMPT_CONFIG_KEYS = frozenset(
    {
        "prompt",
        "model",
        "name",
        "init_ranks",
        "forecast_ranks",
        "plot",
        "buffer_km",
        "reasoning_effort",
    }
)

# Methods a regional request may name, and the keys of a project_hexes block
_REGIONAL_METHODS = ("project_hexes", "create_region")
_PROJECT_HEXES_REQUIRED = ("center_lat", "center_lon", "extent_x_km", "extent_y_km")
_PROJECT_HEXES_KEYS = frozenset(_PROJECT_HEXES_REQUIRED + ("rotation_degrees",))

# Fixed names in a create_region mesh directory. The points file always names
# the region "region", so the mesh name never reaches the tool's parser, and the
# parent is linked under a name that tells the tool it is a static file.
_REGION_POINTS = "region.pts"
_REGION_PARENT = "parent.static.nc"
_REGION_STATIC = "region.static.nc"
_REGION_GRAPH = "region.graph.info"

# Largest radius or semi-axis create_region can use: half the circumference of
# the sphere it measures on, whose radius is 6371229 m. A larger one reaches
# past the opposite side of the globe.
_MAX_REGION_SIZE_KM = math.pi * 6371.229

# Shapes a create_region block may hold, with their required and optional keys
_CREATE_REGION_SHAPES = {
    "circle": (("center_lat", "center_lon", "radius_km"), ()),
    "ellipse": (
        ("center_lat", "center_lon", "semi_major_axis_km", "semi_minor_axis_km"),
        ("orientation_degrees",),
    ),
    "polygon": (("vertices", "inside_lat", "inside_lon"), ()),
    "channel": (("upper_lat", "lower_lat"), ()),
}


class _MeshBehavior:
    """Mesh behavior for generating MPAS meshes.

    Provides a top-level ``generate_mesh()`` action that either downloads the
    requested global mesh and creates named links to its files, or generates a
    regional mesh, with the MPAS-Tools ``project_hexes`` program or by cutting
    it from a global mesh with the MPAS-Limited-Area ``create_region`` program.
    It can also partition the mesh graph for MPI execution. Tools a request
    needs, such as Metis for partitioning, are installed on demand.
    """

    def __init__(
        self,
        work_dir: str,
        metis_version: str = "5.2.1",
        mpas_tools_version: str = "2.0.0",
        limited_area_version: str = "2.2",
        nominatim_contact: Optional[str] = None,
    ):
        """Initialize the mesh behavior.

        Parameters
        ----------
        work_dir : str
            Directory where mesh tools will be installed and logs will be written
        metis_version : str, optional
            Metis version to install, by default "5.2.1"
        mpas_tools_version : str, optional
            MPAS-Tools tag to install, by default "2.0.0"
        limited_area_version : str, optional
            MPAS-Limited-Area version to install, by default "2.2". The tag is
            this version with a ``v`` added, such as ``v2.2``.
        nominatim_contact : str, optional
            Email address of whoever runs the agent, sent to OpenStreetMap's
            Nominatim service when a prompt names a place Natural Earth lacks,
            such as a city. Without it, such places are not looked up.

        Raises
        ------
        ValueError
            If ``metis_version``, ``mpas_tools_version`` or
            ``limited_area_version`` is not a plain version string, or
            ``nominatim_contact`` is not an email address.
        """
        for name, version in (
            ("metis_version", metis_version),
            ("mpas_tools_version", mpas_tools_version),
            ("limited_area_version", limited_area_version),
        ):
            if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
                raise ValueError(f"Invalid {name}: {version!r}")
        if nominatim_contact is not None and (
            not isinstance(nominatim_contact, str)
            or len(nominatim_contact) > 254
            or not _CONTACT_RE.fullmatch(nominatim_contact)
        ):
            raise ValueError("nominatim_contact must be an email address")

        # Tasks run elsewhere with their own working directory, so use an absolute path
        self.work_dir = Path(os.path.abspath(Path(work_dir).expanduser()))
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metis state
        self.metis_version = metis_version
        self.metis_built = False

        # MPAS-Tools state (the project_hexes and grid_rotate programs)
        self.mpas_tools_version = mpas_tools_version
        self.mpas_tools_built = False

        # MPAS-Limited-Area state (the create_region program)
        self.limited_area_version = limited_area_version
        self.limited_area_built = False

        # Plotting state (the map layers plots use)
        self.plotting_built = False

        # Prompt state (the boundaries places are looked up in)
        self.nominatim_contact = nominatim_contact
        self.prompting_built = False

        # Serializes tool installs, so concurrent requests that need a tool wait
        # for one install instead of each starting their own
        self._install_lock = asyncio.Lock()

        # Serializes downloads of each global mesh to each directory, so a
        # request never replaces a mesh file another one is reading
        self._download_locks: Dict[Any, asyncio.Lock] = {}

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
        be pickled, and workers do not need the locks.
        """
        state = self.__dict__.copy()
        del state["_install_lock"]
        del state["_download_locks"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state on a worker, with fresh locks."""
        self.__dict__.update(state)
        self._install_lock = asyncio.Lock()
        self._download_locks = {}

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

    @staticmethod
    def _orientation(block: Dict[str, Any]) -> float:
        """Return an ellipse block's orientation within one turn, by default 0.

        A tiny negative angle gives exactly 360.0 from %, which is the same
        direction as 0.
        """
        orientation = block.get("orientation_degrees", 0) % 360
        return 0.0 if orientation >= 360 else orientation

    def _mesh_identity(self, mesh_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return what makes a validated request's mesh: what an existing one must match.

        That is the resolution, the ``regional`` block, and the version of the
        tool that makes a regional mesh. Rank counts and ``plot`` are not part
        of it, since they add to a mesh without changing it. Optional numbers
        are filled in with their defaults, and an ellipse's orientation is
        written within one turn, as the mesh is made with them, so a request
        that leaves one out matches one that gives it.
        """
        regional = mesh_config.get("regional")
        # As it is read back, so the comparison does not depend on, for
        # example, tuples given for lists
        regional = json.loads(json.dumps(regional))
        if regional is not None and "project_hexes" in regional:
            regional["project_hexes"].setdefault("rotation_degrees", 0.0)
        if regional is not None and "ellipse" in regional.get("create_region", {}):
            block = regional["create_region"]["ellipse"]
            # As the points file writes it
            block["orientation_degrees"] = self._orientation(block)
        identity: Dict[str, Any] = {
            "resolution": mesh_config.get("resolution", "120km"),
            "regional": regional,
        }
        if regional is not None and "project_hexes" in regional:
            identity["mpas_tools_version"] = self.mpas_tools_version
        if regional is not None and "create_region" in regional:
            identity["limited_area_version"] = self.limited_area_version
        return identity

    @staticmethod
    def _mesh_record_path(mesh_dir: Path, mesh_name: str) -> Path:
        """Return the path of a mesh's record, ``<name>.mesh.json``."""
        return mesh_dir / f"{mesh_name}{_MESH_RECORD_SUFFIX}"

    @classmethod
    def _write_mesh_record(
        cls,
        mesh_dir: Path,
        mesh_name: str,
        identity: Dict[str, Any],
        partitions: Dict[int, str],
    ) -> None:
        """Write a mesh's record, with the Metis version of each partition.

        It is written to a temporary file and renamed, so a record is never
        partly written.
        """
        record = {
            "mesh": identity,
            "partitions": {str(rank): version for rank, version in partitions.items()},
        }
        path = cls._mesh_record_path(mesh_dir, mesh_name)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
        try:
            temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def _read_mesh_record(cls, mesh_dir: Path, mesh_name: str) -> Optional[Dict]:
        """Return a finished mesh's record, or None if it has no valid one.

        Only a real directory counts, not a link to one, so a request can never
        take a directory elsewhere for its mesh.
        """
        if mesh_dir.is_symlink() or not mesh_dir.is_dir():
            return None
        try:
            record = json.loads(cls._mesh_record_path(mesh_dir, mesh_name).read_text())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("mesh"), dict)
            or not isinstance(record.get("partitions"), dict)
            or not all(
                re.fullmatch(r"[0-9]{1,9}", rank) and isinstance(version, str)
                for rank, version in record["partitions"].items()
            )
        ):
            return None
        return record

    @staticmethod
    def _mesh_differences(made: Dict[str, Any], requested: Dict[str, Any]) -> List[str]:
        """Return how a recorded mesh differs from a requested one, in words."""

        def kind(identity):
            regional = identity.get("regional")
            if regional is None:
                return "a global mesh"
            ((method, block),) = regional.items()
            if method == "create_region":
                ((shape, _),) = block.items()
                return f"a create_region {shape}"
            return "a project_hexes mesh"

        differences = []
        if made.get("resolution") != requested["resolution"]:
            differences.append(
                f"it is {made.get('resolution')}, not {requested['resolution']}"
            )
        if kind(made) != kind(requested):
            differences.append(f"it is {kind(made)}, not {kind(requested)}")
        elif made.get("regional") != requested["regional"]:

            def numbers(identity):
                ((method, block),) = identity["regional"].items()
                if method == "create_region":
                    ((_, block),) = block.items()
                return block

            old, new = numbers(made), numbers(requested)
            keys = sorted(key for key in {*old, *new} if old.get(key) != new.get(key))
            # Keys set to null in one and left out of the other differ only in
            # how they are written
            differences.append(
                f"its region differs in {', '.join(keys)}"
                if keys
                else "its region differs"
            )
        tools = (
            ("mpas_tools_version", "MPAS-Tools"),
            ("limited_area_version", "MPAS-Limited-Area"),
        )
        for key, tool in tools:
            if key in requested and made.get(key) != requested[key]:
                differences.append(
                    f"it was made with {tool} {made.get(key)}, not {requested[key]}"
                )
        return differences

    def _validate_mesh_config(self, mesh_config: Dict[str, Any]) -> None:
        """Check a mesh request configuration, as passed to ``generate_mesh()``.

        Every action that takes a mesh configuration calls this first, so
        invalid requests fail before any resource check, install or download.
        New keys and rules for them belong here.

        A request with a ``prompt`` is checked as the prompt's own request: it
        needs a ``model``, and the prompt gives its resolution and region.

        Raises
        ------
        ValueError
            If ``mesh_config`` is not a dict, has an unknown key, or has an
            invalid ``resolution``, ``name``, ``init_ranks``, ``forecast_ranks``,
            ``regional`` block or ``plot``. A name must be at most 200 bytes long, must
            not contain control characters, and must not be that of a downloaded
            global mesh file. A ``project_hexes`` mesh must not reach a pole. A
            request with a ``prompt`` must not have a ``resolution`` or
            ``regional`` block, and must have a valid ``model``, ``buffer_km``
            and ``reasoning_effort``.
        """
        if not isinstance(mesh_config, dict):
            raise ValueError(
                f"mesh_config must be a dict, got {type(mesh_config).__name__}"
            )
        if "prompt" in mesh_config:
            self._validate_prompt_config(mesh_config)
            return
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
        if regional is None or "create_region" in regional:
            # A global mesh is downloaded, and create_region cuts a region from
            # one, so only the available resolutions can be used
            if (
                not isinstance(resolution, str)
                or resolution not in self.resolution_cells
            ):
                raise ValueError(
                    f"Resolution {resolution!r} not supported. "
                    f"Available: {list(self.resolution_cells.keys())}"
                )
        else:
            spacing = self._parse_resolution_km(resolution)
            self._check_project_hexes_reach(regional["project_hexes"], spacing)
        self._validate_name(mesh_config.get("name", f"mesh_{resolution}"))
        self._validate_ranks_and_plot(mesh_config)

    @staticmethod
    def _validate_name(mesh_name: Any) -> None:
        """Check a request's mesh name.

        Raises
        ------
        ValueError
            As ``_validate_mesh_config()`` raises for a name.
        """
        if (
            not isinstance(mesh_name, str)
            or mesh_name in ("", ".", "..")
            or Path(mesh_name).name != mesh_name
        ):
            raise ValueError("Mesh name must be a valid file name and not a path")
        if any(unicodedata.category(char) == "Cc" for char in mesh_name):
            raise ValueError("Mesh name must not contain control characters")
        try:
            name_bytes = len(os.fsencode(mesh_name))
        except UnicodeEncodeError:
            raise ValueError("Mesh name must be a valid file name") from None
        if name_bytes > _MAX_MESH_NAME_BYTES:
            raise ValueError(
                f"Mesh name must be at most {_MAX_MESH_NAME_BYTES} bytes long, "
                f"got {name_bytes}"
            )
        if _SHARED_MESH_FILE_RE.fullmatch(mesh_name):
            raise ValueError(
                f"Mesh name {mesh_name!r} is the name of a downloaded global mesh file"
            )

    @staticmethod
    def _validate_ranks_and_plot(mesh_config: Dict[str, Any]) -> None:
        """Check a request's rank counts and ``plot``.

        Raises
        ------
        ValueError
            As ``_validate_mesh_config()`` raises for these.
        """
        for key in ("init_ranks", "forecast_ranks"):
            value = mesh_config.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")
        plot = mesh_config.get("plot", False)
        if not isinstance(plot, bool):
            raise ValueError(f"plot must be True or False, got {plot!r}")

    def _validate_prompt_config(self, mesh_config: Dict[str, Any]) -> None:
        """Check a mesh request made from a prompt.

        Raises
        ------
        ValueError
            As ``_validate_mesh_config()`` raises for a request with a prompt.
        """
        decided = sorted(k for k in ("resolution", "regional") if k in mesh_config)
        if decided:
            raise ValueError(
                f"A mesh_config with a prompt cannot have {' or '.join(decided)}, "
                "which the prompt gives. Put it in the prompt, or leave out the "
                "prompt."
            )
        unknown = sorted(str(k) for k in mesh_config if k not in _PROMPT_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown mesh_config keys with a prompt: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(_PROMPT_CONFIG_KEYS))}"
            )
        if "model" not in mesh_config:
            raise ValueError(
                "A mesh_config with a prompt needs a model, a litellm model "
                "string such as 'bedrock/us.anthropic.claude-sonnet-5-5'"
            )
        self._validate_prompt(
            mesh_config["prompt"],
            mesh_config["model"],
            mesh_config.get("buffer_km", _DEFAULT_BUFFER_KM),
            mesh_config.get("reasoning_effort"),
        )
        if "name" in mesh_config:
            self._validate_name(mesh_config["name"])
        self._validate_ranks_and_plot(mesh_config)

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

    @classmethod
    def _number(cls, config: Dict[str, Any], key: str, where: str) -> float:
        """Return a required or optional value as a finite number.

        Booleans and strings are rejected, since a mesh request must give real
        numbers.
        """
        return cls._finite(config[key], f"{where}.{key}")

    @staticmethod
    def _finite(value: Any, label: str) -> float:
        """Return a value as a finite number, naming it ``label`` if it is not."""
        number = math.nan
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                number = float(value)
            except OverflowError:
                # An integer too large for a float is not a usable number
                pass
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number, got {value!r}")
        return number

    @classmethod
    def _validate_regional(cls, regional: Any) -> None:
        """Check the ``regional`` block of a mesh request.

        Raises
        ------
        ValueError
            If the block is not a dict holding exactly one known method, or
            that method's block is invalid.
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
            cls._validate_create_region(regional["create_region"])
        else:
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
        cls._latitude(config["center_lat"], f"{where}.center_lat")
        cls._longitude(config["center_lon"], f"{where}.center_lon")
        for key in ("extent_x_km", "extent_y_km"):
            cls._positive(config[key], f"{where}.{key}")
        if "rotation_degrees" in config:
            cls._number(config, "rotation_degrees", where)

    @staticmethod
    def _check_project_hexes_reach(config: Dict[str, Any], spacing_km: float) -> None:
        """Check that a valid ``project_hexes`` block's mesh does not reach a pole.

        project_hexes' cone has its apex at the pole, and cells at or past it
        are misplaced. The mesh reaches past ``extent_y_km`` by its boundary
        layers, whose size is set by the cell spacing.

        Raises
        ------
        ValueError
            If the mesh reaches the pole.
        """
        center_lat = float(config["center_lat"])
        apex = lambert_apex_km(center_lat)
        reach = project_hexes_reach_km(float(config["extent_y_km"]), spacing_km)
        if reach >= apex:
            limit = 2 * (apex - project_hexes_reach_km(0.0, spacing_km))
            raise ValueError(
                "regional.project_hexes reaches the pole, which project_hexes "
                f"cannot make a mesh around. At center_lat {center_lat} with "
                f"{spacing_km:g} km cells, extent_y_km must be less than "
                f"{max(limit, 0.0):.1f}. Use create_region instead."
            )

    @classmethod
    def _latitude(cls, value: Any, label: str) -> float:
        """Return a latitude, which must be from -90 to 90 degrees."""
        number = cls._finite(value, label)
        if not -90 <= number <= 90:
            raise ValueError(f"{label} must be between -90 and 90")
        return number

    @classmethod
    def _longitude(cls, value: Any, label: str) -> float:
        """Return a longitude, which must be from -180 to 360 degrees."""
        number = cls._finite(value, label)
        if not -180 <= number <= 360:
            raise ValueError(f"{label} must be between -180 and 360")
        return number

    @classmethod
    def _positive(cls, value: Any, label: str) -> float:
        """Return a size, which must be greater than 0."""
        number = cls._finite(value, label)
        if number <= 0:
            raise ValueError(f"{label} must be greater than 0")
        return number

    @classmethod
    def _validate_plot_view(
        cls,
        title: Any,
        zoom: Any,
        center_lat: Any,
        center_lon: Any,
    ) -> None:
        """Check the optional view settings of ``plot_mesh()``.

        ``zoom`` must be a finite number greater than 0, and ``center_lat`` and
        ``center_lon`` must be given together, as a latitude and a longitude.

        Raises
        ------
        ValueError
            If any of them is invalid.
        """
        if title is not None and not isinstance(title, str):
            raise ValueError(f"title must be a string, got {type(title).__name__}")
        if zoom is not None:
            cls._positive(zoom, "zoom")
        if (center_lat is None) != (center_lon is None):
            raise ValueError("center_lat and center_lon must be given together")
        if center_lat is not None:
            cls._latitude(center_lat, "center_lat")
            cls._longitude(center_lon, "center_lon")

    @classmethod
    def _region_size(cls, value: Any, label: str) -> float:
        """Return a radius or semi-axis, from 0 to half the globe's circumference."""
        number = cls._positive(value, label)
        if number >= _MAX_REGION_SIZE_KM:
            raise ValueError(
                f"{label} must be less than half the circumference of the Earth, "
                f"{_MAX_REGION_SIZE_KM:.1f} km"
            )
        return number

    @classmethod
    def _validate_create_region(cls, config: Any) -> None:
        """Check a ``regional.create_region`` block.

        It must hold exactly one shape: ``circle``, ``ellipse``, ``polygon`` or
        ``channel``, each with its own keys.

        Raises
        ------
        ValueError
            If it is not a dict holding exactly one known shape, or the shape is
            missing a key, has an unknown key, or has a value out of range.
        """
        where = "regional.create_region"
        if not isinstance(config, dict):
            raise ValueError(f"{where} must be a dict, got {type(config).__name__}")
        shapes = ", ".join(_CREATE_REGION_SHAPES)
        unknown = sorted(str(k) for k in config if k not in _CREATE_REGION_SHAPES)
        if unknown:
            raise ValueError(
                f"Unknown {where} keys: {', '.join(unknown)}. Allowed: {shapes}"
            )
        if len(config) != 1:
            raise ValueError(f"{where} must contain exactly one of {shapes}")
        ((shape, block),) = config.items()
        where = f"{where}.{shape}"
        if not isinstance(block, dict):
            raise ValueError(f"{where} must be a dict, got {type(block).__name__}")
        required, optional = _CREATE_REGION_SHAPES[shape]
        allowed = set(required) | set(optional)
        unknown = sorted(str(k) for k in block if k not in allowed)
        if unknown:
            raise ValueError(
                f"Unknown {where} keys: {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        missing = [key for key in required if key not in block]
        if missing:
            raise ValueError(f"{where} is missing required keys: {', '.join(missing)}")

        if shape in ("circle", "ellipse"):
            cls._latitude(block["center_lat"], f"{where}.center_lat")
            cls._longitude(block["center_lon"], f"{where}.center_lon")
        if shape == "circle":
            cls._region_size(block["radius_km"], f"{where}.radius_km")
        elif shape == "ellipse":
            major = cls._region_size(
                block["semi_major_axis_km"], f"{where}.semi_major_axis_km"
            )
            minor = cls._region_size(
                block["semi_minor_axis_km"], f"{where}.semi_minor_axis_km"
            )
            if minor > major:
                raise ValueError(
                    f"{where}.semi_minor_axis_km must not be greater than "
                    "semi_major_axis_km"
                )
            if "orientation_degrees" in block:
                cls._number(block, "orientation_degrees", where)
        elif shape == "polygon":
            cls._latitude(block["inside_lat"], f"{where}.inside_lat")
            cls._longitude(block["inside_lon"], f"{where}.inside_lon")
            vertices = block["vertices"]
            if not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
                raise ValueError(
                    f"{where}.vertices must be a list of at least 3 [lat, lon] pairs"
                )
            for index, vertex in enumerate(vertices):
                label = f"{where}.vertices[{index}]"
                if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
                    raise ValueError(f"{label} must be a [lat, lon] pair")
                cls._latitude(vertex[0], f"{label} latitude")
                cls._longitude(vertex[1], f"{label} longitude")
        else:
            upper = cls._latitude(block["upper_lat"], f"{where}.upper_lat")
            lower = cls._latitude(block["lower_lat"], f"{where}.lower_lat")
            if upper <= lower:
                raise ValueError(f"{where}.upper_lat must be greater than lower_lat")

    @staticmethod
    def _real(value: float) -> str:
        """Format a number for a Fortran namelist.

        Raises
        ------
        ValueError
            If the value is not finite, which the Fortran programs cannot read.
        """
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            raise ValueError(f"Cannot write {value!r} to a namelist")
        return repr(number)

    @classmethod
    def _project_hexes_namelist(
        cls,
        cell_spacing_km: float,
        extent_x_km: float,
        extent_y_km: float,
        center_lat: float,
        center_lon: float,
    ) -> str:
        """Return the ``namelist.projections`` text that project_hexes reads.

        The Lambert conformal cone touches the sphere at the center, unless the
        center is within 0.0001 degrees of the equator, where project_hexes
        would divide by zero or lose all precision. Its cone then touches 0.0001
        degrees from the equator, which moves its cells by at most about 40 m
        on a mesh 10000 km across.
        """
        r = cls._real
        parallel = standard_parallel(center_lat)
        return (
            "&mesh\n"
            f"  cell_spacing_km = {r(cell_spacing_km)}\n"
            f"  mesh_length_x_km = {r(extent_x_km)}\n"
            f"  mesh_length_y_km = {r(extent_y_km)}\n"
            "  earth_radius_km = 6371.229\n"
            "/\n"
            "&projection\n"
            '  projection_type = "lambert_conformal"\n'
            "/\n"
            "&lambert_conformal\n"
            f"  reference_longitude_degrees = {r(center_lon)}\n"
            f"  standard_longitude_degrees = {r(center_lon)}\n"
            f"  reference_latitude_degrees = {r(center_lat)}\n"
            f"  standard_parallel_1_degrees = {r(parallel)}\n"
            f"  standard_parallel_2_degrees = {r(parallel)}\n"
            "/\n"
        )

    @classmethod
    def _rotate_namelist(
        cls, center_lat: float, center_lon: float, rotation_degrees: float
    ) -> str:
        """Return the ``namelist.input`` text that grid_rotate reads.

        The mesh is rotated counter-clockwise about its own center.
        """
        r = cls._real
        return (
            "&input\n"
            f"   config_original_latitude_degrees = {r(center_lat)}\n"
            f"   config_original_longitude_degrees = {r(center_lon)}\n"
            "\n"
            f"   config_new_latitude_degrees = {r(center_lat)}\n"
            f"   config_new_longitude_degrees = {r(center_lon)}\n"
            "   config_birdseye_rotation_counter_clockwise_degrees = "
            f"{r(rotation_degrees)}\n"
            "/\n"
        )

    @classmethod
    def _create_region_points(cls, config: Dict[str, Any]) -> str:
        """Return the points file that create_region reads for a region.

        ``config`` is a validated ``regional.create_region`` block. Sizes are
        given in kilometers and written in meters, which the tool expects. The
        region is always named ``region``.
        """
        r = cls._real
        ((shape, block),) = config.items()
        lines = ["Name: region"]
        # Any angle is written within one turn
        orientation = cls._orientation(block)
        if shape == "circle":
            lines += [
                "Type: circle",
                f"Point: {r(block['center_lat'])}, {r(block['center_lon'])}",
                f"Radius: {r(block['radius_km'] * 1000)}",
            ]
        elif shape == "ellipse":
            lines += [
                "Type: ellipse",
                f"Point: {r(block['center_lat'])}, {r(block['center_lon'])}",
                f"Semi-major-axis: {r(block['semi_major_axis_km'] * 1000)}",
                f"Semi-minor-axis: {r(block['semi_minor_axis_km'] * 1000)}",
                f"Orientation-angle: {r(orientation)}",
            ]
        elif shape == "polygon":
            lines += [
                "Type: custom",
                f"Point: {r(block['inside_lat'])}, {r(block['inside_lon'])}",
            ]
            lines += [f"{r(lat)}, {r(lon)}" for lat, lon in block["vertices"]]
        else:
            lines += [
                "Type: channel",
                f"Upper-lat: {r(block['upper_lat'])}",
                f"Lower-lat: {r(block['lower_lat'])}",
            ]
        return "\n".join(lines) + "\n"

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

    @staticmethod
    def _is_executable(path: Path) -> bool:
        """Whether ``path`` is a regular file that can be run."""
        return (
            path.is_file()
            # Some filesystems report every file as executable to os.access
            and bool(path.stat().st_mode & 0o111)
            and os.access(path, os.X_OK)
        )

    async def _run_install(
        self,
        label: str,
        log_name: str,
        download_task: Callable[..., Any],
        build_task: Callable[..., Any],
        stages: Tuple[str, str] = ("download", "build"),
        executors: Tuple[str, str] = ("service", "compute"),
    ) -> None:
        """Download and build a tool, and report the first failure.

        The download runs on ``service`` and the build on ``compute``, unless
        ``executors`` names others. ``stages`` names the two steps in logs and
        messages. Logs are
        ``<log_name>_<stage>.*`` in ``log_dir``, such as
        ``metis_download.stderr``. The caller holds the install lock and has
        checked that the resources exist.

        Raises
        ------
        RuntimeError
            If either step fails. The message names the log file.
        """
        first, second = stages
        download_future = download_task(
            executor=[executors[0]],
            stdout=str(self.log_dir / f"{log_name}_{first}.stdout"),
            stderr=str(self.log_dir / f"{log_name}_{first}.stderr"),
        )
        build_future = build_task(
            executor=[executors[1]],
            stdout=str(self.log_dir / f"{log_name}_{second}.stdout"),
            stderr=str(self.log_dir / f"{log_name}_{second}.stderr"),
            inputs=[download_future],
        )
        for stage, future in ((first, download_future), (second, build_future)):
            try:
                await asyncio.wrap_future(future)
            except BashExitFailure as e:
                raise RuntimeError(
                    f"{label} {stage} failed (exit {e.exitcode}), "
                    f"see {self.log_dir}/{log_name}_{stage}.stderr"
                ) from e
            except Exception as e:
                raise RuntimeError(f"{label} {stage} failed: {e}") from e

    @property
    def _gpmetis(self) -> Path:
        """Path of the gpmetis executable."""
        return self._metis_prefix / "bin" / "gpmetis"

    def _metis_usable(self) -> bool:
        """Whether a finished Metis install with a runnable gpmetis exists."""
        return (self._metis_prefix / ".installed").is_file() and self._is_executable(
            self._gpmetis
        )

    @property
    def _metis_prefix(self) -> Path:
        """Directory Metis is installed into."""
        return self.work_dir / "metis" / self.metis_version

    @property
    def _mpas_tools_prefix(self) -> Path:
        """Directory the MPAS-Tools programs are installed into."""
        return self.work_dir / "mpas-tools" / self.mpas_tools_version

    @property
    def _project_hexes(self) -> Path:
        """Path of the project_hexes executable."""
        return self._mpas_tools_prefix / "bin" / "project_hexes"

    @property
    def _grid_rotate(self) -> Path:
        """Path of the grid_rotate executable."""
        return self._mpas_tools_prefix / "bin" / "grid_rotate"

    @property
    def _limited_area_prefix(self) -> Path:
        """Directory MPAS-Limited-Area is installed in."""
        return self.work_dir / "limited-area" / self.limited_area_version

    @property
    def _create_region(self) -> Path:
        """Path of the create_region script."""
        return self._limited_area_prefix / "src" / "create_region"

    def _limited_area_usable(self) -> bool:
        """Whether a finished MPAS-Limited-Area install exists.

        An install made when create_region ran in its own environment, in
        ``venv``, was checked with that environment's Python, not the tasks'
        Python, so it is replaced.
        """
        return (
            (self._limited_area_prefix / ".installed").is_file()
            and self._create_region.is_file()
            and not (self._limited_area_prefix / "venv").exists()
        )

    @property
    def _plot_prefix(self) -> Path:
        """Directory the map layers and matplotlib's settings are kept in."""
        return self.work_dir / "plot"

    @property
    def _map_data_dir(self) -> Path:
        """Directory cartopy reads the Natural Earth map layers from."""
        return self._plot_prefix / "natural_earth"

    def _map_layer_paths(self) -> List[Path]:
        """Paths of the map layer files, where cartopy stores and finds them."""
        return [layer_path(self._map_data_dir, layer) for layer in MAP_LAYERS]

    @property
    def _plot_script(self) -> Path:
        """Path of the plotting script, next to this module.

        Tasks read this on the worker, so it is the script of the chiltepin
        installation the worker runs, which the compute resource can read.
        """
        return Path(__file__).with_name("_plot_mesh.py")

    def _prompting_usable(self) -> bool:
        """Whether the boundaries are installed and prompting was checked.

        Every boundary layer must be complete, so an install made before a
        layer was added is replaced.
        """
        return (self._geo_prefix / ".installed").is_file() and all(
            layer_complete(layer_path(self._boundaries_dir, layer))
            for layer in BOUNDARY_LAYERS
        )

    @property
    def _geo_prefix(self) -> Path:
        """Directory the boundaries and Nominatim's answers are kept in."""
        return self.work_dir / "geo"

    @property
    def _boundaries_dir(self) -> Path:
        """Directory the Natural Earth boundaries are downloaded to."""
        return self._geo_prefix / "natural_earth"

    @property
    def _nominatim_cache_dir(self) -> Path:
        """Directory Nominatim's answers are kept in."""
        return self._geo_prefix / "nominatim"

    def _plotting_usable(self) -> bool:
        """Whether the map layers are installed and plotting was checked."""
        return (self._plot_prefix / ".installed").is_file()

    def _mpas_tools_usable(self) -> bool:
        """Whether a finished MPAS-Tools install with runnable programs exists."""
        return (self._mpas_tools_prefix / ".installed").is_file() and all(
            self._is_executable(path)
            for path in (self._project_hexes, self._grid_rotate)
        )

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

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Tools
    # -------------------------------------------------------------------------

    @bash_task
    def _download_mpas_tools(self) -> str:
        """Download the MPAS-Tools source code for the requested tag from GitHub."""
        import textwrap

        build = f"{self.work_dir}/mpas-tools/build/{self.mpas_tools_version}"
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            build={self._q(build)}
            rm -rf "$build"
            mkdir -p "$build"
            cd "$build"
            # Clone only the requested tag, giving up if the transfer stalls
            git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 --branch {self._q(self.mpas_tools_version)} https://github.com/MPAS-Dev/MPAS-Tools.git
            echo "Completed download at $(date)"
            """
        )

    @bash_task
    def _build_mpas_tools(self) -> str:
        """Build project_hexes and grid_rotate, and install them to the prefix."""
        import textwrap

        build = f"{self.work_dir}/mpas-tools/build/{self.mpas_tools_version}"
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started build at $(date)"
            echo "Executing on $(hostname)"
            build={self._q(build)}
            prefix={self._q(self._mpas_tools_prefix)}
            # The Makefiles find netCDF through these, so fail clearly without them
            for tool in nf-config nc-config; do
                if ! command -v "$tool" > /dev/null; then
                    echo "$tool not found. Load netCDF-C and netCDF-Fortran in the environment of the compute resource." >&2
                    exit 1
                fi
            done
            # Start from an empty prefix so a failed earlier install leaves nothing
            rm -rf "$prefix"
            mkdir -p "$prefix/bin"
            tools="$build/MPAS-Tools/mesh_tools"
            make -C "$tools/hex_projection"
            make -C "$tools/grid_rotate"
            cp "$tools/hex_projection/project_hexes" "$tools/grid_rotate/grid_rotate" "$prefix/bin/"
            # The sources are no longer needed. A failed build keeps them to debug.
            cd /
            # Failing to remove the sources must not fail a finished build
            rm -rf "$build" || true
            # Mark the install complete. The agent only uses a marked install.
            touch "$prefix/.installed"
            echo "Completed build at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @bash_task
    def _install_limited_area(self) -> str:
        """Clone MPAS-Limited-Area, on ``service``, which can reach GitHub.

        create_region runs with the Python of the tasks on ``compute``, whose
        ``chiltepin[mpas]`` extra has the numpy and netCDF4 it needs.
        """
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started install at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._limited_area_prefix)}
            # Start from an empty prefix so a failed earlier install leaves nothing
            rm -rf "$prefix"
            mkdir -p "$prefix"
            cd "$prefix"
            # Clone only the requested tag, giving up if the transfer stalls
            git -c advice.detachedHead=false -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --depth 1 --branch {self._q("v" + self.limited_area_version)} https://github.com/MPAS-Dev/MPAS-Limited-Area.git src
            echo "Completed install at $(date)"
            """
        )

    @bash_task
    def _check_limited_area(self) -> str:
        """Check that create_region runs on ``compute``, and mark the install.

        It runs with the Python of the tasks on ``compute``, which must have the
        ``chiltepin[mpas]`` extra.
        """
        import sys
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started check at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._limited_area_prefix)}
            python={self._q(sys.executable)}
            if ! "$python" -c "import numpy, netCDF4"; then
                echo "The Python of the compute resource cannot import numpy and netCDF4, which create_region needs. Install chiltepin[mpas] in the environment of the compute resource." >&2
                exit 1
            fi
            "$python" "$prefix/src/create_region" --help > /dev/null
            # Mark the install complete. The agent only uses a marked install.
            touch "$prefix/.installed"
            echo "Completed check at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — Plotting
    # -------------------------------------------------------------------------

    @bash_task
    def _download_plotting(self) -> str:
        """Download the map layers the plots draw, on ``service``.

        The plotting script downloads them with only the standard library, so
        the ``service`` resource does not need the ``chiltepin[mpas]`` extra.
        Layers already downloaded are kept, and nothing is removed, so plots
        running while plotting is installed again keep their map layers and
        matplotlib settings. Removing ``work_dir/plot`` downloads every layer
        again.
        """
        import sys
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._plot_prefix)}
            # The install counts again only once the check passes
            mkdir -p "$prefix"
            rm -f "$prefix/.installed"
            {self._q(sys.executable)} {self._q(self._plot_script)} --download-map-data {self._q(self._map_data_dir)}
            echo "Completed download at $(date)"
            """
        )

    @bash_task
    def _check_plotting(self) -> str:
        """Check that plotting can run on ``compute``, and mark the install.

        Plotting runs with the Python of the tasks on ``compute``, which must
        have the ``chiltepin[mpas]`` extra.
        """
        import sys
        import textwrap

        # Every part each layer needs, as the plotting script checks
        layers = " ".join(
            self._q(path.with_suffix(suffix))
            for path in self._map_layer_paths()
            for suffix in REQUIRED_PARTS
        )
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started check at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._plot_prefix)}
            if ! MPLBACKEND=Agg {self._q(sys.executable)} -c "import cartopy, matplotlib, netCDF4, numpy"; then
                echo "The Python of the compute resource cannot import the plotting packages. Install chiltepin[mpas] in the environment of the compute resource." >&2
                exit 1
            fi
            for layer in {layers}; do
                if [ ! -s "$layer" ]; then
                    echo "Map layer $layer is missing" >&2
                    exit 1
                fi
            done
            # Mark the install complete. The agent only uses a marked install.
            touch "$prefix/.installed"
            echo "Completed check at $(date)"
            """
        )

    @bash_task
    def _download_boundaries(self) -> str:
        """Download the Natural Earth boundaries places are looked up in.

        It needs only the standard library. Boundaries already downloaded are
        kept, so prompts being read meanwhile keep them.
        """
        import sys
        import textwrap

        code = (
            "import sys; from pathlib import Path; "
            "from chiltepin.agents.mpas._geo_lookup import download_boundaries; "
            "download_boundaries(Path(sys.argv[1]))"
        )
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._geo_prefix)}
            # The install counts again only once the check passes
            mkdir -p "$prefix"
            rm -f "$prefix/.installed"
            {self._q(sys.executable)} -c {self._q(code)} {self._q(self._boundaries_dir)}
            echo "Completed download at $(date)"
            """
        )

    @bash_task
    def _check_prompting(self) -> str:
        """Check that prompts can be read on ``service``, and mark the install.

        The Python of the tasks there must have the ``chiltepin[mpas,llm]``
        extras, and the boundaries must be complete.
        """
        import sys
        import textwrap

        code = (
            "import sys; from pathlib import Path; "
            "import instructor, litellm, shapefile, shapely; "
            "from chiltepin.agents.mpas._geo_lookup import BOUNDARY_LAYERS; "
            "from chiltepin.agents.mpas._plot_mesh import layer_complete, layer_path; "
            "missing = [str(layer_path(Path(sys.argv[1]), layer)) "
            "for layer in BOUNDARY_LAYERS "
            "if not layer_complete(layer_path(Path(sys.argv[1]), layer))]; "
            "sys.exit('Boundaries are missing: ' + ', '.join(missing) "
            "if missing else 0)"
        )
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started check at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._geo_prefix)}
            if ! {self._q(sys.executable)} -c "import instructor, litellm, shapefile, shapely"; then
                echo "The Python of the service resource cannot import the packages prompts need. Install chiltepin[mpas,llm] in the environment of the service resource." >&2
                exit 1
            fi
            {self._q(sys.executable)} -c {self._q(code)} {self._q(self._boundaries_dir)}
            # Mark the install complete. The agent only uses a marked install.
            touch "$prefix/.installed"
            echo "Completed check at $(date)"
            """
        )

    @bash_task
    def _run_plot_mesh(
        self, mesh_path: str, output_path: str, options: List[str]
    ) -> str:
        """Run the plotting script with the Python of the task.

        matplotlib keeps its configuration and font cache in ``work_dir/plot``,
        not in the home directory.
        """
        import sys
        import textwrap

        arguments = " ".join(self._q(option) for option in options)
        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started plot at $(date)"
            echo "Executing on $(hostname)"
            prefix={self._q(self._plot_prefix)}
            export MPLBACKEND=Agg
            export MPLCONFIGDIR="$prefix/matplotlib"
            {self._q(sys.executable)} {self._q(self._plot_script)} {self._q(mesh_path)} {self._q(output_path)} --map-data {self._q(self._map_data_dir)} {arguments}
            echo "Completed plot at $(date)"
            """
        )

    @bash_task
    def _run_create_region(self, mesh_dir: str) -> str:
        """Run create_region in a mesh directory holding the points and parent.

        It runs with the Python of the task, which has the ``chiltepin[mpas]``
        extra.
        """
        import sys
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started create_region at $(date)"
            echo "Executing on $(hostname)"
            cd {self._q(mesh_dir)}
            {self._q(sys.executable)} {self._q(self._create_region)} {self._q(_REGION_POINTS)} {self._q(_REGION_PARENT)}
            echo "Completed create_region at $(date)"
            """
        )

    @bash_task
    def _run_project_hexes(self, mesh_dir: str) -> str:
        """Run project_hexes in a mesh directory holding ``namelist.projections``."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started project_hexes at $(date)"
            echo "Executing on $(hostname)"
            cd {self._q(mesh_dir)}
            {self._q(self._project_hexes)}
            echo "Completed project_hexes at $(date)"
            """
        )

    @bash_task
    def _run_grid_rotate(self, mesh_dir: str, input_file: str, output_file: str) -> str:
        """Run grid_rotate in a mesh directory holding ``namelist.input``.

        The files are named relative to ``mesh_dir``. grid_rotate keeps only the
        first 100 characters of each path, as of MPAS-Tools 2.0.0, so a full
        path could be cut short.
        """
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started grid_rotate at $(date)"
            echo "Executing on $(hostname)"
            cd {self._q(mesh_dir)}
            {self._q(self._grid_rotate)} {self._q(input_file)} {self._q(output_file)}
            echo "Completed grid_rotate at $(date)"
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
            {self._q(self._gpmetis)} -minconn -contig -niter=200 {self._q(mesh_path)} {int(num_ranks)}
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

    @python_task
    def _read_prompt(
        self,
        prompt: str,
        model: str,
        buffer_km: float,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the model what mesh a prompt describes, and return its request.

        Runs on ``service``, whose environment holds the model's credentials.
        If the places of the model's answer cannot be looked up, the model is
        told why, and asked once more. The request is returned as
        ``{"mesh_config": ...}``, or why the prompt cannot be made into a mesh
        as ``{"error": ...}``, which every executor returns as it is. Other
        failures raise a RuntimeError. Neither holds credentials.
        """
        from chiltepin.agents.mpas._geo_lookup import RegionNotFound
        from chiltepin.agents.mpas._mesh_prompt import ask, mesh_config, redact

        resolutions = list(self.resolution_cells)

        def config_of(answer):
            return mesh_config(
                answer,
                resolutions,
                self._boundaries_dir,
                buffer_km,
                nominatim_contact=self.nominatim_contact,
                cache_dir=self._nominatim_cache_dir,
                prompt=prompt,
            )

        try:
            answer = ask(prompt, model, resolutions, reasoning_effort=reasoning_effort)
            try:
                return {"mesh_config": config_of(answer)}
            except RegionNotFound as e:
                answer = ask(
                    prompt,
                    model,
                    resolutions,
                    reasoning_effort=reasoning_effort,
                    correction=(answer, str(e)),
                )
                return {"mesh_config": config_of(answer)}
        except ValueError as e:
            return {"error": redact(str(e))}
        except Exception as e:
            raise RuntimeError(redact(f"{type(e).__name__}: {e}")) from None

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
                await self._run_install(
                    "Metis", "metis", self._download_metis, self._build_metis
                )
                if not self._metis_usable():
                    raise RuntimeError(
                        f"Metis build finished but {self._gpmetis} is not usable, "
                        f"see {self.log_dir}/metis_build.stderr"
                    )
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

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Tools
    # -------------------------------------------------------------------------

    @agent_action
    async def install_mpas_tools(self) -> None:
        """Download and build project_hexes and grid_rotate, unless installed.

        The two programs from MPAS-Tools count as installed once a build has
        finished in ``work_dir``, so a restarted agent reuses them. Requests that
        need them while they are being installed wait for that install. If it
        fails, each waiting request tries again in turn. Called automatically
        when a request needs them, but can be called earlier to have them ready.
        Building needs a Fortran compiler and the netCDF-C and netCDF-Fortran
        libraries in the environment of the ``compute`` resource.

        Raises
        ------
        RuntimeError
            If the ``service`` or ``compute`` resource is not defined, or the
            download or build fails. The message names the log file.
        """
        async with self._install_lock:
            # Check the install on disk each time, so a removed or damaged
            # install is rebuilt instead of failing every later request
            if not self._mpas_tools_usable():
                self.mpas_tools_built = False
                self._require_resources("Installing MPAS-Tools", "service", "compute")
                await self._run_install(
                    "MPAS-Tools",
                    "mpas_tools",
                    self._download_mpas_tools,
                    self._build_mpas_tools,
                )
                if not self._mpas_tools_usable():
                    raise RuntimeError(
                        "MPAS-Tools build finished but project_hexes and "
                        f"grid_rotate are not usable in {self._mpas_tools_prefix}"
                        f"/bin, see {self.log_dir}/mpas_tools_build.stderr"
                    )
            self.mpas_tools_built = True

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @agent_action
    async def install_limited_area(self) -> None:
        """Install MPAS-Limited-Area, unless installed.

        The ``create_region`` program of MPAS-Limited-Area is a Python script
        that needs numpy and netCDF4, which the ``chiltepin[mpas]`` extra
        installs on the ``compute`` resource, where it runs. The tag is cloned
        into ``work_dir`` on the ``service`` resource, and a check on the
        ``compute`` resource confirms that the Python of its tasks runs it. It
        counts as installed once that check has passed, so a restarted agent
        reuses it. Requests that need it while it is being installed wait for
        that install. If it fails, each waiting request tries again in turn.
        Called automatically when a request needs it, but can be called earlier
        to have it ready.

        Raises
        ------
        RuntimeError
            If the ``service`` or ``compute`` resource is not defined, or the
            install or check fails. The message names the log file.
        """
        async with self._install_lock:
            # Check the install on disk each time, so a removed or damaged
            # install is replaced instead of failing every later request
            if not self._limited_area_usable():
                self.limited_area_built = False
                self._require_resources(
                    "Installing MPAS-Limited-Area", "service", "compute"
                )
                await self._run_install(
                    "MPAS-Limited-Area",
                    "limited_area",
                    self._install_limited_area,
                    self._check_limited_area,
                    stages=("install", "check"),
                )
                if not self._limited_area_usable():
                    raise RuntimeError(
                        "MPAS-Limited-Area check finished but the install in "
                        f"{self._limited_area_prefix} is not usable, see "
                        f"{self.log_dir}/limited_area_check.stderr"
                    )
            self.limited_area_built = True

    # -------------------------------------------------------------------------
    # Public agent actions — Plotting
    # -------------------------------------------------------------------------

    @agent_action
    async def install_plotting(self) -> None:
        """Download the map layers plots need and check plotting, unless done.

        Plots are drawn on the ``compute`` resource with numpy, netCDF4,
        matplotlib and cartopy, which the ``chiltepin[mpas]`` extra installs
        there. The Natural Earth map layers the plots draw are downloaded to
        ``work_dir/plot/natural_earth`` on the ``service`` resource, since the
        ``compute`` resource may have no internet access. A check on the
        ``compute`` resource then confirms that its Python has the extra and the
        layers are in place. It counts as installed once that check has passed,
        so a restarted agent reuses it. Requests that need it while it is being
        installed wait for that install. If it fails, each waiting request tries
        again in turn. Called automatically when a plot is requested, but can be
        called earlier to have it ready.

        Raises
        ------
        RuntimeError
            If the ``service`` or ``compute`` resource is not defined, or the
            download or check fails. The message names the log file.
        """
        async with self._install_lock:
            # Check the install on disk each time, so a removed or damaged
            # install is replaced instead of failing every later plot
            if not self._plotting_usable():
                self.plotting_built = False
                self._require_resources("Installing plotting", "service", "compute")
                await self._run_install(
                    "Plotting",
                    "plot",
                    self._download_plotting,
                    self._check_plotting,
                    stages=("download", "check"),
                )
                if not self._plotting_usable():
                    raise RuntimeError(
                        "Plotting check finished but the install in "
                        f"{self._plot_prefix} is not usable, see "
                        f"{self.log_dir}/plot_check.stderr"
                    )
            self.plotting_built = True

    @staticmethod
    def _plot_path(mesh: Path) -> Path:
        """Return the image path of a mesh: ``<name>.png`` next to it.

        ``conus.static.nc`` gives ``conus.png``, and ``x1.2562.grid.nc`` gives
        ``x1.2562.grid.png``. Meshes whose names differ only by these endings,
        such as ``a.nc`` and ``a.static.nc`` in one directory, share an image.
        """
        name = mesh.name.removesuffix(".nc").removesuffix(".static")
        return mesh.with_name(f"{name}.png")

    async def _plot(
        self,
        mesh_path: str,
        title: Optional[str] = None,
        zoom: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
    ) -> Tuple[str, Optional[str]]:
        """Draw a mesh, and return the image's path and any warning about it.

        The image is written under a temporary name and renamed when it is
        complete, so a failed plot leaves an earlier image of the mesh alone.
        The warning names the log of a plot drawn without some map layers.

        Raises
        ------
        ValueError
            If an argument is invalid, or the view settings are given for a
            regional mesh.
        FileNotFoundError
            If the mesh file does not exist.
        RuntimeError
            If plotting cannot be installed, the ``compute`` resource is not
            defined, or the plot fails. The message names the log files.
        """
        if not isinstance(mesh_path, str) or not mesh_path:
            raise ValueError("mesh_path must be a non-empty string")
        self._validate_plot_view(title, zoom, center_lat, center_lon)
        mesh = Path(os.path.abspath(os.path.expanduser(mesh_path)))
        output = self._plot_path(mesh)
        if not mesh.is_file():
            raise FileNotFoundError(f"No mesh file at {mesh}")

        await self.install_plotting()
        self._require_resources("Plotting", "compute")
        call_id = uuid.uuid4().hex[:8]
        log = self.log_dir / f"plot_mesh.{call_id}"
        partial = output.with_name(f".{output.stem}.{call_id}.png")
        # One argument per option, so values starting with "-" stay values
        options: List[str] = []
        if title is not None:
            options.append(f"--title={title}")
        if zoom is not None:
            options.append(f"--zoom={float(zoom)!r}")
        if center_lat is not None:
            options.append(f"--center-lat={float(center_lat)!r}")
            options.append(f"--center-lon={float(center_lon)!r}")

        warning = None
        future = self._run_plot_mesh(
            str(mesh),
            str(partial),
            options,
            executor=["compute"],
            stdout=f"{log}.stdout",
            stderr=f"{log}.stderr",
        )
        try:
            try:
                await asyncio.wrap_future(future)
            except BashExitFailure as e:
                if e.exitcode == EXIT_VIEW_OF_REGIONAL_MESH:
                    raise ValueError(
                        "zoom, center_lat and center_lon apply only to global "
                        f"meshes, and {mesh} is regional"
                    ) from e
                if e.exitcode != EXIT_MISSING_MAP:
                    raise RuntimeError(
                        f"Plotting failed (exit {e.exitcode}), "
                        f"see {log}.stdout and {log}.stderr"
                    ) from e
                warning = f"Plotted without some map layers, see {log}.stderr"
                logger.warning("%s: %s", output, warning)
            except Exception as e:
                raise RuntimeError(f"Plotting failed: {e}") from e
            self._require_output(partial, "plot_mesh", call_id)
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
        return str(output), warning

    @agent_action
    async def plot_mesh(
        self,
        mesh_path: str,
        title: Optional[str] = None,
        zoom: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
    ) -> str:
        """Draw a PNG image of a mesh, and return its path.

        The image is written next to the mesh as ``<name>.png``, replacing an
        earlier one. A regional mesh is drawn on a flat map, with its cells
        colored by boundary layer. A global mesh is drawn on a sphere. Cell
        edges are drawn when cells come out large enough in the image to tell
        apart. Plotting is installed first if it is not already installed, and
        runs on the ``compute`` resource.

        Parameters
        ----------
        mesh_path : str
            Path to the mesh's static or grid file. A relative path or one
            starting with ``~`` is made absolute on the agent.
        title : str, optional
            Title of the image, by default "MPAS mesh: <name>"
        zoom : float, optional
            Zoom of a global mesh's sphere, greater than 0. The default makes
            the globe fill the image, and a larger zoom shows less of it, with
            larger cells.
        center_lat, center_lon : float, optional
            Point of a global mesh to center the view on, given together, by
            default 0N 0E.

        Returns
        -------
        str
            Absolute path of the image. If some map layers are missing, it is
            drawn without them, and a warning naming the log is logged.

        Raises
        ------
        ValueError
            If an argument is invalid, or ``zoom``, ``center_lat`` or
            ``center_lon`` is given for a regional mesh.
        FileNotFoundError
            If the mesh file does not exist.
        RuntimeError
            If plotting cannot be installed, a resource it needs is not defined,
            or the plot fails. The message names the log files.
        """
        path, _ = await self._plot(mesh_path, title, zoom, center_lat, center_lon)
        return path

    # -------------------------------------------------------------------------
    # Public agent actions — Tools
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Public agent actions — Prompts
    # -------------------------------------------------------------------------

    @agent_action
    async def install_prompting(self) -> None:
        """Download the boundaries prompts are read with, and check, unless done.

        Prompts are read on the ``service`` resource, where the model is asked
        and its places are looked up, with the ``chiltepin[mpas,llm]`` extras.
        The Natural Earth countries and states or provinces are downloaded to
        ``work_dir/geo/natural_earth`` there, and a check confirms that the
        extras import and the boundaries are complete. It counts as installed
        once that check has passed. Called automatically when a prompt is read,
        but can be called earlier to have it ready.

        Raises
        ------
        RuntimeError
            If the ``service`` resource is not defined, or the download or
            check fails. The message names the log file.
        """
        async with self._install_lock:
            if not self._prompting_usable():
                self.prompting_built = False
                self._require_resources("Installing prompting", "service")
                await self._run_install(
                    "Prompting",
                    "prompt",
                    self._download_boundaries,
                    self._check_prompting,
                    stages=("download", "check"),
                    executors=("service", "service"),
                )
                if not self._prompting_usable():
                    raise RuntimeError(
                        "Prompting check finished but the install in "
                        f"{self._geo_prefix} is not usable, see "
                        f"{self.log_dir}/prompt_check.stderr"
                    )
            self.prompting_built = True

    @staticmethod
    def _validate_prompt(
        prompt: Any, model: Any, buffer_km: Any, reasoning_effort: Any = None
    ) -> float:
        """Check a prompt action's arguments, and return the margin in km.

        Raises
        ------
        ValueError
            If the prompt is not a non-empty string of at most 10000
            characters, the model is not a non-empty string without spaces,
            ``buffer_km`` is not a number from 0 to 5000, or
            ``reasoning_effort`` is not None or a short lower-case word.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > _MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt must be at most {_MAX_PROMPT_CHARS} characters long"
            )
        if (
            not isinstance(model, str)
            or not model
            or any(char.isspace() or not char.isprintable() for char in model)
        ):
            raise ValueError(
                "model must be a litellm model string, such as "
                "'bedrock/us.anthropic.claude-sonnet-5-5'"
            )
        if isinstance(buffer_km, bool) or not isinstance(buffer_km, (int, float)):
            raise ValueError(f"buffer_km must be a number, got {buffer_km!r}")
        if not math.isfinite(buffer_km) or not 0 <= buffer_km <= _MAX_BUFFER_KM:
            raise ValueError(f"buffer_km must be from 0 to {_MAX_BUFFER_KM:g}")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or not _REASONING_EFFORT_RE.fullmatch(reasoning_effort)
        ):
            raise ValueError(
                "reasoning_effort must be None or a word such as 'none' or 'low'"
            )
        return float(buffer_km)

    @agent_action
    async def mesh_config_from_prompt(
        self,
        prompt: str,
        model: str,
        buffer_km: float = _DEFAULT_BUFFER_KM,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the mesh request a prompt describes, without making the mesh.

        An LLM reads the prompt on the ``service`` resource, and answers with
        a resolution, a name, and the places the mesh covers, and optionally
        a shape, a method and a channel's latitudes. The places are looked up,
        and the region is fitted to the shape, so the request is in the form
        ``generate_mesh()`` takes, without ``plot`` or rank counts.

        Parameters
        ----------
        prompt : str
            What mesh to make, such as "a 15 km mesh over Texas".
        model : str
            The litellm model string, such as
            ``"bedrock/us.anthropic.claude-sonnet-5-5"``. The model's
            credentials must be in the environment of the ``service`` resource.
        buffer_km : float, optional
            Margin around the region, by default 50. It is at least a tenth of
            the region's diameter.
        reasoning_effort : str, optional
            How much the model reasons before answering, passed to litellm,
            such as ``"none"`` to turn reasoning off for Ollama's Qwen and
            Gemma models, or ``"low"`` for GPT-OSS. Reasoning is slow without
            a GPU. By default the model's own setting is used.

        Raises
        ------
        ValueError
            If an argument is invalid, or the prompt asks for a mesh that
            cannot be made, such as a place that is not found.
        RuntimeError
            If prompting cannot be installed, the ``service`` resource is not
            defined, or the model cannot be reached or gives no valid answer.
        """
        margin = self._validate_prompt(prompt, model, buffer_km, reasoning_effort)
        await self.install_prompting()
        self._require_resources("Reading a prompt", "service")
        future = self._read_prompt(
            prompt, model, margin, reasoning_effort, executor=["service"]
        )
        try:
            result = await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(f"Reading the prompt failed: {e}") from None
        if "error" in result:
            raise ValueError(
                f"The prompt cannot be made into a mesh: {result['error']}"
            )
        mesh_config = result["mesh_config"]
        # A request with a prompt would have generate_mesh read one again
        if isinstance(mesh_config, dict) and "prompt" in mesh_config:
            raise RuntimeError(
                "Reading the prompt failed: it gave a request with a prompt, not "
                "a full request"
            )
        self._validate_mesh_config(mesh_config)
        return mesh_config

    @agent_action
    async def ensure_tools_installed(self, mesh_config: Dict[str, Any]) -> None:
        """Install the tools a mesh request needs.

        Can be called early to overlap tool installation with other work.
        ``generate_mesh()`` installs the same tools before it downloads the
        mesh, so a failed install does not waste a download, except that it
        installs plotting only when it plots, so a failed plotting install
        cannot fail the request.

        For a request with a ``prompt``, prompting is installed, and Metis and
        plotting if the request asks for them. The program that makes the
        prompt's regional mesh, if it is one, is known only once the prompt is
        read, so ``generate_mesh()`` installs it.

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
        await self._install_tools(mesh_config, plotting=True)

    async def _install_tools(self, mesh_config: Dict[str, Any], plotting: bool) -> None:
        """Install the tools a validated mesh request needs.

        ``generate_mesh()`` passes ``plotting=False`` and installs plotting only
        when it plots, so that a plotting install that fails cannot fail the
        request.
        """
        if "prompt" in mesh_config:
            await self.install_prompting()
        if self._partition_ranks(mesh_config):
            await self.install_metis()
        regional = mesh_config.get("regional") or {}
        if "project_hexes" in regional:
            await self.install_mpas_tools()
        if "create_region" in regional:
            await self.install_limited_area()
        if plotting and mesh_config.get("plot"):
            await self.install_plotting()

    # -------------------------------------------------------------------------
    # Private mesh generation
    # -------------------------------------------------------------------------

    async def _run_program(
        self, task: Callable[..., Any], program: str, call_id: str, *args: str
    ) -> None:
        """Run a program on ``compute`` and report a failure with its log names.

        Both logs are named, since some programs, such as create_region, print
        their errors to standard output.

        Raises
        ------
        RuntimeError
            If the program fails. The message names the log files.
        """
        log = self.log_dir / f"{program}.{call_id}"
        future = task(
            *args,
            executor=["compute"],
            stdout=f"{log}.stdout",
            stderr=f"{log}.stderr",
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"{program} failed (exit {e.exitcode}), "
                f"see {log}.stdout and {log}.stderr"
            ) from e
        except Exception as e:
            raise RuntimeError(f"{program} failed: {e}") from e

    def _require_output(self, path: Path, program: str, call_id: str) -> None:
        """Check that a program wrote a non-empty file.

        Raises
        ------
        RuntimeError
            If it did not. The message names the log file of that program.
        """
        if not path.is_file() or path.stat().st_size == 0:
            log = self.log_dir / f"{program}.{call_id}"
            raise RuntimeError(
                f"{program} did not write {path.name}, "
                f"see {log}.stdout and {log}.stderr"
            )

    async def _generate_project_hexes_mesh(
        self,
        mesh_dir: Path,
        mesh_name: str,
        resolution: str,
        config: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate a regional mesh with project_hexes, rotated if requested.

        Everything is written in ``mesh_dir``, which the caller created and
        removes if this fails. The inputs the programs read stay there, and the
        mesh and graph end up named after the mesh.

        Parameters
        ----------
        mesh_dir : Path
            Absolute path of the new mesh directory
        mesh_name : str
            Name of the mesh
        resolution : str
            Cell spacing such as ``"15km"``
        config : dict
            A validated ``regional.project_hexes`` block

        Returns
        -------
        dict
            ``{"static": path, "graph": path}``

        Raises
        ------
        RuntimeError
            If the programs cannot be installed or fail, or do not write the
            files they should.
        """
        await self.install_mpas_tools()
        call_id = uuid.uuid4().hex[:8]
        center_lat = float(config["center_lat"])
        center_lon = float(config["center_lon"])
        rotation = float(config.get("rotation_degrees", 0.0))

        (mesh_dir / "namelist.projections").write_text(
            self._project_hexes_namelist(
                self._parse_resolution_km(resolution),
                config["extent_x_km"],
                config["extent_y_km"],
                center_lat,
                center_lon,
            )
        )
        await self._run_program(
            self._run_project_hexes, "project_hexes", call_id, str(mesh_dir)
        )
        mesh_file = mesh_dir / "mpas_hex_mesh.nc"
        graph_file = mesh_dir / "graph.info"
        for path in (mesh_file, graph_file):
            self._require_output(path, "project_hexes", call_id)

        if rotation != 0.0:
            (mesh_dir / "namelist.input").write_text(
                self._rotate_namelist(center_lat, center_lon, rotation)
            )
            rotated_file = mesh_dir / "rotated_mesh.nc"
            await self._run_program(
                self._run_grid_rotate,
                "grid_rotate",
                call_id,
                str(mesh_dir),
                # Names in the directory it runs in, since grid_rotate keeps
                # only the first 100 characters of each path
                mesh_file.name,
                rotated_file.name,
            )
            self._require_output(rotated_file, "grid_rotate", call_id)
            mesh_file.unlink(missing_ok=True)
            mesh_file = rotated_file

        static = mesh_dir / f"{mesh_name}.static.nc"
        graph = mesh_dir / f"{mesh_name}.graph.info"
        mesh_file.replace(static)
        graph_file.replace(graph)
        return {"static": str(static), "graph": str(graph)}

    async def _generate_create_region_mesh(
        self,
        mesh_dir: Path,
        mesh_name: str,
        resolution: str,
        config: Dict[str, Any],
        mesh_data_dir: Path,
    ) -> Dict[str, str]:
        """Cut a regional mesh from a global mesh with create_region.

        The global mesh of ``resolution`` is downloaded to ``mesh_data_dir``
        unless it is already there, and is linked into ``mesh_dir`` while the
        tool runs. Everything else is written in ``mesh_dir``, which the caller
        created and removes if this fails. The points file stays there, and the
        mesh and graph end up named after the mesh.

        Parameters
        ----------
        mesh_dir : Path
            Absolute path of the new mesh directory
        mesh_name : str
            Name of the mesh
        resolution : str
            Resolution of the global mesh to cut from, such as ``"120km"``
        config : dict
            A validated ``regional.create_region`` block
        mesh_data_dir : Path
            Absolute path of the directory holding the downloaded global meshes

        Returns
        -------
        dict
            ``{"static": path, "graph": path}``

        Raises
        ------
        RuntimeError
            If MPAS-Limited-Area cannot be installed, the global mesh cannot be
            downloaded, or create_region fails or does not write its files.
        """
        # Write the points first, so a value the tool cannot take fails at once
        points = self._create_region_points(config)
        (mesh_dir / _REGION_POINTS).write_text(points)
        await self.install_limited_area()
        parent = await self.download_global_mesh(resolution, str(mesh_data_dir))
        call_id = uuid.uuid4().hex[:8]

        link = mesh_dir / _REGION_PARENT
        link.symlink_to(Path(parent["static"]).resolve())
        try:
            await self._run_program(
                self._run_create_region, "create_region", call_id, str(mesh_dir)
            )
        finally:
            link.unlink(missing_ok=True)
        mesh_file = mesh_dir / _REGION_STATIC
        graph_file = mesh_dir / _REGION_GRAPH
        for path in (mesh_file, graph_file):
            self._require_output(path, "create_region", call_id)

        static = mesh_dir / f"{mesh_name}.static.nc"
        graph = mesh_dir / f"{mesh_name}.graph.info"
        mesh_file.replace(static)
        graph_file.replace(graph)
        return {"static": str(static), "graph": str(graph)}

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

        # A second request for the same mesh waits, then finds the files there
        key = (str(request_mesh_data_dir), resolution)
        lock = self._download_locks.setdefault(key, asyncio.Lock())
        async with lock:
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

    async def _generate_mesh_from_prompt(
        self, mesh_config: Dict[str, Any], mesh_data_dir: str, overwrite: bool
    ) -> Dict[str, Any]:
        """Make the mesh a validated request with a prompt describes.

        The prompt is read into a request, the request's own keys are added to
        it, and the mesh is made from that, which is returned with the result
        as ``mesh_config``.
        """
        made_config = await self.mesh_config_from_prompt(
            mesh_config["prompt"],
            mesh_config["model"],
            mesh_config.get("buffer_km", _DEFAULT_BUFFER_KM),
            mesh_config.get("reasoning_effort"),
        )
        for key in ("name", "init_ranks", "forecast_ranks", "plot"):
            if key in mesh_config:
                made_config[key] = mesh_config[key]
        result = await self.generate_mesh(
            dict(made_config), mesh_data_dir, overwrite=overwrite
        )
        return {**result, "mesh_config": made_config}

    @agent_action
    async def generate_mesh(
        self,
        mesh_config: Dict[str, Any],
        mesh_data_dir: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Generate requested mesh.

        Without a ``regional`` block the request is for a global mesh, which is
        downloaded. With one, a regional mesh is generated. Tools the request
        needs, such as Metis for partitioning or the program that makes a
        regional mesh, are installed first if they are not already installed.
        A requested plot is drawn last, once the mesh and its partitions are
        complete.

        A finished mesh keeps a record of the request that made it,
        ``<name>.mesh.json``. A request matching an existing mesh of its name,
        in resolution, region and the version of the tool that made it, reuses
        it: only the partitions it lacks, or that another Metis version made,
        are made, and a plot is drawn if there is none. With ``overwrite``, the
        mesh is made again in a new directory, which replaces whatever has its
        name only once it is complete.

        Parameters
        ----------
        mesh_config : dict
            Optional ``resolution`` (default ``"120km"``) and ``name``
            (default ``"mesh_<resolution>"``). A global mesh, and a regional
            mesh cut from one with ``create_region``, must use one of the
            supported global resolutions. A ``project_hexes`` mesh may use any
            positive number of kilometers, such as ``"2km"``. Optional
            ``init_ranks`` and
            ``forecast_ranks`` are positive integers giving MPI rank counts to
            partition the mesh graph for. A count of 1 needs no partition.
            An optional ``regional`` dict holds exactly one method.
            ``{"project_hexes": {...}}`` needs ``center_lat``, ``center_lon``,
            ``extent_x_km`` and ``extent_y_km`` as numbers, and may have
            ``rotation_degrees`` to rotate the mesh counter-clockwise about its
            center. ``{"create_region": {shape: {...}}}`` holds exactly one
            shape: ``circle`` (``center_lat``, ``center_lon``, ``radius_km``),
            ``ellipse`` (``center_lat``, ``center_lon``, ``semi_major_axis_km``,
            ``semi_minor_axis_km``, optional ``orientation_degrees``),
            ``polygon`` (``vertices`` as ``[lat, lon]`` pairs, ``inside_lat``,
            ``inside_lon``) or ``channel`` (``upper_lat``, ``lower_lat``).
            An optional ``plot`` of True draws an image of the mesh, as
            ``plot_mesh()`` does with its defaults.

            Or a ``prompt`` describing the mesh, such as ``"a 15 km mesh over
            Texas"``, with the ``model`` that reads it, as for
            ``mesh_config_from_prompt()``. The prompt gives the resolution,
            the region and the name, so ``resolution`` and ``regional`` are
            not allowed. A ``name`` replaces the prompt's, and ``init_ranks``,
            ``forecast_ranks`` and ``plot`` are as above. Optional
            ``buffer_km`` and ``reasoning_effort`` are as for
            ``mesh_config_from_prompt()``.
        mesh_data_dir : str
            Directory where this request's mesh files are written
        overwrite : bool, optional
            Whether to make the mesh again and replace whatever has its name in
            ``mesh_data_dir``, by default False. Until the new mesh is complete,
            the old one stays, so a request that fails leaves it.

        Returns
        -------
        dict
            - static: path to static mesh file (in mesh_name subdirectory)
            - graph: path to graph.info file (in mesh_name subdirectory)
            - partitions: dict of {num_ranks: partition_file_path} (in mesh_name
              subdirectory), present only if a partition was requested
            - plot: path to the PNG image of the mesh (in mesh_name
              subdirectory), present only if a plot was requested and drawn
            - plot_error: why the plot failed, or that it was drawn without
              some map layers, present only then. A plot that fails does not
              fail the request.
            - mesh_config: for a request with a prompt, the request it was
              made into, which makes the same mesh again without the prompt
            - reused: True, present only if the mesh existed and was reused

        Raises
        ------
        FileExistsError
            If something of the mesh's name exists in ``mesh_data_dir`` that is
            not a finished mesh made from a request like this one, without
            ``overwrite``.
        ValueError
            If ``mesh_config`` is not a dict, or the name, resolution, a rank
            count or the ``regional`` block is invalid. A name must not be that
            of a downloaded global mesh file such as ``x1.2562.graph.info``.
            For a request with a prompt, also as ``mesh_config_from_prompt()``
            raises.
        RuntimeError
            If a resource the request needs is not defined (``service`` for a
            global or ``create_region`` mesh, ``compute`` for a regional mesh or
            partitioning), or a download, tool install, program run or partition
            fails, or for a request with a prompt, as
            ``mesh_config_from_prompt()`` raises.
        """
        self._validate_mesh_config(mesh_config)
        if not isinstance(overwrite, bool):
            raise ValueError(f"overwrite must be True or False, got {overwrite!r}")
        if "prompt" in mesh_config:
            return await self._generate_mesh_from_prompt(
                mesh_config, mesh_data_dir, overwrite
            )
        ranks = self._partition_ranks(mesh_config)
        regional = mesh_config.get("regional")
        resolution = mesh_config.get("resolution", "120km")
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")

        # Check resources before creating anything, so a request that cannot
        # finish downloads nothing and leaves nothing behind. Installing a tool
        # that is missing checks that it has the resources it needs.
        downloads = regional is None or "create_region" in regional
        needed = ["service"] if downloads else []
        if regional is not None or ranks:
            needed.append("compute")
        self._require_resources("This mesh request", *needed)
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)
        mesh_dir = request_mesh_data_dir / mesh_name
        identity = self._mesh_identity(mesh_config)
        if overwrite:
            result = await self._replace_mesh(
                mesh_dir, mesh_name, mesh_config, identity, ranks
            )
        else:
            try:
                mesh_dir.mkdir()
            except FileExistsError:
                result = await self._reuse_mesh(mesh_dir, mesh_name, identity, ranks)
            else:
                result = await self._make_mesh(
                    mesh_dir,
                    mesh_name,
                    mesh_config,
                    identity,
                    ranks,
                    request_mesh_data_dir,
                )
        if mesh_config.get("plot"):
            image = self._plot_path(Path(result["static"]))
            if result.get("reused") and image.is_file():
                result["plot"] = str(image)
            else:
                # The mesh is complete, so a plot that fails only reports why
                try:
                    plot, warning = await self._plot(result["static"])
                except Exception as e:
                    result["plot_error"] = str(e)
                else:
                    result["plot"] = plot
                    if warning is not None:
                        result["plot_error"] = warning
        return result

    async def _make_mesh(
        self,
        mesh_dir: Path,
        mesh_name: str,
        mesh_config: Dict[str, Any],
        identity: Dict[str, Any],
        ranks: List[int],
        mesh_data_dir: Path,
    ) -> Dict[str, Any]:
        """Make a validated request's mesh and partitions in a new directory.

        The record is written last, once everything else is complete. If
        anything fails, the directory is removed, so the name can be used
        again.
        """
        regional = mesh_config.get("regional")
        resolution = mesh_config.get("resolution", "120km")
        result: Dict[str, Any] = {}
        complete = False
        try:
            # Install tools before downloading so a failure wastes no download
            await self._install_tools(mesh_config, plotting=False)
            if regional is None:
                # Global mesh, downloaded precomputed from UCAR
                paths = await self.download_global_mesh(resolution, str(mesh_data_dir))
                for key, suffix in (("static", "static.nc"), ("graph", "graph.info")):
                    link = mesh_dir / f"{mesh_name}.{suffix}"
                    link.symlink_to(Path(paths[key]).resolve())
                    result[key] = str(link)
            elif "project_hexes" in regional:
                result = await self._generate_project_hexes_mesh(
                    mesh_dir, mesh_name, resolution, regional["project_hexes"]
                )
            else:
                # Cut from the global mesh of the same resolution
                result = await self._generate_create_region_mesh(
                    mesh_dir,
                    mesh_name,
                    resolution,
                    regional["create_region"],
                    mesh_data_dir,
                )
            if ranks:
                result["partitions"] = await self.partition_mesh(result["graph"], ranks)
            self._write_mesh_record(
                mesh_dir,
                mesh_name,
                identity,
                {rank: self.metis_version for rank in ranks},
            )
            complete = True
        finally:
            # Remove the failed mesh completely so the name can be used again
            if not complete:
                self._remove_mesh_dir(mesh_dir)
        return result

    async def _replace_mesh(
        self,
        mesh_dir: Path,
        mesh_name: str,
        mesh_config: Dict[str, Any],
        identity: Dict[str, Any],
        ranks: List[int],
    ) -> Dict[str, Any]:
        """Make a request's mesh again, and put it in place of whatever has its name.

        The mesh is made in a new directory beside ``mesh_dir``. Only once it is
        complete is what has the name moved aside, the new mesh moved in, and
        the old one removed, so a request that fails leaves the old one. The
        files inside are named after the mesh, not its directory, so moving the
        directory leaves them whole.
        """
        tag = uuid.uuid4().hex[:8]
        build_dir = mesh_dir.with_name(f".{mesh_name}.new-{tag}")
        build_dir.mkdir()
        made = await self._make_mesh(
            build_dir, mesh_name, mesh_config, identity, ranks, mesh_dir.parent
        )
        old = mesh_dir.with_name(f".{mesh_name}.old-{tag}")
        replaced = mesh_dir.exists() or mesh_dir.is_symlink()
        try:
            if replaced:
                mesh_dir.rename(old)
            build_dir.rename(mesh_dir)
        except OSError:
            # Put back what had the name, so the request changes nothing
            if replaced and (old.exists() or old.is_symlink()):
                try:
                    old.rename(mesh_dir)
                except OSError:
                    logger.warning(
                        "Could not move %s back to %s, which it replaced", old, mesh_dir
                    )
            self._remove_mesh_dir(build_dir)
            raise
        if replaced:
            self._remove_replaced(old)

        def moved(path: str) -> str:
            return str(mesh_dir / Path(path).relative_to(build_dir))

        result: Dict[str, Any] = {key: moved(made[key]) for key in ("static", "graph")}
        if "partitions" in made:
            result["partitions"] = {
                rank: moved(path) for rank, path in made["partitions"].items()
            }
        return result

    @staticmethod
    def _remove_replaced(old: Path) -> None:
        """Remove what a new mesh replaced: a directory, a file or a link.

        Links are removed without following them, so shared downloaded mesh
        files they point to are left alone. A failure is logged and never
        raised, since the new mesh is already in place.
        """
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            try:
                old.unlink()
            except OSError:
                pass
        if old.exists() or old.is_symlink():
            logger.warning("Could not remove %s, which a new mesh replaced", old)

    async def _reuse_mesh(
        self,
        mesh_dir: Path,
        mesh_name: str,
        identity: Dict[str, Any],
        ranks: List[int],
    ) -> Dict[str, Any]:
        """Return an existing mesh a request matches, with the partitions it asks for.

        Partitions it lacks, or that another Metis version made, are made. They
        leave the record before any is made, so if one fails, the record names
        only partitions that are as it says, and the next request makes the
        others.

        Raises
        ------
        FileExistsError
            If the name is not a finished mesh, or one made from a different
            request.
        """
        record = self._read_mesh_record(mesh_dir, mesh_name)
        replace = "Use another name, or pass overwrite=True to replace it."
        unfinished = FileExistsError(
            f"{mesh_dir} exists but is not a finished mesh: it is incomplete, "
            f"or another request is making it. {replace}"
        )
        if record is None:
            raise unfinished
        try:
            differences = self._mesh_differences(record["mesh"], identity)
        except (AttributeError, TypeError, ValueError):
            # A record no request writes, so not a finished mesh's
            raise unfinished from None
        if differences:
            raise FileExistsError(
                f"Mesh {mesh_name!r} in {mesh_dir.parent} was made from a different "
                f"request: {'; '.join(differences)}. {replace}"
            )
        static = mesh_dir / f"{mesh_name}.static.nc"
        graph = mesh_dir / f"{mesh_name}.graph.info"
        missing = [path.name for path in (static, graph) if not path.is_file()]
        if missing:
            raise FileExistsError(
                f"Mesh {mesh_name!r} in {mesh_dir.parent} is missing "
                f"{', '.join(missing)}. {replace}"
            )
        result: Dict[str, Any] = {"static": str(static), "graph": str(graph)}
        if ranks:
            made = {
                int(rank): version for rank, version in record["partitions"].items()
            }
            needed = [
                rank
                for rank in ranks
                if made.get(rank) != self.metis_version
                or not Path(f"{graph}.part.{rank}").is_file()
            ]
            if needed:
                kept = {rank: v for rank, v in made.items() if rank not in needed}
                self._write_mesh_record(mesh_dir, mesh_name, record["mesh"], kept)
                await self.partition_mesh(str(graph), needed)
                kept.update({rank: self.metis_version for rank in needed})
                self._write_mesh_record(mesh_dir, mesh_name, record["mesh"], kept)
            result["partitions"] = {rank: f"{graph}.part.{rank}" for rank in ranks}
        result["reused"] = True
        return result


@chiltepin_agent()
class MeshAgent(_MeshBehavior):
    """Deployable agent that makes, partitions and plots MPAS meshes.

    Global meshes are downloaded, and regional meshes are generated with
    ``project_hexes`` or cut from a global mesh with ``create_region``, from a
    full request or from a prompt an LLM reads.

    Parameters
    ----------
    work_dir : str
        Directory where mesh tools will be installed and agent logs will be written
    metis_version : str, optional
        Metis version to install for partitioning, by default "5.2.1"
    mpas_tools_version : str, optional
        MPAS-Tools tag to install for ``project_hexes`` meshes, by default "2.0.0"
    limited_area_version : str, optional
        MPAS-Limited-Area version to install for ``create_region`` meshes, by
        default "2.2"
    nominatim_contact : str, optional
        Email address sent to OpenStreetMap's Nominatim service when a prompt
        names a place Natural Earth lacks, such as a city. Without it, such
        places are not looked up.
    """
