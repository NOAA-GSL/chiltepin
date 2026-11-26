# SPDX-License-Identifier: Apache-2.0
"""Files and counts for making a global MPAS mesh of any resolution.

A quasi-uniform global mesh is made as Duda's "Generating meshes for
MPAS-Atmosphere" describes: an icosahedron is divided into a regular set of
points with ``sphtri_subdiv``, JIGSAW meshes the sphere starting from those
points, and ``mkgrid`` makes the MPAS mesh from the points and their
triangulation. This module writes the files that pipeline passes along, in the
formats of the ``mpas_jigsaw`` repository's ``create_hfun.py``,
``convert_jigsaw.py`` and ``create_density.py``, and counts what it makes.

The counts follow the UCAR meshes. Each icosahedron edge is divided into ``N``
parts, which gives ``10 N^2 + 2`` cells, and the UCAR mesh of ``dx`` km has
``N = 7680 / dx``, exactly, for all 16 of them. A mesh of any other resolution
gets the nearest whole ``N``, and so has the spacing ``7680 / N``.

Only the standard library is imported, so the agent can import this without
numpy.
"""

import math
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Tuple

# ``dx = DIVISION_FACTOR / N`` for the UCAR meshes
DIVISION_FACTOR = 7680.0
# Fewest divisions of an icosahedron edge. N = 1 would be the icosahedron's own
# 12 cells, which JIGSAW is not given to mesh.
MIN_DIVISION = 2
# Most divisions: mkgrid numbers the 30 N^2 edges with 32-bit integers, and
# sphtri_subdiv counts 20 N^2 triangles in 32 bits. The edges are the limit, at
# about 0.91 km.
MAX_DIVISION = 8460
# Lines of density written at once
DENSITY_CHUNK = 1_000_000

# Names of the files the pipeline passes along, in the mesh directory
GEOMETRY_FILE = "GEOM.msh"
ICOSAHEDRON_FILE = "ICOS.msh"
INITIAL_FILE = "INIT.msh"
HFUN_FILE = "HFUN.msh"
JIG_FILE = "MESH.jig"
MESH_FILE = "MESH.msh"
# What mkgrid reads
VERTICES_FILE = "SaveVertices"
TRIANGLES_FILE = "SaveTriangles"
DENSITY_FILE = "SaveDensity"
CODE_FILE = "SaveCode"
CONFIG_FILE = "SaveConfig"
# Files of no use once a mesh is complete, and large for a fine one. The sphere,
# the icosahedron, MESH.jig and the logs stay.
INTERMEDIATE_FILES = (
    INITIAL_FILE,
    HFUN_FILE,
    MESH_FILE,
    VERTICES_FILE,
    TRIANGLES_FILE,
    DENSITY_FILE,
    CODE_FILE,
    CONFIG_FILE,
)


def division(resolution_km: float) -> int:
    """Return the number of parts each icosahedron edge is divided into.

    It is the whole number nearest ``7680 / resolution_km``.
    """
    return round(DIVISION_FACTOR / resolution_km)


def division_valid(parts: int) -> bool:
    """Whether a division can be made into a mesh by the tools."""
    return MIN_DIVISION <= parts <= MAX_DIVISION


def cell_count(parts: int) -> int:
    """Return the number of cells of the mesh of an icosahedron divided in ``parts``."""
    return 10 * parts * parts + 2


def nominal_spacing_km(parts: int) -> float:
    """Return the nominal spacing of the mesh, as the UCAR meshes name it."""
    return DIVISION_FACTOR / parts


def hfun_text(spacing_km: float) -> str:
    """Return a JIGSAW mesh-size file holding ``spacing_km`` all over the sphere.

    It is a longitude-latitude grid, as ``create_hfun.py`` writes. The size is
    constant, so the grid can be coarse.
    """
    longitudes = [-math.pi + 2.0 * math.pi * i / 36 for i in range(37)]
    latitudes = [-0.5 * math.pi + math.pi * j / 18 for j in range(19)]
    lines = ["MSHID=3;ellipsoid-grid", "NDIMS=2"]
    lines.append(f"COORD=1;{len(longitudes)}")
    lines += [f"{value:.17g}" for value in longitudes]
    lines.append(f"COORD=2;{len(latitudes)}")
    lines += [f"{value:.17g}" for value in latitudes]
    lines.append(f"VALUE={len(longitudes) * len(latitudes)}; 1")
    lines += [f"{spacing_km:.17g}"] * (len(longitudes) * len(latitudes))
    return "\n".join(lines) + "\n"


def jig_text() -> str:
    """Return the JIGSAW configuration: the sphere, the size file, and the start."""
    return (
        "VERBOSITY=1\n"
        f"GEOM_FILE={GEOMETRY_FILE}\n"
        f"HFUN_FILE={HFUN_FILE}\n"
        "HFUN_SCAL=absolute\n"
        "HFUN_HMAX=inf\n"
        "HFUN_HMIN=0.0\n"
        f"INIT_FILE={INITIAL_FILE}\n"
        f"MESH_FILE={MESH_FILE}\n"
        "MESH_DIMS=2\n"
        "OPTM_QLIM=0.9375\n"
    )


def density_code(spacing_km: float) -> str:
    """Return the source of the mesh density function, which grid.nc records.

    It is the uniform function of Duda's slides, as the ``hfun.py`` of the
    ``mpas_jigsaw`` repository has it: the same size everywhere.
    """
    return (
        f"hfun_min = {spacing_km!r}\n"
        "\n"
        "\n"
        "def get_hfun(lon, lat):\n"
        "    import numpy as np\n"
        "\n"
        "    return np.full_like(lon, hfun_min)\n"
    )


def config_text(
    requested_km: float, parts: int, spacing_km: float, tools: Dict[str, str]
) -> str:
    """Return the configuration grid.nc records, as the ``mpas_jigsaw`` YAML.

    The ``mesh`` block is what that repository's ``mesh_quality.py`` reads. The
    ``chiltepin`` block says how the mesh was made, and is ignored there.
    """
    lines = [
        "mesh:",
        f"  resolution: {spacing_km!r}",
        "chiltepin:",
        f"  requested_resolution_km: {requested_km!r}",
        f"  division: {parts}",
        f"  nominal_spacing_km: {spacing_km!r}",
    ]
    # Quoted, since a commit of only digits would be read as a number
    lines += [f"  {name}: '{commit}'" for name, commit in sorted(tools.items())]
    return "\n".join(lines) + "\n"


def convert_jigsaw(mesh_file: Path, directory: Path) -> Tuple[int, int]:
    """Write the points and triangles of a JIGSAW mesh in the files mkgrid reads.

    ``SaveVertices`` gets the coordinates of each point and ``SaveTriangles``
    the indices of each triangle's points, one per line, as ``convert_jigsaw.py``
    writes them, with JIGSAW's own digits. The file is read a line at a time.

    Returns
    -------
    tuple
        The number of points and the number of triangles.

    Raises
    ------
    ValueError
        If the file lacks points or triangles, is cut short, or is not the
        triangulation of a closed surface, which has ``2 P - 4`` triangles for
        ``P`` points.
    """
    declared = {"POINT": 0, "TRIA3": 0}
    written = {"POINT": 0, "TRIA3": 0}
    names = {"POINT": VERTICES_FILE, "TRIA3": TRIANGLES_FILE}
    with ExitStack() as files:
        source = files.enter_context(open(mesh_file))
        sinks = {
            section: files.enter_context(open(directory / name, "w"))
            for section, name in names.items()
        }
        section, left = "", 0
        for line in source:
            if left > 0:
                # The last column is JIGSAW's marker, and is dropped
                sinks[section].write(line.rsplit(";", 1)[0].replace(";", " ") + "\n")
                written[section] += 1
                left -= 1
            elif line.startswith(("POINT=", "TRIA3=")):
                section, count = line.split("=")
                left = declared[section] = int(count)
    for name, count in declared.items():
        if count == 0:
            raise ValueError(f"{mesh_file} has no {name} section")
        if written[name] != count:
            raise ValueError(f"{mesh_file} has {written[name]} of {count} {name} lines")
    if declared["TRIA3"] != 2 * declared["POINT"] - 4:
        raise ValueError(
            f"{mesh_file} has {declared['POINT']} points and {declared['TRIA3']} "
            "triangles, so it is not the triangulation of a sphere"
        )
    return declared["POINT"], declared["TRIA3"]


def write_density_files(
    directory: Path, points: int, spacing_km: float, config: str
) -> None:
    """Write the density, with its source and configuration, for mkgrid.

    The density is 1 at every point, since the mesh is uniform.
    """
    with open(directory / DENSITY_FILE, "w") as density:
        remaining = points
        while remaining > 0:
            count = min(remaining, DENSITY_CHUNK)
            density.write("1.0\n" * count)
            remaining -= count
    (directory / CODE_FILE).write_text(density_code(spacing_km))
    (directory / CONFIG_FILE).write_text(config)
