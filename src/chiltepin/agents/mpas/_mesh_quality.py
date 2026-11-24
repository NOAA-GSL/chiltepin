# SPDX-License-Identifier: Apache-2.0
"""Checks of an MPAS mesh file, for the mesh agent's ``check_mesh_quality``.

The checks are in two kinds, and the report keeps them apart:

* ``errors`` name a broken mesh: connectivity that contradicts itself, cells
  that are not a Voronoi mesh of their centers, areas that do not add up, or,
  for a global mesh, cell shapes that cannot close a sphere.
* ``warnings`` name red flags about quality, such as many non-hexagons or a
  steep change of cell size. They do not say that a mesh is unusable. Meshes
  with refinement regions made by JIGSAW always have non-hexagons in their
  transition zones, so whether a result is acceptable is for the user to say.

Everything numeric is a measurement in the report, whether or not it raised a
warning. The thresholds are starting points, set from real meshes: each is
several times larger than the worst value found in UCAR meshes, MPAS-Tools
``project_hexes`` meshes, ``create_region`` meshes and meshes made with
JIGSAW and ``mkgrid``.

Only the standard library is imported at the top. numpy and netCDF4 are
imported where the checks run, which is on the ``compute`` resource.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# MPAS's value for the radius of the Earth, in km, which a mesh's lengths are
# scaled to whatever sphere radius the file uses
EARTH_RADIUS_KM = 6371.229

# Items handled at once, so the working arrays stay small for very large meshes
BLOCK = 2_000_000

# Variables every check reads
REQUIRED_VARIABLES = (
    "nEdgesOnCell",
    "cellsOnCell",
    "cellsOnEdge",
    "verticesOnEdge",
    "cellsOnVertex",
    "dcEdge",
    "dvEdge",
    "areaCell",
    "areaTriangle",
    "kiteAreasOnVertex",
    "meshDensity",
    "nominalMinDc",
    "xCell",
    "yCell",
    "zCell",
    "xVertex",
    "yVertex",
    "zVertex",
)

# A global mesh whose density varies by less than this is quasi-uniform
UNIFORM_DENSITY_SPREAD = 1.0e-6
# Pentagons a closed mesh of pentagons and hexagons has, by Euler's formula
GLOBAL_PENTAGONS = 12
# Percent of cells that are not hexagons above which to warn
NON_HEXAGON_WARN_PERCENT = 10.0

# Where the Voronoi edge's plane tilts from the bisector plane of its two cell
# centers (the sine of the angle). Real meshes reach 1.1e-4 in single precision.
TILT_WARN, TILT_ERROR = 1.0e-3, 1.0e-2
# Relative error of a vertex's kites against its triangle's area. Real meshes
# reach 1.8e-7.
KITE_TRIANGLE_WARN, KITE_TRIANGLE_ERROR = 1.0e-5, 1.0e-3
# Relative error of a cell's kites against its area. Real meshes reach 1.8e-6.
KITE_CELL_WARN, KITE_CELL_ERROR = 1.0e-4, 1.0e-2
# Relative error of a global mesh's total area against the sphere's
AREA_SUM_WARN, AREA_SUM_ERROR = 1.0e-5, 1.0e-3
# Change of nominal cell size per distance that Duda's mesh guidelines call safe
# ("a few percent"), and the one above which to warn
GRADIENT_REFERENCE = 0.03
GRADIENT_WARN = 0.05
# Ratio of a Voronoi edge to the distance between its cell centers. A regular
# hexagon has 0.577. The band is reported, and values outside the limits warn.
REGULARITY_BAND = (0.4, 0.8)
REGULARITY_LIMITS = (0.1, 2.0)
# Most cells per rank, over the mean, that a partition may have before a warning.
# Metis with its default tolerance keeps to 1.03 on large meshes, and reached 1.04
# with 19 cells per rank.
PARTITION_IMBALANCE_WARN = 1.10
# A regional mesh's boundary layers, from the edge in
BOUNDARY_LAYERS = 7
# Fraction of the requested finest spacing the mesh's own may differ by
REQUESTED_TOLERANCE = 0.01
# Vertex tolerance of the Delaunay test: the sine of an angle, so a vertex this
# far outside its triangle is still inside
OUTSIDE_TOLERANCE = 1.0e-6

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
# Most bins of the cell size histogram
MAX_BINS = 200


def nice_bin_width(width: float) -> float:
    """Round a positive bin width to the nearest of 1, 2, 2.5, 5 and 10 times a power of 10."""
    scale = 10.0 ** math.floor(math.log10(width))
    choices = [scale * factor for factor in (1.0, 2.0, 2.5, 5.0, 10.0)]
    return min(choices, key=lambda choice: abs(choice - width))


def bin_edges(
    sizes: Any, width: Optional[float] = None, center: Optional[float] = None
) -> Any:
    """Return histogram bin edges for cell sizes, fixed or by the Freedman-Diaconis rule.

    With a ``center``, one bin is centered on it, so a quasi-uniform mesh's
    target size sits in the middle of a bin. The bin count is limited.
    """
    import numpy as np

    low, high = float(sizes.min()), float(sizes.max())
    automatic = width is None
    if automatic:
        q25, q75 = np.percentile(sizes, [25, 75])
        width = 2.0 * (q75 - q25) / sizes.size ** (1.0 / 3.0)
    if not math.isfinite(width) or width <= 0.0:
        width = high - low
    if width == 0.0:
        return np.array([low - 0.5, high + 0.5])
    width = max(width, (high - low) / MAX_BINS)
    if automatic:
        width = nice_bin_width(max(width, (high - low) / 50.0))
    if center is not None:
        first = center + width * math.floor((low - center) / width)
        last = center + width * math.ceil((high - center) / width)
        return np.arange(first - 0.5 * width, last + 1.5 * width, width)
    return np.arange(
        math.floor(low / width) * width, math.ceil(high / width) * width + width, width
    )


def _percentiles(values: Any) -> Dict[str, float]:
    import numpy as np

    return {
        f"p{p:02d}": float(v)
        for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))
    }


def _stats(values: Any) -> Dict[str, float]:
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _normalized(vectors: Any) -> Any:
    """Return vectors scaled to length 1, leaving zero vectors zero."""
    import numpy as np

    length = np.linalg.norm(vectors, axis=1)[:, None]
    return np.divide(vectors, length, out=np.zeros_like(vectors), where=length > 0.0)


def _blocks(count: int):
    for start in range(0, count, BLOCK):
        yield slice(start, min(start + BLOCK, count))


class _Report:
    """What the checks have found so far."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.measurements: Dict[str, Any] = {}

    def level(self, value: float, warn: float, error: float, what: str) -> None:
        """Record an error or a warning for a measurement above its limits."""
        if value > error:
            self.errors.append(f"{what} is {value:.3g}, above the limit {error:g}")
        elif value > warn:
            self.warnings.append(f"{what} is {value:.3g}, above {warn:g}")


def _read(dataset: Any, name: str) -> Any:
    import numpy as np

    return np.asarray(dataset.variables[name][:])


def check_mesh(path: str, resolution_km: Optional[float] = None) -> Dict[str, Any]:
    """Check a mesh file and return the report.

    Parameters
    ----------
    path : str
        An MPAS mesh file: a static or grid file.
    resolution_km : float, optional
        The finest cell spacing that was asked for. The mesh's nominal
        minimum spacing is compared with it.

    Returns
    -------
    dict
        ``passed`` (True when there are no errors), ``summary``, ``errors``,
        ``warnings`` and ``measurements``, all of which are JSON-serializable.

    Raises
    ------
    FileNotFoundError
        If there is no file at ``path``.
    ValueError
        If the file is not a netCDF file, or lacks variables of an MPAS mesh.
    """
    from netCDF4 import Dataset

    mesh = Path(path)
    if not mesh.is_file():
        raise FileNotFoundError(f"No mesh file at {mesh}")
    try:
        dataset = Dataset(str(mesh))
    except OSError as e:
        raise ValueError(f"{mesh} is not a netCDF file: {e}") from e
    with dataset:
        missing = [v for v in REQUIRED_VARIABLES if v not in dataset.variables]
        if missing:
            raise ValueError(
                f"{mesh} is not an MPAS mesh, it lacks: {', '.join(missing)}"
            )
        dataset.set_auto_mask(False)
        report = _Report()
        _check(dataset, report, resolution_km, mesh)
    errors, warnings = report.errors, report.warnings
    counts = f"{len(errors)} error(s) and {len(warnings)} warning(s)"
    return {
        "passed": not errors,
        "summary": ("Passed" if not errors else "Failed") + f" with {counts}",
        "errors": errors,
        "warnings": warnings,
        "measurements": report.measurements,
    }


def _check(
    dataset: Any, report: _Report, resolution_km: Optional[float], mesh: Path
) -> None:
    """Run every check, which stop short of the geometry if indices are invalid."""
    import numpy as np

    radius = float(getattr(dataset, "sphere_radius", 1.0))
    to_km = EARTH_RADIUS_KM / radius
    n_cells = dataset.dimensions["nCells"].size
    n_vertices = dataset.dimensions["nVertices"].size
    n_edges = dataset.dimensions["nEdges"].size
    sides = _read(dataset, "nEdgesOnCell").astype(np.int64)
    on_cell = _read(dataset, "cellsOnCell").astype(np.int64)
    on_edge = _read(dataset, "cellsOnEdge").astype(np.int64)
    vertices_on_edge = _read(dataset, "verticesOnEdge").astype(np.int64)
    on_vertex = _read(dataset, "cellsOnVertex").astype(np.int64)
    interior = np.flatnonzero((on_edge > 0).all(axis=1))
    kind = "global" if interior.size == n_edges else "regional"
    report.measurements.update(
        {
            "kind": kind,
            "cells": int(n_cells),
            "edges": int(n_edges),
            "vertices": int(n_vertices),
            "precision": str(dataset.variables["xCell"].dtype),
            "sphere_radius": radius,
        }
    )
    # Unused slots of the per-cell arrays hold anything, so only the first
    # nEdgesOnCell of each row count
    used = np.arange(on_cell.shape[1])[None, :] < sides[:, None]
    if not _indices_valid(
        report,
        n_cells,
        n_vertices,
        sides,
        on_cell,
        used,
        on_edge,
        vertices_on_edge,
        on_vertex,
    ):
        return
    if interior.size == 0:
        report.errors.append("The mesh has no edge between two cells")
        return
    _connectivity(report, on_cell, used, on_edge, on_vertex, interior)
    _cell_shapes(report, kind, sides, on_cell, used, _read(dataset, "meshDensity"))
    cell_xyz = (
        np.column_stack(
            [
                _read(dataset, name).astype(np.float64)
                for name in ("xCell", "yCell", "zCell")
            ]
        )
        / radius
    )
    vertex_xyz = (
        np.column_stack(
            [
                _read(dataset, name).astype(np.float64)
                for name in ("xVertex", "yVertex", "zVertex")
            ]
        )
        / radius
    )
    _voronoi(
        report, cell_xyz, vertex_xyz, on_edge, vertices_on_edge, on_vertex, interior
    )
    _areas(dataset, report, kind, radius, n_cells, on_vertex)
    _sizes(dataset, report, to_km, on_edge, interior, resolution_km)
    _boundary_layers(dataset, report, kind, on_cell, used)
    _partitions(mesh, report, n_cells, on_edge, interior)


def _indices_valid(
    report,
    n_cells,
    n_vertices,
    sides,
    on_cell,
    used,
    on_edge,
    vertices_on_edge,
    on_vertex,
) -> bool:
    """Record connectivity indices that are out of range, and return whether all are in."""
    checks = (
        ("nEdgesOnCell", (sides < 3) | (sides > on_cell.shape[1])),
        ("cellsOnCell", used & ((on_cell < 0) | (on_cell > n_cells))),
        ("cellsOnEdge", (on_edge < 0) | (on_edge > n_cells)),
        ("verticesOnEdge", (vertices_on_edge < 1) | (vertices_on_edge > n_vertices)),
        ("cellsOnVertex", (on_vertex < 0) | (on_vertex > n_cells)),
    )
    valid = True
    for name, bad in checks:
        count = int(bad.sum())
        if count:
            valid = False
            report.errors.append(f"{name} has {count} value(s) out of range")
    return valid


def _connectivity(report, on_cell, used, on_edge, on_vertex, interior) -> None:
    """Check that neighbors, edges and vertices agree with each other."""
    pairs = on_edge[interior]
    same = int((pairs[:, 0] == pairs[:, 1]).sum())
    if same:
        report.errors.append(f"{same} edge(s) have the same cell on both sides")
    missing = 0
    for block in _blocks(interior.size):
        first, second = pairs[block, 0] - 1, pairs[block, 1] - 1
        # Each cell of an edge must list the other as a neighbor
        listed_first = ((on_cell[first] == pairs[block, 1][:, None]) & used[first]).any(
            axis=1
        )
        listed_second = (
            (on_cell[second] == pairs[block, 0][:, None]) & used[second]
        ).any(axis=1)
        missing += int((~listed_first).sum() + (~listed_second).sum())
    if missing:
        report.errors.append(
            f"{missing} neighbor(s) that cellsOnEdge gives are missing from cellsOnCell"
        )
    # Neighbors that no edge accounts for are as wrong as missing ones
    extra = int(((on_cell > 0) & used).sum()) - 2 * interior.size
    if extra > 0:
        report.errors.append(
            f"cellsOnCell lists {extra} more neighbor(s) than cellsOnEdge gives"
        )
    known = on_vertex > 0
    duplicated = 0
    for a, b in ((0, 1), (1, 2), (0, 2)):
        duplicated += int(
            (known[:, a] & known[:, b] & (on_vertex[:, a] == on_vertex[:, b])).sum()
        )
    if duplicated:
        report.errors.append(f"{duplicated} vertex(es) list a cell twice")
    report.measurements["boundary_edges"] = int(on_edge.shape[0] - interior.size)


def _cell_shapes(report, kind, sides, on_cell, used, density) -> None:
    """Count cell shapes and the defects among them."""
    import numpy as np

    values, counts = np.unique(sides, return_counts=True)
    census = {str(int(v)): int(c) for v, c in zip(values, counts)}
    n_cells = sides.size
    non_hex = int((sides != 6).sum())
    pentagons = int((sides == 5).sum())
    larger = sides >= 7
    touching = 0
    for slot in range(on_cell.shape[1]):
        neighbor = on_cell[:, slot]
        known = used[:, slot] & (neighbor > 0)
        touching += int(
            ((sides == 5) & known & larger[np.where(known, neighbor - 1, 0)]).sum()
        )
    density_spread = float(density.max() / density.min() - 1.0)
    uniform = kind == "global" and density_spread <= UNIFORM_DENSITY_SPREAD
    report.measurements.update(
        {
            "cell_sides": census,
            "non_hexagons": {
                "count": non_hex,
                "percent": 100.0 * non_hex / n_cells,
                "per_1000_cells": 1000.0 * non_hex / n_cells,
            },
            "defects": {
                "pentagons": pentagons,
                "heptagons_or_larger": int(larger.sum()),
                "pentagons_touching_larger": touching,
            },
            "expectation": "quasi_uniform" if uniform else "variable",
        }
    )
    if kind == "global":
        euler = int((6 - sides).sum())
        report.measurements["euler_sum"] = euler
        if euler != GLOBAL_PENTAGONS:
            report.errors.append(
                f"The sum of (6 - sides) over cells is {euler}, and a closed mesh "
                f"has {GLOBAL_PENTAGONS}"
            )
    if uniform and (pentagons != GLOBAL_PENTAGONS or non_hex != pentagons):
        report.warnings.append(
            f"The mesh is quasi-uniform and has {pentagons} pentagons and "
            f"{non_hex - pentagons} other non-hexagons, where the minimum is "
            f"{GLOBAL_PENTAGONS} pentagons and no others"
        )
    if 100.0 * non_hex / n_cells > NON_HEXAGON_WARN_PERCENT:
        report.warnings.append(
            f"{100.0 * non_hex / n_cells:.1f}% of cells are not hexagons, above "
            f"{NON_HEXAGON_WARN_PERCENT:g}%. Transition zones of refined meshes "
            "made with JIGSAW account for some, and the mesh may still suit"
        )
    if int((sides < 5).sum()):
        report.warnings.append(
            f"{int((sides < 5).sum())} cell(s) have fewer than 5 sides"
        )


def _voronoi(
    report, cell_xyz, vertex_xyz, on_edge, vertices_on_edge, on_vertex, interior
) -> None:
    """Check that edges bisect their cell centers, and that vertices are inside triangles."""
    import numpy as np

    tilt = 0.0
    degenerate = 0
    for block in _blocks(interior.size):
        edges = interior[block]
        a = cell_xyz[on_edge[edges, 0] - 1]
        b = cell_xyz[on_edge[edges, 1] - 1]
        u = vertex_xyz[vertices_on_edge[edges, 0] - 1]
        w = vertex_xyz[vertices_on_edge[edges, 1] - 1]
        normal = _normalized(np.cross(u, w))
        direction = _normalized(b - a)
        degenerate += int((np.linalg.norm(np.cross(u, w), axis=1) <= 0.0).sum())
        tilt = max(
            tilt, float(np.linalg.norm(np.cross(normal, direction), axis=1).max())
        )
    report.measurements["voronoi_tilt"] = tilt
    report.level(
        tilt,
        TILT_WARN,
        TILT_ERROR,
        "The tilt of Voronoi edges from the bisector of their cell centers",
    )
    if degenerate:
        report.warnings.append(f"{degenerate} Voronoi edge(s) have zero length")
    full = np.flatnonzero((on_vertex > 0).all(axis=1))
    obtuse = 0
    for block in _blocks(full.size):
        rows = full[block]
        a, b, c = (cell_xyz[on_vertex[rows, k] - 1] for k in range(3))
        v = vertex_xyz[rows]
        orientation = np.sign((a * np.cross(b, c)).sum(axis=1))
        # Sine of the angle from the vertex's plane to each side of the triangle
        inside = np.column_stack(
            [
                (v * _normalized(np.cross(p, q))).sum(axis=1)
                for p, q in ((b, c), (c, a), (a, b))
            ]
        )
        obtuse += int(
            ((inside * orientation[:, None]).min(axis=1) < -OUTSIDE_TOLERANCE).sum()
        )
    report.measurements["obtuse_triangles"] = obtuse
    if obtuse:
        report.warnings.append(
            f"{obtuse} Delaunay triangle(s) are obtuse, so their Voronoi vertex "
            "is outside the triangle"
        )


def _areas(dataset, report, kind, radius, n_cells, on_vertex) -> None:
    """Check that the cell, triangle and kite areas agree."""
    import numpy as np

    area = _read(dataset, "areaCell").astype(np.float64)
    triangle = _read(dataset, "areaTriangle").astype(np.float64)
    kites = _read(dataset, "kiteAreasOnVertex").astype(np.float64)
    sphere = 4.0 * math.pi * radius**2
    total = float(area.sum())
    report.measurements["area_fraction_of_sphere"] = total / sphere
    if kind == "global":
        report.level(
            abs(total / sphere - 1.0),
            AREA_SUM_WARN,
            AREA_SUM_ERROR,
            "The relative difference of the cell areas from the sphere's",
        )
    kite_triangle = float((np.abs(kites.sum(axis=1) - triangle) / triangle).max())
    report.measurements["kite_triangle_error"] = kite_triangle
    report.level(
        kite_triangle,
        KITE_TRIANGLE_WARN,
        KITE_TRIANGLE_ERROR,
        "The relative difference of kite areas from their triangle's area",
    )
    known = on_vertex > 0
    kite_cell = np.bincount(
        on_vertex[known], weights=kites[known], minlength=n_cells + 1
    )[1:]
    kite_cell_error = float((np.abs(kite_cell - area) / area).max())
    report.measurements["kite_cell_error"] = kite_cell_error
    report.level(
        kite_cell_error,
        KITE_CELL_WARN,
        KITE_CELL_ERROR,
        "The relative difference of kite areas from their cell's area",
    )


def _sizes(dataset, report, to_km, on_edge, interior, resolution_km) -> None:
    """Measure spacing, its histogram, shape regularity and the change of size."""
    import numpy as np

    dc = _read(dataset, "dcEdge").astype(np.float64)
    dv = _read(dataset, "dvEdge").astype(np.float64)
    area = _read(dataset, "areaCell").astype(np.float64)
    density = _read(dataset, "meshDensity").astype(np.float64)
    nominal_min = float(np.asarray(_read(dataset, "nominalMinDc")).ravel()[0])
    report.measurements["dc_edge_km"] = _stats(dc * to_km)
    report.measurements["dv_edge_km"] = _stats(dv * to_km)
    report.measurements["dc_edge_percentiles_km"] = _percentiles(dc * to_km)
    # The spacing of a regular hexagonal cell of each cell's area
    spacing = np.sqrt(2.0 * area * to_km**2 / math.sqrt(3.0))
    report.measurements["cell_spacing_km"] = {
        **_stats(spacing),
        **_percentiles(spacing),
    }
    target = nominal_min * to_km
    report.measurements["nominal_min_spacing_km"] = target
    uniform = report.measurements["expectation"] == "quasi_uniform"
    edges = bin_edges(
        spacing,
        width=nice_bin_width(0.1 * target) if uniform else None,
        center=target if uniform else None,
    )
    counts, edges = np.histogram(spacing, bins=edges)
    report.measurements["cell_spacing_histogram"] = {
        "bin_edges_km": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }
    low, high = REGULARITY_LIMITS
    ratio = dv[interior] / dc[interior]
    report.measurements["edge_regularity"] = {
        **_stats(ratio),
        **_percentiles(ratio),
        "regular_hexagon": 1.0 / math.sqrt(3.0),
        "fraction_outside_band": float(
            ((ratio < REGULARITY_BAND[0]) | (ratio > REGULARITY_BAND[1])).mean()
        ),
    }
    if ratio.min() < low or ratio.max() > high:
        report.warnings.append(
            f"The ratio of Voronoi edge to cell-center distance ranges from "
            f"{ratio.min():.3g} to {ratio.max():.3g}, outside {low:g} to {high:g}, "
            "so some cells are very distorted"
        )
    nominal = nominal_min * np.power(1.0 / density, 0.25)
    gradient = (
        np.abs(nominal[on_edge[interior, 0] - 1] - nominal[on_edge[interior, 1] - 1])
        / dc[interior]
    )
    report.measurements["size_gradient"] = {
        "max": float(gradient.max()),
        "p99": float(np.percentile(gradient, 99)),
        "fraction_above_reference": float((gradient > GRADIENT_REFERENCE).mean()),
        "reference": GRADIENT_REFERENCE,
    }
    if gradient.max() > GRADIENT_WARN:
        report.warnings.append(
            f"The nominal cell size changes by up to {gradient.max():.3g} per "
            f"distance, above {GRADIENT_WARN:g}. A few percent is advised"
        )
    if resolution_km is not None:
        difference = abs(target - resolution_km) / resolution_km
        report.measurements["requested_resolution_km"] = float(resolution_km)
        if difference > REQUESTED_TOLERANCE:
            report.warnings.append(
                f"The mesh's nominal finest spacing is {target:.4g} km and "
                f"{resolution_km:g} km was asked for"
            )


def _boundary_layers(dataset, report, kind, on_cell, used) -> None:
    """Check the boundary layers of a regional mesh that has them."""
    import numpy as np

    if kind != "regional" or "bdyMaskCell" not in dataset.variables:
        return
    mask = _read(dataset, "bdyMaskCell").astype(np.int64)
    if mask.max() <= 0:
        report.measurements["boundary_layers"] = None
        return
    layers = {
        str(layer): int((mask == layer).sum())
        for layer in range(1, BOUNDARY_LAYERS + 1)
    }
    report.measurements["boundary_layers"] = {
        "interior": int((mask == 0).sum()),
        **layers,
    }
    empty = [layer for layer, count in layers.items() if count == 0]
    if empty:
        report.warnings.append(
            f"Boundary layer(s) {', '.join(empty)} of {BOUNDARY_LAYERS} have no cells"
        )
    # A cell in layer k touches layer k - 1, or the layer is broken
    stranded = 0
    for layer in range(2, BOUNDARY_LAYERS + 1):
        cells = np.flatnonzero(mask == layer)
        touching = np.zeros(cells.size, dtype=bool)
        for slot in range(on_cell.shape[1]):
            neighbor = on_cell[cells, slot]
            known = used[cells, slot] & (neighbor > 0)
            touching |= known & (mask[np.where(known, neighbor - 1, 0)] == layer - 1)
        stranded += int((~touching).sum())
    if stranded:
        report.warnings.append(
            f"{stranded} boundary-layer cell(s) do not touch the layer inside them"
        )


def _partitions(
    mesh: Path, report: _Report, n_cells: int, on_edge: Any, interior: Any
) -> None:
    """Check the Metis partitions next to a mesh: ``<name>.graph.info.part.<ranks>``."""
    import numpy as np

    stem = mesh.name.removesuffix(".nc").removesuffix(".static")
    prefix = f"{stem}.graph.info.part."
    files = {}
    for path in mesh.parent.iterdir():
        count = path.name.removeprefix(prefix)
        if path.name.startswith(prefix) and count.isdigit() and int(count) > 1:
            files[int(count)] = path
    if not files:
        return
    partitions = {}
    for count in sorted(files):
        name = files[count].name
        try:
            ranks = np.array(files[count].read_text().split(), dtype=np.int64)
        except ValueError:
            report.errors.append(f"{name} does not hold rank numbers")
            continue
        if ranks.size != n_cells:
            report.errors.append(f"{name} has {ranks.size} entries for {n_cells} cells")
            continue
        if ranks.min() < 0 or ranks.max() >= count:
            report.errors.append(f"{name} has rank numbers outside 0 to {count - 1}")
            continue
        cells = np.bincount(ranks, minlength=count)
        first = ranks[on_edge[interior, 0] - 1]
        second = ranks[on_edge[interior, 1] - 1]
        cut = first != second
        # Each pair of ranks that share an edge, once, and so each rank's neighbors
        pairs = np.unique(
            np.minimum(first, second)[cut] * count + np.maximum(first, second)[cut]
        )
        neighbors = np.bincount(
            np.concatenate([pairs // count, pairs % count]), minlength=count
        )
        imbalance = float(cells.max() / cells.mean())
        partitions[str(count)] = {
            "cells_per_rank": _stats(cells),
            "imbalance": imbalance,
            "empty_ranks": int((cells == 0).sum()),
            "edge_cut": int(cut.sum()),
            "edge_cut_fraction": float(cut.mean()),
            "neighbor_ranks": _stats(neighbors),
        }
        if int((cells == 0).sum()):
            report.warnings.append(
                f"{name}: {int((cells == 0).sum())} of {count} ranks have no cells"
            )
        if imbalance > PARTITION_IMBALANCE_WARN:
            report.warnings.append(
                f"{name}: the busiest rank has {imbalance:.3g} times the mean number "
                f"of cells, above {PARTITION_IMBALANCE_WARN:g}"
            )
    if partitions:
        report.measurements["partitions"] = partitions


def format_lines(report: Dict[str, Any]) -> Sequence[str]:
    """Return a short text of a report's errors and warnings, for logs."""
    return (
        [report["summary"]]
        + [f"error: {e}" for e in report["errors"]]
        + [f"warning: {w}" for w in report["warnings"]]
    )
