# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh quality checks.

The meshes are small Voronoi meshes built here from a subdivided icosahedron,
with every variable the checks read, so the tests need no download. Each error
and warning is triggered by changing one thing in a copy of a valid mesh.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from chiltepin.agents.mpas import _mesh_quality as quality

SCRIPT = Path(quality.__file__)


def _icosahedron():
    t = (1.0 + math.sqrt(5.0)) / 2.0
    points = np.array(
        [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t)]
        + [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)],
        dtype=float,
    )
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    return points / np.linalg.norm(points, axis=1)[:, None], faces


def _subdivide(points, faces, levels):
    points = [p for p in points]
    for _ in range(levels):
        midpoints, new_faces = {}, []

        def middle(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                m = points[i] + points[j]
                points.append(m / np.linalg.norm(m))
                midpoints[key] = len(points) - 1
            return midpoints[key]

        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(points), faces


def _triangle_area(a, b, c):
    """Return the area of spherical triangles on the unit sphere."""
    triple = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    dots = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(triple, dots)


def _unit(v):
    return v / np.linalg.norm(v, axis=1)[:, None]


def mesh_arrays(levels=2):
    """Return the arrays of a global Voronoi mesh on the unit sphere."""
    points, faces = _subdivide(*_icosahedron(), levels)
    cells = np.array(faces)
    a, b, c = (points[cells[:, k]] for k in range(3))
    # Counter-clockwise seen from outside, with the vertex at the circumcenter
    flip = np.einsum("ij,ij->i", a, np.cross(b, c)) < 0
    cells[flip] = cells[flip][:, [0, 2, 1]]
    a, b, c = (points[cells[:, k]] for k in range(3))
    vertices = _unit(np.cross(b - a, c - a))
    vertices *= np.sign(np.einsum("ij,ij->i", vertices, a + b + c))[:, None]
    n_cells, n_vertices = len(points), len(cells)
    edge_index, edge_vertices = {}, []
    for v, tri in enumerate(cells):
        for k in range(3):
            key = tuple(sorted((tri[k], tri[(k + 1) % 3])))
            if key not in edge_index:
                edge_index[key] = len(edge_index)
                edge_vertices.append([])
            edge_vertices[edge_index[key]].append(v)
    edge_cells = np.array(list(edge_index))
    edge_vertices = np.array(edge_vertices)
    neighbors = [[] for _ in range(n_cells)]
    for i, j in edge_cells:
        neighbors[i].append(j)
        neighbors[j].append(i)
    cells_on_cell = np.zeros((n_cells, 10), dtype=np.int32)
    for i, row in enumerate(neighbors):
        cells_on_cell[i, : len(row)] = np.array(sorted(row)) + 1
    # Kites: the quadrilateral of a cell center, two edge midpoints and the vertex
    kites = np.zeros((n_vertices, 3))
    for k in range(3):
        p, q, r = cells[:, k], cells[:, (k + 1) % 3], cells[:, (k + 2) % 3]
        m_q = _unit(points[p] + points[q])
        m_r = _unit(points[p] + points[r])
        kites[:, k] = _triangle_area(points[p], m_q, vertices) + _triangle_area(
            points[p], vertices, m_r
        )
    area_cell = np.zeros(n_cells)
    for k in range(3):
        np.add.at(area_cell, cells[:, k], kites[:, k])
    dc = np.arccos(
        np.clip(
            np.einsum("ij,ij->i", points[edge_cells[:, 0]], points[edge_cells[:, 1]]),
            -1,
            1,
        )
    )
    dv = np.arccos(
        np.clip(
            np.einsum(
                "ij,ij->i", vertices[edge_vertices[:, 0]], vertices[edge_vertices[:, 1]]
            ),
            -1,
            1,
        )
    )
    return {
        "points": points,
        "vertices": vertices,
        "cells_on_vertex": cells + 1,
        "cells_on_edge": edge_cells + 1,
        "vertices_on_edge": edge_vertices + 1,
        "cells_on_cell": cells_on_cell,
        "sides": np.array([len(r) for r in neighbors], dtype=np.int32),
        "kites": kites,
        "area_triangle": kites.sum(axis=1),
        "area_cell": area_cell,
        "dc": dc,
        "dv": dv,
    }


def write_mesh(path, arrays, radius=1.0, dtype="f8", mask=None):
    """Write a mesh file holding the variables the checks read."""
    n_cells, n_edges, n_vertices = (
        len(arrays["points"]),
        len(arrays["cells_on_edge"]),
        len(arrays["vertices"]),
    )
    with Dataset(str(path), "w") as ds:
        for name, size in (
            ("nCells", n_cells),
            ("nEdges", n_edges),
            ("nVertices", n_vertices),
            ("maxEdges", 10),
            ("vertexDegree", 3),
            ("one", 1),
            ("three", 3),
            ("two", 2),
        ):
            ds.createDimension(name, size)
        ds.sphere_radius = radius

        def put(name, dims, values, kind=dtype):
            ds.createVariable(name, kind, dims)[:] = values

        for k, axis in enumerate("xyz"):
            put(f"{axis}Cell", ("nCells",), arrays["points"][:, k] * radius)
            put(f"{axis}Vertex", ("nVertices",), arrays["vertices"][:, k] * radius)
        put("nEdgesOnCell", ("nCells",), arrays["sides"], "i4")
        put("cellsOnCell", ("nCells", "maxEdges"), arrays["cells_on_cell"], "i4")
        put("cellsOnEdge", ("nEdges", "two"), arrays["cells_on_edge"], "i4")
        put("verticesOnEdge", ("nEdges", "two"), arrays["vertices_on_edge"], "i4")
        put("cellsOnVertex", ("nVertices", "three"), arrays["cells_on_vertex"], "i4")
        put("dcEdge", ("nEdges",), arrays["dc"] * radius)
        put("dvEdge", ("nEdges",), arrays["dv"] * radius)
        put("areaCell", ("nCells",), arrays["area_cell"] * radius**2)
        put("areaTriangle", ("nVertices",), arrays["area_triangle"] * radius**2)
        put("kiteAreasOnVertex", ("nVertices", "three"), arrays["kites"] * radius**2)
        put("meshDensity", ("nCells",), np.ones(n_cells))
        put("nominalMinDc", ("one",), [arrays["dc"].mean() * radius])
        if mask is not None:
            put("bdyMaskCell", ("nCells",), mask, "i4")
    return path


def cut_region(arrays, keep):
    """Return the arrays of a regional mesh, of the cells ``keep`` is true for.

    Edges and vertices touching a kept cell stay, with 0 where a cell is gone, as
    in a regional MPAS mesh. Also return the boundary layer of each kept cell.
    """
    cell_map = np.cumsum(keep) * keep  # new 1-based index of kept cells, else 0
    old_cells = cell_map[arrays["cells_on_edge"] - 1]
    edge_keep = (old_cells > 0).any(axis=1)
    old_vertex_cells = cell_map[arrays["cells_on_vertex"] - 1]
    vertex_keep = (old_vertex_cells > 0).any(axis=1)
    vertex_map = np.cumsum(vertex_keep) * vertex_keep
    cells_on_cell = np.where(
        arrays["cells_on_cell"] > 0, cell_map[arrays["cells_on_cell"] - 1], 0
    )[keep]
    out = {
        "points": arrays["points"][keep],
        "vertices": arrays["vertices"][vertex_keep],
        "cells_on_vertex": old_vertex_cells[vertex_keep],
        "cells_on_edge": old_cells[edge_keep],
        "vertices_on_edge": vertex_map[arrays["vertices_on_edge"][edge_keep] - 1],
        "cells_on_cell": cells_on_cell.astype(np.int32),
        "sides": arrays["sides"][keep],
        "kites": arrays["kites"][vertex_keep],
        "area_triangle": arrays["area_triangle"][vertex_keep],
        "area_cell": arrays["area_cell"][keep],
        "dc": arrays["dc"][edge_keep],
        "dv": arrays["dv"][edge_keep],
    }
    # Layer 1 has a missing neighbor, and each layer inside touches the one outside
    used = np.arange(10)[None, :] < out["sides"][:, None]
    layer = np.where((used & (cells_on_cell == 0)).any(axis=1), 1, 0)
    for depth in range(2, 9):
        touching = np.zeros(len(layer), dtype=bool)
        for slot in range(10):
            nb = cells_on_cell[:, slot]
            touching |= (
                used[:, slot]
                & (nb > 0)
                & (layer[np.where(nb > 0, nb - 1, 0)] == depth - 1)
            )
        layer = np.where((layer == 0) & touching, depth, layer)
    return out, np.where(layer > quality.BOUNDARY_LAYERS, 0, layer)


@pytest.fixture(scope="module")
def arrays():
    return mesh_arrays(2)


@pytest.fixture(scope="module")
def regional_arrays(arrays):
    big = mesh_arrays(3)
    return cut_region(big, big["points"][:, 2] > 0)


@pytest.fixture
def mesh(tmp_path, arrays):
    return write_mesh(tmp_path / "mesh.nc", arrays)


@pytest.fixture
def regional(tmp_path, regional_arrays):
    region, layers = regional_arrays
    return write_mesh(tmp_path / "region.nc", region, mask=layers)


def change(path, edit):
    """Change a mesh file in place. ``edit`` gets the open dataset."""
    with Dataset(str(path), "a") as ds:
        edit(ds)
    return path


def put(path, name, index, value):
    """Set one element of a variable."""
    return change(path, lambda ds: ds.variables[name].__setitem__(index, value))


def messages(report, kind):
    return " | ".join(report[kind])


def test_module_imports_only_the_standard_library():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('q', {str(SCRIPT)!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print(sorted(m for m in ('numpy', 'netCDF4') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


def test_valid_global_mesh_passes(mesh):
    report = quality.check_mesh(str(mesh))
    assert report["passed"] and report["errors"] == [] and report["warnings"] == []
    assert report["summary"] == "Passed with 0 error(s) and 0 warning(s)"
    m = report["measurements"]
    assert (m["kind"], m["cells"], m["precision"]) == ("global", 162, "float64")
    assert m["cell_sides"] == {"5": 12, "6": 150}
    assert m["euler_sum"] == 12 and m["expectation"] == "quasi_uniform"
    assert m["defects"] == {
        "pentagons": 12,
        "heptagons_or_larger": 0,
        "pentagons_touching_larger": 0,
    }
    assert m["non_hexagons"]["count"] == 12
    assert m["voronoi_tilt"] < 1e-12 and m["obtuse_triangles"] == 0
    assert m["area_fraction_of_sphere"] == pytest.approx(1.0)
    assert m["boundary_edges"] == 0 and "boundary_layers" not in m
    assert m["size_gradient"]["max"] == 0.0
    assert m["edge_regularity"]["regular_hexagon"] == pytest.approx(0.5774, abs=1e-4)
    # A result the agent can return and write as JSON
    assert json.loads(json.dumps(report)) == report


def test_single_precision_mesh_on_the_earth_sphere_passes(tmp_path, arrays):
    path = write_mesh(tmp_path / "f.nc", arrays, radius=6371229.0, dtype="f4")
    report = quality.check_mesh(str(path))
    assert report["passed"] and report["warnings"] == []
    m = report["measurements"]
    assert m["precision"] == "float32" and m["sphere_radius"] == 6371229.0
    assert 1e-9 < m["voronoi_tilt"] < quality.TILT_WARN
    # Spacings are in km whatever the sphere's radius
    unit = write_mesh(tmp_path / "u.nc", arrays)
    assert m["dc_edge_km"]["mean"] == pytest.approx(
        quality.check_mesh(str(unit))["measurements"]["dc_edge_km"]["mean"], rel=1e-5
    )


def test_valid_regional_mesh_passes(regional):
    report = quality.check_mesh(str(regional))
    assert report["passed"] and report["warnings"] == [], report
    m = report["measurements"]
    assert m["kind"] == "regional" and m["boundary_edges"] > 0
    assert "euler_sum" not in m and m["expectation"] == "variable"
    assert 0.4 < m["area_fraction_of_sphere"] < 0.6
    assert m["boundary_layers"]["interior"] > 0
    assert all(m["boundary_layers"][str(layer)] > 0 for layer in range(1, 8))


def test_unused_slots_may_hold_anything(tmp_path, regional_arrays):
    """MPAS-Tools writes garbage after the used neighbors of a cell."""
    region, layers = regional_arrays
    region = dict(region, cells_on_cell=region["cells_on_cell"].copy())
    used = np.arange(10)[None, :] < region["sides"][:, None]
    region["cells_on_cell"][~used] = 1_091_101_775
    path = write_mesh(tmp_path / "garbage.nc", region, mask=layers)
    report = quality.check_mesh(str(path))
    assert report["passed"] and report["warnings"] == []


def test_regional_mesh_without_boundary_layers(tmp_path, regional_arrays):
    region, layers = regional_arrays
    absent = write_mesh(tmp_path / "absent.nc", region)
    assert "boundary_layers" not in quality.check_mesh(str(absent))["measurements"]
    zeros = write_mesh(tmp_path / "zeros.nc", region, mask=np.zeros_like(layers))
    report = quality.check_mesh(str(zeros))
    assert report["measurements"]["boundary_layers"] is None
    assert report["warnings"] == []


def test_missing_boundary_layer_warns(tmp_path, regional_arrays):
    region, layers = regional_arrays
    path = write_mesh(tmp_path / "a.nc", region, mask=np.where(layers == 7, 0, layers))
    report = quality.check_mesh(str(path))
    assert report["passed"]
    assert "Boundary layer(s) 7 of 7 have no cells" in messages(report, "warnings")


def test_stranded_boundary_cell_warns(tmp_path, regional_arrays):
    region, layers = regional_arrays
    stranded = layers.copy()
    stranded[np.flatnonzero(layers == 0)[0]] = 3  # inside, with no layer 2 next to it
    path = write_mesh(tmp_path / "a.nc", region, mask=stranded)
    report = quality.check_mesh(str(path))
    assert "1 boundary-layer cell(s) do not touch the layer inside" in messages(
        report, "warnings"
    )


def test_a_boundary_mask_on_a_global_mesh_is_ignored(mesh):
    change(
        mesh,
        lambda ds: ds.createVariable("bdyMaskCell", "i4", ("nCells",)).__setitem__(
            slice(None), 3
        ),
    )
    assert "boundary_layers" not in quality.check_mesh(str(mesh))["measurements"]


def test_works_in_blocks(mesh, monkeypatch):
    whole = quality.check_mesh(str(mesh))
    monkeypatch.setattr(quality, "BLOCK", 7)
    assert quality.check_mesh(str(mesh)) == whole


def test_missing_file_and_bad_files(tmp_path, mesh):
    with pytest.raises(FileNotFoundError, match="No mesh file"):
        quality.check_mesh(str(tmp_path / "none.nc"))
    text = tmp_path / "text.nc"
    text.write_text("not netCDF")
    with pytest.raises(quality.InvalidMesh, match="is not a netCDF file"):
        quality.check_mesh(str(text))
    bare = tmp_path / "bare.nc"
    with Dataset(str(bare), "w") as ds:
        ds.createDimension("n", 1)
    with pytest.raises(ValueError, match="not an MPAS mesh, it lacks: nEdgesOnCell"):
        quality.check_mesh(str(bare))
    assert issubclass(quality.InvalidMesh, ValueError)


@pytest.mark.parametrize(
    "name, index, value",
    [
        ("nEdgesOnCell", 0, 2),
        ("nEdgesOnCell", 0, 11),
        ("cellsOnCell", (0, 0), 1000),
        ("cellsOnCell", (0, 0), -1),
        ("cellsOnEdge", (0, 0), 1000),
        ("verticesOnEdge", (0, 0), 0),
        ("verticesOnEdge", (0, 0), 1000),
        ("cellsOnVertex", (0, 0), 1000),
    ],
)
def test_out_of_range_indices_are_errors(mesh, name, index, value):
    report = quality.check_mesh(str(put(mesh, name, index, value)))
    assert not report["passed"]
    assert f"{name} has 1 value(s) out of range" in messages(report, "errors")
    # The geometry is not read, since it would index with these
    assert "voronoi_tilt" not in report["measurements"]


def test_a_mesh_without_interior_edges_is_an_error(mesh):
    change(
        mesh, lambda ds: ds.variables["cellsOnEdge"].__setitem__((slice(None), 0), 0)
    )
    report = quality.check_mesh(str(mesh))
    assert report["errors"] == ["The mesh has no edge between two cells"]


def test_an_edge_with_one_cell_on_both_sides(mesh):
    change(
        mesh,
        lambda ds: ds.variables["cellsOnEdge"].__setitem__(
            (0, 1), ds.variables["cellsOnEdge"][0, 0]
        ),
    )
    report = quality.check_mesh(str(mesh))
    assert "1 edge(s) have the same cell on both sides" in messages(report, "errors")


def test_neighbors_that_disagree_with_the_edges(mesh, regional):
    # A neighbor swapped for another cell: one is missing
    with Dataset(str(mesh)) as ds:
        cells = ds["cellsOnCell"][0, :].tolist()
    other = next(c for c in range(2, 163) if c not in cells)
    report = quality.check_mesh(str(put(mesh, "cellsOnCell", (0, 0), other)))
    assert report["errors"] == [
        "1 neighbor(s) that cellsOnEdge gives are missing from cellsOnCell"
    ]
    # A boundary slot filled in: an extra neighbor, and none missing
    with Dataset(str(regional)) as ds:
        used = np.arange(10)[None, :] < ds["nEdgesOnCell"][:][:, None]
        row, slot = map(int, np.argwhere((ds["cellsOnCell"][:] == 0) & used)[0])
    report = quality.check_mesh(str(put(regional, "cellsOnCell", (row, slot), 2)))
    assert report["errors"] == [
        "cellsOnCell lists 1 more neighbor(s) than cellsOnEdge gives"
    ]


def test_a_cell_listed_twice_at_a_vertex(mesh):
    change(
        mesh,
        lambda ds: ds.variables["cellsOnVertex"].__setitem__(
            (0, 1), ds.variables["cellsOnVertex"][0, 0]
        ),
    )
    report = quality.check_mesh(str(mesh))
    assert "1 vertex(es) list a cell twice" in messages(report, "errors")


def test_cell_shapes_that_cannot_close_a_sphere(mesh):
    # The side count no longer matches the neighbors either, which is also reported
    report = quality.check_mesh(str(put(mesh, "nEdgesOnCell", 20, 5)))
    assert "The sum of (6 - sides) over cells is 13" in messages(report, "errors")
    assert report["measurements"]["euler_sum"] == 13


def test_cells_with_fewer_than_five_sides_warn(arrays):
    report = quality._Report()
    sides = np.full(30, 6)
    sides[0] = 4
    quality._cell_shapes(
        report,
        "regional",
        sides,
        np.zeros((30, 10), int),
        sides[:, None] > np.arange(10),
        np.ones(30),
    )
    assert "1 cell(s) have fewer than 5 sides" in report.warnings


def test_non_hexagons_in_a_quasi_uniform_mesh_warn(tmp_path):
    path = write_mesh(tmp_path / "small.nc", mesh_arrays(1))
    report = quality.check_mesh(str(path))
    assert report["passed"]
    assert "28.6% of cells are not hexagons, above 10%" in messages(report, "warnings")
    assert "JIGSAW" in messages(report, "warnings")
    # 42 cells with 12 pentagons is the minimum, so there is no count warning
    assert "quasi-uniform" not in messages(report, "warnings")


def test_extra_pentagons_in_a_quasi_uniform_mesh_warn():
    report = quality._Report()
    sides = np.array([5] * 14 + [6] * 400 + [7, 7])
    on_cell = np.zeros((sides.size, 10), int)
    # A pentagon next to both heptagons counts once, as a pentagon
    on_cell[13, :2] = [sides.size - 1, sides.size]
    used = np.arange(10)[None, :] < sides[:, None]
    quality._cell_shapes(report, "global", sides, on_cell, used, np.ones(sides.size))
    assert "has 14 pentagons and 2 other non-hexagons, where the minimum is 12" in (
        messages({"w": report.warnings}, "w")
    )
    assert report.measurements["defects"]["pentagons_touching_larger"] == 1
    assert report.measurements["defects"]["heptagons_or_larger"] == 2
    assert report.measurements["expectation"] == "quasi_uniform"
    assert report.errors == []


def test_variable_density_is_not_expected_to_have_only_twelve_pentagons(
    arrays, tmp_path
):
    path = write_mesh(tmp_path / "v.nc", arrays)
    density = np.linspace(1.0, 0.9, 162)
    change(
        path, lambda ds: ds.variables["meshDensity"].__setitem__(slice(None), density)
    )
    report = quality.check_mesh(str(path))
    assert report["measurements"]["expectation"] == "variable"
    assert report["measurements"]["size_gradient"]["max"] > 0.0


def test_moved_cell_center_breaks_the_voronoi_property(mesh):
    def move(ds):
        for axis, shift in zip("xyz", (0.03, 0.0, 0.0)):
            ds.variables[f"{axis}Cell"][3] = ds.variables[f"{axis}Cell"][3] + shift

    report = quality.check_mesh(str(change(mesh, move)))
    assert not report["passed"]
    assert "tilt of Voronoi edges" in messages(report, "errors")
    assert report["measurements"]["voronoi_tilt"] > quality.TILT_ERROR


def test_a_small_tilt_is_a_warning(mesh):
    def move(ds):
        ds.variables["xCell"][3] = ds.variables["xCell"][3] + 0.0008

    report = quality.check_mesh(str(change(mesh, move)))
    assert report["passed"]
    assert (
        quality.TILT_WARN < report["measurements"]["voronoi_tilt"] < quality.TILT_ERROR
    )
    assert "above 0.001" in messages(report, "warnings")


def test_zero_length_voronoi_edge_warns(mesh):
    def collapse(ds):
        first, second = ds.variables["verticesOnEdge"][0, :] - 1
        for axis in "xyz":
            var = ds.variables[f"{axis}Vertex"]
            var[second] = var[first]

    report = quality.check_mesh(str(change(mesh, collapse)))
    assert "Voronoi edge(s) have zero length" in messages(report, "warnings")


def test_obtuse_triangle_warns(mesh):
    def outside(ds):
        # The vertex goes to the opposite side of the sphere, outside its triangle
        for axis in "xyz":
            var = ds.variables[f"{axis}Vertex"]
            var[0] = -var[0]

    report = quality.check_mesh(str(change(mesh, outside)))
    assert report["measurements"]["obtuse_triangles"] >= 1
    assert "Delaunay triangle(s) are obtuse" in messages(report, "warnings")


def scale(name, factor):
    return lambda ds: ds.variables[name].__setitem__(
        slice(None), ds.variables[name][:] * factor
    )


def test_cell_areas_that_do_not_fill_the_sphere(mesh):
    warn = quality.check_mesh(str(change(mesh, scale("areaCell", 1.00002))))
    assert warn["passed"]
    assert "cell areas from the sphere's is" in messages(warn, "warnings")
    error = quality.check_mesh(str(change(mesh, scale("areaCell", 1.02))))
    assert "cell areas from the sphere's is" in messages(error, "errors")
    assert "kite areas from their cell's area" in messages(error, "errors")


def test_regional_areas_are_not_compared_with_the_sphere(tmp_path, regional_arrays):
    region, _ = regional_arrays
    path = write_mesh(tmp_path / "r.nc", region)
    assert quality.check_mesh(str(path))["passed"]


def test_kite_areas_that_do_not_add_up(mesh):
    warn = quality.check_mesh(str(change(mesh, scale("areaTriangle", 1.0001))))
    assert warn["passed"]
    assert "kite areas from their triangle's area is" in messages(warn, "warnings")
    error = quality.check_mesh(str(change(mesh, scale("areaTriangle", 1.01))))
    assert "kite areas from their triangle's area is" in messages(error, "errors")


def test_distorted_cells_warn(mesh):
    report = quality.check_mesh(str(put(mesh, "dvEdge", 0, 10.0)))
    assert "so some cells are very distorted" in messages(report, "warnings")
    assert report["measurements"]["edge_regularity"]["max"] > 10.0


def test_steep_change_of_size_warns(mesh):
    change(
        mesh,
        lambda ds: ds.variables["meshDensity"].__setitem__(
            slice(None), np.r_[1e-4, np.ones(161)]
        ),
    )
    report = quality.check_mesh(str(mesh))
    assert "A few percent is advised" in messages(report, "warnings")
    gradient = report["measurements"]["size_gradient"]
    assert gradient["max"] > quality.GRADIENT_WARN
    assert 0 < gradient["fraction_above_reference"] < 1


def test_requested_resolution(mesh):
    nominal = quality.check_mesh(str(mesh))["measurements"]["nominal_min_spacing_km"]
    same = quality.check_mesh(str(mesh), resolution_km=nominal * 1.005)
    assert same["warnings"] == []
    assert same["measurements"]["requested_resolution_km"] == nominal * 1.005
    other = quality.check_mesh(str(mesh), resolution_km=nominal * 2)
    assert "was asked for" in messages(other, "warnings")
    assert (
        "requested_resolution_km" not in quality.check_mesh(str(mesh))["measurements"]
    )


def test_errors_and_warnings_make_the_summary(mesh):
    change(mesh, scale("areaTriangle", 1.01))
    report = quality.check_mesh(str(mesh))
    assert report["summary"].startswith("Failed with 1 error(s)")
    lines = quality.format_lines(report)
    assert lines[0] == report["summary"]
    assert lines[1].startswith("error: ")


def test_nice_bin_width():
    assert quality.nice_bin_width(0.9) == 1.0
    assert quality.nice_bin_width(1.7) == 2.0
    assert quality.nice_bin_width(2.4) == 2.5
    assert quality.nice_bin_width(3.9) == 5.0
    assert quality.nice_bin_width(80.0) == 100.0


def test_bin_edges():
    sizes = np.linspace(10.0, 14.0, 1000)
    # By the Freedman-Diaconis rule, with a width from the nice values
    automatic = quality.bin_edges(sizes)
    assert automatic[0] <= 10.0 and automatic[-1] >= 14.0
    assert len(automatic) < 60
    # A fixed width centered on a value puts a bin around it
    centered = quality.bin_edges(sizes, width=0.5, center=12.0)
    assert any(abs((a + b) / 2 - 12.0) < 1e-9 for a, b in zip(centered, centered[1:]))
    assert centered[0] <= 10.0 and centered[-1] >= 14.0
    fixed = quality.bin_edges(sizes, width=1.0)
    assert list(fixed) == [10.0, 11.0, 12.0, 13.0, 14.0]
    # No spread: one bin around the value. A zero width from the quartiles too.
    assert list(quality.bin_edges(np.full(5, 3.0))) == [2.5, 3.5]
    # Too many bins are widened
    wide = quality.bin_edges(np.r_[np.zeros(100), 1e6], width=1.0)
    assert len(wide) <= quality.MAX_BINS + 3
    # A bad width falls back to the range
    assert len(quality.bin_edges(sizes, width=float("nan"))) <= 4
    assert len(quality.bin_edges(sizes, width=-1.0)) <= 4


def write_partition(path, ranks):
    path.write_text("".join(f"{rank}\n" for rank in ranks))
    return path


def sliced_ranks(arrays, count):
    """Return ranks that cut the cells into ``count`` slices along x."""
    order = np.argsort(arrays["points"][:, 0], kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order)) * count // len(order)
    return ranks


def test_partitions_are_measured(mesh, arrays):
    for count in (4, 6):
        write_partition(
            mesh.with_name(f"mesh.graph.info.part.{count}"), sliced_ranks(arrays, count)
        )
    report = quality.check_mesh(str(mesh))
    assert report["passed"] and report["warnings"] == []
    partitions = report["measurements"]["partitions"]
    assert list(partitions) == ["4", "6"]
    four = partitions["4"]
    ranks = sliced_ranks(arrays, 4)
    pairs = arrays["cells_on_edge"] - 1
    assert four["edge_cut"] == int((ranks[pairs[:, 0]] != ranks[pairs[:, 1]]).sum())
    assert four["edge_cut_fraction"] == pytest.approx(four["edge_cut"] / len(pairs))
    assert four["cells_per_rank"] == {"min": 40.0, "mean": 40.5, "max": 41.0}
    assert four["imbalance"] == pytest.approx(41 / 40.5)
    assert four["empty_ranks"] == 0
    # Slices along an axis touch the next and the one after it at most
    assert 1.0 <= four["neighbor_ranks"]["min"] <= four["neighbor_ranks"]["max"] <= 3.0


def test_partitions_of_a_static_file_are_found_by_its_name(tmp_path, arrays):
    path = write_mesh(tmp_path / "g.static.nc", arrays)
    write_partition(tmp_path / "g.graph.info.part.3", sliced_ranks(arrays, 3))
    write_partition(tmp_path / "other.graph.info.part.3", [0, 1, 2])
    report = quality.check_mesh(str(path))
    assert list(report["measurements"]["partitions"]) == ["3"]


def test_files_that_are_not_partitions_are_ignored(mesh, arrays):
    for name in ("part.", "part.abc", "part.1", "part.0"):
        write_partition(mesh.with_name(f"mesh.graph.info.{name}"), [0] * 162)
    mesh.with_name("mesh.graph.info").write_text("162 0\n")
    assert "partitions" not in quality.check_mesh(str(mesh))["measurements"]


def test_an_unbalanced_partition_warns(mesh):
    ranks = np.arange(162) % 4
    ranks[:100] = 0
    write_partition(mesh.with_name("mesh.graph.info.part.4"), ranks)
    report = quality.check_mesh(str(mesh))
    assert report["passed"]
    assert "the busiest rank has 2.86 times the mean number of cells" in messages(
        report, "warnings"
    )


def test_ranks_without_cells_warn(mesh):
    write_partition(mesh.with_name("mesh.graph.info.part.200"), np.arange(162))
    report = quality.check_mesh(str(mesh))
    assert "38 of 200 ranks have no cells" in messages(report, "warnings")
    assert report["measurements"]["partitions"]["200"]["empty_ranks"] == 38


@pytest.mark.parametrize(
    "ranks, message",
    [
        ([0, 1] * 10, "has 20 entries for 162 cells"),
        ([0, 1, 4] * 54, "has rank numbers outside 0 to 3"),
        ([-1, 0, 1] * 54, "has rank numbers outside 0 to 3"),
        (["a"] * 162, "does not hold rank numbers"),
    ],
)
def test_partitions_that_do_not_fit_the_mesh_are_errors(mesh, ranks, message):
    write_partition(mesh.with_name("mesh.graph.info.part.4"), ranks)
    report = quality.check_mesh(str(mesh))
    assert not report["passed"]
    assert f"mesh.graph.info.part.4 {message}" in messages(report, "errors")
    assert "partitions" not in report["measurements"]


def test_values_that_are_not_finite_are_errors(mesh):
    report = quality.check_mesh(str(put(mesh, "xCell", 3, float("nan"))))
    assert report["errors"] == ["xCell has 1 value(s) that are not finite numbers"]
    report = quality.check_mesh(str(put(mesh, "areaTriangle", 0, float("inf"))))
    assert "areaTriangle has 1 value(s) that are not finite numbers" in messages(
        report, "errors"
    )
    # The report is plain JSON, with no NaN to write
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize(
    "name, index",
    [
        ("dcEdge", 0),
        ("areaCell", 0),
        ("areaTriangle", 1),
        ("meshDensity", 2),
        ("nominalMinDc", 0),
    ],
)
def test_values_that_must_be_positive(mesh, name, index):
    report = quality.check_mesh(str(put(mesh, name, index, 0.0)))
    assert report["errors"] == [f"{name} has 1 value(s) that are not positive"]
    assert not report["passed"]


def test_no_sphere_radius_is_assumed_to_be_one(mesh):
    change(mesh, lambda ds: ds.delncattr("sphere_radius"))
    report = quality.check_mesh(str(mesh))
    assert report["passed"]
    assert report["warnings"] == [
        "The file has no sphere_radius attribute, so lengths and areas are "
        "taken to be on a sphere of radius 1"
    ]
    assert report["measurements"]["sphere_radius"] == 1.0


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
def test_sphere_radius_must_be_positive(mesh, radius):
    change(mesh, lambda ds: setattr(ds, "sphere_radius", radius))
    report = quality.check_mesh(str(mesh))
    assert report["errors"] == [f"sphere_radius is {radius}, and must be positive"]
    assert "voronoi_tilt" not in report["measurements"]


def clockwise(arrays):
    """Return the arrays with each vertex's cells listed clockwise."""
    order = [0, 2, 1]
    return dict(
        arrays,
        cells_on_vertex=arrays["cells_on_vertex"][:, order],
        kites=arrays["kites"][:, order],
    )


def test_the_orientation_of_triangles_does_not_matter(tmp_path, arrays):
    path = write_mesh(tmp_path / "cw.nc", clockwise(arrays))
    report = quality.check_mesh(str(path))
    assert report["passed"] and report["warnings"] == []
    assert report["measurements"]["obtuse_triangles"] == 0

    def outside(ds):
        for axis in "xyz":
            var = ds.variables[f"{axis}Vertex"]
            var[0] = -var[0]

    # A vertex outside its triangle is found whichever way the triangle is listed
    for orient, name in ((clockwise(arrays), "cw2.nc"), (arrays, "ccw.nc")):
        path = change(write_mesh(tmp_path / name, orient), outside)
        assert quality.check_mesh(str(path))["measurements"]["obtuse_triangles"] >= 1


def test_tilt_limits_rise_with_the_noise_of_the_coordinates():
    assert quality.tilt_limits(0.0) == (quality.TILT_WARN, quality.TILT_ERROR)
    assert quality.tilt_limits(1e-5) == (quality.TILT_WARN, quality.TILT_ERROR)
    assert quality.tilt_limits(1e-2) == pytest.approx((0.2, 2.0))


def test_single_precision_is_judged_by_its_noise(tmp_path, arrays):
    """Cells a ten-thousandth as far apart have a hundred times the noise to tilt by."""
    small = dict(arrays, dc=arrays["dc"] * 1e-4, dv=arrays["dv"] * 1e-4)
    path = write_mesh(tmp_path / "fine.nc", small, radius=6371229.0, dtype="f4")

    def move(ds):
        ds.variables["xCell"][3] = ds.variables["xCell"][3] + 0.004 * 6371229.0

    report = quality.check_mesh(str(change(path, move)))
    measurements = report["measurements"]
    assert measurements["voronoi_tilt"] > quality.TILT_ERROR
    assert measurements["coordinate_noise"] > 1e-3
    assert measurements["voronoi_tilt_limits"] == dict(
        zip(("warn", "error"), quality.tilt_limits(measurements["coordinate_noise"]))
    )
    assert report["passed"] and "tilt" not in messages(report, "warnings")


def test_a_partition_that_cannot_be_read_is_an_error(mesh):
    mesh.with_name("mesh.graph.info.part.4").mkdir()
    report = quality.check_mesh(str(mesh))
    assert not report["passed"]
    assert "mesh.graph.info.part.4 cannot be read:" in messages(report, "errors")


def test_partitions_are_read_in_blocks(mesh, arrays, monkeypatch):
    write_partition(mesh.with_name("mesh.graph.info.part.4"), sliced_ranks(arrays, 4))
    whole = quality.check_mesh(str(mesh))
    monkeypatch.setattr(quality, "BLOCK", 5)
    assert quality.check_mesh(str(mesh)) == whole
    # An empty partition file has no entries
    mesh.with_name("mesh.graph.info.part.6").write_text("")
    assert "mesh.graph.info.part.6 has 0 entries for 162 cells" in messages(
        quality.check_mesh(str(mesh)), "errors"
    )
