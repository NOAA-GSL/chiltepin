# SPDX-License-Identifier: Apache-2.0
"""Tests of the files and counts for making a global mesh of any resolution."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from chiltepin.agents.mpas import _global_mesh as global_mesh
from chiltepin.agents.mpas.mesh import _MeshBehavior

SCRIPT = Path(global_mesh.__file__)

# A JIGSAW mesh of a tetrahedron, which is the smallest closed triangulation,
# written as JIGSAW does, with a section the converter does not use
TETRAHEDRON = """\
# a comment
MSHID=3;EUCLIDEAN-MESH
NDIMS=3
POINT=4
1.5;0.25;-3.125;0
-1.0;2.0e-3;0.5;0
0.0;-1.0;0.125;0
2.0;1.0;1.0;0
EDGE2=1
0;1;7
TRIA3=4
0;1;2;0
0;2;3;0
0;3;1;0
1;3;2;0
"""


def test_module_imports_only_the_standard_library():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('g', {str(SCRIPT)!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "print(sorted(m for m in ('numpy', 'netCDF4') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


# Divisions and counts


def test_the_division_reproduces_every_ucar_mesh(tmp_path):
    """Each UCAR mesh of dx km is 7680 / dx divisions, with 10 N^2 + 2 cells."""
    for name, cells in _MeshBehavior(str(tmp_path / "work")).resolution_cells.items():
        parts = global_mesh.division(float(name.removesuffix("km")))
        assert global_mesh.cell_count(parts) == cells, name
        assert global_mesh.nominal_spacing_km(parts) == pytest.approx(
            float(name.removesuffix("km")), rel=1e-12
        )


@pytest.mark.parametrize(
    "resolution, parts",
    [(120, 64), (10, 768), (3.75, 2048), (7.3, 1052), (400, 19), (7000, 1)],
)
def test_division(resolution, parts):
    assert global_mesh.division(resolution) == parts


def test_the_division_may_differ_from_the_resolution():
    # 7.3 km is 1052.05 parts, so the mesh has the spacing of 1052 parts
    parts = global_mesh.division(7.3)
    assert global_mesh.nominal_spacing_km(parts) == pytest.approx(7.3004, abs=1e-4)
    assert global_mesh.cell_count(parts) == 11067042


@pytest.mark.parametrize(
    "parts, valid",
    [(1, False), (2, True), (64, True), (8460, True), (8461, False), (0, False)],
)
def test_the_divisions_the_tools_can_make(parts, valid):
    assert global_mesh.division_valid(parts) is valid


def test_the_largest_division_keeps_the_edge_numbers_in_32_bits():
    edges = 3 * global_mesh.cell_count(global_mesh.MAX_DIVISION)
    assert edges < 2**31
    assert 3 * global_mesh.cell_count(global_mesh.MAX_DIVISION + 1) >= 2**31


# The files JIGSAW reads


def test_the_size_file_is_a_constant_grid():
    lines = global_mesh.hfun_text(7.5).splitlines()
    assert lines[:3] == ["MSHID=3;ellipsoid-grid", "NDIMS=2", "COORD=1;37"]
    longitudes = [float(v) for v in lines[3:40]]
    assert lines[40] == "COORD=2;19"
    latitudes = [float(v) for v in lines[41:60]]
    assert longitudes[0] == pytest.approx(-3.141592653589793)
    assert longitudes[-1] == pytest.approx(3.141592653589793)
    assert latitudes[0] == pytest.approx(-1.5707963267948966)
    assert latitudes[-1] == pytest.approx(1.5707963267948966)
    assert lines[60] == "VALUE=703; 1"
    values = lines[61:]
    assert len(values) == 37 * 19 and set(values) == {"7.5"}


def test_the_configuration_starts_from_the_divided_icosahedron():
    text = global_mesh.jig_text()
    settings = dict(line.split("=") for line in text.splitlines())
    assert settings == {
        "VERBOSITY": "1",
        "GEOM_FILE": "GEOM.msh",
        "HFUN_FILE": "HFUN.msh",
        "HFUN_SCAL": "absolute",
        "HFUN_HMAX": "inf",
        "HFUN_HMIN": "0.0",
        "INIT_FILE": "INIT.msh",
        "MESH_FILE": "MESH.msh",
        "MESH_DIMS": "2",
        "OPTM_QLIM": "0.9375",
    }


# The files mkgrid reads


def test_the_density_source_is_the_uniform_function():
    namespace = {}
    exec(global_mesh.density_code(10.0), namespace)  # noqa: S102
    import numpy as np

    assert namespace["hfun_min"] == 10.0
    assert list(namespace["get_hfun"](np.zeros(3), np.zeros(3))) == [10.0] * 3


def test_the_configuration_is_yaml_the_fork_reads():
    # The last is all digits, which YAML would read as a number if it were not quoted
    tools = {"jigsaw": "a" * 40, "mkgrid": "1e5" + "0" * 37, "sphtri_subdiv": "7" * 40}
    text = global_mesh.config_text(7.3, 1052, 7.3003802281368824, tools)
    config = yaml.safe_load(text)
    # What mpas_jigsaw's mesh_quality.py reads
    assert config["mesh"] == {"resolution": 7.3003802281368824}
    assert config["chiltepin"] == {
        "requested_resolution_km": 7.3,
        "division": 1052,
        "nominal_spacing_km": 7.3003802281368824,
        **tools,
    }


def test_the_conversion_writes_what_the_fork_script_writes(tmp_path):
    mesh = tmp_path / "MESH.msh"
    mesh.write_text(TETRAHEDRON)
    assert global_mesh.convert_jigsaw(mesh, tmp_path) == (4, 4)
    assert (tmp_path / "SaveVertices").read_text() == (
        "1.5 0.25 -3.125\n-1.0 2.0e-3 0.5\n0.0 -1.0 0.125\n2.0 1.0 1.0\n"
    )
    assert (tmp_path / "SaveTriangles").read_text() == ("0 1 2\n0 2 3\n0 3 1\n1 3 2\n")


def reference_conversion(mesh, directory):
    """The mpas_jigsaw convert_jigsaw.py, which reads a line at a time."""
    import re

    out = None
    listing, i, n = False, 0, 1
    with open(mesh) as handle:
        for line in handle:
            if not listing or i >= n:
                if i > n:
                    i, listing = 0, False
                    out.close()
                if re.search(r"^POINT", line):
                    i, n, listing = 0, int(line.split("=")[1]), True
                    out = open(directory / "ref_vertices", "w")
                elif re.search(r"^TRIA", line):
                    i, n, listing = 0, int(line.split("=")[1]), True
                    out = open(directory / "ref_triangles", "w")
            else:
                vals = line.split(";")
                out.write(f"{vals[0]} {vals[1]} {vals[2]}\n")
                i += 1
    out.close()


def test_the_conversion_matches_the_fork_script(tmp_path):
    mesh = tmp_path / "MESH.msh"
    mesh.write_text(TETRAHEDRON)
    reference_conversion(mesh, tmp_path)
    global_mesh.convert_jigsaw(mesh, tmp_path)
    assert (tmp_path / "SaveVertices").read_bytes() == (
        tmp_path / "ref_vertices"
    ).read_bytes()
    assert (tmp_path / "SaveTriangles").read_bytes() == (
        tmp_path / "ref_triangles"
    ).read_bytes()


@pytest.mark.parametrize(
    "text, message",
    [
        ("MSHID=3\nTRIA3=1\n0;1;2;0\n", "has no POINT section"),
        ("POINT=1\n1;2;3;0\n", "has no TRIA3 section"),
        ("POINT=4\n1;2;3;0\n1;2;3;0\n1;2;3;0\n", "has 3 of 4 POINT lines"),
        (TETRAHEDRON.rsplit("\n", 2)[0] + "\n", "has 3 of 4 TRIA3 lines"),
        (
            TETRAHEDRON.replace("TRIA3=4", "TRIA3=2").rsplit("\n", 3)[0] + "\n",
            "so it is not the triangulation of a sphere",
        ),
    ],
)
def test_a_jigsaw_mesh_that_is_not_a_sphere_is_refused(tmp_path, text, message):
    mesh = tmp_path / "MESH.msh"
    mesh.write_text(text)
    with pytest.raises(ValueError, match=message):
        global_mesh.convert_jigsaw(mesh, tmp_path)


def test_the_density_files(tmp_path, monkeypatch):
    monkeypatch.setattr(global_mesh, "DENSITY_CHUNK", 3)
    global_mesh.write_density_files(tmp_path, 7, 12.5, "mesh:\n  resolution: 12.5\n")
    assert (tmp_path / "SaveDensity").read_text() == "1.0\n" * 7
    assert (tmp_path / "SaveCode").read_text() == global_mesh.density_code(12.5)
    assert (tmp_path / "SaveConfig").read_text() == "mesh:\n  resolution: 12.5\n"


def test_the_names_the_pipeline_uses():
    assert global_mesh.INTERMEDIATE_FILES == (
        "INIT.msh",
        "HFUN.msh",
        "MESH.msh",
        "SaveVertices",
        "SaveTriangles",
        "SaveDensity",
        "SaveCode",
        "SaveConfig",
    )
