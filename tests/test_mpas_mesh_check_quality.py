# SPDX-License-Identifier: Apache-2.0
"""Real quality checks of MPAS meshes through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs internet access to UCAR and GitHub, and its ``compute`` resource needs
chiltepin with the ``mpas`` extra, which the test extra includes. The regional
meshes also build MPAS-Tools, which needs make, a Fortran compiler and netCDF on
``compute``, and install MPAS-Limited-Area.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import json
from pathlib import Path

import pytest

# Versions the tests install, passed to the agent so the test and the agent agree
VERSIONS = {"mpas_tools_version": "2.0.0", "limited_area_version": "2.2"}
GLOBAL = {"resolution": "480km", "name": "globe"}
HEXES = {
    "resolution": "50km",
    "name": "hexes",
    "regional": {
        "project_hexes": {
            "center_lat": 40.0,
            "center_lon": -105.0,
            "extent_x_km": 1000,
            "extent_y_km": 800,
        }
    },
}
CUT = {
    "resolution": "480km",
    "name": "cut",
    "regional": {
        "create_region": {
            "circle": {"center_lat": 39.0, "center_lon": -105.5, "radius_km": 1500}
        }
    },
}


@pytest.mark.asyncio
async def test_real_quality_checks(mesh_test_directory, launch_mesh_agent):
    """Check a global mesh and the two kinds of regional mesh."""
    mesh_data_dir = str(mesh_test_directory / "meshes")
    async with launch_mesh_agent("first", **VERSIONS) as agent:
        globe = await agent.generate_mesh(GLOBAL, mesh_data_dir)
        hexes = await agent.generate_mesh(HEXES, mesh_data_dir)
        cut = await agent.generate_mesh(CUT, mesh_data_dir)
        # The UCAR 480 km mesh has the 12 pentagons a quasi-uniform mesh has
        report = await agent.check_mesh_quality(globe["static"], resolution_km=480)
        assert report["passed"] and report["warnings"] == [], report
        measurements = report["measurements"]
        assert measurements["kind"] == "global"
        assert measurements["expectation"] == "quasi_uniform"
        assert measurements["cell_sides"] == {"5": 12, "6": 2550}
        assert measurements["euler_sum"] == 12
        written = Path(report["report_file"])
        assert written == Path(globe["static"]).with_name("globe.quality.json")
        assert json.loads(written.read_text()) == report
        # Regional meshes pass without the quasi-uniform expectations
        for result in (hexes, cut):
            report = await agent.check_mesh_quality(result["static"])
            assert report["passed"], report
            assert report["measurements"]["kind"] == "regional"
            assert report["measurements"]["expectation"] == "variable"
        # create_region meshes have their seven boundary layers
        layers = report["measurements"]["boundary_layers"]
        assert all(layers[str(layer)] > 0 for layer in range(1, 8))
        # A file that is not a mesh is refused, and one that does not exist
        text = Path(mesh_data_dir) / "not_a_mesh.nc"
        text.write_text("text")
        with pytest.raises(ValueError, match="is not a netCDF file"):
            await agent.check_mesh_quality(str(text))
        with pytest.raises(FileNotFoundError):
            await agent.check_mesh_quality(str(Path(mesh_data_dir) / "none.nc"))
