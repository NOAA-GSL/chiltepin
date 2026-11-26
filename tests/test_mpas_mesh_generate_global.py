# SPDX-License-Identifier: Apache-2.0
"""Real global meshes at resolutions that are not downloaded, through the agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs internet access to GitHub, ``compute`` needs CMake, C and C++ compilers,
MPI and PnetCDF, which JIGSAW, mkgrid and sphtri_subdiv are built with, and the
chiltepin ``mpas`` extra, and ``mpi`` runs mkgrid on the same MPI and PnetCDF.
Partitioning builds Metis, which needs cmake and a C compiler on ``compute``.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import json
import re
from pathlib import Path

import pytest
import yaml
from netCDF4 import Dataset

# 7680 / 400 = 19.2 divisions of an icosahedron edge, so 19, with 10 * 19^2 + 2
# cells, spaced 7680 / 19 = 404.2 km
REQUEST = {"resolution": "400km", "name": "globe", "init_ranks": 4}
PARTS = 19
CELLS = 3612
SPACING = 7680 / PARTS


@pytest.mark.asyncio
async def test_real_global_mesh(mesh_test_directory, launch_mesh_agent):
    """Make a global mesh, check it, partition it, reuse it and replace it."""
    mesh_data_dir = str(mesh_test_directory / "meshes")
    work = mesh_test_directory / "work"
    async with launch_mesh_agent("first", with_mpi=True) as agent:
        # A resolution the tools cannot make is refused before any work
        with pytest.raises(ValueError, match="parts per edge"):
            await agent.generate_mesh({"resolution": "0.5km"}, mesh_data_dir)
        assert not (mesh_test_directory / "meshes").exists()

        result = await agent.generate_mesh(REQUEST, mesh_data_dir)
        mesh_dir = mesh_test_directory / "meshes" / "globe"
        assert result["static"] == str(mesh_dir / "globe.static.nc")
        assert result["graph"] == str(mesh_dir / "globe.graph.info")
        assert Path(result["partitions"][4]).is_file()
        generated = result["generated"]
        assert (generated["division"], generated["cells"]) == (PARTS, CELLS)
        assert generated["nominal_spacing_km"] == pytest.approx(SPACING)
        assert generated["requested_resolution_km"] == 400.0
        assert all(re.fullmatch("[0-9a-f]{40}", c) for c in generated["tools"].values())

        # The mesh is a quasi-uniform one: 12 pentagons and the rest hexagons
        quality = result["quality"]
        assert quality["passed"] and quality["warnings"] == [], quality
        measurements = quality["measurements"]
        assert measurements["cells"] == CELLS
        assert measurements["cell_sides"] == {"5": 12, "6": CELLS - 12}
        assert measurements["euler_sum"] == 12
        assert Path(quality["report_file"]).is_file()

        # What it was made from is in the file, and the large files are gone
        with Dataset(result["static"]) as mesh:
            assert mesh.dimensions["nCells"].size == CELLS
            # On the unit sphere mkgrid writes, lengths are in radians
            assert mesh.sphere_radius == 1.0
            assert float(mesh["nominalMinDc"][0]) * 6371.229 == pytest.approx(SPACING)
            config = yaml.safe_load(mesh.getncattr("mesh_config"))
            assert config["mesh"] == {"resolution": pytest.approx(SPACING)}
            assert config["chiltepin"]["division"] == PARTS
            assert config["chiltepin"]["jigsaw"] == generated["tools"]["jigsaw"]
            code = mesh["densityFunctionCode"][:].tobytes().decode()
            assert f"hfun_min = {SPACING!r}" in code
        assert sorted(path.name for path in mesh_dir.iterdir() if path.is_file()) == [
            "GEOM.msh",
            "ICOS.msh",
            "MESH.jig",
            "MESH.log",
            "globe.graph.info",
            "globe.graph.info.part.4",
            "globe.mesh.json",
            "globe.quality.json",
            "globe.static.nc",
        ]
        record = json.loads((mesh_dir / "globe.mesh.json").read_text())
        assert record["generated"] == generated

        # Every program wrote its logs, on the resources it ran on
        logs = work / "logs"
        for program in ("sphtri_subdiv", "jigsaw", "mkgrid"):
            assert len(list(logs.glob(f"{program}.*.stderr"))) == 1, program
        assert "Completed mkgrid" in next(logs.glob("mkgrid.*.stdout")).read_text()

        # The partitions are measured with the mesh
        report = await agent.check_mesh_quality(result["static"])
        partition = report["measurements"]["partitions"]["4"]
        assert partition["empty_ranks"] == 0 and 1.0 <= partition["imbalance"] < 1.1

        # A request like it finds the mesh, and makes nothing
        runs = len(list(logs.glob("jigsaw.*.stderr")))
        again = await agent.generate_mesh(REQUEST, mesh_data_dir)
        assert again["reused"] is True and again["generated"] == generated
        assert again["quality"]["passed"]
        assert len(list(logs.glob("jigsaw.*.stderr"))) == runs

    # A new agent finds it too, and with overwrite makes it again
    async with launch_mesh_agent("restart", with_mpi=True) as agent:
        again = await agent.generate_mesh(REQUEST, mesh_data_dir)
        assert again["reused"] is True
        replaced = await agent.generate_mesh(REQUEST, mesh_data_dir, overwrite=True)
        assert "reused" not in replaced
        assert replaced["generated"]["cells"] == CELLS
        assert replaced["quality"]["passed"]
    assert len(list((work / "logs").glob("jigsaw.*.stderr"))) == runs + 1
