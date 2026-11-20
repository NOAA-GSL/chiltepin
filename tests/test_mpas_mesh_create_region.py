# SPDX-License-Identifier: Apache-2.0
"""Real create_region meshes cut from a global mesh through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs git and internet access to GitHub and UCAR, and its ``compute`` resource,
which runs the create_region program, needs chiltepin with the ``mpas`` extra,
which the test extra includes. Partitioning also builds Metis, which needs git,
wget, tar and internet access on ``service``, and cmake and a C compiler on
``compute``.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import asyncio
from pathlib import Path

import pytest

# Versions the tests install, passed to the agent so the test and the agent agree
METIS_VERSION = "5.2.1"
LIMITED_AREA_VERSION = "2.2"

# Shapes cut from the 480km global mesh, and the number of cells each gives.
# Each count includes the 8 layers of boundary cells the program adds around a
# region. A wrong conversion of kilometers to the meters the program expects
# would change them, or make the cut fail.
SHAPES = {
    "circle": (
        {"circle": {"center_lat": 39.0, "center_lon": -105.5, "radius_km": 1500}},
        344,
    ),
    "ellipse": (
        {
            "ellipse": {
                "center_lat": 39.0,
                "center_lon": -105.5,
                "semi_major_axis_km": 2000,
                "semi_minor_axis_km": 1000,
                "orientation_degrees": 30,
            }
        },
        358,
    ),
    "polygon": (
        {
            "polygon": {
                "vertices": [[30, -120], [30, -90], [48, -90], [48, -120]],
                "inside_lat": 39.0,
                "inside_lon": -105.0,
            }
        },
        328,
    ),
    "channel": ({"channel": {"upper_lat": 23.5, "lower_lat": -23.5}}, 2081),
}


def cut(name, shape):
    """Return a create_region request at 480km for one of the shapes."""
    return {
        "resolution": "480km",
        "name": name,
        "regional": {"create_region": SHAPES[shape][0]},
    }


def cell_count(graph):
    """Return the number of cells, which is the first number of a graph file."""
    return int(Path(graph).read_text().split()[0])


def assert_valid_mesh(result, mesh_data_dir, name, shape, partitions=()):
    """Check the files of a create_region ``generate_mesh`` result."""
    mesh_dir = Path(mesh_data_dir) / name
    expected_keys = {"static", "graph"} | ({"partitions"} if partitions else set())
    assert set(result) == expected_keys
    # Named after the mesh, even under a directory named like "init_runs"
    assert result["static"] == str(mesh_dir / f"{name}.static.nc")
    assert result["graph"] == str(mesh_dir / f"{name}.graph.info")
    assert Path(result["static"]).read_bytes()[:3] == b"CDF"
    cells = cell_count(result["graph"])
    assert cells == SHAPES[shape][1]
    assert set(result.get("partitions", {})) == set(partitions)
    for rank, path in result.get("partitions", {}).items():
        assert path == f"{result['graph']}.part.{rank}"
        # One partition number per mesh cell, using every requested part
        parts = [int(value) for value in Path(path).read_text().split()]
        assert len(parts) == cells
        assert set(parts) == set(range(rank))
    # The points file stays, and the link to the parent mesh does not
    names = {path.name for path in mesh_dir.iterdir()}
    assert names == {
        f"{name}.static.nc",
        f"{name}.graph.info",
        "region.pts",
    } | {Path(path).name for path in result.get("partitions", {}).values()}


@pytest.mark.asyncio
async def test_real_create_region_mesh(mesh_test_directory, launch_mesh_agent):
    """Install MPAS-Limited-Area and cut each shape from the 480km mesh."""
    directory = mesh_test_directory
    versions = {
        "metis_version": METIS_VERSION,
        "limited_area_version": LIMITED_AREA_VERSION,
    }
    # The program names its output after the parent's full path, and a path
    # containing "init" or "grid" used to give the wrong file name
    mesh_data_dir = str(directory / "init_runs" / "meshes")
    work = directory / "work"
    prefix = work / "limited-area" / LIMITED_AREA_VERSION
    program = prefix / "src" / "create_region"
    install_log = work / "logs" / "limited_area_install.stdout"

    async with launch_mesh_agent("first", **versions) as agent:
        # Concurrent requests that both need MPAS-Limited-Area and the global
        # mesh, one of which also needs Metis for partitioning
        circle, ellipse = await asyncio.gather(
            agent.generate_mesh(cut("circle", "circle"), mesh_data_dir),
            agent.generate_mesh(
                {**cut("ellipse", "ellipse"), "forecast_ranks": 4}, mesh_data_dir
            ),
        )
        assert_valid_mesh(circle, mesh_data_dir, "circle", "circle")
        assert_valid_mesh(ellipse, mesh_data_dir, "ellipse", "ellipse", {4})
        for shape in ("polygon", "channel"):
            result = await agent.generate_mesh(cut(shape, shape), mesh_data_dir)
            assert_valid_mesh(result, mesh_data_dir, shape, shape)
        assert (prefix / ".installed").is_file()
        assert program.is_file()
        # create_region runs with the tasks' Python, from the chiltepin[mpas] extra
        assert not (prefix / "venv").exists()

        # The global mesh was downloaded once and is shared by every cut
        assert {path.name for path in Path(mesh_data_dir).iterdir()} == {
            "circle",
            "ellipse",
            "polygon",
            "channel",
            "x1.2562.static.nc",
            "x1.2562.graph.info",
        }

        # A shape smaller than a cell makes the program fail, which leaves
        # nothing behind, and the name can be used again
        tiny = cut("tiny", "circle")
        tiny["regional"]["create_region"] = {
            "circle": {"center_lat": 39.0, "center_lon": -105.5, "radius_km": 1}
        }
        with pytest.raises(RuntimeError, match=r"create_region failed \(exit"):
            await agent.generate_mesh(tiny, mesh_data_dir)
        assert not (Path(mesh_data_dir) / "tiny").exists()
        assert_valid_mesh(
            await agent.generate_mesh(cut("tiny", "circle"), mesh_data_dir),
            mesh_data_dir,
            "tiny",
            "circle",
        )

        # An explicit install does not change the installation
        installed = (program.stat().st_mtime_ns, install_log.stat().st_mtime_ns)
        await agent.install_limited_area()

    # A new agent has no memory of the install, and finds it on disk
    async with launch_mesh_agent("restart", **versions) as agent:
        again = await agent.generate_mesh(cut("again", "circle"), mesh_data_dir)
        assert_valid_mesh(again, mesh_data_dir, "again", "circle")
    assert (program.stat().st_mtime_ns, install_log.stat().st_mtime_ns) == installed

    # Every run wrote its own logs: seven runs of create_region, including the
    # one that failed and the two that asked at the same time
    logs = work / "logs"
    assert len(list(logs.glob("create_region.*.stderr"))) == 7
    assert (logs / "limited_area_check.stdout").is_file()
