# SPDX-License-Identifier: Apache-2.0
"""Real regional mesh generation through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs git and internet access to GitHub, and its ``compute`` resource needs make,
a Fortran compiler, and netCDF-C and netCDF-Fortran, which the MPAS-Tools
programs are built with. Partitioning also builds Metis, which needs git, wget,
tar and internet access on ``service``, and cmake and a C compiler on
``compute``.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import chiltepin.configure
from chiltepin import AgentRuntime, Workflow
from chiltepin.agents.mpas import MeshAgent

# Versions the tests install, passed to the agent so the test and the agent agree
METIS_VERSION = "5.2.1"
MPAS_TOOLS_VERSION = "2.0.0"

# A region of about 1000 cells at 50km
HEXES = {
    "center_lat": 40.0,
    "center_lon": -105.0,
    "extent_x_km": 1000,
    "extent_y_km": 800,
}


def regional(name, **changes):
    """Return a regional mesh request at 50km."""
    return {
        "resolution": "50km",
        "name": name,
        "regional": {"project_hexes": {**HEXES, **changes}},
    }


@pytest.fixture(scope="module")
def directory(config_file, request):
    """Provide a directory that the platform's compute nodes can also see.

    It is removed afterwards unless a test in this module failed, so the install
    and run logs remain available for debugging.
    """
    failed_before = request.session.testsfailed
    output_dir = Path(__file__).parent.resolve() / "test_output"
    output_dir.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="mpas-regional-", dir=output_dir))
    yield path
    if request.session.testsfailed == failed_before:
        shutil.rmtree(path, ignore_errors=True)


@asynccontextmanager
async def launch_agent(config_file, directory, tag):
    """Launch a MeshAgent that runs its tasks on the platform's resources.

    The agent runs locally and sends its install and run tasks to the platform's
    ``service`` and ``compute`` resources. All launches share ``directory/work``
    so a later one finds what an earlier one installed.
    """
    platform = chiltepin.configure.parse_file(config_file)
    src = Path(__file__).parent.parent.resolve() / "src"
    agent_config = {}
    for name in ("service", "compute"):
        resource = dict(platform[name])
        resource["environment"] = list(resource.get("environment", [])) + [
            f"export PYTHONPATH={src}${{PYTHONPATH:+:$PYTHONPATH}}"
        ]
        agent_config[name] = resource
    controller_config = {"agent-executor": {"provider": "localhost"}}
    workflow = Workflow(controller_config, run_dir=str(directory / f"runinfo-{tag}"))
    with workflow:
        runtime = AgentRuntime(workflow)
        async with await runtime.manager() as manager:
            yield await manager.launch(
                MeshAgent,
                executor="agent-executor",
                kwargs={
                    "work_dir": str(directory / "work"),
                    "metis_version": METIS_VERSION,
                    "mpas_tools_version": MPAS_TOOLS_VERSION,
                },
                agent_workflow_config=agent_config,
                agent_workflow_run_dir=str(directory / f"agent-runinfo-{tag}"),
            )


def cell_count(graph):
    """Return the number of cells, which is the first number of a graph file."""
    return int(Path(graph).read_text().split()[0])


def assert_valid_mesh(result, mesh_data_dir, name, partitions=()):
    """Check the files of a regional ``generate_mesh`` result."""
    mesh_dir = Path(mesh_data_dir) / name
    expected_keys = {"static", "graph"} | ({"partitions"} if partitions else set())
    assert set(result) == expected_keys
    assert result["static"] == str(mesh_dir / f"{name}.static.nc")
    assert result["graph"] == str(mesh_dir / f"{name}.graph.info")
    # The static file is NetCDF, either the classic or the HDF5-based format
    header = Path(result["static"]).read_bytes()[:4]
    assert header.startswith(b"CDF") or header.startswith(b"\x89HDF")
    cells = cell_count(result["graph"])
    assert cells > 100
    assert set(result.get("partitions", {})) == set(partitions)
    for rank, path in result.get("partitions", {}).items():
        assert path == f"{result['graph']}.part.{rank}"
        # One partition number per mesh cell, using every requested part
        parts = [int(value) for value in Path(path).read_text().split()]
        assert len(parts) == cells
        assert set(parts) == set(range(rank))
    # The inputs stay, and the intermediate files do not
    names = {path.name for path in mesh_dir.iterdir()}
    assert {f"{name}.static.nc", f"{name}.graph.info", "namelist.projections"} <= names
    assert not names & {"mpas_hex_mesh.nc", "graph.info", "rotated_mesh.nc"}
    return cells


@pytest.mark.asyncio
async def test_real_regional_mesh(config_file, directory):
    """Build the MPAS-Tools programs and generate regional meshes with them."""
    mesh_data_dir = str(directory / "meshes")
    work = directory / "work"
    prefix = work / "mpas-tools" / MPAS_TOOLS_VERSION
    programs = [prefix / "bin" / "project_hexes", prefix / "bin" / "grid_rotate"]
    download_log = work / "logs" / "mpas_tools_download.stdout"

    async with launch_agent(config_file, directory, "first") as agent:
        # Concurrent requests that both need the programs, one of which also
        # needs Metis for partitioning
        plain, rotated = await asyncio.gather(
            agent.generate_mesh(regional("plain"), mesh_data_dir),
            agent.generate_mesh(
                {**regional("rotated", rotation_degrees=30), "forecast_ranks": 4},
                mesh_data_dir,
            ),
        )
        cells = assert_valid_mesh(plain, mesh_data_dir, "plain")
        assert assert_valid_mesh(rotated, mesh_data_dir, "rotated", {4}) == cells

        # Rotating moves the cells but not how they connect
        assert Path(plain["graph"]).read_bytes() == Path(rotated["graph"]).read_bytes()
        plain_mesh = Path(plain["static"]).read_bytes()
        rotated_mesh = Path(rotated["static"]).read_bytes()
        assert len(plain_mesh) == len(rotated_mesh)
        assert plain_mesh != rotated_mesh
        assert (work / "mpas-tools" / "build" / MPAS_TOOLS_VERSION).exists() is False
        assert (prefix / ".installed").is_file()
        assert all(program.is_file() for program in programs)
        # No global mesh is downloaded for a regional one
        assert {path.name for path in Path(mesh_data_dir).iterdir()} == {
            "plain",
            "rotated",
        }

        # A request whose region is too small for the program leaves nothing
        # behind, and the name can be used again
        with pytest.raises(RuntimeError, match=r"project_hexes (failed|did not write)"):
            await agent.generate_mesh(
                regional("small", extent_x_km=100, extent_y_km=100), mesh_data_dir
            )
        assert not (Path(mesh_data_dir) / "small").exists()
        assert_valid_mesh(
            await agent.generate_mesh(regional("small"), mesh_data_dir),
            mesh_data_dir,
            "small",
        )

        # An explicit install does not change the installation
        installed = [program.stat().st_mtime_ns for program in programs]
        installed.append(download_log.stat().st_mtime_ns)
        await agent.install_mpas_tools()

    # A new agent has no memory of the install, and finds the programs on disk
    async with launch_agent(config_file, directory, "restart") as agent:
        again = await agent.generate_mesh(regional("again"), mesh_data_dir)
        assert assert_valid_mesh(again, mesh_data_dir, "again") == cell_count(
            plain["graph"]
        )
    after = [program.stat().st_mtime_ns for program in programs]
    after.append(download_log.stat().st_mtime_ns)
    assert after == installed

    # Every run wrote its own logs: five runs of project_hexes, including the
    # one that failed and the two that asked at the same time, and one rotation
    logs = work / "logs"
    assert len(list(logs.glob("project_hexes.*.stderr"))) == 5
    assert len(list(logs.glob("grid_rotate.*.stderr"))) == 1
    assert (logs / "mpas_tools_build.stdout").is_file()
