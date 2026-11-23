# SPDX-License-Identifier: Apache-2.0

"""Global mesh downloads and deployment through Chiltepin's agent runtime."""

import asyncio
import io
import json
import math
import pickle
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.error import URLError

import pytest
import pytest_asyncio
from parsl.app.errors import BashExitFailure
from parsl.errors import NoDataFlowKernelError

from chiltepin import AgentRuntime, Workflow
from chiltepin.agents import mpas
from chiltepin.agents.mpas import MeshAgent
from chiltepin.agents.mpas import _plot_mesh as plot_script
from chiltepin.agents.mpas import mesh as mesh_module
from chiltepin.agents.mpas._plot_mesh import (
    EXIT_MISSING_MAP,
    EXIT_VIEW_OF_REGIONAL_MESH,
    REQUIRED_PARTS,
)
from chiltepin.agents.mpas.mesh import _MeshBehavior

RESOLUTIONS = {
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


def mesh_files(cells=2562):
    return {
        f"x1.{cells}.static.nc": b"CDF\x02mesh fixture",
        f"x1.{cells}.graph.info": f"{cells} 3\n1 2\n".encode(),
    }


def archive_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def mock_download(monkeypatch, content):
    download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(content))
    monkeypatch.setattr(urllib.request, "urlopen", download)
    return download


def run_download(behavior, directory, resolution="480km"):
    """Run the task body synchronously for filesystem and archive tests."""
    _MeshBehavior._download_global_mesh.func(behavior, resolution, str(directory))


# The real lookup, which the autouse fixture below replaces
real_configured_resources = _MeshBehavior._configured_resources


@pytest.fixture(autouse=True)
def no_resource_check(monkeypatch):
    """Skip the resource check, since the workflow loaded by a fixture is partial.

    Tests of the check itself set the resources they want.
    """
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: None)
    )


def use_real_resource_lookup(monkeypatch):
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(real_configured_resources)
    )


def define_resources(monkeypatch, *names):
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: set(names))
    )


@pytest.fixture
def behavior(tmp_path):
    return _MeshBehavior(str(tmp_path / "work"))


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def mesh_agent(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mpas-agent-runtime")
    config = {
        "service": {
            "provider": "localhost",
            "cores_per_node": 1,
            "max_workers_per_node": 1,
        }
    }
    controller_config = {
        "agent-executor": config["service"],
        "service": config["service"],
    }
    workflow = Workflow(controller_config, run_dir=str(directory / "runinfo"))
    with workflow:
        runtime = AgentRuntime(workflow)
        async with await runtime.manager() as manager:
            agent = await manager.launch(
                MeshAgent,
                executor="agent-executor",
                kwargs={"work_dir": str(directory / "work")},
                agent_workflow_config=config,
                agent_workflow_run_dir=str(directory / "agent-runinfo"),
            )
            yield agent


def test_public_api_exports_only_agent():
    assert mpas.__all__ == ["MeshAgent"]
    assert mpas.MeshAgent is MeshAgent
    assert not hasattr(mpas, "MeshBehavior")
    assert not hasattr(mpas, "_MeshBehavior")


def test_behavior_configuration(behavior):
    assert behavior.log_dir == behavior.work_dir / "logs"
    assert behavior.log_dir.is_dir()
    assert behavior.resolution_cells == RESOLUTIONS
    assert behavior.mesh_data_url == (
        "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_download_coarsest_mesh(tmp_path, mesh_agent):
    """Download from UCAR, reuse the cached mesh, and repair a missing graph."""
    output = tmp_path / "meshes"
    result = await mesh_agent.download_global_mesh("480km", str(output))

    assert result == {
        "static": str(output / "x1.2562.static.nc"),
        "graph": str(output / "x1.2562.graph.info"),
    }
    assert {path.name for path in output.iterdir()} == set(mesh_files())
    for path in result.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    with Path(result["static"]).open("rb") as static:
        assert static.read(3) == b"CDF"
    with Path(result["graph"]).open() as graph:
        assert int(graph.readline().split()[0]) == 2562

    # Cached files should not be replaced or modified by a repeated request.
    before = {key: Path(path).stat() for key, path in result.items()}
    assert await mesh_agent.download_global_mesh("480km", str(output)) == result
    for key, path in result.items():
        after = Path(path).stat()
        assert after.st_ino == before[key].st_ino
        assert after.st_mtime_ns == before[key].st_mtime_ns
        assert after.st_size == before[key].st_size

    # A missing graph should trigger a fresh download restoring both mesh files.
    contents = {key: Path(path).read_bytes() for key, path in result.items()}
    Path(result["graph"]).unlink()
    assert await mesh_agent.download_global_mesh("480km", str(output)) == result
    for key, path in result.items():
        assert Path(path).is_file()
        assert not Path(path).is_symlink()
        assert Path(path).read_bytes() == contents[key]
    assert {path.name for path in output.iterdir()} == set(mesh_files())


def test_home_and_relative_paths(monkeypatch, tmp_path, behavior):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    first = behavior._resolve_mesh_data_dir("~/meshes")
    second = behavior._resolve_mesh_data_dir("meshes")
    assert first == second == tmp_path / "meshes"
    assert first.is_absolute()
    assert first.is_dir()


@pytest.mark.asyncio(loop_scope="module")
async def test_invalid_resolution_fails_before_submission(tmp_path, mesh_agent):
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="not available for download"):
        await mesh_agent.download_global_mesh("unknown", str(output))
    assert not output.exists()


def test_missing_graph_preserves_existing_mesh(monkeypatch, tmp_path, behavior):
    files = mesh_files()
    files.pop("x1.2562.graph.info")
    download = mock_download(monkeypatch, archive_bytes(files))
    output = tmp_path / "meshes"
    output.mkdir()
    original = output / "x1.2562.static.nc"
    original.write_bytes(b"keep original")

    with pytest.raises(ValueError, match="nonempty regular file"):
        run_download(behavior, output)
    download.assert_called_once_with(
        f"{behavior.mesh_data_url}/x1.2562_static.tar.gz", timeout=60
    )

    assert original.read_bytes() == b"keep original"
    assert list(output.iterdir()) == [original]

    mock_download(monkeypatch, archive_bytes(mesh_files()))
    run_download(behavior, output)
    for name, content in mesh_files().items():
        assert (output / name).read_bytes() == content
    assert {path.name for path in output.iterdir()} == set(mesh_files())


def test_download_to_file_path_fails_explicitly(tmp_path, behavior):
    destination = tmp_path / "not-a-directory"
    destination.write_text("keep this file")
    with pytest.raises(FileExistsError):
        run_download(behavior, destination)
    assert destination.read_text() == "keep this file"


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError("timed out")])
def test_network_failure_cleans_staging(monkeypatch, tmp_path, behavior, error):
    monkeypatch.setattr(urllib.request, "urlopen", Mock(side_effect=error))
    output = tmp_path / "meshes"
    with pytest.raises(type(error)):
        run_download(behavior, output)
    assert not list(output.iterdir())


@pytest.mark.asyncio(loop_scope="module")
async def test_task_routing_and_error_chaining(monkeypatch, tmp_path, behavior):
    # Simulate a successful task to check output paths and routing to "service".
    future = Future()
    future.set_result(None)
    task = Mock(return_value=future)
    monkeypatch.setattr(behavior, "_download_global_mesh", task)
    output = tmp_path / "meshes"
    result = await behavior.download_global_mesh("480km", str(output))
    assert result == {
        "static": str(output / "x1.2562.static.nc"),
        "graph": str(output / "x1.2562.graph.info"),
    }
    task.assert_called_once_with("480km", str(output), executor=["service"])

    # Exercise cache reuse and named links locally, where worker coverage is visible.
    for name, content in mesh_files().items():
        (output / name).write_bytes(content)
    run_download(behavior, output)
    named = await behavior.generate_mesh(
        {"resolution": "480km", "name": "forecast"}, str(output)
    )
    assert Path(named["static"]).resolve() == Path(result["static"])
    assert Path(named["graph"]).resolve() == Path(result["graph"])
    task.assert_called_with("480km", str(output), executor=["service"])

    # Simulate a failed task and verify the original error remains the cause.
    failed = Future()
    original = URLError("offline")
    failed.set_exception(original)
    task.return_value = failed
    with pytest.raises(RuntimeError, match="Global mesh download failed") as raised:
        await behavior.download_global_mesh("480km", str(output))
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    ("config", "resolution", "name"),
    [
        ({}, "120km", "mesh_120km"),
        ({"resolution": "480km"}, "480km", "mesh_480km"),
        ({"resolution": "480km", "name": "forecast"}, "480km", "forecast"),
    ],
)
@pytest.mark.asyncio(loop_scope="module")
async def test_generate_mesh_links(tmp_path, mesh_agent, config, resolution, name):
    output = tmp_path / "meshes"
    output.mkdir()
    cells = RESOLUTIONS[resolution]
    paths = {}
    for key, filename in (
        ("static", f"x1.{cells}.static.nc"),
        ("graph", f"x1.{cells}.graph.info"),
    ):
        path = output / filename
        path.write_bytes(b"mesh data")
        paths[key] = str(path)
    expected = {
        "static": str(output / name / f"{name}.static.nc"),
        "graph": str(output / name / f"{name}.graph.info"),
    }
    result = await mesh_agent.generate_mesh(config, str(output))
    assert result == expected
    # The same request again reuses the mesh
    again = await mesh_agent.generate_mesh(config, str(output))
    assert again == {**expected, "reused": True}
    assert Path(result["static"]).is_symlink()
    assert Path(result["static"]).resolve() == Path(paths["static"])
    assert Path(result["graph"]).is_symlink()
    assert Path(result["graph"]).resolve() == Path(paths["graph"])
    assert Path(result["static"]).read_bytes() == b"mesh data"
    assert Path(result["graph"]).read_bytes() == b"mesh data"


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling-symlink"])
@pytest.mark.asyncio(loop_scope="module")
async def test_existing_mesh_name_is_not_modified(tmp_path, mesh_agent, kind):
    output = tmp_path / "meshes"
    output.mkdir()
    target = tmp_path / "external"
    target.mkdir()
    original = target / "forecast.static.nc"
    original.write_bytes(b"keep original")
    named = output / "forecast"
    if kind == "directory":
        named.mkdir()
        (named / "marker").write_bytes(b"keep marker")
    elif kind == "file":
        named.write_bytes(b"keep file")
    else:
        named.symlink_to(target if kind == "symlink" else tmp_path / "missing")
    before = named.lstat()
    with pytest.raises(FileExistsError):
        await mesh_agent.generate_mesh({"name": "forecast"}, str(output))
    assert named.lstat() == before
    assert original.read_bytes() == b"keep original"
    assert list(output.iterdir()) == [named]


@pytest.mark.asyncio(loop_scope="module")
async def test_failed_link_creation_cleans_named_directory(
    monkeypatch, tmp_path, behavior
):
    output = tmp_path / "meshes"
    output.mkdir()
    for name, content in mesh_files().items():
        (output / name).write_bytes(content)
    future = Future()
    future.set_result(None)
    monkeypatch.setattr(behavior, "_download_global_mesh", Mock(return_value=future))
    symlink_to = Path.symlink_to

    def fail_graph_link(path, target):
        if path.name == "forecast.graph.info":
            raise PermissionError("cannot create graph link")
        symlink_to(path, target)

    monkeypatch.setattr(Path, "symlink_to", fail_graph_link)
    with pytest.raises(PermissionError, match="cannot create graph link"):
        await behavior.generate_mesh(
            {"resolution": "480km", "name": "forecast"}, str(output)
        )
    assert not (output / "forecast").exists()
    for name, content in mesh_files().items():
        assert (output / name).read_bytes() == content
    monkeypatch.setattr(Path, "symlink_to", symlink_to)
    result = await behavior.generate_mesh(
        {"resolution": "480km", "name": "forecast"}, str(output)
    )
    assert Path(result["graph"]).is_symlink()


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "/absolute", "a/b", None]
)
@pytest.mark.asyncio(loop_scope="module")
async def test_generate_mesh_rejects_invalid_name(tmp_path, mesh_agent, name):
    output = tmp_path / "meshes"
    with pytest.raises(ValueError, match="Mesh name"):
        await mesh_agent.generate_mesh({"name": name}, str(output))
    assert not output.exists()


@pytest.mark.asyncio(loop_scope="module")
async def test_generate_mesh_rejects_unsupported_resolution(tmp_path, mesh_agent):
    output = tmp_path / "meshes"
    with pytest.raises(ValueError, match="not supported"):
        await mesh_agent.generate_mesh({"resolution": "unknown"}, str(output))
    assert not output.exists()


@pytest.mark.asyncio(loop_scope="module")
async def test_generate_mesh_download_failure_creates_no_links(
    tmp_path, behavior, mesh_agent
):
    served = tmp_path / "served"
    served.mkdir()
    (served / "x1.2562_static.tar.gz").write_bytes(b"not an archive")
    behavior.mesh_data_url = served.as_uri()
    output = tmp_path / "meshes"
    with pytest.raises(RuntimeError, match="Global mesh download failed") as raised:
        await behavior.generate_mesh({"resolution": "480km"}, str(output))
    assert isinstance(raised.value.__cause__, tarfile.ReadError)
    assert not (output / "mesh_480km").exists()
    assert not list(output.iterdir())


# ---------------------------------------------------------------------------
# Metis installation and mesh partitioning
# ---------------------------------------------------------------------------


def done_future(result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def mark_metis_installed(behavior):
    """Create what a finished Metis build leaves in work_dir."""
    prefix = behavior.work_dir / "metis" / behavior.metis_version
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "bin" / "gpmetis").write_text("#!/bin/sh\n")
    (prefix / "bin" / "gpmetis").chmod(0o755)
    (prefix / ".installed").touch()
    return prefix / "bin" / "gpmetis"


def mark_mpas_tools_installed(behavior):
    """Create what a finished MPAS-Tools build leaves in work_dir."""
    prefix = behavior.work_dir / "mpas-tools" / behavior.mpas_tools_version
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    for name in ("project_hexes", "grid_rotate"):
        (prefix / "bin" / name).write_text("#!/bin/sh\n")
        (prefix / "bin" / name).chmod(0o755)
    (prefix / ".installed").touch()
    return prefix


def mock_install_tasks(
    monkeypatch,
    behavior,
    tool,
    mark,
    download=None,
    build=None,
    names=("_download_{}", "_build_{}"),
):
    """Replace a tool's download and build tasks with mocks.

    ``names`` gives the two task names, with ``{}`` standing for ``tool``.
    """

    def run_build(**kwargs):
        # A successful build leaves a finished install behind
        for dependency in kwargs.get("inputs", []):
            if dependency.done() and dependency.exception() is not None:
                return done_future(error=dependency.exception())
        if build is None:
            mark(behavior)
        return done_future(error=build)

    download_task = Mock(return_value=done_future(error=download))
    build_task = Mock(side_effect=run_build)
    monkeypatch.setattr(behavior, names[0].format(tool), download_task)
    monkeypatch.setattr(behavior, names[1].format(tool), build_task)
    return download_task, build_task


def mark_limited_area_installed(behavior):
    """Create what a finished MPAS-Limited-Area install leaves in work_dir.

    Like the install, it starts from an empty install directory.
    """
    prefix = behavior.work_dir / "limited-area" / behavior.limited_area_version
    shutil.rmtree(prefix, ignore_errors=True)
    (prefix / "src").mkdir(parents=True, exist_ok=True)
    (prefix / "src" / "create_region").write_text("#!/usr/bin/env python\n")
    (prefix / ".installed").touch()
    return prefix


def mock_limited_area_tasks(monkeypatch, behavior, install=None, check=None):
    """Replace the MPAS-Limited-Area install and check tasks with mocks."""
    return mock_install_tasks(
        monkeypatch,
        behavior,
        "limited_area",
        mark_limited_area_installed,
        install,
        check,
        names=("_install_{}", "_check_{}"),
    )


def mark_plotting_installed(behavior):
    """Create what a finished plotting install leaves in work_dir."""
    prefix = behavior.work_dir / "plot"
    for path in behavior._map_layer_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in REQUIRED_PARTS:
            path.with_suffix(suffix).write_bytes(b"shapefile")
    (prefix / ".installed").touch()
    return prefix


def mock_plotting_tasks(monkeypatch, behavior, download=None, check=None):
    """Replace the map download and plotting check tasks with mocks."""
    return mock_install_tasks(
        monkeypatch,
        behavior,
        "plotting",
        mark_plotting_installed,
        download,
        check,
        names=("_download_{}", "_check_{}"),
    )


def fake_python(monkeypatch, tmp_path, body):
    """Make the tasks' Python a shell script running ``body``.

    Scripts the tasks build name the Python running them, so this stands in for
    the Python of a resource.
    """
    path = tmp_path / "fake-python"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(path))
    return path


def mock_metis_tasks(monkeypatch, behavior, download=None, build=None):
    """Replace the Metis download and build tasks with mocks."""
    return mock_install_tasks(
        monkeypatch, behavior, "metis", mark_metis_installed, download, build
    )


def mock_mpas_tools_tasks(monkeypatch, behavior, download=None, build=None):
    """Replace the MPAS-Tools download and build tasks with mocks."""
    return mock_install_tasks(
        monkeypatch, behavior, "mpas_tools", mark_mpas_tools_installed, download, build
    )


def fake_partition_task(fail_ranks=()):
    """Return a mock for _partition_mesh that writes partition files."""

    def run(mesh_path, rank, **kwargs):
        if rank in fail_ranks:
            return done_future(error=BashExitFailure("partition", 1))
        Path(f"{mesh_path}.part.{rank}").write_text("partition")
        return done_future()

    return Mock(side_effect=run)


@pytest.fixture
def mesh_download(monkeypatch, tmp_path, behavior):
    """Provide cached 480km mesh files and a mocked global download task."""
    output = tmp_path / "meshes"
    output.mkdir()
    for name, content in mesh_files().items():
        (output / name).write_bytes(content)
    download = Mock(return_value=done_future())
    monkeypatch.setattr(behavior, "_download_global_mesh", download)
    return output, download


PARTITIONED = {
    "resolution": "480km",
    "name": "forecast",
    "init_ranks": 4,
    "forecast_ranks": 8,
}


# Configuration


@pytest.mark.parametrize(
    "version", ["", "../x", "5.2.1; rm -rf /", "a5", None, 5, "5.2.1\n", "v5.2.1"]
)
def test_invalid_metis_version(tmp_path, version):
    with pytest.raises(ValueError, match="Invalid metis_version"):
        _MeshBehavior(str(tmp_path / "work"), metis_version=version)


@pytest.mark.parametrize(
    "version", ["", "../x", "2.0.0; rm -rf /", "a2", None, 2, "2.0.0\n", "$(id)"]
)
def test_invalid_mpas_tools_version(tmp_path, version):
    with pytest.raises(ValueError, match="Invalid mpas_tools_version"):
        _MeshBehavior(str(tmp_path / "work"), mpas_tools_version=version)


@pytest.mark.parametrize(
    "version", ["", "../x", "2.2; rm -rf /", "v2.2", None, 2.2, "2.2\n", "$(id)"]
)
def test_invalid_limited_area_version(tmp_path, version):
    with pytest.raises(ValueError, match="Invalid limited_area_version"):
        _MeshBehavior(str(tmp_path / "work"), limited_area_version=version)


def test_work_dir_is_made_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _MeshBehavior("rel/../work").work_dir == tmp_path / "work"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, []),
        ({"init_ranks": 1, "forecast_ranks": 1}, []),
        ({"init_ranks": 4, "forecast_ranks": 16}, [4, 16]),
        ({"init_ranks": 8, "forecast_ranks": 8}, [8]),
    ],
)
def test_partition_ranks(config, expected):
    assert _MeshBehavior._partition_ranks(config) == expected


@pytest.mark.parametrize("value", [0, -2, 2.5, "4", True])
@pytest.mark.parametrize("key", ["init_ranks", "forecast_ranks"])
def test_validate_mesh_config_rejects_invalid_ranks(behavior, key, value):
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        behavior._validate_mesh_config({key: value})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (None, "must be a dict"),
        ([], "must be a dict"),
        ({"forecast_rank": 4}, "Unknown mesh_config keys: forecast_rank"),
        ({"resolution": "1km"}, "not supported"),
        ({"resolution": ["480km"]}, "not supported"),
        ({"name": ""}, "valid file name"),
        ({"name": "a/b"}, "valid file name"),
        ({"name": 5}, "valid file name"),
        ({"name": "x1.2562.graph.info"}, "downloaded global mesh file"),
    ],
)
def test_validate_mesh_config_rejects_invalid(behavior, config, message):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config(config)


@pytest.mark.parametrize("name", ["a" * 200, "\u00e9" * 100, "x" * 199 + "."])
def test_validate_mesh_config_accepts_names_up_to_the_limit(behavior, name):
    behavior._validate_mesh_config({"name": name})


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("a" * 201, "at most 200 bytes long, got 201"),
        ("\u00e9" * 101, "at most 200 bytes long, got 202"),
        ("\U0001f600" * 51, "at most 200 bytes long, got 204"),
        ("bad\ud800", "valid file name"),
    ],
)
def test_validate_mesh_config_rejects_long_or_unencodable_names(
    behavior, name, message
):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config({"name": name})
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config({**regional_config(), "name": name})


async def test_longest_name_gets_every_file_it_needs(
    monkeypatch, mesh_download, behavior, tmp_path
):
    """The longest allowed name must leave room for the longest file suffix."""
    output, _ = mesh_download
    mark_metis_installed(behavior)
    monkeypatch.setattr(behavior, "_partition_mesh", fake_partition_task())
    name = "n" * 200
    config = {"resolution": "480km", "name": name, "forecast_ranks": 1048576}
    result = await behavior.generate_mesh(config, str(output))
    assert Path(result["partitions"][1048576]).is_file()


def test_validate_mesh_config_accepts_valid(behavior):
    behavior._validate_mesh_config({})
    behavior._validate_mesh_config(PARTITIONED)
    behavior._validate_mesh_config({"resolution": "480km", "init_ranks": 1})


HEXES = {
    "center_lat": 40.0,
    "center_lon": -105.0,
    "extent_x_km": 500,
    "extent_y_km": 400,
}


# One valid block for each create_region shape
SHAPES = {
    "circle": {"center_lat": 39.0, "center_lon": -105.5, "radius_km": 1500},
    "ellipse": {
        "center_lat": 39.0,
        "center_lon": -105.5,
        "semi_major_axis_km": 2000,
        "semi_minor_axis_km": 1000,
        "orientation_degrees": 30,
    },
    "polygon": {
        "vertices": [[30, -120], [30, -90], [48, -90], [48, -120]],
        "inside_lat": 39.0,
        "inside_lon": -105.0,
    },
    "channel": {"upper_lat": 23.5, "lower_lat": -23.5},
}


def create_region_config(shape="circle", name="cut", **changes):
    """Return a valid create_region request, with the shape's block changed."""
    block = {**SHAPES[shape], **changes}
    return {
        "resolution": "480km",
        "name": name,
        "regional": {
            "create_region": {shape: {k: v for k, v in block.items() if v is not ...}}
        },
    }


def regional_config(**changes):
    """Return a valid regional request, with the project_hexes block changed."""
    block = {**HEXES, **changes}
    return {
        "resolution": "15km",
        "name": "region",
        "regional": {"project_hexes": {k: v for k, v in block.items() if v is not ...}},
    }


@pytest.mark.parametrize(
    "resolution", ["15km", "7.5km", "2km", "0.5km", "100km", "12345km"]
)
def test_validate_mesh_config_accepts_any_regional_resolution(behavior, resolution):
    # On the equator, where no pole limits how far coarse cells' boundary
    # layers reach
    config = regional_config(center_lat=0.0)
    config["resolution"] = resolution
    behavior._validate_mesh_config(config)


def test_validate_mesh_config_accepts_regional_defaults_and_rotation(behavior):
    behavior._validate_mesh_config(
        {"regional": {"project_hexes": {**HEXES, "rotation_degrees": -30.5}}}
    )
    # Far south, short of the pole, which project_hexes cannot reach
    behavior._validate_mesh_config(regional_config(center_lon=270, center_lat=-80))


def test_global_resolutions_stay_limited_to_the_available_meshes(behavior):
    with pytest.raises(ValueError, match="not supported"):
        behavior._validate_mesh_config({"resolution": "2km"})


@pytest.mark.parametrize(
    "resolution",
    [
        "0km",
        "0.0km",
        "-5km",
        "15",
        "km",
        "",
        "1e3km",
        "15 km",
        "1.km",
        None,
        15,
        ["1km"],
    ]
    + ["9" * 400 + "km"],
)
def test_validate_mesh_config_rejects_invalid_regional_resolution(behavior, resolution):
    config = regional_config()
    config["resolution"] = resolution
    with pytest.raises(ValueError, match="is not valid"):
        behavior._validate_mesh_config(config)


@pytest.mark.parametrize(
    ("regional", "message"),
    [
        ("project_hexes", "regional must be a dict"),
        ([], "regional must be a dict"),
        ({}, "exactly one of 'project_hexes' or 'create_region'"),
        ({"create_region": {}}, "must contain exactly one of circle, ellipse"),
        (
            {"project_hexes": HEXES, "create_region": {}},
            "exactly one of 'project_hexes' or 'create_region'",
        ),
        ({"project_hex": HEXES}, "Unknown regional keys: project_hex"),
        ({"project_hexes": None}, "regional.project_hexes must be a dict"),
        ({"project_hexes": []}, "regional.project_hexes must be a dict"),
        ({"project_hexes": {}}, "missing required keys: center_lat, center_lon"),
        (
            {"project_hexes": {**HEXES, "extent": 5}},
            "Unknown regional.project_hexes keys: extent",
        ),
    ],
)
def test_validate_mesh_config_rejects_invalid_regional(behavior, regional, message):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config({"resolution": "15km", "regional": regional})


@pytest.mark.parametrize("key", list(HEXES) + ["rotation_degrees"])
@pytest.mark.parametrize(
    "value",
    [True, "40", "40,", None, float("nan"), float("inf"), 10**400, -(2**1024)],
)
def test_validate_mesh_config_rejects_non_numbers_in_project_hexes(
    behavior, key, value
):
    with pytest.raises(ValueError, match=f"regional.project_hexes.{key} must be a"):
        behavior._validate_mesh_config(regional_config(**{key: value}))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("center_lat", 90.1, "between -90 and 90"),
        ("center_lat", -91, "between -90 and 90"),
        ("center_lon", -180.5, "between -180 and 360"),
        ("center_lon", 360.5, "between -180 and 360"),
        ("extent_x_km", 0, "greater than 0"),
        ("extent_y_km", -1.5, "greater than 0"),
    ],
)
def test_validate_mesh_config_rejects_out_of_range_project_hexes(
    behavior, key, value, message
):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config(regional_config(**{key: value}))


# The cone's apex is the pole, R cot(lat) from the center: 1123.4 km at 80
# degrees, and R / sqrt(3) = 3678.4 km at 60. A 15km mesh reaches 120 km past
# half its height, with its 7 boundary layers and a cell for its offset rows.
@pytest.mark.parametrize(
    ("center_lat", "extent_y_km", "resolution", "limit"),
    [
        (90, 1, "15km", "0.0"),
        (-90, 1, "15km", "0.0"),
        (80, 2007, "15km", "2006.8"),
        (-60, 7117, "15km", "7116.9"),
        # Coarser cells have wider boundary layers
        (80, 1500, "60km", "1286.8"),
    ],
)
def test_validate_mesh_config_rejects_project_hexes_reaching_a_pole(
    behavior, center_lat, extent_y_km, resolution, limit
):
    config = regional_config(center_lat=center_lat, extent_y_km=extent_y_km)
    config["resolution"] = resolution
    with pytest.raises(
        ValueError, match=f"reaches the pole.*less than {limit}. Use create_region"
    ):
        behavior._validate_mesh_config(config)


@pytest.mark.parametrize(
    ("center_lat", "extent_y_km"),
    [(80, 2006), (-60, 7116), (0, 20000), (1e-6, 9000)],
)
def test_validate_mesh_config_accepts_project_hexes_short_of_a_pole(
    behavior, center_lat, extent_y_km
):
    config = regional_config(center_lat=center_lat, extent_y_km=extent_y_km)
    behavior._validate_mesh_config(config)


def test_validate_mesh_config_requires_every_project_hexes_key(behavior):
    for key in HEXES:
        with pytest.raises(ValueError, match=f"missing required keys: {key}"):
            behavior._validate_mesh_config(regional_config(**{key: ...}))


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize(
    "config", [{}, PARTITIONED, "regional_config", "create_region_config"]
)
def test_validate_mesh_config_accepts_plot(behavior, config, plot):
    base = {
        "regional_config": regional_config,
        "create_region_config": create_region_config,
    }
    config = base[config]() if isinstance(config, str) else config
    behavior._validate_mesh_config({**config, "plot": plot})


@pytest.mark.parametrize("plot", [1, 0, "true", None, "yes", [True]])
def test_validate_mesh_config_rejects_non_boolean_plot(behavior, plot):
    with pytest.raises(ValueError, match="plot must be True or False"):
        behavior._validate_mesh_config({"plot": plot})


@pytest.mark.parametrize(
    "view",
    [
        {},
        {"title": "A mesh"},
        {"title": ""},
        {"zoom": 2},
        {"zoom": 0.5},
        {"zoom": 1e6},
        {"center_lat": 50.0, "center_lon": 10.0},
        {"center_lat": -90, "center_lon": 360},
        {"title": "x", "zoom": 3.5, "center_lat": 0, "center_lon": -180},
    ],
)
def test_validate_plot_view_accepts_valid(view):
    settings = {"title": None, "zoom": None, "center_lat": None, "center_lon": None}
    _MeshBehavior._validate_plot_view(**{**settings, **view})


@pytest.mark.parametrize(
    ("view", "message"),
    [
        ({"title": 5}, "title must be a string"),
        ({"title": b"x"}, "title must be a string"),
        ({"zoom": 0}, "zoom must be greater than 0"),
        ({"zoom": -1}, "zoom must be greater than 0"),
        ({"zoom": True}, "zoom must be a finite number"),
        ({"zoom": "2"}, "zoom must be a finite number"),
        ({"zoom": float("inf")}, "zoom must be a finite number"),
        ({"zoom": float("nan")}, "zoom must be a finite number"),
        ({"center_lat": 50.0}, "must be given together"),
        ({"center_lon": 10.0}, "must be given together"),
        ({"center_lat": 91, "center_lon": 0}, "center_lat must be between -90 and 90"),
        ({"center_lat": 0, "center_lon": 361}, "center_lon must be between -180"),
        ({"center_lat": "0", "center_lon": 0}, "center_lat must be a finite number"),
        ({"center_lat": 0, "center_lon": None}, "must be given together"),
    ],
)
def test_validate_plot_view_rejects_invalid(view, message):
    settings = {"title": None, "zoom": None, "center_lat": None, "center_lon": None}
    with pytest.raises(ValueError, match=message):
        _MeshBehavior._validate_plot_view(**{**settings, **view})


async def test_plot_false_is_an_ordinary_request(mesh_download, behavior):
    output, download = mesh_download
    result = await behavior.generate_mesh(
        {"resolution": "480km", "name": "g", "plot": False}, str(output)
    )
    assert set(result) == {"static", "graph"}
    download.assert_called_once()


@pytest.mark.parametrize("shape", list(SHAPES))
def test_validate_mesh_config_accepts_each_create_region_shape(behavior, shape):
    behavior._validate_mesh_config(create_region_config(shape))


def test_validate_mesh_config_accepts_create_region_edge_values(behavior):
    behavior._validate_mesh_config(
        create_region_config("ellipse", orientation_degrees=...)
    )
    behavior._validate_mesh_config(
        create_region_config("ellipse", semi_minor_axis_km=2000)
    )
    behavior._validate_mesh_config(
        create_region_config("circle", center_lat=-90, center_lon=360)
    )
    behavior._validate_mesh_config(
        create_region_config("polygon", vertices=((1, 2), (3, 4), (5, 6)))
    )
    behavior._validate_mesh_config(
        create_region_config("channel", upper_lat=90, lower_lat=-90)
    )


@pytest.mark.parametrize(
    ("shape", "key"),
    [
        ("circle", "radius_km"),
        ("ellipse", "semi_major_axis_km"),
        ("ellipse", "semi_minor_axis_km"),
    ],
)
def test_create_region_sizes_must_stay_within_half_the_globe(behavior, shape, key):
    limit = math.pi * 6371.229
    # The major axis must be valid when the minor one is the one tested
    major = {"semi_major_axis_km": limit - 0.1} if key == "semi_minor_axis_km" else {}
    for value in (limit, 1e306):
        with pytest.raises(ValueError, match=f"{key} must be less than half the circ"):
            behavior._validate_mesh_config(
                create_region_config(shape, **major, **{key: value})
            )
    # Just under the limit is allowed
    changes = {"semi_major_axis_km": limit - 0.1} if shape == "ellipse" else {}
    changes[key] = limit - 0.1
    behavior._validate_mesh_config(create_region_config(shape, **changes))


@pytest.mark.parametrize(
    ("angle", "written"),
    [
        (390, "30.0"),
        (-330, "30.0"),
        (720, "0.0"),
        (-0.0, "0.0"),
        (359.5, "359.5"),
        (-1e-20, "0.0"),
    ],
)
def test_create_region_orientation_is_written_as_one_turn(angle, written):
    block = {**SHAPES["ellipse"], "orientation_degrees": angle}
    text = _MeshBehavior._create_region_points({"ellipse": block})
    assert f"Orientation-angle: {written}\n" in text


def test_huge_create_region_orientation_is_still_a_finite_angle():
    block = {**SHAPES["ellipse"], "orientation_degrees": 1e300}
    text = _MeshBehavior._create_region_points({"ellipse": block})
    angle = float(text.split("Orientation-angle: ")[1])
    assert 0 <= angle < 360


@pytest.mark.parametrize("resolution", ["2km", "50km", "0.5km"])
def test_create_region_needs_a_downloadable_resolution(behavior, resolution):
    config = create_region_config()
    config["resolution"] = resolution
    with pytest.raises(ValueError, match="not supported. Available"):
        behavior._validate_mesh_config(config)


@pytest.mark.parametrize(
    ("create_region", "message"),
    [
        (None, "regional.create_region must be a dict"),
        ([], "regional.create_region must be a dict"),
        ({}, "must contain exactly one of circle, ellipse, polygon, channel"),
        (
            {"circle": SHAPES["circle"], "channel": SHAPES["channel"]},
            "must contain exactly one of",
        ),
        ({"square": {}}, "Unknown regional.create_region keys: square"),
        ({"circle": None}, "regional.create_region.circle must be a dict"),
        (
            {"circle": {**SHAPES["circle"], "radius": 5}},
            "Unknown regional.create_region.circle keys: radius",
        ),
        (
            {"circle": {"center_lat": 1}},
            "circle is missing required keys: center_lon, radius_km",
        ),
    ],
)
def test_validate_mesh_config_rejects_invalid_create_region(
    behavior, create_region, message
):
    config = {"resolution": "480km", "regional": {"create_region": create_region}}
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config(config)


@pytest.mark.parametrize(
    ("shape", "key"),
    [(shape, key) for shape, block in SHAPES.items() for key in block],
)
@pytest.mark.parametrize("value", [True, "40", None, float("nan"), 10**400])
def test_validate_mesh_config_rejects_non_numbers_in_create_region(
    behavior, shape, key, value
):
    with pytest.raises(ValueError, match=f"create_region.{shape}.{key}"):
        behavior._validate_mesh_config(create_region_config(shape, **{key: value}))


@pytest.mark.parametrize(
    ("shape", "changes", "message"),
    [
        ("circle", {"center_lat": 90.5}, "center_lat must be between -90 and 90"),
        ("circle", {"center_lon": -181}, "center_lon must be between -180 and 360"),
        ("circle", {"radius_km": 0}, "radius_km must be greater than 0"),
        ("ellipse", {"semi_major_axis_km": -1}, "semi_major_axis_km must be greater"),
        ("ellipse", {"semi_minor_axis_km": 0}, "semi_minor_axis_km must be greater"),
        (
            "ellipse",
            {"semi_minor_axis_km": 2001},
            "semi_minor_axis_km must not be greater than semi_major_axis_km",
        ),
        ("polygon", {"inside_lat": -91}, "inside_lat must be between -90 and 90"),
        ("polygon", {"inside_lon": 400}, "inside_lon must be between -180 and 360"),
        ("channel", {"upper_lat": 95}, "upper_lat must be between -90 and 90"),
        ("channel", {"lower_lat": 23.5}, "upper_lat must be greater than lower_lat"),
        ("channel", {"lower_lat": 30}, "upper_lat must be greater than lower_lat"),
    ],
)
def test_validate_mesh_config_rejects_out_of_range_create_region(
    behavior, shape, changes, message
):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config(create_region_config(shape, **changes))


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        ("30,-120 30,-90 48,-90", "vertices must be a list of at least 3"),
        ([[30, -120], [30, -90]], "vertices must be a list of at least 3"),
        (None, "vertices must be a list of at least 3"),
        ([[30, -120], [30, -90], [48]], r"vertices\[2\] must be a \[lat, lon\] pair"),
        ([[30, -120], [30, -90], "48,-90"], r"vertices\[2\] must be a \[lat, lon\]"),
        ([[30, -120], [30, -90], [48, -90, 0]], r"vertices\[2\] must be a \[lat"),
        ([[30, -120], [True, -90], [48, -90]], r"vertices\[1\] latitude must be a fin"),
        ([[30, -120], [30, "x"], [48, -90]], r"vertices\[1\] longitude must be a fin"),
        ([[30, -120], [91, -90], [48, -90]], r"vertices\[1\] latitude must be betw"),
        ([[30, -120], [30, 361], [48, -90]], r"vertices\[1\] longitude must be betw"),
    ],
)
def test_validate_mesh_config_rejects_invalid_polygon_vertices(
    behavior, vertices, message
):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config(
            create_region_config("polygon", vertices=vertices)
        )


@pytest.mark.parametrize("name", ["a\nb", "tab\there", "nul\x00", "del\x7f", "c1\x85"])
@pytest.mark.parametrize("config", [{}, regional_config(), create_region_config()])
def test_validate_mesh_config_rejects_control_characters_in_names(
    behavior, name, config
):
    with pytest.raises(ValueError, match="must not contain control characters"):
        behavior._validate_mesh_config({**config, "name": name})


async def test_invalid_regional_mesh_creates_nothing(tmp_path, behavior):
    output = tmp_path / "meshes"
    with pytest.raises(ValueError, match="missing required keys"):
        await behavior.generate_mesh(
            {"regional": {"create_region": {"circle": {}}}}, str(output)
        )
    assert not output.exists()


# Task scripts


def test_metis_scripts(tmp_path):
    behavior = _MeshBehavior(str(tmp_path / "work dir"))
    download = _MeshBehavior._download_metis.func(behavior)
    build = _MeshBehavior._build_metis.func(behavior)
    assert f"build='{behavior.work_dir}/metis/build/5.2.1'" in download
    assert "/refs/tags/v5.2.1.tar.gz" in download
    assert f"prefix='{behavior.work_dir}/metis/5.2.1'" in build
    # A failed earlier install is removed, and a finished one is marked
    assert build.index('rm -rf "$prefix"') < build.index("make install")
    # The sources are removed after a successful build, before it is marked
    assert build.rindex('rm -rf "$build"') < build.index('touch "$prefix/.installed"')
    lines = [line.strip() for line in build.splitlines()]
    marker = lines.index('touch "$prefix/.installed"')
    assert marker > max(i for i, line in enumerate(lines) if line == "make install")
    # Only echo commands may follow the marker
    assert all(line.startswith("echo ") for line in lines[marker + 1 :] if line)


def test_plotting_scripts(tmp_path):
    behavior = _MeshBehavior(str(tmp_path / "work dir"))
    download = _MeshBehavior._download_plotting.func(behavior)
    check = _MeshBehavior._check_plotting.func(behavior)
    prefix = f"{behavior.work_dir}/plot"
    assert f"prefix='{prefix}'" in download
    assert f"prefix='{prefix}'" in check
    # The map layers are downloaded by the plotting script, with the Python of
    # the task, after the install's marker is removed. Nothing else is removed,
    # so plots running meanwhile keep their map layers.
    fetch = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(behavior._plot_script))} "
        f"--download-map-data {shlex.quote(str(behavior._map_data_dir))}"
    )
    assert fetch in download
    assert download.index('rm -f "$prefix/.installed"') < download.index(fetch)
    assert "rm -rf" not in download
    assert download.rstrip().endswith('echo "Completed download at $(date)"')
    # The install is marked only after the imports and every layer are checked,
    # with the Python of the task
    lines = [line.strip() for line in check.splitlines()]
    marker = lines.index('touch "$prefix/.installed"')
    imports = next(i for i, line in enumerate(lines) if "import cartopy" in line)
    assert shlex.quote(sys.executable) in lines[imports]
    layers = next(i for i, line in enumerate(lines) if line.startswith("for layer"))
    assert imports < layers < marker
    for path in behavior._map_layer_paths():
        for suffix in REQUIRED_PARTS:
            assert shlex.quote(str(path.with_suffix(suffix))) in check
    assert all(line.startswith("echo ") for line in lines[marker + 1 :] if line)


def test_plotting_download_writes_the_layers(monkeypatch, tmp_path, behavior):
    """Run the download script with a stand-in Python that writes the layers."""
    record = tmp_path / "record"
    fake_python(
        monkeypatch,
        tmp_path,
        f'printf "%s\\n" "$@" > {shlex.quote(str(record))}',
    )
    prefix = behavior.work_dir / "plot"
    (prefix / "matplotlib").mkdir(parents=True)
    (prefix / ".installed").touch()
    script = _MeshBehavior._download_plotting.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert record.read_text().splitlines() == [
        str(behavior._plot_script),
        "--download-map-data",
        str(behavior._map_data_dir),
    ]
    # The earlier install no longer counts, but its files are kept
    assert not (prefix / ".installed").exists()
    assert (prefix / "matplotlib").is_dir()


@pytest.mark.parametrize(
    "problem", [None, "imports", "missing_layer", "empty_layer", "missing_dbf"]
)
def test_plotting_check_marks_only_a_working_install(
    monkeypatch, tmp_path, behavior, problem
):
    prefix = mark_plotting_installed(behavior)
    (prefix / ".installed").unlink()
    fake_python(monkeypatch, tmp_path, "exit 1" if problem == "imports" else "exit 0")
    layer = behavior._map_layer_paths()[2]
    if problem == "missing_dbf":
        layer = layer.with_suffix(".dbf")
    if problem in ("missing_layer", "missing_dbf"):
        layer.unlink()
    elif problem == "empty_layer":
        layer.write_bytes(b"")
    script = _MeshBehavior._check_plotting.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (done.returncode == 0) == (problem is None)
    assert (prefix / ".installed").is_file() == (problem is None)
    if problem == "imports":
        assert "Install chiltepin[mpas]" in done.stderr
    elif problem is not None:
        assert f"Map layer {layer} is missing" in done.stderr


def test_plotting_check_imports_with_the_real_packages(monkeypatch, tmp_path, behavior):
    # The tests have the chiltepin[mpas] extra, so the real check passes
    prefix = mark_plotting_installed(behavior)
    (prefix / ".installed").unlink()
    script = _MeshBehavior._check_plotting.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (prefix / ".installed").is_file()


def test_map_layers_are_where_cartopy_stores_them(behavior):
    data_dir = behavior.work_dir / "plot" / "natural_earth" / "shapefiles"
    assert behavior._map_layer_paths() == [
        data_dir / "natural_earth" / "physical" / "ne_10m_coastline.shp",
        data_dir
        / "natural_earth"
        / "cultural"
        / "ne_10m_admin_0_boundary_lines_land.shp",
        data_dir
        / "natural_earth"
        / "cultural"
        / "ne_10m_admin_1_states_provinces_lines.shp",
        data_dir / "natural_earth" / "physical" / "ne_110m_coastline.shp",
    ]


def test_limited_area_scripts(tmp_path):
    behavior = _MeshBehavior(str(tmp_path / "work dir"), limited_area_version="2.1")
    install = _MeshBehavior._install_limited_area.func(behavior)
    check = _MeshBehavior._check_limited_area.func(behavior)
    prefix = f"{behavior.work_dir}/limited-area/2.1"
    assert f"prefix='{prefix}'" in install
    assert f"prefix='{prefix}'" in check
    # The tag has a v, and the clone goes to src in the prefix
    assert "--depth 1 --branch v2.1 https://github.com/MPAS-Dev/MPAS-Limited-Area" in (
        install
    )
    assert "MPAS-Limited-Area.git src" in install
    # Only the source is installed. create_region runs with the Python of the
    # tasks, which has the chiltepin[mpas] extra.
    assert "venv" not in install
    assert "pip install" not in install
    assert install.rstrip().endswith('echo "Completed install at $(date)"')
    assert f"python={shlex.quote(sys.executable)}" in check
    # A failed earlier install is removed before anything is cloned
    assert install.index('rm -rf "$prefix"') < install.index("git ")
    # The install is marked only after the imports and the script are checked
    lines = [line.strip() for line in check.splitlines()]
    marker = lines.index('touch "$prefix/.installed"')
    imports = next(i for i, line in enumerate(lines) if "import numpy, netCDF4" in line)
    runs = next(i for i, line in enumerate(lines) if 'create_region" --help' in line)
    assert imports < runs < marker
    assert all(line.startswith("echo ") for line in lines[marker + 1 :] if line)


def test_limited_area_check_fails_if_create_region_does_not_run(
    monkeypatch, tmp_path, behavior
):
    prefix = mark_limited_area_installed(behavior)
    (prefix / ".installed").unlink()
    # Imports work, and running the script fails
    fake_python(monkeypatch, tmp_path, '[ "$1" = "-c" ] && exit 0\nexit 3')
    script = _MeshBehavior._check_limited_area.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 3
    assert not (prefix / ".installed").exists()


@pytest.mark.parametrize("works", [True, False])
def test_limited_area_check_marks_only_a_working_install(
    monkeypatch, tmp_path, behavior, works
):
    prefix = mark_limited_area_installed(behavior)
    (prefix / ".installed").unlink()
    # A fake Python that either runs everything or fails every import
    fake_python(monkeypatch, tmp_path, "exit 0" if works else "exit 1")
    script = _MeshBehavior._check_limited_area.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (done.returncode == 0) == works
    assert (prefix / ".installed").is_file() == works
    if not works:
        assert "cannot import numpy and netCDF4" in done.stderr
        assert "Install chiltepin[mpas]" in done.stderr


def test_limited_area_check_imports_with_the_real_packages(
    monkeypatch, tmp_path, behavior
):
    # The tests have the chiltepin[mpas] extra, so the imports pass, and a stand-in
    # create_region shows the script is run with the tasks' Python
    prefix = mark_limited_area_installed(behavior)
    (prefix / ".installed").unlink()
    (prefix / "src" / "create_region").write_text("import netCDF4, numpy\n")
    script = _MeshBehavior._check_limited_area.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (prefix / ".installed").is_file()


def test_mpas_tools_scripts(tmp_path):
    behavior = _MeshBehavior(str(tmp_path / "work dir"), mpas_tools_version="2.1.0")
    download = _MeshBehavior._download_mpas_tools.func(behavior)
    build = _MeshBehavior._build_mpas_tools.func(behavior)
    assert f"build='{behavior.work_dir}/mpas-tools/build/2.1.0'" in download
    assert "--depth 1 --branch 2.1.0 https://github.com/MPAS-Dev/MPAS-Tools" in download
    assert f"prefix='{behavior.work_dir}/mpas-tools/2.1.0'" in build
    lines = [line.strip() for line in build.splitlines()]
    # Missing netCDF tools fail before an existing install is touched
    assert build.index("nf-config") < build.index('rm -rf "$prefix"')
    # A failed earlier install is removed before the build, and the programs are
    # copied out and the sources removed before the install is marked
    assert build.index('rm -rf "$prefix"') < build.index("make -C")
    last_make = max(i for i, line in enumerate(lines) if line.startswith("make -C"))
    copy = next(i for i, line in enumerate(lines) if line.startswith("cp "))
    sources = lines.index('rm -rf "$build" || true')
    marker = lines.index('touch "$prefix/.installed"')
    assert last_make < copy < sources < marker
    assert "hex_projection/project_hexes" in lines[copy]
    assert "grid_rotate/grid_rotate" in lines[copy]
    # Only echo commands may follow the marker
    assert all(line.startswith("echo ") for line in lines[marker + 1 :] if line)


def test_mpas_tools_build_without_netcdf_keeps_an_existing_install(tmp_path, behavior):
    prefix = mark_mpas_tools_installed(behavior)
    # A PATH holding only what the script uses before the netCDF check
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("date", "hostname"):
        found = shutil.which(name)
        if found is None:
            pytest.skip(f"{name} is not available")
        (tools / name).symlink_to(found)
    script = _MeshBehavior._build_mpas_tools.func(behavior)
    done = subprocess.run(
        [shutil.which("bash"), "-c", script],
        env={"PATH": str(tools)},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert "nf-config not found" in done.stderr
    assert (prefix / ".installed").is_file()


@pytest.mark.parametrize(
    "name", ["m$(touch {pwn})", "b`touch {pwn}`", 'q"$(touch {pwn})', "$HOME"]
)
def test_partition_script_does_not_expand_mesh_path(tmp_path, behavior, name):
    pwned = tmp_path / "PWNED"
    mesh_path = str(tmp_path / name.format(pwn=pwned)) + ".graph.info"
    record = tmp_path / "args"
    # The script runs gpmetis from the install, so put a fake one there
    fake = mark_metis_installed(behavior)
    fake.write_text(
        f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > {shlex.quote(str(record))}\n'
    )

    script = _MeshBehavior._partition_mesh.func(behavior, mesh_path, 4)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert not pwned.exists()
    assert record.read_text().splitlines()[-2:] == [mesh_path, "4"]


# Installing Metis


async def test_install_metis_reuses_a_finished_install(monkeypatch, behavior):
    mark_metis_installed(behavior)
    download, build = mock_metis_tasks(monkeypatch, behavior)
    await behavior.install_metis()
    assert behavior.metis_built
    download.assert_not_called()
    build.assert_not_called()


async def test_install_metis_ignores_an_unfinished_install(monkeypatch, behavior):
    gpmetis = mark_metis_installed(behavior)
    (gpmetis.parent.parent / ".installed").unlink()
    download, build = mock_metis_tasks(monkeypatch, behavior)
    await behavior.install_metis()
    download.assert_called_once()
    build.assert_called_once()


async def test_concurrent_installs_run_once(monkeypatch, behavior):
    # Keep the download unfinished while the other requests arrive, so they
    # would start installs of their own if nothing made them wait
    pending, build_pending = Future(), Future()
    download, build = mock_metis_tasks(monkeypatch, behavior)
    download.return_value = pending
    build.side_effect = lambda **kwargs: build_pending
    requests = [asyncio.create_task(behavior.install_metis()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert download.call_count == 1
    pending.set_result(None)
    mark_metis_installed(behavior)
    build_pending.set_result(None)
    await asyncio.gather(*requests)
    download.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "metis_download.stdout"),
        stderr=str(behavior.log_dir / "metis_download.stderr"),
    )
    build.assert_called_once_with(
        executor=["compute"],
        stdout=str(behavior.log_dir / "metis_build.stdout"),
        stderr=str(behavior.log_dir / "metis_build.stderr"),
        inputs=[download.return_value],
    )
    assert behavior._gpmetis == (
        behavior.work_dir / "metis" / "5.2.1" / "bin" / "gpmetis"
    )


@pytest.mark.parametrize("stage", ["download", "build"])
async def test_failed_install_is_reported_and_retried(monkeypatch, behavior, stage):
    failure = BashExitFailure("task", 3)
    download, build = mock_metis_tasks(
        monkeypatch,
        behavior,
        download=failure if stage == "download" else None,
        build=failure if stage == "build" else None,
    )
    # Each waiting request reports the failure, after trying again itself
    results = await asyncio.gather(
        *(behavior.install_metis() for _ in range(2)), return_exceptions=True
    )
    for result in results:
        assert isinstance(result, RuntimeError)
        assert f"Metis {stage} failed (exit 3)" in str(result)
        assert f"metis_{stage}.stderr" in str(result)
        assert result.__cause__ is failure
    assert download.call_count == 2
    assert not behavior.metis_built

    download.return_value = done_future()
    build.side_effect = lambda **kwargs: (
        mark_metis_installed(behavior),
        done_future(),
    )[1]
    await behavior.install_metis()
    assert behavior.metis_built


async def test_behavior_is_picklable_while_requests_wait(behavior):
    """Task submission pickles the behavior while an install holds the lock."""
    await behavior._install_lock.acquire()
    waiter = asyncio.create_task(behavior._install_lock.acquire())
    await asyncio.sleep(0.05)

    download_lock = behavior._download_locks.setdefault("key", asyncio.Lock())
    await download_lock.acquire()
    download_waiter = asyncio.create_task(download_lock.acquire())
    await asyncio.sleep(0.05)

    restored = pickle.loads(pickle.dumps(behavior))
    assert restored.work_dir == behavior.work_dir
    assert not restored._install_lock.locked()
    assert restored._download_locks == {}

    behavior._install_lock.release()
    await waiter
    behavior._install_lock.release()
    download_lock.release()
    await download_waiter
    download_lock.release()


# Installing MPAS-Tools


async def test_install_mpas_tools_reuses_a_finished_install(monkeypatch, behavior):
    mark_mpas_tools_installed(behavior)
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.install_mpas_tools()
    assert behavior.mpas_tools_built
    download.assert_not_called()
    build.assert_not_called()


@pytest.mark.parametrize(
    "damage", ["marker", "project_hexes", "grid_rotate", "not_executable"]
)
async def test_install_mpas_tools_rebuilds_a_damaged_install(
    monkeypatch, behavior, damage
):
    prefix = mark_mpas_tools_installed(behavior)
    if damage == "marker":
        (prefix / ".installed").unlink()
    elif damage == "not_executable":
        (prefix / "bin" / "grid_rotate").chmod(0o644)
    else:
        (prefix / "bin" / damage).unlink()
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.install_mpas_tools()
    download.assert_called_once()
    build.assert_called_once()
    assert behavior.mpas_tools_built


async def test_install_mpas_tools_runs_the_download_on_service_and_build_on_compute(
    monkeypatch, behavior
):
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.install_mpas_tools()
    download.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "mpas_tools_download.stdout"),
        stderr=str(behavior.log_dir / "mpas_tools_download.stderr"),
    )
    build.assert_called_once_with(
        executor=["compute"],
        stdout=str(behavior.log_dir / "mpas_tools_build.stdout"),
        stderr=str(behavior.log_dir / "mpas_tools_build.stderr"),
        inputs=[download.return_value],
    )


async def test_concurrent_mpas_tools_installs_run_once(monkeypatch, behavior):
    # Keep the download unfinished while the other requests arrive
    pending, build_pending = Future(), Future()
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    download.return_value = pending
    build.side_effect = lambda **kwargs: build_pending
    requests = [asyncio.create_task(behavior.install_mpas_tools()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert download.call_count == 1
    pending.set_result(None)
    mark_mpas_tools_installed(behavior)
    build_pending.set_result(None)
    await asyncio.gather(*requests)
    assert download.call_count == 1
    assert build.call_count == 1


@pytest.mark.parametrize("stage", ["download", "build"])
async def test_failed_mpas_tools_install_is_reported_and_retried(
    monkeypatch, behavior, stage
):
    failure = BashExitFailure("task", 3)
    download, build = mock_mpas_tools_tasks(
        monkeypatch,
        behavior,
        download=failure if stage == "download" else None,
        build=failure if stage == "build" else None,
    )
    with pytest.raises(RuntimeError, match=f"MPAS-Tools {stage} failed") as info:
        await behavior.install_mpas_tools()
    assert f"mpas_tools_{stage}.stderr" in str(info.value)
    assert info.value.__cause__ is failure
    assert not behavior.mpas_tools_built

    download.return_value = done_future()
    build.side_effect = lambda **kwargs: (
        mark_mpas_tools_installed(behavior),
        done_future(),
    )[1]
    await behavior.install_mpas_tools()
    assert behavior.mpas_tools_built


async def test_mpas_tools_build_that_leaves_no_programs_is_an_error(
    monkeypatch, behavior
):
    monkeypatch.setattr(
        behavior, "_download_mpas_tools", Mock(return_value=done_future())
    )
    monkeypatch.setattr(behavior, "_build_mpas_tools", Mock(return_value=done_future()))
    with pytest.raises(RuntimeError, match="not usable"):
        await behavior.install_mpas_tools()
    assert not behavior.mpas_tools_built


@pytest.mark.parametrize(
    ("defined", "missing"),
    [(["compute"], "service"), (["service"], "compute"), ([], "service")],
)
async def test_install_mpas_tools_checks_resources_before_submitting(
    monkeypatch, behavior, defined, missing
):
    define_resources(monkeypatch, "_parsl_internal", *defined)
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.install_mpas_tools()
    download.assert_not_called()
    build.assert_not_called()


# Installing MPAS-Limited-Area


async def test_install_limited_area_reuses_a_finished_install(monkeypatch, behavior):
    mark_limited_area_installed(behavior)
    install, check = mock_limited_area_tasks(monkeypatch, behavior)
    await behavior.install_limited_area()
    assert behavior.limited_area_built
    install.assert_not_called()
    check.assert_not_called()


@pytest.mark.parametrize("damage", ["marker", "script", "old_venv"])
async def test_install_limited_area_replaces_a_damaged_install(
    monkeypatch, behavior, damage
):
    prefix = mark_limited_area_installed(behavior)
    if damage == "marker":
        (prefix / ".installed").unlink()
    elif damage == "script":
        (prefix / "src" / "create_region").unlink()
    else:
        # An install from when create_region ran in its own environment
        (prefix / "venv" / "bin").mkdir(parents=True)
    install, check = mock_limited_area_tasks(monkeypatch, behavior)
    await behavior.install_limited_area()
    install.assert_called_once()
    check.assert_called_once()
    assert behavior.limited_area_built


async def test_install_limited_area_runs_on_service_and_checks_on_compute(
    monkeypatch, behavior
):
    install, check = mock_limited_area_tasks(monkeypatch, behavior)
    await behavior.install_limited_area()
    install.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "limited_area_install.stdout"),
        stderr=str(behavior.log_dir / "limited_area_install.stderr"),
    )
    check.assert_called_once_with(
        executor=["compute"],
        stdout=str(behavior.log_dir / "limited_area_check.stdout"),
        stderr=str(behavior.log_dir / "limited_area_check.stderr"),
        inputs=[install.return_value],
    )


async def test_concurrent_limited_area_installs_run_once(monkeypatch, behavior):
    # Keep the install unfinished while the other requests arrive
    pending, check_pending = Future(), Future()
    install, check = mock_limited_area_tasks(monkeypatch, behavior)
    install.return_value = pending
    check.side_effect = lambda **kwargs: check_pending
    requests = [asyncio.create_task(behavior.install_limited_area()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert install.call_count == 1
    pending.set_result(None)
    mark_limited_area_installed(behavior)
    check_pending.set_result(None)
    await asyncio.gather(*requests)
    assert install.call_count == 1
    assert check.call_count == 1


@pytest.mark.parametrize("stage", ["install", "check"])
async def test_failed_limited_area_install_is_reported_and_retried(
    monkeypatch, behavior, stage
):
    failure = BashExitFailure("task", 3)
    install, check = mock_limited_area_tasks(
        monkeypatch,
        behavior,
        install=failure if stage == "install" else None,
        check=failure if stage == "check" else None,
    )
    with pytest.raises(
        RuntimeError, match=f"MPAS-Limited-Area {stage} failed \\(exit 3\\)"
    ) as info:
        await behavior.install_limited_area()
    assert f"limited_area_{stage}.stderr" in str(info.value)
    assert info.value.__cause__ is failure
    assert not behavior.limited_area_built

    install.return_value = done_future()
    check.side_effect = lambda **kwargs: (
        mark_limited_area_installed(behavior),
        done_future(),
    )[1]
    await behavior.install_limited_area()
    assert behavior.limited_area_built


async def test_limited_area_check_that_leaves_no_install_is_an_error(
    monkeypatch, behavior
):
    monkeypatch.setattr(
        behavior, "_install_limited_area", Mock(return_value=done_future())
    )
    monkeypatch.setattr(
        behavior, "_check_limited_area", Mock(return_value=done_future())
    )
    with pytest.raises(RuntimeError, match="is not usable"):
        await behavior.install_limited_area()
    assert not behavior.limited_area_built


@pytest.mark.parametrize(
    ("defined", "missing"),
    [(["compute"], "service"), (["service"], "compute"), ([], "service")],
)
async def test_install_limited_area_checks_resources_before_submitting(
    monkeypatch, behavior, defined, missing
):
    define_resources(monkeypatch, "_parsl_internal", *defined)
    install, check = mock_limited_area_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.install_limited_area()
    install.assert_not_called()
    check.assert_not_called()


async def test_install_limited_area_shares_the_lock_with_other_installs(
    monkeypatch, behavior
):
    # An MPAS-Tools install holding the lock makes the Limited-Area install wait
    pending = Future()
    tools_download, _ = mock_mpas_tools_tasks(monkeypatch, behavior)
    tools_download.return_value = pending
    install, _ = mock_limited_area_tasks(monkeypatch, behavior)
    tools = asyncio.create_task(behavior.install_mpas_tools())
    limited = asyncio.create_task(behavior.install_limited_area())
    for _ in range(10):
        await asyncio.sleep(0)
    install.assert_not_called()
    pending.set_result(None)
    await asyncio.gather(tools, limited)
    install.assert_called_once()


# Installing plotting


async def test_install_plotting_reuses_a_finished_install(monkeypatch, behavior):
    mark_plotting_installed(behavior)
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    await behavior.install_plotting()
    assert behavior.plotting_built
    install.assert_not_called()
    check.assert_not_called()


async def test_install_plotting_replaces_an_unfinished_install(monkeypatch, behavior):
    prefix = mark_plotting_installed(behavior)
    (prefix / ".installed").unlink()
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    await behavior.install_plotting()
    install.assert_called_once()
    check.assert_called_once()
    assert behavior.plotting_built


async def test_install_plotting_runs_on_service_and_checks_on_compute(
    monkeypatch, behavior
):
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    await behavior.install_plotting()
    install.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "plot_download.stdout"),
        stderr=str(behavior.log_dir / "plot_download.stderr"),
    )
    check.assert_called_once_with(
        executor=["compute"],
        stdout=str(behavior.log_dir / "plot_check.stdout"),
        stderr=str(behavior.log_dir / "plot_check.stderr"),
        inputs=[install.return_value],
    )


async def test_concurrent_plotting_installs_run_once(monkeypatch, behavior):
    pending, check_pending = Future(), Future()
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    install.return_value = pending
    check.side_effect = lambda **kwargs: check_pending
    requests = [asyncio.create_task(behavior.install_plotting()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert install.call_count == 1
    pending.set_result(None)
    mark_plotting_installed(behavior)
    check_pending.set_result(None)
    await asyncio.gather(*requests)
    assert install.call_count == 1
    assert check.call_count == 1


@pytest.mark.parametrize("stage", ["download", "check"])
async def test_failed_plotting_install_is_reported_and_retried(
    monkeypatch, behavior, stage
):
    failure = BashExitFailure("task", 3)
    install, check = mock_plotting_tasks(
        monkeypatch,
        behavior,
        download=failure if stage == "download" else None,
        check=failure if stage == "check" else None,
    )
    with pytest.raises(
        RuntimeError, match=f"Plotting {stage} failed \\(exit 3\\)"
    ) as info:
        await behavior.install_plotting()
    assert f"plot_{stage}.stderr" in str(info.value)
    assert info.value.__cause__ is failure
    assert not behavior.plotting_built

    install.return_value = done_future()
    check.side_effect = lambda **kwargs: (
        mark_plotting_installed(behavior),
        done_future(),
    )[1]
    await behavior.install_plotting()
    assert behavior.plotting_built


async def test_plotting_check_that_leaves_no_install_is_an_error(monkeypatch, behavior):
    monkeypatch.setattr(
        behavior, "_download_plotting", Mock(return_value=done_future())
    )
    monkeypatch.setattr(behavior, "_check_plotting", Mock(return_value=done_future()))
    with pytest.raises(RuntimeError, match="is not usable"):
        await behavior.install_plotting()
    assert not behavior.plotting_built


@pytest.mark.parametrize(
    ("defined", "missing"),
    [(["compute"], "service"), (["service"], "compute"), ([], "service")],
)
async def test_install_plotting_checks_resources_before_submitting(
    monkeypatch, behavior, defined, missing
):
    define_resources(monkeypatch, "_parsl_internal", *defined)
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.install_plotting()
    install.assert_not_called()
    check.assert_not_called()


async def test_install_plotting_shares_the_lock_with_other_installs(
    monkeypatch, behavior
):
    pending = Future()
    limited_install, _ = mock_limited_area_tasks(monkeypatch, behavior)
    limited_install.return_value = pending
    install, _ = mock_plotting_tasks(monkeypatch, behavior)
    limited = asyncio.create_task(behavior.install_limited_area())
    plotting = asyncio.create_task(behavior.install_plotting())
    for _ in range(10):
        await asyncio.sleep(0)
    install.assert_not_called()
    pending.set_result(None)
    await asyncio.gather(limited, plotting)
    install.assert_called_once()


@pytest.mark.parametrize(
    ("config", "plotting"),
    [
        ({"resolution": "480km", "plot": True}, True),
        ({"resolution": "480km", "plot": False}, False),
        ({"resolution": "480km"}, False),
        ({**regional_config(), "plot": True}, True),
    ],
)
async def test_ensure_tools_installed_installs_plotting_for_plots(
    monkeypatch, behavior, config, plotting
):
    mark_mpas_tools_installed(behavior)
    install, _ = mock_plotting_tasks(monkeypatch, behavior)
    await behavior.ensure_tools_installed(config)
    assert install.called == plotting


async def test_mesh_requests_leave_plotting_to_the_plot_step(monkeypatch, behavior):
    # generate_mesh installs tools with plotting=False, so a failed plotting
    # install cannot fail the request
    install, _ = mock_plotting_tasks(monkeypatch, behavior)
    await behavior._install_tools({"resolution": "480km", "plot": True}, plotting=False)
    install.assert_not_called()


# Plotting


def fake_plot_task(behavior, monkeypatch, code=0, writes=True):
    """Replace the plotting task with a fake that acts like the script.

    It writes a PNG to the path it is given unless ``writes`` is false, and
    exits with ``code``.
    """

    def run(mesh_path, output_path, options, **kwargs):
        assert Path(mesh_path).is_file()
        if writes and code != EXIT_VIEW_OF_REGIONAL_MESH:
            Path(output_path).write_bytes(b"\x89PNG new image")
        if code:
            return done_future(error=BashExitFailure("plot", code))
        return done_future()

    task = Mock(side_effect=run)
    monkeypatch.setattr(behavior, "_run_plot_mesh", task)
    return task


@pytest.fixture
def plotting(behavior, tmp_path):
    """Provide installed plotting and a mesh file to plot."""
    mark_plotting_installed(behavior)
    mesh = tmp_path / "meshes" / "cut" / "cut.static.nc"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"CDF mesh")
    return mesh


@pytest.mark.parametrize(
    ("name", "image"),
    [
        ("conus.static.nc", "conus.png"),
        ("x1.2562.static.nc", "x1.2562.png"),
        ("x1.2562.grid.nc", "x1.2562.grid.png"),
        ("mesh", "mesh.png"),
        # An image never replaces its mesh
        ("picture.png", "picture.png.png"),
    ],
)
def test_plot_path(name, image):
    assert _MeshBehavior._plot_path(Path("/m") / name) == Path("/m") / image


def test_plot_script_is_next_to_the_mesh_module(behavior):
    assert behavior._plot_script == Path(mesh_module.__file__).with_name(
        "_plot_mesh.py"
    )
    assert behavior._plot_script.is_file()


async def test_plot_mesh(monkeypatch, behavior, plotting):
    task = fake_plot_task(behavior, monkeypatch)
    image = await behavior.plot_mesh(str(plotting))
    assert image == str(plotting.with_name("cut.png"))
    assert Path(image).read_bytes() == b"\x89PNG new image"
    # Only the image is added, under its final name
    assert {path.name for path in plotting.parent.iterdir()} == {
        "cut.static.nc",
        "cut.png",
    }
    task.assert_called_once()
    mesh_arg, partial, options = task.call_args.args
    assert mesh_arg == str(plotting)
    assert Path(partial).parent == plotting.parent
    assert Path(partial).name.startswith(".cut.") and partial.endswith(".png")
    assert options == []
    assert task.call_args.kwargs["executor"] == ["compute"]
    assert task.call_args.kwargs["stderr"].startswith(
        str(behavior.log_dir / "plot_mesh.")
    )


async def test_plot_mesh_passes_the_view(monkeypatch, behavior, plotting):
    task = fake_plot_task(behavior, monkeypatch)
    await behavior.plot_mesh(
        str(plotting), title="A title", zoom=4, center_lat=50, center_lon=-10.5
    )
    assert task.call_args.args[2] == [
        "--title=A title",
        "--zoom=4.0",
        "--center-lat=50.0",
        "--center-lon=-10.5",
    ]


async def test_plot_mesh_passes_values_starting_with_a_dash(
    monkeypatch, behavior, plotting
):
    task = fake_plot_task(behavior, monkeypatch)
    await behavior.plot_mesh(
        str(plotting), title="-t --zoom 3", center_lat=-1e-05, center_lon=-0.0
    )
    options = task.call_args.args[2]
    assert options == [
        "--title=-t --zoom 3",
        "--center-lat=-1e-05",
        "--center-lon=-0.0",
    ]
    # As the script reads them
    args = plot_script.parse_args(["m.nc", "o.png", "--map-data", "/d", *options])
    assert (args.title, args.center_lat, args.center_lon) == ("-t --zoom 3", -1e-05, 0)


async def test_plot_mesh_makes_relative_paths_absolute(monkeypatch, behavior, plotting):
    fake_plot_task(behavior, monkeypatch)
    monkeypatch.chdir(plotting.parent)
    assert await behavior.plot_mesh("cut.static.nc") == str(
        plotting.with_name("cut.png")
    )


async def test_plot_mesh_replaces_an_earlier_image(monkeypatch, behavior, plotting):
    plotting.with_name("cut.png").write_bytes(b"old")
    fake_plot_task(behavior, monkeypatch)
    await behavior.plot_mesh(str(plotting))
    assert plotting.with_name("cut.png").read_bytes() == b"\x89PNG new image"


@pytest.mark.parametrize("code", [1, 139])
async def test_failed_plot_names_its_logs_and_keeps_an_earlier_image(
    monkeypatch, behavior, plotting, code
):
    plotting.with_name("cut.png").write_bytes(b"old")
    task = fake_plot_task(behavior, monkeypatch, code=code)
    with pytest.raises(
        RuntimeError, match=f"Plotting failed \\(exit {code}\\)"
    ) as info:
        await behavior.plot_mesh(str(plotting))
    assert task.call_args.kwargs["stdout"] in str(info.value)
    assert task.call_args.kwargs["stderr"] in str(info.value)
    # The partial image is removed, and the earlier one kept
    assert {path.name for path in plotting.parent.iterdir()} == {
        "cut.static.nc",
        "cut.png",
    }
    assert plotting.with_name("cut.png").read_bytes() == b"old"


async def test_unexpected_plot_error_becomes_a_runtime_error(
    monkeypatch, behavior, plotting
):
    failure = OSError("worker lost")
    monkeypatch.setattr(
        behavior, "_run_plot_mesh", Mock(return_value=done_future(error=failure))
    )
    with pytest.raises(RuntimeError, match="Plotting failed: worker lost") as info:
        await behavior.plot_mesh(str(plotting))
    assert info.value.__cause__ is failure


async def test_plot_without_some_map_layers_still_gives_an_image(
    monkeypatch, behavior, plotting
):
    warning = Mock()
    monkeypatch.setattr(mesh_module.logger, "warning", warning)
    task = fake_plot_task(behavior, monkeypatch, code=EXIT_MISSING_MAP)
    image = await behavior.plot_mesh(str(plotting))
    assert Path(image).read_bytes() == b"\x89PNG new image"
    warning.assert_called_once()
    assert "without some map layers" in warning.call_args.args[2]
    assert task.call_args.kwargs["stderr"] in warning.call_args.args[2]
    path, message = await behavior._plot(str(plotting))
    assert path == image
    assert "without some map layers" in message


async def test_view_of_a_regional_mesh_is_rejected(monkeypatch, behavior, plotting):
    plotting.with_name("cut.png").write_bytes(b"old")
    fake_plot_task(behavior, monkeypatch, code=EXIT_VIEW_OF_REGIONAL_MESH)
    with pytest.raises(ValueError, match="apply only to global meshes, and .* is"):
        await behavior.plot_mesh(str(plotting), zoom=2)
    assert plotting.with_name("cut.png").read_bytes() == b"old"
    assert len(list(plotting.parent.iterdir())) == 2


async def test_plot_that_writes_nothing_is_an_error(monkeypatch, behavior, plotting):
    task = fake_plot_task(behavior, monkeypatch, writes=False)
    with pytest.raises(RuntimeError, match="plot_mesh did not write") as info:
        await behavior.plot_mesh(str(plotting))
    assert task.call_args.kwargs["stdout"] in str(info.value)
    assert not plotting.with_name("cut.png").exists()


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"mesh_path": ""}, ValueError, "non-empty string"),
        ({"mesh_path": None}, ValueError, "non-empty string"),
        ({"zoom": 0}, ValueError, "zoom must be greater than 0"),
        ({"center_lat": 10}, ValueError, "must be given together"),
        ({"title": 5}, ValueError, "title must be a string"),
        ({"mesh_path": "missing.static.nc"}, FileNotFoundError, "No mesh file"),
    ],
)
async def test_plot_mesh_rejects_invalid_input_before_any_work(
    monkeypatch, tmp_path, behavior, plotting, arguments, error, message
):
    monkeypatch.chdir(tmp_path)
    install, _ = mock_plotting_tasks(monkeypatch, behavior)
    task = fake_plot_task(behavior, monkeypatch)
    arguments = {"mesh_path": str(plotting), **arguments}
    with pytest.raises(error, match=message):
        await behavior.plot_mesh(**arguments)
    task.assert_not_called()
    install.assert_not_called()


async def test_plot_mesh_installs_plotting_first(monkeypatch, tmp_path, behavior):
    mesh = tmp_path / "m.static.nc"
    mesh.write_bytes(b"CDF")
    install, check = mock_plotting_tasks(monkeypatch, behavior)
    fake_plot_task(behavior, monkeypatch)
    await behavior.plot_mesh(str(mesh))
    install.assert_called_once()
    check.assert_called_once()


async def test_plot_mesh_needs_compute(monkeypatch, behavior, plotting):
    define_resources(monkeypatch, "service")
    task = fake_plot_task(behavior, monkeypatch)
    with pytest.raises(RuntimeError, match="Plotting needs the resource 'compute'"):
        await behavior.plot_mesh(str(plotting))
    task.assert_not_called()


@pytest.mark.parametrize(
    "name", ["m$(touch {pwn})", "b`touch {pwn}`", 'q"$(touch {pwn})', "$HOME", "a b"]
)
def test_plot_script_runs_without_expansion(monkeypatch, tmp_path, behavior, name):
    pwned = tmp_path / "PWNED"
    directory = tmp_path / name.format(pwn=pwned)
    directory.mkdir(parents=True)
    record = tmp_path / "record"
    prefix = mark_plotting_installed(behavior)
    fake_python(
        monkeypatch,
        tmp_path,
        f'{{ echo "$MPLBACKEND $MPLCONFIGDIR"; for a in "$@"; do printf "%s\\n" "$a"; done; }}'
        f" > {shlex.quote(str(record))}",
    )
    mesh, output = str(directory / "m.static.nc"), str(directory / ".m.1.png")
    options = [f"--title={name}", "--zoom=2.0"]
    script = _MeshBehavior._run_plot_mesh.func(behavior, mesh, output, options)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert not pwned.exists()
    assert record.read_text().splitlines() == [
        f"Agg {prefix / 'matplotlib'}",
        str(behavior._plot_script),
        mesh,
        output,
        "--map-data",
        str(prefix / "natural_earth"),
        *options,
    ]


@pytest.mark.parametrize("code", [1, 2, EXIT_MISSING_MAP, EXIT_VIEW_OF_REGIONAL_MESH])
def test_plot_script_passes_the_exit_code_on(monkeypatch, tmp_path, behavior, code):
    mark_plotting_installed(behavior)
    fake_python(monkeypatch, tmp_path, f"exit {code}")
    script = _MeshBehavior._run_plot_mesh.func(
        behavior, str(tmp_path / "m.nc"), str(tmp_path / ".m.png"), []
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == code


# Plotting through generate_mesh


async def test_generate_mesh_with_a_plot(monkeypatch, mesh_download, behavior):
    output, _ = mesh_download
    mark_plotting_installed(behavior)
    task = fake_plot_task(behavior, monkeypatch)
    config = {"resolution": "480km", "name": "g", "plot": True}
    result = await behavior.generate_mesh(config, str(output))
    mesh_dir = output / "g"
    assert result == {
        "static": str(mesh_dir / "g.static.nc"),
        "graph": str(mesh_dir / "g.graph.info"),
        "plot": str(mesh_dir / "g.png"),
    }
    assert (mesh_dir / "g.png").read_bytes() == b"\x89PNG new image"
    # The mesh's own file is plotted, with the defaults
    mesh_arg, _, options = task.call_args.args
    assert mesh_arg == str(mesh_dir / "g.static.nc")
    assert options == []


async def test_plot_comes_after_the_partitions(monkeypatch, mesh_download, behavior):
    output, _ = mesh_download
    mark_metis_installed(behavior)
    mark_plotting_installed(behavior)
    order = []
    partition = fake_partition_task()
    partition_run = partition.side_effect
    partition.side_effect = lambda *a, **k: (
        order.append("partition"),
        partition_run(*a, **k),
    )[1]
    monkeypatch.setattr(behavior, "_partition_mesh", partition)
    plot = fake_plot_task(behavior, monkeypatch)
    plot_run = plot.side_effect
    plot.side_effect = lambda *a, **k: (order.append("plot"), plot_run(*a, **k))[1]
    result = await behavior.generate_mesh({**PARTITIONED, "plot": True}, str(output))
    assert order == ["partition", "partition", "plot"]
    assert set(result) == {"static", "graph", "partitions", "plot"}


@pytest.mark.parametrize("code", [1, 139])
async def test_failed_plot_does_not_fail_the_request(
    monkeypatch, mesh_download, behavior, code
):
    output, _ = mesh_download
    mark_plotting_installed(behavior)
    fake_plot_task(behavior, monkeypatch, code=code)
    config = {"resolution": "480km", "name": "g", "plot": True}
    result = await behavior.generate_mesh(config, str(output))
    assert set(result) == {"static", "graph", "plot_error"}
    assert f"Plotting failed (exit {code})" in result["plot_error"]
    # The mesh is kept, and no image
    assert {path.name for path in (output / "g").iterdir()} == {
        "g.static.nc",
        "g.graph.info",
        "g.mesh.json",
    }


async def test_plot_without_some_map_layers_reports_it(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    mark_plotting_installed(behavior)
    fake_plot_task(behavior, monkeypatch, code=EXIT_MISSING_MAP)
    config = {"resolution": "480km", "name": "g", "plot": True}
    result = await behavior.generate_mesh(config, str(output))
    assert result["plot"] == str(output / "g" / "g.png")
    assert "without some map layers" in result["plot_error"]


async def test_failed_plotting_install_does_not_fail_the_request(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    mock_plotting_tasks(monkeypatch, behavior, check=BashExitFailure("task", 1))
    task = fake_plot_task(behavior, monkeypatch)
    config = {"resolution": "480km", "name": "g", "plot": True}
    result = await behavior.generate_mesh(config, str(output))
    assert "Plotting check failed" in result["plot_error"]
    assert "plot" not in result
    assert (output / "g" / "g.static.nc").is_symlink()
    task.assert_not_called()


async def test_plot_without_compute_does_not_fail_a_global_request(
    monkeypatch, mesh_download, behavior
):
    # A global mesh without partitions needs only service, and a plot that
    # cannot run reports why
    output, _ = mesh_download
    mark_plotting_installed(behavior)
    define_resources(monkeypatch, "service")
    task = fake_plot_task(behavior, monkeypatch)
    config = {"resolution": "480km", "name": "g", "plot": True}
    result = await behavior.generate_mesh(config, str(output))
    assert "Plotting needs the resource 'compute'" in result["plot_error"]
    assert (output / "g" / "g.static.nc").is_symlink()
    task.assert_not_called()


async def test_failed_mesh_is_not_plotted(monkeypatch, mesh_download, behavior):
    output, _ = mesh_download
    mark_metis_installed(behavior)
    mark_plotting_installed(behavior)
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={8})
    )
    task = fake_plot_task(behavior, monkeypatch)
    with pytest.raises(RuntimeError, match="partition failed for 8 ranks"):
        await behavior.generate_mesh({**PARTITIONED, "plot": True}, str(output))
    task.assert_not_called()
    assert not (output / "forecast").exists()


async def test_regional_meshes_are_plotted(monkeypatch, regional, behavior, cutting):
    mark_plotting_installed(behavior)
    task = fake_plot_task(behavior, monkeypatch)
    hexes = await behavior.generate_mesh(
        {**regional_config(), "plot": True}, str(regional.output)
    )
    assert hexes["plot"] == str(regional.output / "region" / "region.png")
    cut = await behavior.generate_mesh(
        {**create_region_config(), "plot": True}, str(cutting.output)
    )
    assert cut["plot"] == str(cutting.output / "cut" / "cut.png")
    assert [call.args[0] for call in task.call_args_list] == [
        hexes["static"],
        cut["static"],
    ]


# Generating project_hexes meshes

PROJECT_HEXES_NAMELIST = """\
&mesh
  cell_spacing_km = 15.0
  mesh_length_x_km = 500.0
  mesh_length_y_km = 400.0
  earth_radius_km = 6371.229
/
&projection
  projection_type = "lambert_conformal"
/
&lambert_conformal
  reference_longitude_degrees = -105.0
  standard_longitude_degrees = -105.0
  reference_latitude_degrees = 40.0
  standard_parallel_1_degrees = 40.0
  standard_parallel_2_degrees = 40.0
/
"""

ROTATE_NAMELIST = """\
&input
   config_original_latitude_degrees = 40.0
   config_original_longitude_degrees = -105.0

   config_new_latitude_degrees = 40.0
   config_new_longitude_degrees = -105.0
   config_birdseye_rotation_counter_clockwise_degrees = -30.5
/
"""


def test_project_hexes_namelist():
    assert (
        _MeshBehavior._project_hexes_namelist(15.0, 500, 400, 40, -105)
        == PROJECT_HEXES_NAMELIST
    )


def test_project_hexes_namelist_on_the_equator():
    # project_hexes would divide by the sine of a parallel on the equator
    text = _MeshBehavior._project_hexes_namelist(15.0, 500, 400, 0, -105)
    assert "reference_latitude_degrees = 0.0\n" in text
    assert "standard_parallel_1_degrees = 0.0001\n" in text
    assert "standard_parallel_2_degrees = 0.0001\n" in text
    # So does a center nearer the equator than that, on its own side
    text = _MeshBehavior._project_hexes_namelist(15.0, 500, 400, -1e-6, -105)
    assert "standard_parallel_1_degrees = -0.0001\n" in text
    # A center that far from it keeps its own parallels
    text = _MeshBehavior._project_hexes_namelist(15.0, 500, 400, -0.0002, -105)
    assert "standard_parallel_1_degrees = -0.0002\n" in text


def test_rotate_namelist():
    assert _MeshBehavior._rotate_namelist(40, -105, -30.5) == ROTATE_NAMELIST


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_namelists_reject_non_finite_numbers(value):
    with pytest.raises(ValueError, match="Cannot write"):
        _MeshBehavior._project_hexes_namelist(15.0, value, 400, 40, -105)
    with pytest.raises(ValueError, match="Cannot write"):
        _MeshBehavior._rotate_namelist(40, -105, value)


def test_namelist_numbers_keep_their_precision():
    text = _MeshBehavior._project_hexes_namelist(0.0001, 1e22, 0.125, -33.3, 270.0)
    assert "cell_spacing_km = 0.0001\n" in text
    assert "mesh_length_x_km = 1e+22\n" in text
    assert "reference_latitude_degrees = -33.3\n" in text


@pytest.mark.parametrize(
    "name", ["m$(touch {pwn})", "b`touch {pwn}`", 'q"$(touch {pwn})', "$HOME", "a b"]
)
def test_program_scripts_run_in_the_mesh_directory_without_expansion(
    tmp_path, behavior, name
):
    pwned = tmp_path / "PWNED"
    mesh_dir = tmp_path / name.format(pwn=pwned)
    mesh_dir.mkdir(parents=True)
    record = tmp_path / "record"
    prefix = mark_mpas_tools_installed(behavior)
    for program in ("project_hexes", "grid_rotate"):
        fake = prefix / "bin" / program
        fake.write_text(
            "#!/bin/sh\n"
            f'{{ pwd; for a in "$@"; do printf "%s\\n" "$a"; done; }} '
            f">> {shlex.quote(str(record))}\n"
        )
    inputs = [str(mesh_dir / "in.nc"), str(mesh_dir / "out.nc")]
    scripts = [
        _MeshBehavior._run_project_hexes.func(behavior, str(mesh_dir)),
        _MeshBehavior._run_grid_rotate.func(behavior, str(mesh_dir), *inputs),
    ]
    for script in scripts:
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    assert not pwned.exists()
    assert record.read_text().splitlines() == [str(mesh_dir)] + [str(mesh_dir)] + inputs


def fake_program_tasks(behavior, monkeypatch, fail=None):
    """Replace the project_hexes and grid_rotate tasks with fakes.

    ``fail`` maps a program name to the exception its task fails with. Returns
    the two mocks, which record what the fakes were called with.
    """
    fail = fail or {}

    def project_hexes(mesh_dir, **kwargs):
        if "project_hexes" in fail:
            return done_future(error=fail["project_hexes"])
        directory = Path(mesh_dir)
        assert (directory / "namelist.projections").is_file()
        (directory / "mpas_hex_mesh.nc").write_text("mesh")
        (directory / "graph.info").write_text("graph")
        return done_future()

    def grid_rotate(mesh_dir, input_file, output_file, **kwargs):
        if "grid_rotate" in fail:
            return done_future(error=fail["grid_rotate"])
        assert (Path(mesh_dir) / "namelist.input").is_file()
        Path(output_file).write_text("rotated " + Path(input_file).read_text())
        return done_future()

    hexes, rotate = Mock(side_effect=project_hexes), Mock(side_effect=grid_rotate)
    monkeypatch.setattr(behavior, "_run_project_hexes", hexes)
    monkeypatch.setattr(behavior, "_run_grid_rotate", rotate)
    return hexes, rotate


@pytest.fixture
def mesh_dir(tmp_path, behavior):
    """Provide a new mesh directory and an installed set of MPAS-Tools."""
    mark_mpas_tools_installed(behavior)
    directory = tmp_path / "meshes" / "region"
    directory.mkdir(parents=True)
    return directory


async def test_project_hexes_mesh_without_rotation(monkeypatch, behavior, mesh_dir):
    hexes, rotate = fake_program_tasks(behavior, monkeypatch)
    result = await behavior._generate_project_hexes_mesh(
        mesh_dir, "region", "15km", HEXES
    )
    assert result == {
        "static": str(mesh_dir / "region.static.nc"),
        "graph": str(mesh_dir / "region.graph.info"),
    }
    assert (mesh_dir / "region.static.nc").read_text() == "mesh"
    assert (mesh_dir / "region.graph.info").read_text() == "graph"
    # The inputs stay, and the intermediate names are gone
    assert (mesh_dir / "namelist.projections").read_text() == PROJECT_HEXES_NAMELIST
    assert {path.name for path in mesh_dir.iterdir()} == {
        "namelist.projections",
        "region.static.nc",
        "region.graph.info",
    }
    rotate.assert_not_called()
    hexes.assert_called_once()
    assert hexes.call_args.args == (str(mesh_dir),)
    assert hexes.call_args.kwargs["executor"] == ["compute"]
    assert hexes.call_args.kwargs["stderr"].startswith(
        str(behavior.log_dir / "project_hexes.")
    )


async def test_project_hexes_mesh_with_rotation(monkeypatch, behavior, mesh_dir):
    hexes, rotate = fake_program_tasks(behavior, monkeypatch)
    result = await behavior._generate_project_hexes_mesh(
        mesh_dir, "region", "15km", {**HEXES, "rotation_degrees": -30.5}
    )
    assert (mesh_dir / "region.static.nc").read_text() == "rotated mesh"
    assert (mesh_dir / "region.graph.info").read_text() == "graph"
    assert result["static"] == str(mesh_dir / "region.static.nc")
    assert (mesh_dir / "namelist.input").read_text() == ROTATE_NAMELIST
    assert {path.name for path in mesh_dir.iterdir()} == {
        "namelist.projections",
        "namelist.input",
        "region.static.nc",
        "region.graph.info",
    }
    rotate.assert_called_once()
    assert rotate.call_args.args == (
        str(mesh_dir),
        str(mesh_dir / "mpas_hex_mesh.nc"),
        str(mesh_dir / "rotated_mesh.nc"),
    )
    assert rotate.call_args.kwargs["executor"] == ["compute"]


async def test_zero_rotation_does_not_run_grid_rotate(monkeypatch, behavior, mesh_dir):
    _, rotate = fake_program_tasks(behavior, monkeypatch)
    await behavior._generate_project_hexes_mesh(
        mesh_dir, "region", "15km", {**HEXES, "rotation_degrees": 0}
    )
    rotate.assert_not_called()


@pytest.mark.parametrize("program", ["project_hexes", "grid_rotate"])
async def test_program_failure_names_its_log(monkeypatch, behavior, mesh_dir, program):
    failure = BashExitFailure("task", 2)
    hexes, rotate = fake_program_tasks(behavior, monkeypatch, {program: failure})
    with pytest.raises(RuntimeError, match=f"{program} failed \\(exit 2\\)") as info:
        await behavior._generate_project_hexes_mesh(
            mesh_dir, "region", "15km", {**HEXES, "rotation_degrees": 10}
        )
    task = hexes if program == "project_hexes" else rotate
    assert str(task.call_args.kwargs["stderr"]) in str(info.value)
    assert info.value.__cause__ is failure
    assert not (mesh_dir / "region.static.nc").exists()


async def test_unexpected_program_error_becomes_a_runtime_error(
    monkeypatch, behavior, mesh_dir
):
    failure = OSError("worker lost")
    fake_program_tasks(behavior, monkeypatch, {"project_hexes": failure})
    with pytest.raises(RuntimeError, match="project_hexes failed: worker lost") as info:
        await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", HEXES)
    assert info.value.__cause__ is failure


@pytest.mark.parametrize("missing", ["mpas_hex_mesh.nc", "graph.info"])
@pytest.mark.parametrize("empty", [False, True])
async def test_missing_program_output_is_an_error(
    monkeypatch, behavior, mesh_dir, missing, empty
):
    hexes, _ = fake_program_tasks(behavior, monkeypatch)
    write = hexes.side_effect

    def damaged(directory, **kwargs):
        future = write(directory, **kwargs)
        if empty:
            (Path(directory) / missing).write_text("")
        else:
            (Path(directory) / missing).unlink()
        return future

    hexes.side_effect = damaged
    with pytest.raises(RuntimeError, match=f"did not write {missing}"):
        await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", HEXES)


async def test_missing_project_hexes_output_blames_it_and_skips_the_rotation(
    monkeypatch, behavior, mesh_dir
):
    hexes, rotate = fake_program_tasks(behavior, monkeypatch)
    hexes.side_effect = lambda directory, **kwargs: done_future()
    config = {**HEXES, "rotation_degrees": 30}
    with pytest.raises(RuntimeError, match="project_hexes did not write") as info:
        await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", config)
    assert str(hexes.call_args.kwargs["stderr"]) in str(info.value)
    rotate.assert_not_called()


@pytest.mark.parametrize("empty", [False, True])
async def test_missing_grid_rotate_output_blames_it(
    monkeypatch, behavior, mesh_dir, empty
):
    _, rotate = fake_program_tasks(behavior, monkeypatch)

    def writes_nothing(mesh_dir, input_file, output_file, **kwargs):
        if empty:
            Path(output_file).write_text("")
        return done_future()

    rotate.side_effect = writes_nothing
    config = {**HEXES, "rotation_degrees": 30}
    with pytest.raises(
        RuntimeError, match="grid_rotate did not write rotated_mesh.nc"
    ) as info:
        await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", config)
    assert str(rotate.call_args.kwargs["stderr"]) in str(info.value)


async def test_project_hexes_mesh_installs_the_programs_first(
    monkeypatch, tmp_path, behavior
):
    mesh_dir = tmp_path / "region"
    mesh_dir.mkdir()
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    fake_program_tasks(behavior, monkeypatch)
    await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", HEXES)
    download.assert_called_once()
    build.assert_called_once()


async def test_each_program_run_has_its_own_logs(monkeypatch, behavior, mesh_dir):
    hexes, _ = fake_program_tasks(behavior, monkeypatch)
    second = mesh_dir.parent / "second"
    second.mkdir()
    await behavior._generate_project_hexes_mesh(mesh_dir, "region", "15km", HEXES)
    await behavior._generate_project_hexes_mesh(second, "second", "15km", HEXES)
    logs = [call.kwargs["stderr"] for call in hexes.call_args_list]
    assert len(set(logs)) == 2


# Cutting create_region meshes

POINTS = {
    "circle": "Name: region\nType: circle\nPoint: 39.0, -105.5\nRadius: 1500000.0\n",
    "ellipse": (
        "Name: region\nType: ellipse\nPoint: 39.0, -105.5\n"
        "Semi-major-axis: 2000000.0\nSemi-minor-axis: 1000000.0\n"
        "Orientation-angle: 30.0\n"
    ),
    "polygon": (
        "Name: region\nType: custom\nPoint: 39.0, -105.0\n"
        "30.0, -120.0\n30.0, -90.0\n48.0, -90.0\n48.0, -120.0\n"
    ),
    "channel": "Name: region\nType: channel\nUpper-lat: 23.5\nLower-lat: -23.5\n",
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_create_region_points(shape):
    config = create_region_config(shape)["regional"]["create_region"]
    assert _MeshBehavior._create_region_points(config) == POINTS[shape]


def test_create_region_points_default_orientation_and_precision():
    block = {**SHAPES["ellipse"], "semi_major_axis_km": 0.0015}
    del block["orientation_degrees"]
    text = _MeshBehavior._create_region_points({"ellipse": block})
    assert "Orientation-angle: 0.0\n" in text
    assert "Semi-major-axis: 1.5\n" in text
    tuples = {**SHAPES["polygon"], "vertices": ((1, 2), (3.25, -4), (5, 359.5))}
    text = _MeshBehavior._create_region_points({"polygon": tuples})
    assert text.endswith("1.0, 2.0\n3.25, -4.0\n5.0, 359.5\n")


@pytest.mark.parametrize(
    "name", ["m$(touch {pwn})", "b`touch {pwn}`", 'q"$(touch {pwn})', "$HOME", "a b"]
)
def test_create_region_script_runs_in_the_mesh_directory_without_expansion(
    monkeypatch, tmp_path, behavior, name
):
    pwned = tmp_path / "PWNED"
    mesh_dir = tmp_path / name.format(pwn=pwned)
    mesh_dir.mkdir(parents=True)
    record = tmp_path / "record"
    prefix = mark_limited_area_installed(behavior)
    fake_python(
        monkeypatch,
        tmp_path,
        f'{{ pwd; for a in "$@"; do printf "%s\\n" "$a"; done; }} '
        f"> {shlex.quote(str(record))}",
    )
    script = _MeshBehavior._run_create_region.func(behavior, str(mesh_dir))
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert not pwned.exists()
    assert record.read_text().splitlines() == [
        str(mesh_dir),
        str(prefix / "src" / "create_region"),
        "region.pts",
        "parent.static.nc",
    ]


def fake_create_region_task(behavior, monkeypatch, fail=None, writes=None):
    """Replace the create_region task with a fake that writes the region files.

    The fake checks that the points file and the parent link are in place while
    it runs. ``fail`` is an exception for it to fail with, and ``writes`` the
    file names it writes, by default both outputs.
    """
    seen = {}

    def run(mesh_dir, **kwargs):
        directory = Path(mesh_dir)
        seen["points"] = (directory / "region.pts").read_text()
        link = directory / "parent.static.nc"
        seen["parent"] = link.resolve() if link.is_symlink() else None
        if fail is not None:
            (directory / "region.static.nc").write_text("partial")
            return done_future(error=fail)
        for name in (
            writes if writes is not None else ("region.static.nc", "region.graph.info")
        ):
            (directory / name).write_text(f"cut {name}")
        return done_future()

    task = Mock(side_effect=run)
    monkeypatch.setattr(behavior, "_run_create_region", task)
    return task, seen


@pytest.fixture
def region(monkeypatch, tmp_path, behavior, mesh_download):
    """Provide a new mesh directory, cached global meshes, and installed tools."""
    output, download = mesh_download
    mark_limited_area_installed(behavior)
    directory = output / "cut"
    directory.mkdir()
    return SimpleNamespace(output=output, mesh_dir=directory, download=download)


async def test_create_region_mesh(monkeypatch, behavior, region):
    task, seen = fake_create_region_task(behavior, monkeypatch)
    config = create_region_config("ellipse")["regional"]["create_region"]
    result = await behavior._generate_create_region_mesh(
        region.mesh_dir, "cut", "480km", config, region.output
    )
    assert result == {
        "static": str(region.mesh_dir / "cut.static.nc"),
        "graph": str(region.mesh_dir / "cut.graph.info"),
    }
    assert (region.mesh_dir / "cut.static.nc").read_text() == "cut region.static.nc"
    assert (region.mesh_dir / "cut.graph.info").read_text() == "cut region.graph.info"
    # The tool saw the points and the shared parent mesh, and only the points
    # file stays
    assert seen["points"] == POINTS["ellipse"]
    assert seen["parent"] == (region.output / "x1.2562.static.nc").resolve()
    assert {path.name for path in region.mesh_dir.iterdir()} == {
        "region.pts",
        "cut.static.nc",
        "cut.graph.info",
    }
    # The parent mesh is the global mesh of the requested resolution
    region.download.assert_called_once()
    assert region.download.call_args.args == ("480km", str(region.output))
    assert (region.output / "x1.2562.static.nc").read_bytes().startswith(b"CDF")
    task.assert_called_once()
    assert task.call_args.args == (str(region.mesh_dir),)
    assert task.call_args.kwargs["executor"] == ["compute"]
    assert task.call_args.kwargs["stderr"].startswith(
        str(behavior.log_dir / "create_region.")
    )


@pytest.mark.parametrize("name", ["parent", "region"])
async def test_create_region_mesh_named_like_its_working_files(
    monkeypatch, behavior, region, name
):
    fake_create_region_task(behavior, monkeypatch)
    config = create_region_config()["regional"]["create_region"]
    result = await behavior._generate_create_region_mesh(
        region.mesh_dir, name, "480km", config, region.output
    )
    assert Path(result["static"]).read_text() == "cut region.static.nc"
    assert Path(result["graph"]).read_text() == "cut region.graph.info"
    assert {path.name for path in region.mesh_dir.iterdir()} == {
        "region.pts",
        f"{name}.static.nc",
        f"{name}.graph.info",
    }


async def test_create_region_failure_names_its_log_and_removes_the_link(
    monkeypatch, behavior, region
):
    failure = BashExitFailure("task", 255)
    task, _ = fake_create_region_task(behavior, monkeypatch, fail=failure)
    config = create_region_config()["regional"]["create_region"]
    with pytest.raises(
        RuntimeError, match="create_region failed \\(exit 255\\)"
    ) as info:
        await behavior._generate_create_region_mesh(
            region.mesh_dir, "cut", "480km", config, region.output
        )
    assert str(task.call_args.kwargs["stderr"]) in str(info.value)
    assert str(task.call_args.kwargs["stdout"]) in str(info.value)
    assert info.value.__cause__ is failure
    assert not (region.mesh_dir / "parent.static.nc").is_symlink()
    assert (region.output / "x1.2562.static.nc").is_file()


async def test_unexpected_create_region_error_becomes_a_runtime_error(
    monkeypatch, behavior, region
):
    failure = OSError("worker lost")
    fake_create_region_task(behavior, monkeypatch, fail=failure)
    config = create_region_config()["regional"]["create_region"]
    with pytest.raises(RuntimeError, match="create_region failed: worker lost"):
        await behavior._generate_create_region_mesh(
            region.mesh_dir, "cut", "480km", config, region.output
        )


@pytest.mark.parametrize(
    ("writes", "missing"),
    [
        ((), "region.static.nc"),
        (("region.static.nc",), "region.graph.info"),
        (("region.graph.info",), "region.static.nc"),
    ],
)
async def test_missing_create_region_output_is_an_error(
    monkeypatch, behavior, region, writes, missing
):
    task, _ = fake_create_region_task(behavior, monkeypatch, writes=writes)
    config = create_region_config()["regional"]["create_region"]
    with pytest.raises(RuntimeError, match=f"create_region did not write {missing}"):
        await behavior._generate_create_region_mesh(
            region.mesh_dir, "cut", "480km", config, region.output
        )


async def test_empty_create_region_output_is_an_error(monkeypatch, behavior, region):
    task, _ = fake_create_region_task(behavior, monkeypatch)
    write = task.side_effect

    def empty(directory, **kwargs):
        future = write(directory, **kwargs)
        (Path(directory) / "region.graph.info").write_text("")
        return future

    task.side_effect = empty
    config = create_region_config()["regional"]["create_region"]
    with pytest.raises(RuntimeError, match="did not write region.graph.info"):
        await behavior._generate_create_region_mesh(
            region.mesh_dir, "cut", "480km", config, region.output
        )


async def test_create_region_mesh_installs_and_downloads_first(
    monkeypatch, tmp_path, behavior
):
    output = tmp_path / "meshes"
    mesh_dir = output / "cut"
    mesh_dir.mkdir(parents=True)
    install, check = mock_limited_area_tasks(monkeypatch, behavior)

    def download(resolution, directory, **kwargs):
        for name, content in mesh_files().items():
            (Path(directory) / name).write_bytes(content)
        return done_future()

    monkeypatch.setattr(behavior, "_download_global_mesh", Mock(side_effect=download))
    fake_create_region_task(behavior, monkeypatch)
    config = create_region_config()["regional"]["create_region"]
    await behavior._generate_create_region_mesh(
        mesh_dir, "cut", "480km", config, output
    )
    install.assert_called_once()
    check.assert_called_once()
    assert (output / "x1.2562.static.nc").is_file()


async def test_each_create_region_run_has_its_own_logs(monkeypatch, behavior, region):
    task, _ = fake_create_region_task(behavior, monkeypatch)
    second = region.output / "second"
    second.mkdir()
    config = create_region_config()["regional"]["create_region"]
    for directory in (region.mesh_dir, second):
        await behavior._generate_create_region_mesh(
            directory, directory.name, "480km", config, region.output
        )
    assert len({call.kwargs["stderr"] for call in task.call_args_list}) == 2


async def test_downloads_of_the_same_mesh_do_not_overlap(
    monkeypatch, tmp_path, behavior
):
    pending = Future()
    download = Mock(side_effect=[pending, done_future(), done_future()])
    monkeypatch.setattr(behavior, "_download_global_mesh", download)
    output = tmp_path / "meshes"
    first = asyncio.create_task(behavior.download_global_mesh("480km", str(output)))
    second = asyncio.create_task(behavior.download_global_mesh("480km", str(output)))
    # Another directory, or another mesh, does not wait
    other = asyncio.create_task(
        behavior.download_global_mesh("480km", str(tmp_path / "other"))
    )
    for _ in range(10):
        await asyncio.sleep(0)
    assert download.call_count == 2
    assert {call.args[1] for call in download.call_args_list} == {
        str(output),
        str(tmp_path / "other"),
    }
    pending.set_result(None)
    await asyncio.gather(first, second, other)
    assert download.call_count == 3


async def test_failed_global_download_leaves_no_create_region_mesh(
    monkeypatch, tmp_path, behavior
):
    mark_limited_area_installed(behavior)
    failure = URLError("unreachable")
    monkeypatch.setattr(
        behavior, "_download_global_mesh", Mock(return_value=done_future(error=failure))
    )
    task, _ = fake_create_region_task(behavior, monkeypatch)
    output = tmp_path / "meshes"
    with pytest.raises(RuntimeError, match="Global mesh download failed"):
        await behavior.generate_mesh(create_region_config(), str(output))
    task.assert_not_called()
    assert list(output.iterdir()) == []


# Generating create_region meshes through generate_mesh


@pytest.fixture
def cutting(monkeypatch, behavior, mesh_download):
    """Provide cached global meshes, installed tools, and a fake create_region."""
    output, download = mesh_download
    mark_limited_area_installed(behavior)
    task, seen = fake_create_region_task(behavior, monkeypatch)
    return SimpleNamespace(output=output, download=download, task=task, seen=seen)


@pytest.mark.parametrize("shape", list(SHAPES))
async def test_generate_create_region_mesh(cutting, behavior, shape):
    result = await behavior.generate_mesh(
        create_region_config(shape), str(cutting.output)
    )
    mesh_dir = cutting.output / "cut"
    assert result == {
        "static": str(mesh_dir / "cut.static.nc"),
        "graph": str(mesh_dir / "cut.graph.info"),
    }
    assert cutting.seen["points"] == POINTS[shape]
    assert {path.name for path in mesh_dir.iterdir()} == {
        "region.pts",
        "cut.static.nc",
        "cut.graph.info",
        "cut.mesh.json",
    }
    # The global mesh it was cut from stays shared in mesh_data_dir
    cutting.download.assert_called_once()
    assert cutting.download.call_args.args[0] == "480km"
    assert {path.name for path in cutting.output.iterdir()} == {"cut"} | set(
        mesh_files()
    )


async def test_generate_create_region_mesh_with_default_name(cutting, behavior):
    config = create_region_config()
    del config["name"]
    result = await behavior.generate_mesh(config, str(cutting.output))
    assert result["static"] == str(
        cutting.output / "mesh_480km" / "mesh_480km.static.nc"
    )


async def test_generate_create_region_mesh_with_partitions(
    monkeypatch, cutting, behavior
):
    mark_metis_installed(behavior)
    task = fake_partition_task()
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    config = {**create_region_config("polygon"), "init_ranks": 4, "forecast_ranks": 8}
    result = await behavior.generate_mesh(config, str(cutting.output))
    graph = result["graph"]
    assert result["partitions"] == {4: f"{graph}.part.4", 8: f"{graph}.part.8"}
    assert Path(f"{graph}.part.4").is_file()
    assert {call.args[0] for call in task.call_args_list} == {graph}


async def test_failed_create_region_mesh_is_removed_and_can_be_retried(
    monkeypatch, cutting, behavior
):
    fake_create_region_task(behavior, monkeypatch, fail=BashExitFailure("task", 255))
    with pytest.raises(RuntimeError, match="create_region failed"):
        await behavior.generate_mesh(create_region_config(), str(cutting.output))
    # Nothing is left of the mesh, and the shared global mesh is untouched
    assert {path.name for path in cutting.output.iterdir()} == set(mesh_files())
    assert all((cutting.output / name).stat().st_size > 0 for name in mesh_files())

    fake_create_region_task(behavior, monkeypatch)
    result = await behavior.generate_mesh(create_region_config(), str(cutting.output))
    assert Path(result["static"]).is_file()


async def test_failed_create_region_partition_removes_the_whole_mesh(
    monkeypatch, cutting, behavior
):
    mark_metis_installed(behavior)
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={8})
    )
    config = {**create_region_config(), "init_ranks": 4, "forecast_ranks": 8}
    with pytest.raises(RuntimeError, match="partition failed for 8 ranks"):
        await behavior.generate_mesh(config, str(cutting.output))
    assert {path.name for path in cutting.output.iterdir()} == set(mesh_files())


async def test_existing_create_region_mesh_name_fails_before_anything_runs(
    cutting, behavior
):
    (cutting.output / "cut").mkdir()
    with pytest.raises(FileExistsError):
        await behavior.generate_mesh(create_region_config(), str(cutting.output))
    cutting.task.assert_not_called()
    cutting.download.assert_not_called()
    assert list((cutting.output / "cut").iterdir()) == []


async def test_create_region_mesh_installs_before_downloading(
    monkeypatch, behavior, mesh_download
):
    output, download = mesh_download
    order = []
    install, _ = mock_limited_area_tasks(monkeypatch, behavior)
    install.side_effect = lambda **kwargs: (order.append("install"), done_future())[1]
    download.side_effect = lambda *a, **k: (order.append("download"), done_future())[1]
    fake_create_region_task(behavior, monkeypatch)
    await behavior.generate_mesh(create_region_config(), str(output))
    assert order == ["install", "download"]


async def test_failed_install_leaves_no_create_region_mesh(
    monkeypatch, behavior, mesh_download
):
    output, download = mesh_download
    mock_limited_area_tasks(monkeypatch, behavior, check=BashExitFailure("task", 1))
    task, _ = fake_create_region_task(behavior, monkeypatch)
    with pytest.raises(RuntimeError, match="MPAS-Limited-Area check failed"):
        await behavior.generate_mesh(create_region_config(), str(output))
    task.assert_not_called()
    download.assert_not_called()
    assert {path.name for path in output.iterdir()} == set(mesh_files())


@pytest.mark.parametrize(
    ("config", "defined", "missing"),
    [
        (create_region_config(), ["compute"], "service"),
        (create_region_config(), ["service"], "compute"),
        ({**create_region_config(), "forecast_ranks": 4}, ["service"], "compute"),
    ],
)
async def test_create_region_mesh_checks_resources_before_any_work(
    monkeypatch, cutting, behavior, config, defined, missing
):
    define_resources(monkeypatch, *defined)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.generate_mesh(config, str(cutting.output))
    cutting.task.assert_not_called()
    cutting.download.assert_not_called()
    assert not (cutting.output / "cut").exists()


async def test_installed_create_region_mesh_needs_service_and_compute(
    monkeypatch, cutting, behavior
):
    define_resources(monkeypatch, "service", "compute")
    result = await behavior.generate_mesh(create_region_config(), str(cutting.output))
    assert Path(result["static"]).is_file()


# Generating regional meshes through generate_mesh


@pytest.fixture
def regional(monkeypatch, tmp_path, behavior):
    """Provide an output directory, installed tools, and fake programs."""
    mark_mpas_tools_installed(behavior)
    hexes, rotate = fake_program_tasks(behavior, monkeypatch)
    download = Mock(return_value=done_future())
    monkeypatch.setattr(behavior, "_download_global_mesh", download)
    return SimpleNamespace(
        output=tmp_path / "meshes", hexes=hexes, rotate=rotate, download=download
    )


async def test_generate_regional_mesh(regional, behavior):
    result = await behavior.generate_mesh(regional_config(), str(regional.output))
    mesh_dir = regional.output / "region"
    assert result == {
        "static": str(mesh_dir / "region.static.nc"),
        "graph": str(mesh_dir / "region.graph.info"),
    }
    assert (mesh_dir / "region.static.nc").read_text() == "mesh"
    # No global mesh is downloaded for a regional one
    regional.download.assert_not_called()
    assert {path.name for path in regional.output.iterdir()} == {"region"}
    regional.rotate.assert_not_called()


@pytest.mark.parametrize("resolution", ["2km", "7.5km", "0.5km", "480km"])
async def test_generate_regional_mesh_uses_the_requested_resolution(
    regional, behavior, resolution
):
    config = regional_config()
    config["resolution"] = resolution
    await behavior.generate_mesh(config, str(regional.output))
    text = (regional.output / "region" / "namelist.projections").read_text()
    assert f"cell_spacing_km = {float(resolution[:-2])!r}\n" in text


async def test_generate_regional_mesh_with_rotation(regional, behavior):
    config = regional_config(rotation_degrees=45)
    result = await behavior.generate_mesh(config, str(regional.output))
    assert (regional.output / "region" / "region.static.nc").read_text() == (
        "rotated mesh"
    )
    assert result["static"].endswith("region.static.nc")
    regional.rotate.assert_called_once()


async def test_generate_regional_mesh_with_default_name(regional, behavior):
    config = regional_config()
    del config["name"]
    result = await behavior.generate_mesh(config, str(regional.output))
    assert result["static"] == str(
        regional.output / "mesh_15km" / "mesh_15km.static.nc"
    )


async def test_generate_regional_mesh_with_partitions(monkeypatch, regional, behavior):
    mark_metis_installed(behavior)
    task = fake_partition_task()
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    config = {**regional_config(), "init_ranks": 4, "forecast_ranks": 8}
    result = await behavior.generate_mesh(config, str(regional.output))
    graph = result["graph"]
    assert result["partitions"] == {4: f"{graph}.part.4", 8: f"{graph}.part.8"}
    # The partitions are written next to the graph, in the mesh directory
    assert Path(f"{graph}.part.8").is_file()
    assert {call.args[0] for call in task.call_args_list} == {graph}


async def test_none_regional_block_means_a_global_mesh(
    monkeypatch, mesh_download, behavior
):
    output, download = mesh_download
    result = await behavior.generate_mesh(
        {"resolution": "480km", "name": "global", "regional": None}, str(output)
    )
    download.assert_called_once()
    assert result["static"] == str(output / "global" / "global.static.nc")


@pytest.mark.parametrize("fail", ["project_hexes", "grid_rotate"])
async def test_failed_regional_mesh_is_removed_and_can_be_retried(
    monkeypatch, regional, behavior, fail
):
    failure = BashExitFailure("task", 1)
    hexes, rotate = fake_program_tasks(behavior, monkeypatch, {fail: failure})
    config = regional_config(rotation_degrees=45)
    with pytest.raises(RuntimeError, match=f"{fail} failed"):
        await behavior.generate_mesh(config, str(regional.output))
    # Nothing is left behind, including the namelists and intermediate meshes
    assert list(regional.output.iterdir()) == []

    fake_program_tasks(behavior, monkeypatch)
    result = await behavior.generate_mesh(config, str(regional.output))
    assert Path(result["static"]).is_file()


async def test_failed_regional_partition_removes_the_whole_mesh(
    monkeypatch, regional, behavior
):
    mark_metis_installed(behavior)
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={8})
    )
    config = {**regional_config(), "init_ranks": 4, "forecast_ranks": 8}
    with pytest.raises(RuntimeError, match="partition failed for 8 ranks"):
        await behavior.generate_mesh(config, str(regional.output))
    assert list(regional.output.iterdir()) == []


async def test_existing_regional_mesh_name_fails_before_anything_runs(
    regional, behavior
):
    (regional.output / "region").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        await behavior.generate_mesh(regional_config(), str(regional.output))
    regional.hexes.assert_not_called()
    assert list((regional.output / "region").iterdir()) == []


async def test_regional_mesh_installs_the_programs_before_running(
    monkeypatch, tmp_path, behavior
):
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    hexes, _ = fake_program_tasks(behavior, monkeypatch)
    await behavior.generate_mesh(regional_config(), str(tmp_path / "meshes"))
    download.assert_called_once()
    build.assert_called_once()
    hexes.assert_called_once()


async def test_failed_install_leaves_no_regional_mesh(monkeypatch, tmp_path, behavior):
    failure = BashExitFailure("task", 3)
    mock_mpas_tools_tasks(monkeypatch, behavior, build=failure)
    hexes, _ = fake_program_tasks(behavior, monkeypatch)
    output = tmp_path / "meshes"
    with pytest.raises(RuntimeError, match="MPAS-Tools build failed"):
        await behavior.generate_mesh(regional_config(), str(output))
    hexes.assert_not_called()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    ("config", "defined", "missing"),
    [
        (regional_config(), ["service"], "compute"),
        (regional_config(), [], "compute"),
        ({**regional_config(), "forecast_ranks": 4}, ["service"], "compute"),
    ],
)
async def test_regional_mesh_checks_resources_before_any_work(
    monkeypatch, regional, behavior, config, defined, missing
):
    define_resources(monkeypatch, *defined)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.generate_mesh(config, str(regional.output))
    regional.hexes.assert_not_called()
    assert not regional.output.exists()


async def test_installed_regional_mesh_needs_only_compute(
    monkeypatch, regional, behavior
):
    define_resources(monkeypatch, "compute")
    result = await behavior.generate_mesh(regional_config(), str(regional.output))
    assert Path(result["static"]).is_file()


# Partitioning


async def test_partition_mesh_returns_absolute_paths(monkeypatch, tmp_path, behavior):
    mark_metis_installed(behavior)
    task = fake_partition_task()
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    monkeypatch.chdir(tmp_path)

    assert await behavior.partition_mesh("graph.info", 4) == str(
        tmp_path / "graph.info.part.4"
    )
    many = await behavior.partition_mesh("-graph.info", [16, 4, 4])
    assert many == {
        4: str(tmp_path / "-graph.info.part.4"),
        16: str(tmp_path / "-graph.info.part.16"),
    }
    for call in task.call_args_list:
        assert Path(call.args[0]).is_absolute()
        assert call.kwargs["executor"] == ["compute"]


@pytest.mark.parametrize(
    ("mesh_path", "ranks", "message"),
    [
        ("/graph.info", 1, "greater than 1"),
        ("/graph.info", [4, 0], "greater than 1"),
        ("/graph.info", [2, "x"], "greater than 1"),
        ("/graph.info", True, "greater than 1"),
        ("/graph.info", [], "must not be empty"),
        ("/graph.info", list(range(2, 19)), "At most 16 distinct"),
        ("", 4, "non-empty string"),
        (None, 4, "non-empty string"),
    ],
)
async def test_partition_mesh_rejects_invalid_input(
    monkeypatch, behavior, mesh_path, ranks, message
):
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(ValueError, match=message):
        await behavior.partition_mesh(mesh_path, ranks)
    download.assert_not_called()


async def test_partition_failure_names_its_log(monkeypatch, tmp_path, behavior):
    mark_metis_installed(behavior)
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={4})
    )
    with pytest.raises(
        RuntimeError,
        match=r"4 ranks \(exit 1\).*partition_mesh\.4\.[0-9a-f]{8}\.stderr",
    ):
        await behavior.partition_mesh(str(tmp_path / "graph.info"), [4, 8])


async def test_unexpected_partition_error_becomes_a_runtime_error(
    monkeypatch, tmp_path, behavior
):
    mark_metis_installed(behavior)
    failure = OSError("worker lost")
    monkeypatch.setattr(
        behavior, "_partition_mesh", Mock(return_value=done_future(error=failure))
    )
    with pytest.raises(
        RuntimeError, match="Metis partition failed for 4 ranks: worker lost"
    ) as info:
        await behavior.partition_mesh(str(tmp_path / "graph.info"), 4)
    assert info.value.__cause__ is failure


async def test_concurrent_partitions_use_separate_logs(monkeypatch, tmp_path, behavior):
    mark_metis_installed(behavior)
    task = fake_partition_task()
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    graph = str(tmp_path / "graph.info")
    await asyncio.gather(
        behavior.partition_mesh(graph, 4), behavior.partition_mesh(graph, 4)
    )
    logs = [call.kwargs["stderr"] for call in task.call_args_list]
    assert len(set(logs)) == 2


# Generating partitioned meshes


async def test_generate_mesh_partitions(monkeypatch, mesh_download, behavior):
    output, _ = mesh_download
    mark_metis_installed(behavior)
    monkeypatch.setattr(behavior, "_partition_mesh", fake_partition_task())
    result = await behavior.generate_mesh(PARTITIONED, str(output))
    graph = output / "forecast" / "forecast.graph.info"
    assert result == {
        "static": str(output / "forecast" / "forecast.static.nc"),
        "graph": str(graph),
        "partitions": {4: f"{graph}.part.4", 8: f"{graph}.part.8"},
    }
    assert Path(f"{graph}.part.8").is_file()


async def test_generate_mesh_without_partitions_needs_no_metis(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    config = {"resolution": "480km", "init_ranks": 1, "forecast_ranks": 1}
    assert set(await behavior.generate_mesh(config, str(output))) == {
        "static",
        "graph",
    }
    download.assert_not_called()


async def test_generate_mesh_installs_metis_before_download(
    monkeypatch, mesh_download, behavior
):
    output, mesh = mesh_download
    order = []
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    download.side_effect = lambda **kw: order.append("metis") or done_future()
    mesh.side_effect = lambda *a, **kw: order.append("mesh") or done_future()
    monkeypatch.setattr(behavior, "_partition_mesh", fake_partition_task())
    await behavior.generate_mesh({"resolution": "480km", "init_ranks": 4}, str(output))
    assert order == ["metis", "mesh"]


async def test_generate_mesh_install_failure_leaves_nothing(
    monkeypatch, mesh_download, behavior
):
    output, mesh = mesh_download
    mock_metis_tasks(monkeypatch, behavior, download=BashExitFailure("download", 1))
    with pytest.raises(RuntimeError, match="Metis download failed"):
        await behavior.generate_mesh(PARTITIONED, str(output))
    mesh.assert_not_called()
    assert not (output / "forecast").exists()


async def test_generate_mesh_partition_failure_cleans_up(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    mark_metis_installed(behavior)
    task = fake_partition_task(fail_ranks={8})
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    with pytest.raises(RuntimeError, match="8 ranks"):
        await behavior.generate_mesh(PARTITIONED, str(output))
    # The 4-rank partition that succeeded is removed with the rest of the mesh
    assert {path.name for path in output.iterdir()} == set(mesh_files())

    task.side_effect = fake_partition_task().side_effect
    assert set(
        (await behavior.generate_mesh(PARTITIONED, str(output)))["partitions"]
    ) == {
        4,
        8,
    }


async def test_cleanup_failures_do_not_hide_the_error(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    mark_metis_installed(behavior)
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={8})
    )
    warning = Mock()
    monkeypatch.setattr(mesh_module.logger, "warning", warning)
    # Removing the directory fails and leaves it behind
    monkeypatch.setattr(mesh_module.shutil, "rmtree", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Metis partition failed for 8 ranks"):
        await behavior.generate_mesh(PARTITIONED, str(output))
    warning.assert_called_once()
    assert "remove it before requesting this mesh again" in warning.call_args.args[0]


async def test_failed_request_removes_the_whole_mesh_directory(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    mark_metis_installed(behavior)

    def partition_then_fail(mesh_path, rank, **kwargs):
        # A partial file and an unexpected file left by the failed tool
        Path(f"{mesh_path}.part.{rank}").write_text("partial")
        Path(mesh_path).with_name("gpmetis.log").write_text("log")
        return done_future(error=BashExitFailure("partition", 1))

    task = Mock(side_effect=partition_then_fail)
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    with pytest.raises(RuntimeError, match="partition failed"):
        await behavior.generate_mesh(PARTITIONED, str(output))
    assert not (output / "forecast").exists()
    # The shared downloaded files are untouched
    assert {path.name for path in output.iterdir()} == set(mesh_files())
    assert all((output / name).stat().st_size > 0 for name in mesh_files())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (None, "mesh_config must be a dict"),
        ([], "mesh_config must be a dict"),
        ({"forecast_ranks": 0}, "forecast_ranks must be a positive integer"),
        ({"name": "x1.2562.graph.info"}, "downloaded global mesh file"),
        ({"name": "x1.40962.static.nc"}, "downloaded global mesh file"),
        ({"resolution": ["480km"]}, "not supported"),
    ],
)
async def test_generate_mesh_rejects_bad_config_before_any_work(
    monkeypatch, tmp_path, behavior, config, message
):
    output = tmp_path / "meshes"
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(ValueError, match=message):
        await behavior.generate_mesh(config, str(output))
    assert not output.exists()
    download.assert_not_called()


async def test_existing_name_fails_before_install(monkeypatch, mesh_download, behavior):
    output, _ = mesh_download
    (output / "forecast").mkdir()
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(FileExistsError):
        await behavior.generate_mesh(PARTITIONED, str(output))
    download.assert_not_called()


@pytest.mark.parametrize(
    "config", [{"forecast_rank": 4}, {"resolution": "480km", "ranks": 4}]
)
async def test_unknown_mesh_config_keys_are_rejected(
    monkeypatch, tmp_path, behavior, config
):
    output = tmp_path / "meshes"
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(ValueError, match="Unknown mesh_config keys"):
        await behavior.generate_mesh(config, str(output))
    assert not output.exists()
    download.assert_not_called()


@pytest.mark.parametrize("damage", ["binary", "not_executable"])
async def test_install_metis_rebuilds_a_marked_install_without_a_binary(
    monkeypatch, behavior, damage
):
    gpmetis = mark_metis_installed(behavior)
    if damage == "binary":
        gpmetis.unlink()
    else:
        gpmetis.chmod(0o644)
    download, build = mock_metis_tasks(monkeypatch, behavior)
    await behavior.install_metis()
    download.assert_called_once()
    build.assert_called_once()
    assert behavior.metis_built


async def test_install_metis_notices_an_install_removed_while_running(
    monkeypatch, behavior
):
    gpmetis = mark_metis_installed(behavior)
    await behavior.install_metis()
    gpmetis.unlink()
    download, build = mock_metis_tasks(monkeypatch, behavior)
    await behavior.install_metis()
    build.assert_called_once()


async def test_build_that_leaves_no_binary_is_an_error(monkeypatch, behavior):
    download = Mock(return_value=done_future())
    build = Mock(return_value=done_future())
    monkeypatch.setattr(behavior, "_download_metis", download)
    monkeypatch.setattr(behavior, "_build_metis", build)
    with pytest.raises(RuntimeError, match="not usable"):
        await behavior.install_metis()
    assert not behavior.metis_built


@pytest.mark.parametrize(
    ("defined", "missing"),
    [(["compute"], "service"), (["service"], "compute"), ([], "service")],
)
async def test_install_metis_checks_resources_before_submitting(
    monkeypatch, behavior, defined, missing
):
    define_resources(monkeypatch, "_parsl_internal", *defined)
    download, build = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.install_metis()
    download.assert_not_called()
    build.assert_not_called()


async def test_install_metis_does_not_need_resources_once_installed(
    monkeypatch, behavior
):
    mark_metis_installed(behavior)
    define_resources(monkeypatch)
    await behavior.install_metis()
    assert behavior.metis_built


async def test_partition_mesh_checks_for_compute_first(monkeypatch, behavior):
    define_resources(monkeypatch, "service")
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match="Partitioning needs.*'compute'"):
        await behavior.partition_mesh("/tmp/x.graph.info", 4)
    download.assert_not_called()


@pytest.mark.parametrize(
    ("config", "defined", "missing"),
    [
        (PARTITIONED, ["service"], "compute"),
        (PARTITIONED, ["compute"], "service"),
        ({"resolution": "480km"}, ["compute"], "service"),
    ],
)
async def test_generate_mesh_checks_resources_before_any_work(
    monkeypatch, mesh_download, behavior, config, defined, missing
):
    output, download = mesh_download
    define_resources(monkeypatch, *defined)
    metis_download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.generate_mesh(config, str(output))
    download.assert_not_called()
    metis_download.assert_not_called()
    assert not (output / config.get("name", "mesh_480km")).exists()


async def test_generate_mesh_without_partitions_needs_only_service(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    define_resources(monkeypatch, "service")
    result = await behavior.generate_mesh({"resolution": "480km"}, str(output))
    assert "partitions" not in result


async def test_resource_check_uses_the_loaded_workflow(
    monkeypatch, mesh_agent, behavior
):
    # mesh_agent keeps a workflow with "agent-executor" and "service" loaded
    use_real_resource_lookup(monkeypatch)
    assert {"agent-executor", "service"} <= _MeshBehavior._configured_resources()
    behavior._require_resources("Testing", "service")
    with pytest.raises(RuntimeError, match="'compute'"):
        behavior._require_resources("Testing", "service", "compute")


async def test_missing_workflow_skips_the_check_with_a_warning(monkeypatch, behavior):
    use_real_resource_lookup(monkeypatch)

    def no_workflow():
        raise NoDataFlowKernelError("none")

    monkeypatch.setattr(mesh_module.parsl, "dfk", no_workflow)
    warning = Mock()
    monkeypatch.setattr(mesh_module.logger, "warning", warning)
    behavior._require_resources("Testing", "compute")
    warning.assert_called_once()
    assert "resources are not checked" in warning.call_args.args[0]


async def test_unexpected_lookup_errors_are_not_hidden(monkeypatch, behavior):
    use_real_resource_lookup(monkeypatch)

    def broken():
        raise AttributeError("renamed")

    monkeypatch.setattr(mesh_module.parsl, "dfk", broken)
    with pytest.raises(AttributeError):
        behavior._require_resources("Testing", "compute")


@pytest.mark.parametrize("stage", ["download", "build"])
async def test_unexpected_task_errors_become_runtime_errors(
    monkeypatch, behavior, stage
):
    failure = OSError("worker lost")
    mock_metis_tasks(
        monkeypatch,
        behavior,
        download=failure if stage == "download" else None,
        build=failure if stage == "build" else None,
    )
    with pytest.raises(RuntimeError, match=f"Metis {stage} failed: ") as info:
        await behavior.install_metis()
    assert info.value.__cause__ is failure


def test_build_script_survives_a_failed_source_cleanup():
    build = _MeshBehavior._build_metis.func(_MeshBehavior("/tmp/w"))
    assert 'rm -rf "$build" || true' in build


@pytest.mark.parametrize(
    "config", [{"name": "a/b", "forecast_ranks": 4}, {"resolution": "1km"}]
)
async def test_ensure_tools_installed_validates_the_config(
    monkeypatch, behavior, config
):
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    with pytest.raises(ValueError):
        await behavior.ensure_tools_installed(config)
    download.assert_not_called()


async def test_ensure_tools_installed(monkeypatch, behavior):
    download, _ = mock_metis_tasks(monkeypatch, behavior)
    tools_download, _ = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.ensure_tools_installed({"resolution": "480km"})
    download.assert_not_called()
    tools_download.assert_not_called()
    await behavior.ensure_tools_installed({"init_ranks": 2})
    download.assert_called_once()
    tools_download.assert_not_called()


@pytest.mark.parametrize(
    ("config", "metis", "mpas_tools", "limited_area"),
    [
        (regional_config(), False, True, False),
        ({**regional_config(), "forecast_ranks": 4}, True, True, False),
        ({"resolution": "480km", "forecast_ranks": 4}, True, False, False),
        (create_region_config(), False, False, True),
        ({**create_region_config("polygon"), "init_ranks": 8}, True, False, True),
    ],
)
async def test_ensure_tools_installed_for_regional_requests(
    monkeypatch, behavior, config, metis, mpas_tools, limited_area
):
    metis_download, _ = mock_metis_tasks(monkeypatch, behavior)
    tools_download, _ = mock_mpas_tools_tasks(monkeypatch, behavior)
    limited_install, _ = mock_limited_area_tasks(monkeypatch, behavior)
    await behavior.ensure_tools_installed(config)
    assert metis_download.called == metis
    assert tools_download.called == mpas_tools
    assert limited_install.called == limited_area


# Reusing meshes


def read_record(mesh_dir, name):
    return json.loads((mesh_dir / f"{name}.mesh.json").read_text())


@pytest.fixture
def partitioning(monkeypatch, behavior):
    """Installed Metis and a fake partition task."""
    mark_metis_installed(behavior)
    task = fake_partition_task()
    monkeypatch.setattr(behavior, "_partition_mesh", task)
    return task


async def test_finished_mesh_keeps_a_record_of_its_request(
    mesh_download, partitioning, behavior
):
    output, _ = mesh_download
    await behavior.generate_mesh(PARTITIONED, str(output))
    version = behavior.metis_version
    assert read_record(output / "forecast", "forecast") == {
        "mesh": {"resolution": "480km", "regional": None},
        "partitions": {"4": version, "8": version},
    }


async def test_same_request_reuses_the_mesh(mesh_download, partitioning, behavior):
    output, download = mesh_download
    made = await behavior.generate_mesh(PARTITIONED, str(output))
    download.reset_mock()
    partitioning.reset_mock()
    again = await behavior.generate_mesh(PARTITIONED, str(output))
    assert again == {**made, "reused": True}
    download.assert_not_called()
    partitioning.assert_not_called()


async def test_reuse_makes_only_the_partitions_a_mesh_lacks(
    mesh_download, partitioning, behavior
):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast", "init_ranks": 4}
    await behavior.generate_mesh(config, str(output))
    partitioning.reset_mock()
    result = await behavior.generate_mesh(PARTITIONED, str(output))
    assert [call.args[1] for call in partitioning.call_args_list] == [8]
    graph = output / "forecast" / "forecast.graph.info"
    assert result["partitions"] == {4: f"{graph}.part.4", 8: f"{graph}.part.8"}
    assert set(read_record(output / "forecast", "forecast")["partitions"]) == {
        "4",
        "8",
    }


async def test_reuse_remakes_partitions_of_another_metis_version_or_lost(
    mesh_download, partitioning, behavior
):
    output, _ = mesh_download
    await behavior.generate_mesh(PARTITIONED, str(output))
    mesh_dir = output / "forecast"
    record = read_record(mesh_dir, "forecast")
    record["partitions"]["4"] = "5.1.0"
    (mesh_dir / "forecast.mesh.json").write_text(json.dumps(record))
    (mesh_dir / "forecast.graph.info.part.8").unlink()
    partitioning.reset_mock()
    await behavior.generate_mesh(PARTITIONED, str(output))
    assert sorted(call.args[1] for call in partitioning.call_args_list) == [4, 8]
    assert read_record(mesh_dir, "forecast")["partitions"] == {
        "4": behavior.metis_version,
        "8": behavior.metis_version,
    }


async def test_failed_partition_of_a_reused_mesh_leaves_it(
    monkeypatch, mesh_download, partitioning, behavior
):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast", "init_ranks": 4}
    await behavior.generate_mesh(config, str(output))
    mesh_dir = output / "forecast"
    before = {path.name for path in mesh_dir.iterdir()}
    monkeypatch.setattr(
        behavior, "_partition_mesh", fake_partition_task(fail_ranks={8})
    )
    with pytest.raises(RuntimeError, match="partition failed for 8 ranks"):
        await behavior.generate_mesh(PARTITIONED, str(output))
    assert {path.name for path in mesh_dir.iterdir()} == before
    assert read_record(mesh_dir, "forecast")["partitions"] == {
        "4": behavior.metis_version
    }


async def test_reuse_of_regional_meshes(regional, cutting, behavior):
    hexes = await behavior.generate_mesh(regional_config(), str(cutting.output))
    cut = await behavior.generate_mesh(create_region_config(), str(cutting.output))
    regional.hexes.reset_mock()
    cutting.task.reset_mock()
    again = await behavior.generate_mesh(regional_config(), str(cutting.output))
    assert again == {**hexes, "reused": True}
    again = await behavior.generate_mesh(create_region_config(), str(cutting.output))
    assert again == {**cut, "reused": True}
    regional.hexes.assert_not_called()
    cutting.task.assert_not_called()
    # The tool's version is recorded with the mesh
    assert read_record(cutting.output / "region", "region")["mesh"] == {
        "resolution": "15km",
        "regional": regional_config()["regional"],
        "mpas_tools_version": behavior.mpas_tools_version,
    }
    assert (
        read_record(cutting.output / "cut", "cut")["mesh"]["limited_area_version"]
        == behavior.limited_area_version
    )


@pytest.mark.parametrize(
    ("config", "difference"),
    [
        (create_region_config(), None),
        ({**create_region_config(), "resolution": "120km"}, "it is 480km, not 120km"),
        (
            create_region_config("ellipse"),
            "it is a create_region circle, not a create_region ellipse",
        ),
        (
            create_region_config(radius_km=500),
            "its region differs",
        ),
        ({"resolution": "480km", "name": "cut"}, "not a global mesh"),
    ],
)
async def test_request_unlike_the_existing_mesh_is_refused(
    cutting, behavior, config, difference
):
    await behavior.generate_mesh(create_region_config(), str(cutting.output))
    before = {path.name for path in (cutting.output / "cut").iterdir()}
    cutting.task.reset_mock()
    if difference is None:
        # The same request, as a check on the cases below
        assert (await behavior.generate_mesh(config, str(cutting.output)))["reused"]
        return
    with pytest.raises(FileExistsError, match="made from a different request") as error:
        await behavior.generate_mesh(config, str(cutting.output))
    assert difference in str(error.value)
    assert "overwrite=True" in str(error.value)
    cutting.task.assert_not_called()
    assert {path.name for path in (cutting.output / "cut").iterdir()} == before


async def test_mesh_of_another_tool_version_is_refused(tmp_path, regional, behavior):
    await behavior.generate_mesh(regional_config(), str(regional.output))
    newer = _MeshBehavior(str(behavior.work_dir), mpas_tools_version="2.1.0")
    with pytest.raises(FileExistsError, match="made with MPAS-Tools 2.0.0, not 2.1.0"):
        await newer.generate_mesh(regional_config(), str(regional.output))


@pytest.mark.parametrize(
    "damage",
    [
        lambda record: record.unlink(),
        lambda record: record.write_text("not json"),
        lambda record: record.write_text("[]"),
        lambda record: record.write_text('{"mesh": {}, "partitions": {"x": "5"}}'),
        lambda record: record.write_text('{"mesh": [], "partitions": {}}'),
    ],
)
async def test_mesh_without_a_valid_record_is_not_finished(
    mesh_download, behavior, damage
):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast"}
    await behavior.generate_mesh(config, str(output))
    damage(output / "forecast" / "forecast.mesh.json")
    with pytest.raises(FileExistsError, match="is not a finished mesh"):
        await behavior.generate_mesh(config, str(output))


async def test_link_to_a_finished_mesh_is_not_reused(tmp_path, mesh_download, behavior):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast"}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name, content in mesh_files().items():
        (elsewhere / name).write_bytes(content)
    await behavior.generate_mesh(config, str(elsewhere))
    (output / "forecast").symlink_to(elsewhere / "forecast")
    with pytest.raises(FileExistsError, match="is not a finished mesh"):
        await behavior.generate_mesh(config, str(output))


async def test_mesh_missing_its_files_is_refused(mesh_download, behavior):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast"}
    await behavior.generate_mesh(config, str(output))
    # The shared global mesh the link points to is gone
    (output / "x1.2562.graph.info").unlink()
    with pytest.raises(FileExistsError, match="is missing forecast.graph.info"):
        await behavior.generate_mesh(config, str(output))


async def test_reused_mesh_keeps_its_plot_or_draws_one(
    monkeypatch, mesh_download, behavior
):
    output, _ = mesh_download
    config = {"resolution": "480km", "name": "forecast"}
    image = output / "forecast" / "forecast.png"

    async def plot(static):
        image.write_bytes(b"png")
        return str(image), None

    draw = AsyncMock(side_effect=plot)
    monkeypatch.setattr(behavior, "_plot", draw)
    await behavior.generate_mesh(config, str(output))
    # Without an image, one is drawn
    result = await behavior.generate_mesh({**config, "plot": True}, str(output))
    assert result["plot"] == str(image)
    draw.assert_awaited_once()
    # With one, it is kept
    result = await behavior.generate_mesh({**config, "plot": True}, str(output))
    assert result["plot"] == str(image)
    draw.assert_awaited_once()
