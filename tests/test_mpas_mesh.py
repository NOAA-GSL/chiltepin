# SPDX-License-Identifier: Apache-2.0

"""Global mesh downloads and deployment through Chiltepin's agent runtime."""

import asyncio
import io
import pickle
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
import pytest_asyncio
from parsl.app.errors import BashExitFailure
from parsl.errors import NoDataFlowKernelError

from chiltepin import AgentRuntime, Workflow
from chiltepin.agents import mpas
from chiltepin.agents.mpas import MeshAgent
from chiltepin.agents.mpas import mesh as mesh_module
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
    with pytest.raises(FileExistsError):
        await mesh_agent.generate_mesh(config, str(output))
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


def mock_install_tasks(monkeypatch, behavior, tool, mark, download=None, build=None):
    """Replace a tool's download and build tasks with mocks."""

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
    monkeypatch.setattr(behavior, f"_download_{tool}", download_task)
    monkeypatch.setattr(behavior, f"_build_{tool}", build_task)
    return download_task, build_task


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
    config = regional_config()
    config["resolution"] = resolution
    behavior._validate_mesh_config(config)


def test_validate_mesh_config_accepts_regional_defaults_and_rotation(behavior):
    behavior._validate_mesh_config(
        {"regional": {"project_hexes": {**HEXES, "rotation_degrees": -30.5}}}
    )
    behavior._validate_mesh_config(regional_config(center_lon=270, center_lat=-90))


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
        ({"create_region": {}}, "'create_region' is not supported yet"),
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
@pytest.mark.parametrize("value", [True, "40", "40,", None, float("nan"), float("inf")])
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


def test_validate_mesh_config_requires_every_project_hexes_key(behavior):
    for key in HEXES:
        with pytest.raises(ValueError, match=f"missing required keys: {key}"):
            behavior._validate_mesh_config(regional_config(**{key: ...}))


async def test_regional_mesh_is_rejected_before_anything_is_created(tmp_path, behavior):
    # Temporary until regional meshes can be generated
    output = tmp_path / "meshes"
    with pytest.raises(RuntimeError, match="cannot be generated yet"):
        await behavior.generate_mesh(regional_config(), str(output))
    assert not output.exists()


async def test_invalid_regional_mesh_creates_nothing(tmp_path, behavior):
    output = tmp_path / "meshes"
    with pytest.raises(ValueError, match="not supported yet"):
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
    fake = tmp_path / "gpmetis"
    fake.write_text(
        f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > {shlex.quote(str(record))}\n'
    )
    fake.chmod(0o755)
    behavior.gpmetis_path = fake

    script = _MeshBehavior._partition_mesh.func(behavior, mesh_path, 4)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert not pwned.exists()
    assert record.read_text().splitlines()[-2:] == [mesh_path, "4"]


# Installing Metis


async def test_install_metis_reuses_a_finished_install(monkeypatch, behavior):
    gpmetis = mark_metis_installed(behavior)
    download, build = mock_metis_tasks(monkeypatch, behavior)
    await behavior.install_metis()
    assert behavior.metis_built
    assert behavior.gpmetis_path == gpmetis
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
    assert behavior.gpmetis_path == (
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

    restored = pickle.loads(pickle.dumps(behavior))
    assert restored.work_dir == behavior.work_dir
    assert not restored._install_lock.locked()

    behavior._install_lock.release()
    await waiter
    behavior._install_lock.release()


# Installing MPAS-Tools


async def test_install_mpas_tools_reuses_a_finished_install(monkeypatch, behavior):
    prefix = mark_mpas_tools_installed(behavior)
    download, build = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.install_mpas_tools()
    assert behavior.mpas_tools_built
    assert behavior.project_hexes_path == prefix / "bin" / "project_hexes"
    assert behavior.grid_rotate_path == prefix / "bin" / "grid_rotate"
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
    ("config", "metis", "mpas_tools"),
    [
        (regional_config(), False, True),
        ({**regional_config(), "forecast_ranks": 4}, True, True),
        ({"resolution": "480km", "forecast_ranks": 4}, True, False),
    ],
)
async def test_ensure_tools_installed_for_regional_requests(
    monkeypatch, behavior, config, metis, mpas_tools
):
    metis_download, _ = mock_metis_tasks(monkeypatch, behavior)
    tools_download, _ = mock_mpas_tools_tasks(monkeypatch, behavior)
    await behavior.ensure_tools_installed(config)
    assert metis_download.called == metis
    assert tools_download.called == mpas_tools
