# SPDX-License-Identifier: Apache-2.0

"""Global mesh downloads and deployment through Chiltepin's agent runtime."""

import io
import tarfile
import urllib.request
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest
import pytest_asyncio

from chiltepin import AgentRuntime, Workflow
from chiltepin.agents import mpas
from chiltepin.agents.mpas import MeshAgent
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
