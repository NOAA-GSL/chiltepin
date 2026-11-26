# SPDX-License-Identifier: Apache-2.0
"""Tests of the installs of the tools that make global meshes.

JIGSAW, mkgrid and sphtri_subdiv are installed as Metis and MPAS-Tools are. The
tasks are stood in, and their scripts are run against a local Git repository and
with stand-ins for the build tools. The real integration test installs them.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest
from parsl.app.errors import BashExitFailure

from chiltepin.agents.mpas.mesh import _MeshBehavior

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class Tool:
    """What differs among the three installs."""

    def __init__(self, name, action, task, log, label, version, built, prefix, files):
        self.name = name
        self.action = action
        self.task = task  # "jigsaw" in _download_jigsaw
        self.log = log
        self.label = label
        self.version = version  # the agent's attribute, and launch argument
        self.built = built
        self.prefix = prefix  # the agent's property
        self.files = files  # executable first, then the other files it needs


TOOLS = {
    "jigsaw": Tool(
        "jigsaw", "install_jigsaw", "jigsaw", "jigsaw", "JIGSAW",
        "jigsaw_version", "jigsaw_built", "_jigsaw_prefix", ["bin/jigsaw"],
    ),
    "mkgrid": Tool(
        "mkgrid", "install_mkgrid", "mkgrid", "mkgrid", "mkgrid",
        "mkgrid_version", "mkgrid_built", "_mkgrid_prefix",
        ["bin/mkgrid", "share/ICOS.msh", "share/GEOM.msh"],
    ),
    "sphtri_subdiv": Tool(
        "sphtri_subdiv", "install_sphtri_subdiv", "sphtri_subdiv", "sphtri_subdiv",
        "sphtri_subdiv", "mesh_tools_version", "sphtri_subdiv_built",
        "_sphtri_subdiv_prefix", ["bin/sphtri_subdiv"],
    ),
}  # fmt: skip
EACH_TOOL = pytest.mark.parametrize("tool", list(TOOLS.values()), ids=list(TOOLS))


@pytest.fixture(autouse=True)
def no_resource_check(monkeypatch):
    """Skip the resource check. Tests of the check set the resources they want."""
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: None)
    )


def define_resources(monkeypatch, *names):
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: set(names))
    )


@pytest.fixture
def behavior(tmp_path):
    return _MeshBehavior(str(tmp_path / "work"))


def done_future(result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def mark_installed(behavior, tool, commit=COMMIT):
    """Create what a finished install leaves in work_dir."""
    prefix = getattr(behavior, tool.prefix)
    shutil.rmtree(prefix, ignore_errors=True)
    for index, name in enumerate(tool.files):
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        if index == 0:
            path.chmod(0o755)
    (prefix / ".installed").write_text(commit + "\n")
    return prefix


def mock_tasks(monkeypatch, behavior, tool, download=None, build=None):
    """Replace a tool's download and build tasks with mocks."""

    def run_build(**kwargs):
        # A successful build leaves a finished install behind
        for dependency in kwargs.get("inputs", []):
            if dependency.done() and dependency.exception() is not None:
                return done_future(error=dependency.exception())
        if build is None:
            mark_installed(behavior, tool)
        return done_future(error=build)

    download_task = Mock(return_value=done_future(error=download))
    build_task = Mock(side_effect=run_build)
    monkeypatch.setattr(behavior, f"_download_{tool.task}", download_task)
    monkeypatch.setattr(behavior, f"_build_{tool.task}", build_task)
    return download_task, build_task


# Configuration


def test_defaults(behavior):
    assert behavior.jigsaw_version == "3777fe7f00815295a94065cdcaca27bcb3c2106e"
    assert behavior.mkgrid_version == "feature/nesting"
    assert behavior.mesh_tools_version == "4967f0fbb278584c11b3057c3eb4396d199f030b"
    assert behavior.jigsaw_url == "https://github.com/dengwirda/jigsaw.git"
    assert behavior.mesh_tools_url == "https://github.com/mgduda/mesh_tools.git"
    assert behavior.mkgrid_url.endswith("/christopherwharrop-noaa/mpas_jigsaw.git")
    assert not (
        behavior.jigsaw_built or behavior.mkgrid_built or behavior.sphtri_subdiv_built
    )


@EACH_TOOL
@pytest.mark.parametrize("ref", ["v1.2", "feature/nesting", COMMIT, "release-2.0_a"])
def test_a_branch_tag_or_commit_is_accepted(tmp_path, tool, ref):
    behavior = _MeshBehavior(str(tmp_path / "work"), **{tool.version: ref})
    assert getattr(behavior, tool.version) == ref


@EACH_TOOL
@pytest.mark.parametrize(
    "ref",
    [
        "", "../x", "a..b", "a//b", "-x", "--upload-pack=x", ".build", "x/", "x.", "x.lock",
        "a b", "x;rm -rf /", "$(id)", "a\n", "a@{1}", "a:b", "/abs", None, 5,
    ],
)  # fmt: skip
def test_an_invalid_ref_is_refused(tmp_path, tool, ref):
    with pytest.raises(ValueError, match=f"Invalid {tool.version}"):
        _MeshBehavior(str(tmp_path / "work"), **{tool.version: ref})


@EACH_TOOL
def test_a_ref_named_build_does_not_collide_with_the_sources(tmp_path, tool):
    """The sources are fetched under .build, which no ref can be named."""
    behavior = _MeshBehavior(str(tmp_path / "work"), **{tool.version: "build"})
    prefix = getattr(behavior, tool.prefix)
    assert prefix.name == "build"
    script = getattr(_MeshBehavior, f"_download_{tool.task}").func(behavior)
    assert f"{prefix.parent}/.build/build" in script
    assert f"{prefix}" != f"{prefix.parent}/.build/build"


@EACH_TOOL
def test_a_ref_must_fit_a_directory_name(tmp_path, tool):
    long_ref = "a" * 200
    behavior = _MeshBehavior(str(tmp_path / "work"), **{tool.version: long_ref})
    assert getattr(behavior, tool.version) == long_ref
    # The second is 133 characters, and 265 once its slashes are written as %2F
    for ref in ("a" * 201, "a/" * 66 + "a"):
        with pytest.raises(ValueError, match=f"Invalid {tool.version}"):
            _MeshBehavior(str(tmp_path / "work"), **{tool.version: ref})


def test_the_directory_of_a_ref_holds_no_separator(tmp_path):
    behavior = _MeshBehavior(str(tmp_path / "work"), mkgrid_version="feature/nesting")
    assert behavior._mkgrid_prefix == tmp_path / "work" / "mkgrid" / "feature%2Fnesting"
    assert behavior._ref_dir("v1") == "v1"


def test_an_install_dir_for_each_version(tmp_path):
    work = tmp_path / "work"
    one = _MeshBehavior(str(work), jigsaw_version="v1")
    two = _MeshBehavior(str(work), jigsaw_version="v2")
    assert one._jigsaw_prefix != two._jigsaw_prefix


@pytest.mark.parametrize(
    "text, commit",
    [(f"{COMMIT}\n", COMMIT), (f"  {COMMIT}  ", COMMIT), ("", None), ("\n", None)],
)
def test_the_marker_holds_the_commit(tmp_path, text, commit):
    (tmp_path / ".installed").write_text(text)
    assert _MeshBehavior._installed_commit(tmp_path) == commit


def test_no_marker_means_no_commit(tmp_path):
    assert _MeshBehavior._installed_commit(tmp_path) is None
    assert _MeshBehavior._installed_commit(tmp_path / "none") is None


def test_agent_state_with_the_new_tools_is_picklable(behavior):
    import pickle

    again = pickle.loads(pickle.dumps(behavior))
    assert again.jigsaw_version == behavior.jigsaw_version
    assert again._mkgrid_prefix == behavior._mkgrid_prefix


# What a usable install is


@EACH_TOOL
def test_a_finished_install_is_usable(behavior, tool):
    usable = getattr(behavior, f"_{tool.name}_usable")
    assert not usable()
    mark_installed(behavior, tool)
    assert usable()


@EACH_TOOL
@pytest.mark.parametrize("damage", ["marker", "empty_marker", "not_executable", "file"])
def test_a_damaged_install_is_not_usable(behavior, tool, damage):
    if damage == "file" and len(tool.files) == 1:
        pytest.skip("this tool has only its program")
    prefix = mark_installed(behavior, tool)
    if damage == "marker":
        (prefix / ".installed").unlink()
    elif damage == "empty_marker":
        (prefix / ".installed").write_text("")
    elif damage == "not_executable":
        (prefix / tool.files[0]).chmod(0o644)
    else:
        (prefix / tool.files[1]).unlink()
    assert not getattr(behavior, f"_{tool.name}_usable")()


# The install actions


@EACH_TOOL
async def test_a_finished_install_is_reused(monkeypatch, behavior, tool):
    mark_installed(behavior, tool)
    download, build = mock_tasks(monkeypatch, behavior, tool)
    await getattr(behavior, tool.action)()
    assert getattr(behavior, tool.built)
    download.assert_not_called()
    build.assert_not_called()


@EACH_TOOL
async def test_a_damaged_install_is_rebuilt(monkeypatch, behavior, tool):
    prefix = mark_installed(behavior, tool)
    (prefix / ".installed").unlink()
    download, build = mock_tasks(monkeypatch, behavior, tool)
    await getattr(behavior, tool.action)()
    download.assert_called_once()
    build.assert_called_once()
    assert getattr(behavior, tool.built)
    assert behavior._installed_commit(prefix) == COMMIT


@EACH_TOOL
async def test_the_download_runs_on_service_and_the_build_on_compute(
    monkeypatch, behavior, tool
):
    download, build = mock_tasks(monkeypatch, behavior, tool)
    await getattr(behavior, tool.action)()
    download.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / f"{tool.log}_download.stdout"),
        stderr=str(behavior.log_dir / f"{tool.log}_download.stderr"),
    )
    build.assert_called_once_with(
        executor=["compute"],
        stdout=str(behavior.log_dir / f"{tool.log}_build.stdout"),
        stderr=str(behavior.log_dir / f"{tool.log}_build.stderr"),
        inputs=[download.return_value],
    )


@EACH_TOOL
async def test_concurrent_installs_run_once(monkeypatch, behavior, tool):
    pending, build_pending = Future(), Future()
    download, build = mock_tasks(monkeypatch, behavior, tool)
    download.return_value = pending
    build.side_effect = lambda **kwargs: build_pending
    requests = [asyncio.create_task(getattr(behavior, tool.action)()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert download.call_count == 1
    pending.set_result(None)
    mark_installed(behavior, tool)
    build_pending.set_result(None)
    await asyncio.gather(*requests)
    assert download.call_count == 1 and build.call_count == 1


@EACH_TOOL
@pytest.mark.parametrize("stage", ["download", "build"])
async def test_a_failed_install_is_reported_and_retried(
    monkeypatch, behavior, tool, stage
):
    failure = BashExitFailure("task", 3)
    download, build = mock_tasks(
        monkeypatch,
        behavior,
        tool,
        download=failure if stage == "download" else None,
        build=failure if stage == "build" else None,
    )
    with pytest.raises(RuntimeError, match=f"{tool.label} {stage} failed") as info:
        await getattr(behavior, tool.action)()
    assert f"{tool.log}_{stage}.stderr" in str(info.value)
    assert info.value.__cause__ is failure
    assert not getattr(behavior, tool.built)
    download.return_value = done_future()
    build.side_effect = lambda **kwargs: (
        mark_installed(behavior, tool),
        done_future(),
    )[1]
    await getattr(behavior, tool.action)()
    assert getattr(behavior, tool.built)


@EACH_TOOL
async def test_a_build_that_leaves_nothing_usable_is_an_error(
    monkeypatch, behavior, tool
):
    monkeypatch.setattr(
        behavior, f"_download_{tool.task}", Mock(return_value=done_future())
    )
    monkeypatch.setattr(
        behavior, f"_build_{tool.task}", Mock(return_value=done_future())
    )
    with pytest.raises(RuntimeError, match="build finished but .* not usable"):
        await getattr(behavior, tool.action)()
    assert not getattr(behavior, tool.built)


@EACH_TOOL
@pytest.mark.parametrize(
    ("defined", "missing"),
    [(["compute"], "service"), (["service"], "compute"), ([], "service")],
)
async def test_resources_are_checked_before_anything_is_submitted(
    monkeypatch, behavior, tool, defined, missing
):
    define_resources(monkeypatch, "_parsl_internal", *defined)
    download, build = mock_tasks(monkeypatch, behavior, tool)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await getattr(behavior, tool.action)()
    download.assert_not_called()
    build.assert_not_called()


async def test_all_three_are_installed_in_turn(monkeypatch, behavior):
    calls = []
    for tool in TOOLS.values():
        monkeypatch.setattr(
            behavior,
            tool.action,
            Mock(side_effect=lambda name=tool.name: _record(calls, name)),
        )
    await behavior.install_global_mesh_tools()
    assert calls == ["jigsaw", "mkgrid", "sphtri_subdiv"]


async def _record(calls, name):
    calls.append(name)


async def test_a_failed_install_stops_the_rest(monkeypatch, behavior):
    async def fail():
        raise RuntimeError("mkgrid build failed")

    monkeypatch.setattr(behavior, "install_jigsaw", Mock(side_effect=lambda: _noop()))
    monkeypatch.setattr(behavior, "install_mkgrid", fail)
    later = Mock(side_effect=lambda: _noop())
    monkeypatch.setattr(behavior, "install_sphtri_subdiv", later)
    with pytest.raises(RuntimeError, match="mkgrid build failed"):
        await behavior.install_global_mesh_tools()
    later.assert_not_called()


async def _noop():
    return None


async def test_installing_everything_with_the_tasks_stood_in(monkeypatch, behavior):
    for tool in TOOLS.values():
        mock_tasks(monkeypatch, behavior, tool)
    await behavior.install_global_mesh_tools()
    assert behavior.jigsaw_built and behavior.mkgrid_built
    assert behavior.sphtri_subdiv_built


# The scripts


@pytest.fixture
def scratch():
    """A directory in the system's temporary directory, for tests that run scripts.

    The scripts remove a directory, make it and enter it at once. A filesystem
    that does not keep up with that fails some of the time, which Docker
    Desktop's mount of the host's directories does, and the tests directory may
    be on one.
    """
    with tempfile.TemporaryDirectory() as path:
        yield Path(path)


def git(directory, *arguments):
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repository(scratch):
    """A repository with two commits, a branch and a tag on the first."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    path = scratch / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "file.txt").write_text("one\n")
    git(path, "add", "file.txt")
    git(path, "commit", "-q", "-m", "one")
    first = git(path, "rev-parse", "HEAD")
    git(path, "tag", "v1")
    git(path, "branch", "feature/old")
    (path / "file.txt").write_text("two\n")
    git(path, "commit", "-q", "-am", "two")
    # Any commit may be asked for, as GitHub allows
    git(path, "config", "uploadpack.allowAnySHA1InWant", "true")
    return path, first, git(path, "rev-parse", "HEAD")


@EACH_TOOL
@pytest.mark.parametrize("which", ["tag", "branch", "commit", "default"])
def test_the_download_fetches_the_requested_revision(scratch, repository, tool, which):
    path, first, second = repository
    ref, expected = {
        "tag": ("v1", first),
        "branch": ("feature/old", first),
        "commit": (first, first),
        "default": ("main", second),
    }[which]
    behavior = _MeshBehavior(str(scratch / "work dir"), **{tool.version: ref})
    setattr(
        behavior, f"{tool.name.replace('sphtri_subdiv', 'mesh_tools')}_url", str(path)
    )
    script = getattr(_MeshBehavior, f"_download_{tool.task}").func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    build = next(
        (behavior.work_dir / tool.task.replace("sphtri_subdiv", "mesh-tools")).glob(
            ".build/*"
        )
    )
    assert (build / "commit").read_text().strip() == expected
    text = "one\n" if expected == first else "two\n"
    assert (build / "src" / "file.txt").read_text() == text
    assert f"Fetched commit {expected}" in done.stdout


@EACH_TOOL
def test_the_download_starts_from_empty_sources(scratch, repository, tool):
    path, first, _ = repository
    behavior = _MeshBehavior(str(scratch / "work"), **{tool.version: "main"})
    setattr(
        behavior, f"{tool.name.replace('sphtri_subdiv', 'mesh_tools')}_url", str(path)
    )
    script = getattr(_MeshBehavior, f"_download_{tool.task}").func(behavior)
    for _ in range(2):
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


def test_only_the_requested_revision_is_fetched(scratch, repository):
    path, first, second = repository
    behavior = _MeshBehavior(str(scratch / "work"), jigsaw_version="main")
    # A file URL, since git ignores --depth for a path on the same filesystem
    behavior.jigsaw_url = path.as_uri()
    script = _MeshBehavior._download_jigsaw.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    src = next((behavior.work_dir / "jigsaw").glob(".build/*")) / "src"
    assert git(src, "rev-parse", "--is-shallow-repository") == "true"
    assert git(src, "rev-list", "--count", "HEAD") == "1"
    assert git(src, "rev-parse", "HEAD") == second


def test_the_download_of_a_missing_ref_fails(scratch, repository):
    path, _, _ = repository
    behavior = _MeshBehavior(str(scratch / "work"), jigsaw_version="nothing")
    behavior.jigsaw_url = str(path)
    script = _MeshBehavior._download_jigsaw.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode != 0
    assert "Completed download" not in done.stdout


def test_the_download_scripts_name_the_repositories(scratch):
    behavior = _MeshBehavior(str(scratch / "work dir"))
    jigsaw = _MeshBehavior._download_jigsaw.func(behavior)
    assert (
        f"build='{behavior.work_dir}/jigsaw/.build/{behavior.jigsaw_version}'" in jigsaw
    )
    assert "remote add origin https://github.com/dengwirda/jigsaw.git" in jigsaw
    assert f"fetch -q --depth 1 origin {behavior.jigsaw_version}" in jigsaw
    mkgrid = _MeshBehavior._download_mkgrid.func(behavior)
    assert f"{behavior.work_dir}/mkgrid/.build/feature%2Fnesting" in mkgrid
    assert "origin feature/nesting" in mkgrid
    tools = _MeshBehavior._download_sphtri_subdiv.func(behavior)
    assert "remote add origin https://github.com/mgduda/mesh_tools.git" in tools
    # Transfers that stall are abandoned
    assert "http.lowSpeedTime=60" in jigsaw


def lines_of(script):
    return [line.strip() for line in script.splitlines()]


@pytest.mark.parametrize("name", ["jigsaw", "mkgrid", "sphtri_subdiv"])
def test_a_build_removes_the_old_install_only_after_checking_its_tools(scratch, name):
    behavior = _MeshBehavior(str(scratch / "work dir"))
    lines = lines_of(getattr(_MeshBehavior, f"_build_{name}").func(behavior))
    remove = lines.index('rm -rf "$prefix"')
    # Missing tools are found before an existing install is touched
    assert max(i for i, line in enumerate(lines) if "not found" in line) < remove
    # The sources go before the install is marked, with the commit, last
    sources = lines.index('rm -rf "$build" || true')
    marker = lines.index('echo "$commit" > "$prefix/.installed"')
    assert remove < sources < marker
    assert all(line.startswith("echo ") for line in lines[marker + 1 :] if line)


def fake_tools(scratch, **scripts):
    """Make a PATH of commands the scripts use, with stand-ins for some."""
    tools = scratch / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("date", "hostname", "cat", "rm", "mkdir", "cp", "chmod"):
        found = shutil.which(name)
        if found is None:
            pytest.skip(f"{name} is not available")
        if not (tools / name).exists():
            (tools / name).symlink_to(found)
    for name, body in scripts.items():
        (tools / name).write_text("#!/bin/bash\n" + body + "\n")
        (tools / name).chmod(0o755)
    return tools


def run_script(script, tools, **env):
    return subprocess.run(
        [shutil.which("bash"), "-c", script],
        env={"PATH": str(tools), **env},
        capture_output=True,
        text=True,
    )


def sources(behavior, tool_dir, files):
    """Create the downloaded sources a build starts from."""
    build = (
        behavior.work_dir
        / tool_dir
        / ".build"
        / behavior._ref_dir(
            {
                "jigsaw": behavior.jigsaw_version,
                "mkgrid": behavior.mkgrid_version,
                "mesh-tools": behavior.mesh_tools_version,
            }[tool_dir]
        )
    )
    (build / "src").mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (build / "src" / name).parent.mkdir(parents=True, exist_ok=True)
        (build / "src" / name).write_text(text)
    (build / "commit").write_text(COMMIT + "\n")
    return build


# A cmake that records how it was run, and installs a program where asked
FAKE_CMAKE = """\
echo "cmake $*" >> "$CMAKE_LOG"
echo "PNETCDF=${PNETCDF:-}" >> "$CMAKE_LOG"
for argument in "$@"; do
    case "$argument" in -DCMAKE_INSTALL_PREFIX=*) prefix="${argument#*=}" ;; esac
done
if [ "$1" = "--build" ]; then
    mkdir -p "$(cat install_prefix)/bin"
    for program in $PROGRAMS; do echo "#!/bin/sh" > "$(cat install_prefix)/bin/$program"; chmod +x "$(cat install_prefix)/bin/$program"; done
else
    echo "$prefix" > install_prefix
fi
"""


@pytest.mark.parametrize("name", ["jigsaw", "mkgrid"])
def test_a_cmake_build_installs_and_marks_the_commit(scratch, name):
    tool = TOOLS[name]
    behavior = _MeshBehavior(str(scratch / "work dir"))
    build = sources(
        behavior,
        name,
        {"ICOS.msh": "icosahedron\n", "GEOM.msh": "sphere\n", "kept.txt": "x"},
    )
    tools = fake_tools(
        scratch,
        cmake=FAKE_CMAKE,
        mpicc="exit 0",
        **{"pnetcdf-config": 'echo "/opt/pnetcdf"', "c++": "exit 0"},
    )
    log = scratch / "cmake.log"
    script = getattr(_MeshBehavior, f"_build_{tool.task}").func(behavior)
    done = run_script(script, tools, CMAKE_LOG=str(log), PROGRAMS=name)
    assert done.returncode == 0, done.stderr
    prefix = getattr(behavior, tool.prefix)
    assert (prefix / "bin" / name).is_file()
    assert (prefix / ".installed").read_text() == COMMIT + "\n"
    assert not build.exists()
    recorded = log.read_text()
    assert f"-DCMAKE_INSTALL_PREFIX={prefix}" in recorded
    assert "-DCMAKE_BUILD_TYPE=Release" in recorded
    assert "cmake .. -DCMAKE_BUILD_TYPE=Release" in recorded
    assert "--build . --target install -- -j4" in recorded
    if name == "mkgrid":
        # PnetCDF is found from pnetcdf-config, and the pipeline's files are kept
        assert "PNETCDF=/opt/pnetcdf" in recorded
        assert (prefix / "share" / "ICOS.msh").read_text() == "icosahedron\n"
        assert (prefix / "share" / "GEOM.msh").read_text() == "sphere\n"
        assert behavior._mkgrid_usable()


def test_pnetcdf_set_by_the_environment_is_not_replaced(scratch):
    behavior = _MeshBehavior(str(scratch / "work"))
    sources(behavior, "mkgrid", {"ICOS.msh": "i", "GEOM.msh": "g"})
    tools = fake_tools(
        scratch, cmake=FAKE_CMAKE, mpicc="exit 0", **{"pnetcdf-config": "echo /other"}
    )
    log = scratch / "cmake.log"
    script = _MeshBehavior._build_mkgrid.func(behavior)
    done = run_script(
        script, tools, CMAKE_LOG=str(log), PROGRAMS="mkgrid", PNETCDF="/site/pnetcdf"
    )
    assert done.returncode == 0, done.stderr
    assert "PNETCDF=/site/pnetcdf" in log.read_text()


@pytest.mark.parametrize(
    "name, missing, available, message",
    [
        ("jigsaw", "cmake", {"c++": "exit 0"}, "cmake not found"),
        ("jigsaw", "c++", {"cmake": "exit 0"}, "c++ not found"),
        ("mkgrid", "cmake", {"mpicc": "exit 0"}, "cmake not found"),
        ("mkgrid", "mpicc", {"cmake": "exit 0"}, "mpicc not found"),
        ("mkgrid", "pnetcdf-config", {"cmake": "exit 0", "mpicc": "exit 0"}, "PnetCDF not found"),
        ("sphtri_subdiv", "c++", {}, "c++ not found"),
        ("mkgrid", "pnetcdf-config", {"cmake": "exit 0", "mpicc": "exit 0", "pnetcdf-config": "exit 3"}, "did not give the prefix of PnetCDF"),
        ("mkgrid", "pnetcdf-config", {"cmake": "exit 0", "mpicc": "exit 0", "pnetcdf-config": "echo"}, "did not give the prefix of PnetCDF"),
    ],
)  # fmt: skip
def test_a_missing_tool_fails_clearly_and_keeps_an_existing_install(
    scratch, name, missing, available, message
):
    tool = TOOLS[name]
    behavior = _MeshBehavior(str(scratch / "work"))
    prefix = mark_installed(behavior, tool)
    tools = fake_tools(scratch, **available)
    script = getattr(_MeshBehavior, f"_build_{tool.task}").func(behavior)
    done = run_script(script, tools)
    assert done.returncode == 1
    assert message in done.stderr
    assert (prefix / ".installed").is_file()


def test_sphtri_subdiv_is_compiled_from_one_file(scratch):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler is available")
    behavior = _MeshBehavior(str(scratch / "work"))
    program = '#include <cstdio>\nint main() { std::puts("subdivided"); return 0; }\n'
    build = sources(
        behavior, "mesh-tools", {"sphtri_subdiv/sphtri_subdiv.cpp": program}
    )
    tools = fake_tools(scratch)
    script = _MeshBehavior._build_sphtri_subdiv.func(behavior)
    done = run_script(
        script,
        tools,
        CXX=compiler,
        # What the compiler itself needs to run
        PATH=f"{tools}:{os.path.dirname(compiler)}:/usr/bin:/bin",
    )
    assert done.returncode == 0, done.stderr
    assert behavior._sphtri_subdiv_usable()
    ran = subprocess.run([str(behavior._sphtri_subdiv)], capture_output=True, text=True)
    assert ran.stdout.strip() == "subdivided"
    assert behavior._installed_commit(behavior._sphtri_subdiv_prefix) == COMMIT
    assert not build.exists()


@pytest.mark.parametrize("name", ["jigsaw", "sphtri_subdiv"])
def test_a_compiler_with_a_wrapper_is_found_by_its_first_word(scratch, name):
    tool = TOOLS[name]
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler is available")
    behavior = _MeshBehavior(str(scratch / "work"))
    sources(
        behavior,
        {"jigsaw": "jigsaw", "sphtri_subdiv": "mesh-tools"}[name],
        {"sphtri_subdiv/sphtri_subdiv.cpp": "int main() { return 0; }\n"},
    )
    tools = fake_tools(
        scratch,
        cmake=FAKE_CMAKE,
        # A wrapper that runs what it is given, as ccache does
        ccache='exec "$@"',
    )
    log = scratch / "cmake.log"
    script = getattr(_MeshBehavior, f"_build_{tool.task}").func(behavior)
    path = f"{tools}:{os.path.dirname(compiler)}:/usr/bin:/bin"
    done = run_script(
        script,
        tools,
        PATH=path,
        CXX=f"ccache {compiler}",
        CMAKE_LOG=str(log),
        PROGRAMS="jigsaw",
    )
    assert done.returncode == 0, done.stderr
    assert behavior._installed_commit(getattr(behavior, tool.prefix)) == COMMIT
    # A wrapper that is not there is named, not the compiler
    missing = run_script(script, tools, PATH=path, CXX=f"nowrapper {compiler}")
    assert missing.returncode == 1 and "nowrapper not found" in missing.stderr


@pytest.mark.parametrize("failure", ["cmake", "files"])
def test_a_failed_cmake_build_leaves_no_marker_and_keeps_the_sources(scratch, failure):
    behavior = _MeshBehavior(str(scratch / "work"))
    files = {} if failure == "files" else {"ICOS.msh": "i", "GEOM.msh": "g"}
    build = sources(behavior, "mkgrid", files)
    tools = fake_tools(
        scratch,
        cmake="exit 1" if failure == "cmake" else FAKE_CMAKE,
        mpicc="exit 0",
        **{"pnetcdf-config": "echo /opt/pnetcdf"},
    )
    script = _MeshBehavior._build_mkgrid.func(behavior)
    done = run_script(
        script, tools, CMAKE_LOG=str(scratch / "cmake.log"), PROGRAMS="mkgrid"
    )
    assert done.returncode != 0
    assert not (behavior._mkgrid_prefix / ".installed").exists()
    assert not behavior._mkgrid_usable()
    # Kept to debug
    assert (build / "src").is_dir() and (build / "commit").is_file()


@pytest.mark.parametrize("name", ["jigsaw", "sphtri_subdiv"])
def test_a_compiler_setting_of_only_spaces_is_the_default(scratch, name):
    tool = TOOLS[name]
    behavior = _MeshBehavior(str(scratch / "work"))
    prefix = mark_installed(behavior, tool)
    tools = fake_tools(scratch, cmake="exit 0")
    script = getattr(_MeshBehavior, f"_build_{tool.task}").func(behavior)
    done = run_script(script, tools, CXX="   ")
    # c++ is not available here, and is the one named, not an unset variable
    assert done.returncode == 1
    assert "c++ not found" in done.stderr and "unbound" not in done.stderr
    assert (prefix / ".installed").is_file()
