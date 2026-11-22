# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh agent's prompt actions, with their tasks stood in.

The tasks' scripts are run with a stand-in Python, and the prompt task's body
is run with the model and the lookup stood in. tests/test_mpas_mesh_prompt.py
tests reading the model's answer, and the real integration test reads a prompt
with a real model.
"""

import asyncio
import shlex
import subprocess
import sys
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from chiltepin.agents.mpas import _mesh_prompt
from chiltepin.agents.mpas import mesh as mesh_module
from chiltepin.agents.mpas._geo_lookup import BOUNDARY_LAYERS, RegionNotFound
from chiltepin.agents.mpas._plot_mesh import REQUIRED_PARTS, layer_path
from chiltepin.agents.mpas.mesh import MeshAgent, _MeshBehavior

MODEL = "bedrock/us.anthropic.claude-sonnet-5-5"
CONFIG = {
    "resolution": "15km",
    "name": "texas",
    "regional": {
        "project_hexes": {
            "center_lat": 31.0,
            "center_lon": -99.0,
            "extent_x_km": 1500,
            "extent_y_km": 1300,
        }
    },
}


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


def fake_python(monkeypatch, tmp_path, body):
    """Make the tasks' Python a shell script running ``body``."""
    path = tmp_path / "fake-python"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(path))
    return path


def write_boundaries(behavior):
    """Write every part each boundary layer needs."""
    for layer in BOUNDARY_LAYERS:
        path = layer_path(behavior._boundaries_dir, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in REQUIRED_PARTS:
            path.with_suffix(suffix).write_bytes(b"shapefile")


def mark_prompting_installed(behavior):
    write_boundaries(behavior)
    (behavior._geo_prefix / ".installed").touch()


def mock_prompting_tasks(monkeypatch, behavior, download=None, check=None):
    """Replace the boundaries download and the check with mocks."""
    download_task = Mock(return_value=done_future(error=download))

    def run_check(**kwargs):
        if check is None:
            mark_prompting_installed(behavior)
        return done_future(error=check)

    check_task = Mock(side_effect=run_check)
    monkeypatch.setattr(behavior, "_download_boundaries", download_task)
    monkeypatch.setattr(behavior, "_check_prompting", check_task)
    return download_task, check_task


# The agent's argument


def test_nominatim_contact(tmp_path):
    assert _MeshBehavior(str(tmp_path)).nominatim_contact is None
    behavior = _MeshBehavior(str(tmp_path), nominatim_contact="me@example.org")
    assert behavior.nominatim_contact == "me@example.org"
    agent = MeshAgent(str(tmp_path), nominatim_contact="a.b@c.d")
    assert agent._behavior.nominatim_contact == "a.b@c.d"


@pytest.mark.parametrize(
    "contact",
    ["", "me", "me@", "@example.org", "me@example", "me @example.org", "a\n@b.c", 5],
)
def test_nominatim_contact_must_be_an_email_address(tmp_path, contact):
    with pytest.raises(ValueError, match="nominatim_contact must be an email"):
        _MeshBehavior(str(tmp_path), nominatim_contact=contact)


def test_long_nominatim_contact(tmp_path):
    with pytest.raises(ValueError, match="nominatim_contact"):
        _MeshBehavior(str(tmp_path), nominatim_contact="a" * 250 + "@b.cd")


# The install's scripts


def test_boundaries_download_script(monkeypatch, tmp_path, behavior):
    record = tmp_path / "record"
    fake_python(
        monkeypatch, tmp_path, f'printf "%s\\n" "$@" > {shlex.quote(str(record))}'
    )
    (behavior._geo_prefix / "natural_earth").mkdir(parents=True)
    (behavior._geo_prefix / ".installed").touch()
    script = _MeshBehavior._download_boundaries.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    arguments = record.read_text().splitlines()
    assert arguments[0] == "-c"
    assert "download_boundaries" in arguments[1]
    assert arguments[2] == str(behavior._boundaries_dir)
    # The install no longer counts, but nothing downloaded before is removed
    assert not (behavior._geo_prefix / ".installed").exists()
    assert (behavior._geo_prefix / "natural_earth").is_dir()
    assert "rm -rf" not in script


def test_boundaries_download_with_the_real_python(monkeypatch, tmp_path, behavior):
    # The script's own code runs, with the download stood in
    calls = []
    from chiltepin.agents.mpas import _geo_lookup

    monkeypatch.setattr(
        _geo_lookup, "download_map_data", lambda path, layers: calls.append(path)
    )
    script = _MeshBehavior._download_boundaries.func(behavior)
    code = shlex.split(script.splitlines()[-2])[2]
    exec(code.replace("sys.argv[1]", repr(str(behavior._boundaries_dir))), {})
    assert calls == [behavior._boundaries_dir]


@pytest.mark.parametrize("problem", [None, "imports", "boundaries"])
def test_prompting_check_marks_only_a_working_install(
    monkeypatch, tmp_path, behavior, problem
):
    behavior._geo_prefix.mkdir(parents=True)
    if problem == "imports":
        fake_python(monkeypatch, tmp_path, "exit 1")
    elif problem == "boundaries":
        # Imports work, and the boundaries check fails
        fake_python(monkeypatch, tmp_path, '[ "$2" = "" ] && exit 0\nexit 3')
    else:
        fake_python(monkeypatch, tmp_path, "exit 0")
    script = _MeshBehavior._check_prompting.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (done.returncode == 0) == (problem is None)
    assert (behavior._geo_prefix / ".installed").is_file() == (problem is None)
    if problem == "imports":
        assert "Install chiltepin[mpas,llm]" in done.stderr


@pytest.mark.parametrize("complete", [True, False])
def test_prompting_check_with_the_real_packages(behavior, complete):
    # The tests have both extras, so the real check runs, and finds the
    # boundaries complete or not
    write_boundaries(behavior)
    if not complete:
        path = layer_path(behavior._boundaries_dir, BOUNDARY_LAYERS[1])
        path.with_suffix(".dbf").unlink()
    script = _MeshBehavior._check_prompting.func(behavior)
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert (done.returncode == 0) == complete, done.stderr
    assert (behavior._geo_prefix / ".installed").is_file() == complete
    if not complete:
        assert "Boundaries are missing" in done.stderr
        assert "admin_1_states_provinces" in done.stderr


# Installing


async def test_install_prompting_runs_both_steps_on_service(monkeypatch, behavior):
    download, check = mock_prompting_tasks(monkeypatch, behavior)
    await behavior.install_prompting()
    assert behavior.prompting_built
    download.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "prompt_download.stdout"),
        stderr=str(behavior.log_dir / "prompt_download.stderr"),
    )
    check.assert_called_once_with(
        executor=["service"],
        stdout=str(behavior.log_dir / "prompt_check.stdout"),
        stderr=str(behavior.log_dir / "prompt_check.stderr"),
        inputs=[download.return_value],
    )


@pytest.mark.parametrize("problem", ["marker", "layer"])
async def test_install_prompting_replaces_an_unfinished_install(
    monkeypatch, behavior, problem
):
    # An install without a marker, or made before a boundary layer was added,
    # is replaced
    mark_prompting_installed(behavior)
    if problem == "marker":
        (behavior._geo_prefix / ".installed").unlink()
    else:
        path = layer_path(behavior._boundaries_dir, BOUNDARY_LAYERS[0])
        path.with_suffix(".shp").unlink()
    download, check = mock_prompting_tasks(monkeypatch, behavior)
    await behavior.install_prompting()
    download.assert_called_once()
    check.assert_called_once()


async def test_install_prompting_reuses_a_finished_install(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    download, check = mock_prompting_tasks(monkeypatch, behavior)
    await behavior.install_prompting()
    assert behavior.prompting_built
    download.assert_not_called()
    check.assert_not_called()


async def test_prompting_check_that_leaves_no_install_is_an_error(
    monkeypatch, behavior
):
    monkeypatch.setattr(
        behavior, "_download_boundaries", Mock(return_value=done_future())
    )
    monkeypatch.setattr(behavior, "_check_prompting", Mock(return_value=done_future()))
    with pytest.raises(RuntimeError, match="is not usable.*prompt_check.stderr"):
        await behavior.install_prompting()
    assert not behavior.prompting_built


async def test_install_prompting_needs_service(monkeypatch, behavior):
    define_resources(monkeypatch, "compute")
    download, _ = mock_prompting_tasks(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match="'service'"):
        await behavior.install_prompting()
    download.assert_not_called()


async def test_concurrent_prompting_installs_run_once(monkeypatch, behavior):
    pending = Future()
    download, _ = mock_prompting_tasks(monkeypatch, behavior)
    download.return_value = pending
    requests = [asyncio.create_task(behavior.install_prompting()) for _ in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    pending.set_result(None)
    await asyncio.gather(*requests)
    assert download.call_count == 1


# Reading a prompt on service


@pytest.fixture
def stand_ins(monkeypatch):
    """Replace the model and the request builder, recording their arguments.

    Set ``failures`` to errors the request builder raises, in turn, before it
    returns the request.
    """
    calls = {"ask": [], "failures": []}

    def ask(prompt, model, resolutions, reasoning_effort=None, correction=None):
        calls["ask"].append((prompt, model, resolutions, reasoning_effort, correction))
        return f"answer {len(calls['ask'])}"

    def mesh_config(answer, downloadable, data_dir, buffer_km, **kwargs):
        calls["mesh_config"] = (answer, downloadable, data_dir, buffer_km, kwargs)
        if calls["failures"]:
            raise calls["failures"].pop(0)
        return dict(CONFIG)

    monkeypatch.setattr(_mesh_prompt, "ask", ask)
    monkeypatch.setattr(_mesh_prompt, "mesh_config", mesh_config)
    return calls


def test_read_prompt(tmp_path, stand_ins):
    behavior = _MeshBehavior(str(tmp_path), nominatim_contact="me@example.org")
    result = _MeshBehavior._read_prompt.func(behavior, "Texas", MODEL, 75.0, "low")
    assert result == {"mesh_config": CONFIG}
    resolutions = list(behavior.resolution_cells)
    assert stand_ins["ask"] == [("Texas", MODEL, resolutions, "low", None)]
    assert stand_ins["mesh_config"] == (
        "answer 1",
        resolutions,
        behavior._boundaries_dir,
        75.0,
        {
            "nominatim_contact": "me@example.org",
            "cache_dir": behavior._nominatim_cache_dir,
            "prompt": "Texas",
        },
    )


def test_read_prompt_asks_again_when_places_are_not_found(behavior, stand_ins):
    stand_ins["failures"] = [RegionNotFound("No place is named 'Florida'")]
    result = _MeshBehavior._read_prompt.func(behavior, "Florida", MODEL, 50.0, "none")
    assert result == {"mesh_config": CONFIG}
    # The model is told its answer and why it could not be used
    first, second = stand_ins["ask"]
    assert first[4] is None
    assert second[3] == "none"
    assert second[4] == ("answer 1", "No place is named 'Florida'")
    assert stand_ins["mesh_config"][0] == "answer 2"


def test_read_prompt_asks_again_only_once(behavior, stand_ins):
    stand_ins["failures"] = [RegionNotFound("first"), RegionNotFound("second")]
    result = _MeshBehavior._read_prompt.func(behavior, "Florida", MODEL, 50.0)
    assert result == {"error": "second"}
    assert len(stand_ins["ask"]) == 2


def test_read_prompt_does_not_ask_again_for_other_problems(behavior, stand_ins):
    # Such as a region too small for its cells, which other places cannot fix
    stand_ins["failures"] = [ValueError("Ask for smaller cells")]
    result = _MeshBehavior._read_prompt.func(behavior, "Colorado", MODEL, 50.0)
    assert result == {"error": "Ask for smaller cells"}
    assert len(stand_ins["ask"]) == 1


def test_read_prompt_errors_hold_no_credentials(monkeypatch, behavior, stand_ins):
    monkeypatch.setenv("SOME_API_KEY", "key-123456789")
    # A prompt that cannot be made into a mesh is returned as its reason
    stand_ins["failures"] = [ValueError("No place is named key-123456789")]
    result = _MeshBehavior._read_prompt.func(behavior, "Texas", MODEL, 50.0)
    assert result == {"error": "No place is named <redacted>"}
    # Anything else is raised, without the original error
    stand_ins["failures"] = [OSError("unreachable with key-123456789")]
    with pytest.raises(RuntimeError) as info:
        _MeshBehavior._read_prompt.func(behavior, "Texas", MODEL, 50.0)
    assert str(info.value) == "OSError: unreachable with <redacted>"
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


# The actions


def fake_read_prompt(monkeypatch, behavior, result=None, error=None):
    """Replace the prompt task with one returning ``result``, or raising ``error``.

    ``result`` is by default the request of CONFIG.
    """
    result = result or {"mesh_config": dict(CONFIG)}
    task = Mock(return_value=done_future(result, error))
    monkeypatch.setattr(behavior, "_read_prompt", task)
    return task


async def test_mesh_config_from_prompt(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    task = fake_read_prompt(monkeypatch, behavior)
    assert await behavior.mesh_config_from_prompt("Texas", MODEL, 75) == CONFIG
    task.assert_called_once_with("Texas", MODEL, 75.0, None, executor=["service"])


async def test_mesh_config_from_prompt_passes_the_reasoning_effort(
    monkeypatch, behavior
):
    mark_prompting_installed(behavior)
    task = fake_read_prompt(monkeypatch, behavior)
    await behavior.mesh_config_from_prompt("Texas", MODEL, reasoning_effort="none")
    task.assert_called_once_with("Texas", MODEL, 50.0, "none", executor=["service"])


async def test_mesh_config_from_prompt_installs_prompting_first(monkeypatch, behavior):
    download, _ = mock_prompting_tasks(monkeypatch, behavior)
    fake_read_prompt(monkeypatch, behavior)
    await behavior.mesh_config_from_prompt("Texas", MODEL)
    download.assert_called_once()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("", MODEL), "prompt must be a non-empty string"),
        (("  ", MODEL), "prompt must be a non-empty string"),
        ((5, MODEL), "prompt must be a non-empty string"),
        (("x" * 10001, MODEL), "at most 10000 characters"),
        (("Texas", ""), "model must be a litellm model string"),
        (("Texas", "a model"), "model must be a litellm model string"),
        (("Texas", "model\x00"), "model must be a litellm model string"),
        (("Texas", None), "model must be a litellm model string"),
        (("Texas", MODEL, "50"), "buffer_km must be a number"),
        (("Texas", MODEL, True), "buffer_km must be a number"),
        (("Texas", MODEL, -1), "buffer_km must be from 0 to 5000"),
        (("Texas", MODEL, 5001), "buffer_km must be from 0 to 5000"),
        (("Texas", MODEL, float("nan")), "buffer_km must be from 0 to 5000"),
        (("Texas", MODEL, 50, ""), "reasoning_effort must be None or a word"),
        (("Texas", MODEL, 50, "None; x"), "reasoning_effort must be None or a word"),
        (("Texas", MODEL, 50, 1), "reasoning_effort must be None or a word"),
    ],
)
async def test_mesh_config_from_prompt_rejects(
    monkeypatch, behavior, arguments, message
):
    task = fake_read_prompt(monkeypatch, behavior)
    with pytest.raises(ValueError, match=message):
        await behavior.mesh_config_from_prompt(*arguments)
    task.assert_not_called()


async def test_prompt_that_cannot_be_made_into_a_mesh(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    fake_read_prompt(monkeypatch, behavior, {"error": "No place is named 'X'"})
    with pytest.raises(
        ValueError, match="cannot be made into a mesh: No place is named 'X'"
    ) as info:
        await behavior.mesh_config_from_prompt("X", MODEL)
    assert info.value.__cause__ is None


async def test_prompt_that_cannot_be_read(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    fake_read_prompt(monkeypatch, behavior, error=RuntimeError("model refused"))
    with pytest.raises(RuntimeError, match="Reading the prompt failed: model refused"):
        await behavior.mesh_config_from_prompt("Texas", MODEL)


async def test_invalid_request_from_a_prompt_is_rejected(monkeypatch, behavior):
    # The request is checked as any other is, whatever the task returned
    mark_prompting_installed(behavior)
    fake_read_prompt(monkeypatch, behavior, {"mesh_config": {"resolution": "7km"}})
    with pytest.raises(ValueError, match="not supported"):
        await behavior.mesh_config_from_prompt("Texas", MODEL)


async def test_mesh_config_from_prompt_needs_service(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    define_resources(monkeypatch, "compute")
    task = fake_read_prompt(monkeypatch, behavior)
    with pytest.raises(RuntimeError, match="'service'"):
        await behavior.mesh_config_from_prompt("Texas", MODEL)
    task.assert_not_called()


def fake_full_generate(monkeypatch, behavior):
    """Make only full requests with a mock, leaving prompts to generate_mesh."""
    generate_mesh = behavior.generate_mesh
    made = Mock(return_value=done_future({"static": "/m/texas.static.nc"}))

    async def generate(mesh_config, mesh_data_dir):
        if "prompt" in mesh_config:
            return await generate_mesh(mesh_config, mesh_data_dir)
        return await asyncio.wrap_future(made(mesh_config, mesh_data_dir))

    monkeypatch.setattr(behavior, "generate_mesh", generate)
    return made


@pytest.mark.parametrize(
    ("options", "changes"),
    [
        ({}, {}),
        ({"name": "lone_star"}, {"name": "lone_star"}),
        ({"plot": True}, {"plot": True}),
        ({"plot": False}, {"plot": False}),
        (
            {"init_ranks": 4, "forecast_ranks": 36},
            {"init_ranks": 4, "forecast_ranks": 36},
        ),
    ],
)
async def test_generate_mesh_from_a_prompt(monkeypatch, behavior, options, changes):
    mark_prompting_installed(behavior)
    task = fake_read_prompt(monkeypatch, behavior)
    made = fake_full_generate(monkeypatch, behavior)
    request = {
        "prompt": "Texas",
        "model": MODEL,
        "buffer_km": 60,
        "reasoning_effort": "low",
        **options,
    }
    result = await behavior.generate_mesh(request, "/meshes")
    expected = {**CONFIG, **changes}
    # What generate_mesh returns, with the request the prompt became, which
    # makes the same mesh without the prompt
    assert result == {"static": "/m/texas.static.nc", "mesh_config": expected}
    made.assert_called_once_with(expected, "/meshes")
    task.assert_called_once_with("Texas", MODEL, 60.0, "low", executor=["service"])


async def test_generate_mesh_from_a_prompt_by_default(monkeypatch, behavior):
    mark_prompting_installed(behavior)
    task = fake_read_prompt(monkeypatch, behavior)
    fake_full_generate(monkeypatch, behavior)
    await behavior.generate_mesh({"prompt": "Texas", "model": MODEL}, "/meshes")
    task.assert_called_once_with("Texas", MODEL, 50.0, None, executor=["service"])


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        (
            {"prompt": "Texas", "model": MODEL, "resolution": "15km"},
            "with a prompt cannot have resolution, which the prompt gives",
        ),
        (
            {"prompt": "Texas", "model": MODEL, "regional": {}, "resolution": "1km"},
            "cannot have regional or resolution",
        ),
        ({"prompt": "Texas", "model": MODEL, "size": 3}, "Unknown mesh_config keys"),
        ({"prompt": "Texas"}, "with a prompt needs a model"),
        ({"prompt": "", "model": MODEL}, "prompt must be a non-empty string"),
        ({"prompt": "Texas", "model": "a b"}, "model must be a litellm"),
        ({"prompt": "Texas", "model": MODEL, "buffer_km": -1}, "buffer_km must be"),
        (
            {"prompt": "Texas", "model": MODEL, "reasoning_effort": "LOTS"},
            "reasoning_effort must be",
        ),
        ({"prompt": "Texas", "model": MODEL, "name": "a/b"}, "valid file name"),
        ({"prompt": "Texas", "model": MODEL, "init_ranks": 0}, "init_ranks must be"),
        ({"prompt": "Texas", "model": MODEL, "plot": "yes"}, "plot must be True"),
    ],
)
async def test_generate_mesh_rejects_prompt_requests(
    monkeypatch, behavior, request_, message
):
    task = fake_read_prompt(monkeypatch, behavior)
    with pytest.raises(ValueError, match=message):
        await behavior.generate_mesh(request_, "/meshes")
    task.assert_not_called()


@pytest.mark.parametrize(
    ("request_", "installed"),
    [
        ({}, ["prompting"]),
        ({"init_ranks": 4, "plot": True}, ["prompting", "metis", "plotting"]),
    ],
)
async def test_tools_for_a_prompt_request(monkeypatch, behavior, request_, installed):
    # The program for the prompt's region is known only once it is read
    calls = []
    for tool in ("prompting", "metis", "plotting", "mpas_tools", "limited_area"):

        async def install(tool=tool):
            calls.append(tool)

        monkeypatch.setattr(behavior, f"install_{tool}", install)
    task = fake_read_prompt(monkeypatch, behavior)
    request = {"prompt": "Texas", "model": MODEL, **request_}
    await behavior.ensure_tools_installed(request)
    assert calls == installed
    task.assert_not_called()


def test_prompting_paths(behavior):
    assert behavior._geo_prefix == behavior.work_dir / "geo"
    assert behavior._boundaries_dir == behavior.work_dir / "geo" / "natural_earth"
    assert behavior._nominatim_cache_dir == behavior.work_dir / "geo" / "nominatim"
    assert mesh_module._MAX_PROMPT_CHARS == 10_000
