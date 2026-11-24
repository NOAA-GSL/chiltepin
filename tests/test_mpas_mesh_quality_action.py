# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh agent's check_mesh_quality action, with its task stood in.

tests/test_mpas_mesh_quality.py tests the checks, and the real integration test
checks a real mesh through a launched agent.
"""

import json
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest
from netCDF4 import Dataset

from chiltepin.agents.mpas import _mesh_quality
from chiltepin.agents.mpas import mesh as mesh_module
from chiltepin.agents.mpas.mesh import _MeshBehavior

REPORT = {
    "passed": True,
    "summary": "Passed with 0 error(s) and 1 warning(s)",
    "errors": [],
    "warnings": ["Something to weigh"],
    "measurements": {"kind": "global", "cells": 2562},
}


@pytest.fixture(autouse=True)
def no_resource_check(monkeypatch):
    """Skip the resource check. Tests of the check set the resources they want."""
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: None)
    )


@pytest.fixture
def behavior(tmp_path):
    return _MeshBehavior(str(tmp_path / "work"))


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "meshes" / "conus.static.nc"
    path.parent.mkdir()
    path.write_bytes(b"not read here")
    return path


def done_future(result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def fake_check(monkeypatch, behavior, result=None, error=None):
    """Replace the task with one returning ``result``, or raising ``error``."""
    task = Mock(
        return_value=done_future(
            result if result is not None else {"report": dict(REPORT)}, error
        )
    )
    monkeypatch.setattr(behavior, "_check_quality", task)
    return task


async def test_check_mesh_quality(monkeypatch, behavior, mesh):
    task = fake_check(monkeypatch, behavior)
    report = await behavior.check_mesh_quality(str(mesh))
    task.assert_called_once_with(str(mesh), None, executor=["compute"])
    expected = mesh.with_name("conus.quality.json")
    assert report == {**REPORT, "report_file": str(expected)}
    assert json.loads(expected.read_text()) == report
    assert sorted(p.name for p in mesh.parent.iterdir()) == [
        "conus.quality.json",
        "conus.static.nc",
    ]


async def test_resolution_is_passed_on(monkeypatch, behavior, mesh):
    task = fake_check(monkeypatch, behavior)
    await behavior.check_mesh_quality(str(mesh), 15)
    task.assert_called_once_with(str(mesh), 15.0, executor=["compute"])


async def test_relative_and_home_paths_are_made_absolute(
    monkeypatch, behavior, mesh, tmp_path
):
    task = fake_check(monkeypatch, behavior)
    monkeypatch.chdir(tmp_path)
    await behavior.check_mesh_quality("meshes/conus.static.nc")
    assert task.call_args.args[0] == str(mesh)
    monkeypatch.setenv("HOME", str(tmp_path))
    await behavior.check_mesh_quality("~/meshes/conus.static.nc")
    assert task.call_args.args[0] == str(mesh)


async def test_a_new_report_replaces_the_old_one(monkeypatch, behavior, mesh):
    mesh.with_name("conus.quality.json").write_text("old")
    fake_check(monkeypatch, behavior)
    report = await behavior.check_mesh_quality(str(mesh))
    assert json.loads(mesh.with_name("conus.quality.json").read_text()) == report


async def test_report_names_follow_the_plot_names(monkeypatch, behavior, tmp_path):
    fake_check(monkeypatch, behavior)
    for name, report in (
        ("x1.2562.grid.nc", "x1.2562.grid.quality.json"),
        ("plain", "plain.quality.json"),
    ):
        path = tmp_path / name
        path.write_bytes(b"")
        result = await behavior.check_mesh_quality(str(path))
        assert result["report_file"] == str(tmp_path / report)


async def test_the_report_is_returned_if_it_cannot_be_written(
    monkeypatch, behavior, mesh
):
    fake_check(monkeypatch, behavior)
    logger = Mock()
    monkeypatch.setattr(mesh_module, "logger", logger)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "write_text", refuse)
    report = await behavior.check_mesh_quality(str(mesh))
    assert report["report_file"] is None and report["passed"]
    assert "Could not write the quality report" in logger.warning.call_args.args[0]
    assert [p.name for p in mesh.parent.iterdir()] == ["conus.static.nc"]


async def test_summary_errors_and_warnings_are_logged(monkeypatch, behavior, mesh):
    fake_check(
        monkeypatch,
        behavior,
        {"report": {**REPORT, "errors": ["Broken"], "passed": False}},
    )
    logger = Mock()
    monkeypatch.setattr(mesh_module, "logger", logger)
    report = await behavior.check_mesh_quality(str(mesh))
    assert not report["passed"]
    lines = [call.args[2] for call in logger.info.call_args_list]
    assert lines == [REPORT["summary"], "error: Broken", "warning: Something to weigh"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("",), "mesh_path must be a non-empty string"),
        ((5,), "mesh_path must be a non-empty string"),
        ((None,), "mesh_path must be a non-empty string"),
        (("m.nc", "15"), "resolution_km must be a finite number"),
        (("m.nc", True), "resolution_km must be a finite number"),
        (("m.nc", 0), "resolution_km must be greater than 0"),
        (("m.nc", -3.0), "resolution_km must be greater than 0"),
        (("m.nc", float("inf")), "resolution_km must be a finite number"),
    ],
)
async def test_invalid_arguments(monkeypatch, behavior, arguments, message):
    task = fake_check(monkeypatch, behavior)
    with pytest.raises(ValueError, match=message):
        await behavior.check_mesh_quality(*arguments)
    task.assert_not_called()


async def test_a_missing_file(monkeypatch, behavior, tmp_path):
    task = fake_check(monkeypatch, behavior)
    with pytest.raises(FileNotFoundError, match="No mesh file at"):
        await behavior.check_mesh_quality(str(tmp_path / "none.nc"))
    task.assert_not_called()


async def test_the_compute_resource_is_needed(monkeypatch, behavior, mesh):
    task = fake_check(monkeypatch, behavior)
    monkeypatch.setattr(
        _MeshBehavior, "_configured_resources", staticmethod(lambda: {"service"})
    )
    with pytest.raises(RuntimeError, match="Checking mesh quality needs the resource"):
        await behavior.check_mesh_quality(str(mesh))
    task.assert_not_called()


async def test_a_file_that_is_not_a_mesh_is_a_value_error(monkeypatch, behavior, mesh):
    fake_check(monkeypatch, behavior, {"error": "it lacks x", "invalid": True})
    with pytest.raises(ValueError, match="Cannot check .*conus.static.nc: it lacks x"):
        await behavior.check_mesh_quality(str(mesh))
    assert not mesh.with_name("conus.quality.json").exists()


async def test_a_task_that_cannot_run_is_a_runtime_error(monkeypatch, behavior, mesh):
    fake_check(monkeypatch, behavior, {"error": "no numpy", "invalid": False})
    with pytest.raises(RuntimeError, match="Cannot check .*: no numpy"):
        await behavior.check_mesh_quality(str(mesh))
    fake_check(monkeypatch, behavior, error=OSError("node lost"))
    with pytest.raises(RuntimeError, match="Checking the mesh failed: node lost"):
        await behavior.check_mesh_quality(str(mesh))
    assert not mesh.with_name("conus.quality.json").exists()


def test_the_task_returns_the_report(monkeypatch, behavior, tmp_path):
    check = Mock(return_value=dict(REPORT))
    monkeypatch.setattr(_mesh_quality, "check_mesh", check)
    result = _MeshBehavior._check_quality.func(behavior, "/m.nc", 12.0)
    assert result == {"report": REPORT}
    check.assert_called_once_with("/m.nc", 12.0)


def test_the_task_returns_why_a_file_cannot_be_checked(monkeypatch, behavior, tmp_path):
    bare = tmp_path / "bare.nc"
    with Dataset(str(bare), "w") as ds:
        ds.createDimension("n", 1)
    result = _MeshBehavior._check_quality.func(behavior, str(bare), None)
    assert result["invalid"] is True
    assert "not an MPAS mesh" in result["error"]
    monkeypatch.setattr(
        _mesh_quality,
        "check_mesh",
        Mock(side_effect=ImportError("No module named 'numpy'")),
    )
    result = _MeshBehavior._check_quality.func(behavior, "/m.nc", None)
    assert result["invalid"] is False
    assert "No module named 'numpy'. Install chiltepin[mpas]" in result["error"]
