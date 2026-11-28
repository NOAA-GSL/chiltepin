# SPDX-License-Identifier: Apache-2.0
"""Tests of making a global mesh at a resolution that is not downloaded.

The programs are stood in by tasks that write the files they would, so the
requests, the files that are left, the record and every failure are tested
without them. The real integration test makes a mesh with the tools.
"""

import json
import shutil
import subprocess
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from netCDF4 import Dataset  # noqa: F401  (the mpas extra is needed to run these)
from parsl.app.errors import BashExitFailure

from chiltepin.agents.mpas import _global_mesh as global_mesh
from chiltepin.agents.mpas import mesh as mesh_module
from chiltepin.agents.mpas.mesh import _MeshBehavior

COMMITS = {"jigsaw": "1" * 40, "mkgrid": "2" * 40, "sphtri_subdiv": "3" * 40}
REPORT = {
    "passed": True,
    "summary": "Passed with 0 error(s) and 0 warning(s)",
    "errors": [],
    "warnings": [],
    "measurements": {"cells": 11067042},
    "report_file": "unused",
}
TETRAHEDRON = """\
MSHID=3;EUCLIDEAN-MESH
NDIMS=3
POINT=4
1.5;0.25;-3.125;0
-1.0;2.0e-3;0.5;0
0.0;-1.0;0.125;0
2.0;1.0;1.0;0
TRIA3=4
0;1;2;0
0;2;3;0
0;3;1;0
1;3;2;0
"""


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


def done_future(result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def mark_tools_installed(behavior):
    """Create what finished installs of the three tools leave in work_dir."""
    for name, prefix, files in (
        ("jigsaw", behavior._jigsaw_prefix, ["bin/jigsaw"]),
        (
            "mkgrid",
            behavior._mkgrid_prefix,
            ["bin/mkgrid", "share/ICOS.msh", "share/GEOM.msh"],
        ),
        ("sphtri_subdiv", behavior._sphtri_subdiv_prefix, ["bin/sphtri_subdiv"]),
    ):
        for file in files:
            path = prefix / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name} {file}\n")
            if file.startswith("bin/"):
                path.chmod(0o755)
        (prefix / ".installed").write_text(COMMITS[name] + "\n")


@pytest.fixture
def behavior(tmp_path, monkeypatch):
    agent = _MeshBehavior(str(tmp_path / "work"))
    mark_tools_installed(agent)
    monkeypatch.setattr(agent, "install_global_mesh_tools", AsyncMock())
    return agent


class Pipeline:
    """Stand-ins for the programs of the pipeline, which record how they are called."""

    def __init__(self, monkeypatch, behavior, report=None):
        self.calls = []
        self.fail = {}  # step -> an error its future raises
        self.skip = set()  # steps that leave out their output
        self.prepared = {"points": 11067042, "triangles": 22134080}
        behavior_ = behavior
        self.report = dict(report or REPORT)

        def sphtri(mesh_dir, parts, **kwargs):
            self.calls.append(("sphtri_subdiv", mesh_dir, parts, kwargs))
            if "sphtri_subdiv" not in self.skip:
                (Path(mesh_dir) / "INIT.msh").write_text("init\n")
            return done_future(error=self.fail.get("sphtri_subdiv"))

        def jigsaw(mesh_dir, **kwargs):
            self.calls.append(("jigsaw", mesh_dir, kwargs))
            if "jigsaw" not in self.skip:
                (Path(mesh_dir) / "MESH.msh").write_text(TETRAHEDRON)
            return done_future(error=self.fail.get("jigsaw"))

        def prepare(mesh_dir, requested, parts, spacing, tools, **kwargs):
            self.calls.append(("prepare", mesh_dir, requested, parts, spacing, tools))
            for name in ("SaveVertices", "SaveTriangles", "SaveDensity"):
                (Path(mesh_dir) / name).write_text("data\n")
            return done_future(self.prepared, self.fail.get("prepare"))

        def mkgrid(mesh_dir, spacing_m, **kwargs):
            self.calls.append(("mkgrid", mesh_dir, spacing_m, kwargs))
            for name in ("grid.nc", "graph.info"):
                if name not in self.skip:
                    (Path(mesh_dir) / name).write_text(f"{name}\n")
            return done_future(error=self.fail.get("mkgrid"))

        async def quality(path, resolution_km=None):
            self.calls.append(("quality", path, resolution_km))
            # The report is written next to the mesh, as the action writes it
            report_file = _MeshBehavior._quality_path(Path(path))
            report = {**self.report, "report_file": str(report_file)}
            report_file.write_text(json.dumps(report))
            return report

        monkeypatch.setattr(behavior_, "_run_sphtri_subdiv", Mock(side_effect=sphtri))
        monkeypatch.setattr(behavior_, "_run_jigsaw", Mock(side_effect=jigsaw))
        monkeypatch.setattr(
            behavior_, "_prepare_mkgrid_inputs", Mock(side_effect=prepare)
        )
        monkeypatch.setattr(behavior_, "_run_mkgrid", Mock(side_effect=mkgrid))
        monkeypatch.setattr(behavior_, "check_mesh_quality", quality)

    def steps(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def pipeline(monkeypatch, behavior):
    return Pipeline(monkeypatch, behavior)


CONFIG = {"resolution": "7.3km"}
PARTS = 1052


async def test_a_mesh_is_made_with_the_tools(behavior, pipeline, tmp_path):
    meshes = tmp_path / "meshes"
    result = await behavior.generate_mesh(CONFIG, str(meshes))
    mesh_dir = meshes / "mesh_7.3km"
    spacing = 7680 / PARTS
    assert pipeline.steps() == [
        "sphtri_subdiv",
        "jigsaw",
        "prepare",
        "mkgrid",
        "quality",
    ]
    # The icosahedron is cut into one part fewer than the mesh has
    sphtri = pipeline.calls[0]
    assert sphtri[1:3] == (str(mesh_dir), PARTS - 1)
    assert sphtri[3]["executor"] == ["compute"]
    assert pipeline.calls[1][1] == str(mesh_dir)
    assert pipeline.calls[2][1:] == (
        str(mesh_dir),
        7.3,
        PARTS,
        spacing,
        COMMITS,
    )
    # mkgrid runs on the mpi resource, given the nominal spacing in meters
    mkgrid = pipeline.calls[3]
    assert mkgrid[2] == pytest.approx(spacing * 1000.0)
    assert mkgrid[3]["executor"] == ["mpi"]
    assert mkgrid[3]["chiltepin_task_geometry"] == {
        "num_nodes": 1,
        "num_ranks": 4,
        "ranks_per_node": 4,
    }
    assert pipeline.calls[4][1:] == (str(mesh_dir / "mesh_7.3km.static.nc"), spacing)
    assert result["static"] == str(mesh_dir / "mesh_7.3km.static.nc")
    assert result["graph"] == str(mesh_dir / "mesh_7.3km.graph.info")
    assert result["quality"] == {
        **REPORT,
        "report_file": str(mesh_dir / "mesh_7.3km.quality.json"),
    }
    assert result["generated"] == {
        "requested_resolution_km": 7.3,
        "division": PARTS,
        "nominal_spacing_km": spacing,
        "cells": 11067042,
        "expected_cells": 11067042,
        "tools": COMMITS,
    }
    assert "reused" not in result


async def test_the_directory_keeps_only_what_the_mesh_needs(
    behavior, pipeline, tmp_path
):
    await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    mesh_dir = tmp_path / "meshes" / "mesh_7.3km"
    assert sorted(path.name for path in mesh_dir.iterdir()) == [
        "GEOM.msh",
        "ICOS.msh",
        "MESH.jig",
        "mesh_7.3km.graph.info",
        "mesh_7.3km.mesh.json",
        "mesh_7.3km.quality.json",
        "mesh_7.3km.static.nc",
    ]
    # The sphere and the icosahedron are those the install keeps
    assert (mesh_dir / "ICOS.msh").read_text() == "mkgrid share/ICOS.msh\n"
    assert (mesh_dir / "GEOM.msh").read_text() == "mkgrid share/GEOM.msh\n"
    assert (mesh_dir / "MESH.jig").read_text() == global_mesh.jig_text()


async def test_the_size_file_and_configuration_are_written_before_the_programs(
    behavior, pipeline, tmp_path
):
    seen = {}
    original = behavior._run_jigsaw.side_effect

    def jigsaw(mesh_dir, **kwargs):
        seen["jig"] = (Path(mesh_dir) / "MESH.jig").read_text()
        seen["hfun"] = (Path(mesh_dir) / "HFUN.msh").read_text()
        return original(mesh_dir, **kwargs)

    behavior._run_jigsaw.side_effect = jigsaw
    await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert seen["jig"] == global_mesh.jig_text()
    assert seen["hfun"] == global_mesh.hfun_text(7680 / PARTS)


async def test_the_record_holds_how_the_mesh_was_made(behavior, pipeline, tmp_path):
    await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    record = json.loads(
        (tmp_path / "meshes" / "mesh_7.3km" / "mesh_7.3km.mesh.json").read_text()
    )
    assert record["mesh"] == {
        "resolution": "7.3km",
        "regional": None,
        "division": PARTS,
        "jigsaw_version": behavior.jigsaw_version,
        "mkgrid_version": "feature/nesting",
        "mesh_tools_version": behavior.mesh_tools_version,
    }
    assert record["generated"]["tools"] == COMMITS


async def test_a_named_mesh(behavior, pipeline, tmp_path):
    result = await behavior.generate_mesh(
        {"resolution": "7.3km", "name": "global_7"}, str(tmp_path / "meshes")
    )
    assert result["static"].endswith("/global_7/global_7.static.nc")


@pytest.mark.parametrize(
    "ranks, per_node, geometry",
    [
        (4, None, (1, 4, 4)),
        (8, 4, (2, 8, 4)),
        (10, 4, (3, 10, 4)),
        (1, None, (1, 1, 1)),
    ],
)
async def test_mkgrid_is_run_on_the_ranks_asked_for(
    tmp_path, monkeypatch, ranks, per_node, geometry
):
    agent = _MeshBehavior(
        str(tmp_path / "work"), mkgrid_ranks=ranks, mkgrid_ranks_per_node=per_node
    )
    mark_tools_installed(agent)
    monkeypatch.setattr(agent, "install_global_mesh_tools", AsyncMock())
    pipeline = Pipeline(monkeypatch, agent)
    await agent.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    nodes, total, each = geometry
    assert pipeline.calls[3][3]["chiltepin_task_geometry"] == {
        "num_nodes": nodes,
        "num_ranks": total,
        "ranks_per_node": each,
    }


async def test_the_tools_are_installed_first(behavior, pipeline, tmp_path):
    await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    behavior.install_global_mesh_tools.assert_awaited()


async def test_a_resolution_ucar_has_is_downloaded_not_made(
    behavior, pipeline, tmp_path, monkeypatch
):
    download = AsyncMock(
        return_value={"static": "x1.static.nc", "graph": "x1.graph.info"}
    )
    monkeypatch.setattr(behavior, "download_global_mesh", download)
    # A symlink to files that do not exist is enough for the request
    (tmp_path / "x1.static.nc").write_text("s")
    (tmp_path / "x1.graph.info").write_text("g")
    download.return_value = {
        "static": str(tmp_path / "x1.static.nc"),
        "graph": str(tmp_path / "x1.graph.info"),
    }
    result = await behavior.generate_mesh({"resolution": "10km"}, str(tmp_path / "m"))
    download.assert_awaited_once()
    assert pipeline.calls == []
    assert "generated" not in result and "quality" not in result


# Failures


@pytest.mark.parametrize(
    "step, program",
    [
        ("sphtri_subdiv", "sphtri_subdiv"),
        ("jigsaw", "jigsaw"),
        ("mkgrid", "mkgrid"),
    ],
)
async def test_a_failed_program_fails_the_request_and_frees_the_name(
    behavior, pipeline, tmp_path, step, program
):
    meshes = tmp_path / "meshes"
    pipeline.fail[step] = BashExitFailure(step, 3)
    with pytest.raises(RuntimeError, match=f"{program} failed \\(exit 3\\)") as info:
        await behavior.generate_mesh(CONFIG, str(meshes))
    assert f"{program}." in str(info.value) and ".stderr" in str(info.value)
    assert not (meshes / "mesh_7.3km").exists()
    # The same request works once the program does
    pipeline.fail.clear()
    result = await behavior.generate_mesh(CONFIG, str(meshes))
    assert Path(result["static"]).is_file()


@pytest.mark.parametrize(
    "step, missing",
    [
        ("sphtri_subdiv", "INIT.msh"),
        ("jigsaw", "MESH.msh"),
        ("grid.nc", "grid.nc"),
        ("graph.info", "graph.info"),
    ],
)
async def test_a_program_that_writes_nothing_is_an_error(
    behavior, pipeline, tmp_path, step, missing
):
    pipeline.skip.add(step)
    with pytest.raises(RuntimeError, match=f"did not write {missing}"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert not (tmp_path / "meshes" / "mesh_7.3km").exists()


async def test_inputs_that_cannot_be_prepared_are_an_error(
    behavior, pipeline, tmp_path
):
    pipeline.fail["prepare"] = OSError("disk full")
    with pytest.raises(RuntimeError, match="Preparing the inputs of mkgrid failed"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert pipeline.steps() == ["sphtri_subdiv", "jigsaw", "prepare"]


async def test_a_jigsaw_mesh_that_is_not_a_sphere_is_an_error(
    behavior, pipeline, tmp_path
):
    pipeline.prepared = {"error": "has 3 of 4 POINT lines"}
    with pytest.raises(RuntimeError, match="JIGSAW's mesh cannot be used: has 3 of 4"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert "mkgrid" not in pipeline.steps()


async def test_a_mesh_with_errors_in_its_report_is_not_kept(
    behavior, pipeline, tmp_path
):
    pipeline.report = {
        **REPORT,
        "passed": False,
        "errors": ["The sum of (6 - sides) over cells is 13", "areas disagree"],
    }
    with pytest.raises(
        RuntimeError,
        match=r"has errors: The sum of \(6 - sides\) over cells is 13; areas disagree",
    ):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert not (tmp_path / "meshes" / "mesh_7.3km").exists()


async def test_a_mesh_with_warnings_is_kept(behavior, pipeline, tmp_path):
    pipeline.report = {**REPORT, "warnings": ["12 pentagons expected"]}
    result = await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert result["quality"]["warnings"] == ["12 pentagons expected"]


async def test_a_failed_install_fails_the_request(behavior, pipeline, tmp_path):
    behavior.install_global_mesh_tools.side_effect = RuntimeError("JIGSAW build failed")
    with pytest.raises(RuntimeError, match="JIGSAW build failed"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert pipeline.calls == []
    assert not (tmp_path / "meshes" / "mesh_7.3km").exists()


@pytest.mark.parametrize("missing", ["service", "compute", "mpi"])
async def test_the_resources_are_checked_before_anything_is_made(
    monkeypatch, behavior, pipeline, tmp_path, missing
):
    resources = {"service", "compute", "mpi", "_parsl_internal"} - {missing}
    define_resources(monkeypatch, *resources)
    with pytest.raises(RuntimeError, match=f"'{missing}'"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert pipeline.calls == [] and not (tmp_path / "meshes").exists()


async def test_downloading_a_ucar_mesh_does_not_need_mpi(
    monkeypatch, behavior, pipeline, tmp_path
):
    define_resources(monkeypatch, "service", "_parsl_internal")
    monkeypatch.setattr(
        behavior,
        "download_global_mesh",
        AsyncMock(return_value={"static": "/s", "graph": "/g"}),
    )
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: None)
    result = await behavior.generate_mesh({"resolution": "480km"}, str(tmp_path / "m"))
    assert result["static"].endswith("mesh_480km.static.nc")


# Reuse and replacement


async def test_a_finished_mesh_is_reused(behavior, pipeline, tmp_path):
    meshes = str(tmp_path / "meshes")
    first = await behavior.generate_mesh(CONFIG, meshes)
    calls = len(pipeline.calls)
    second = await behavior.generate_mesh(CONFIG, meshes)
    assert len(pipeline.calls) == calls
    assert second["reused"] is True
    assert second["generated"] == first["generated"]
    # The report the first run wrote next to the mesh
    assert second["quality"] == first["quality"]


async def test_a_reused_replaced_mesh_names_where_its_report_is_now(
    behavior, pipeline, tmp_path
):
    meshes = str(tmp_path / "meshes")
    await behavior.generate_mesh(CONFIG, meshes)
    await behavior.generate_mesh(CONFIG, meshes, overwrite=True)
    again = await behavior.generate_mesh(CONFIG, meshes)
    report_file = again["quality"]["report_file"]
    # The stored report was written before the new mesh's directory was renamed
    stored = json.loads(Path(report_file).read_text())
    assert ".new-" in stored["report_file"]
    assert report_file == str(
        tmp_path / "meshes" / "mesh_7.3km" / "mesh_7.3km.quality.json"
    )
    assert Path(report_file).is_file()


async def test_a_reused_mesh_has_no_report_if_there_is_none(
    behavior, pipeline, tmp_path
):
    meshes = str(tmp_path / "meshes")
    await behavior.generate_mesh(CONFIG, meshes)
    for text in (None, "not json", "[1]"):
        report = tmp_path / "meshes" / "mesh_7.3km" / "mesh_7.3km.quality.json"
        report.unlink(missing_ok=True)
        if text is not None:
            report.write_text(text)
        assert "quality" not in await behavior.generate_mesh(CONFIG, meshes)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"jigsaw_version": "v1.0"}, "it was made with JIGSAW"),
        ({"mkgrid_version": "v2.0"}, "it was made with mkgrid"),
        ({"mesh_tools_version": "main"}, "it was made with mesh_tools"),
    ],
)
async def test_a_mesh_made_with_other_tools_is_not_reused(
    behavior, pipeline, tmp_path, monkeypatch, changes, message
):
    meshes = str(tmp_path / "meshes")
    await behavior.generate_mesh(CONFIG, meshes)
    other = _MeshBehavior(str(tmp_path / "work"), **changes)
    mark_tools_installed(other)
    with pytest.raises(FileExistsError, match=message):
        await other.generate_mesh(CONFIG, meshes)


async def test_a_mesh_of_another_division_is_not_reused(behavior, pipeline, tmp_path):
    meshes = str(tmp_path / "meshes")
    await behavior.generate_mesh({**CONFIG, "name": "m"}, meshes)
    with pytest.raises(FileExistsError) as info:
        await behavior.generate_mesh({"resolution": "7.4km", "name": "m"}, meshes)
    assert f"it has {PARTS} divisions, not 1038" in str(info.value)
    # The mesh is the same however its resolution is written, so it is not said
    assert "it is 7.3km" not in str(info.value)


@pytest.mark.parametrize("written", ["7.30km", "7.3000km", "7.301km", "07.3km"])
async def test_a_resolution_written_another_way_finds_the_same_mesh(
    behavior, pipeline, tmp_path, written
):
    """7.301km is 1051.9 divisions, which is the 1052 of 7.3km."""
    meshes = str(tmp_path / "meshes")
    first = await behavior.generate_mesh({**CONFIG, "name": "m"}, meshes)
    again = await behavior.generate_mesh({"resolution": written, "name": "m"}, meshes)
    assert again["reused"] is True and again["generated"] == first["generated"]


async def test_partitions_are_added_to_a_reused_mesh_and_keep_how_it_was_made(
    behavior, pipeline, tmp_path, monkeypatch
):
    meshes = str(tmp_path / "meshes")
    first = await behavior.generate_mesh(CONFIG, meshes)

    async def partition(graph, ranks):
        for rank in ranks:
            Path(f"{graph}.part.{rank}").write_text("p")
        return {rank: f"{graph}.part.{rank}" for rank in ranks}

    monkeypatch.setattr(behavior, "partition_mesh", partition)
    again = await behavior.generate_mesh({**CONFIG, "init_ranks": 4}, meshes)
    assert again["reused"] and again["generated"] == first["generated"]
    record = json.loads(
        (Path(first["static"]).parent / "mesh_7.3km.mesh.json").read_text()
    )
    assert record["generated"] == first["generated"]
    assert record["partitions"] == {"4": behavior.metis_version}


async def test_a_mesh_can_be_replaced(behavior, pipeline, tmp_path):
    meshes = str(tmp_path / "meshes")
    first = await behavior.generate_mesh(CONFIG, meshes)
    runs = pipeline.steps().count("jigsaw")
    result = await behavior.generate_mesh(CONFIG, meshes, overwrite=True)
    assert pipeline.steps().count("jigsaw") == runs + 1
    assert Path(result["static"]).is_file() and "reused" not in result
    # How it was made and its report come back, with the report's path in the
    # directory the mesh was moved to, not the one it was made in
    assert result["generated"] == first["generated"]
    assert result["quality"]["passed"]
    assert result["quality"]["report_file"] == str(
        Path(result["static"]).with_name("mesh_7.3km.quality.json")
    )
    assert ".new-" not in result["quality"]["report_file"]


async def test_a_replaced_mesh_whose_report_could_not_be_written(
    behavior, pipeline, tmp_path, monkeypatch
):
    async def quality(path, resolution_km=None):
        return {**REPORT, "report_file": None}

    monkeypatch.setattr(behavior, "check_mesh_quality", quality)
    meshes = str(tmp_path / "meshes")
    await behavior.generate_mesh(CONFIG, meshes)
    result = await behavior.generate_mesh(CONFIG, meshes, overwrite=True)
    assert result["quality"]["report_file"] is None


# Requests


@pytest.mark.parametrize(
    "resolution", ["7.3km", "2km", "0.91km", "5000km", "480km", "3.75km", "999.9km"]
)
def test_any_resolution_the_tools_can_make_is_accepted(behavior, resolution):
    behavior._validate_mesh_config({"resolution": resolution})


@pytest.mark.parametrize(
    "resolution, message",
    [
        ("0.5km", "15360 parts per edge. A global mesh can be made at 0.91km to"),
        ("6000km", "1 parts per edge. A global mesh can be made at"),
        ("abc", "is not valid"),
        ("7.3", "is not valid"),
        ("-5km", "is not valid"),
        ("0km", "is not valid"),
        (7.3, "is not valid"),
        (None, "is not valid"),
        (["10km"], "is not valid"),
    ],
)
def test_a_resolution_that_cannot_be_made_is_refused(behavior, resolution, message):
    with pytest.raises(ValueError, match=message):
        behavior._validate_mesh_config({"resolution": resolution})


def test_the_limits_of_the_tools_are_in_the_message(behavior):
    with pytest.raises(ValueError) as info:
        behavior._validate_mesh_config({"resolution": "0.5km"})
    assert "0.91km to 5120km" in str(info.value)
    # The ends are made
    behavior._validate_mesh_config({"resolution": "5120km"})
    behavior._validate_mesh_config({"resolution": f"{7680 / 8460.4}km"})


def test_create_region_still_cuts_only_a_downloaded_mesh(behavior):
    config = {
        "resolution": "7.3km",
        "regional": {
            "create_region": {
                "circle": {"center_lat": 39, "center_lon": -105, "radius_km": 400}
            }
        },
    }
    with pytest.raises(ValueError, match="not supported. Available"):
        behavior._validate_mesh_config(config)


def test_project_hexes_is_unchanged(behavior):
    behavior._validate_mesh_config(
        {
            "resolution": "2km",
            "regional": {
                "project_hexes": {
                    "center_lat": 40.0,
                    "center_lon": -100.0,
                    "extent_x_km": 500,
                    "extent_y_km": 500,
                }
            },
        }
    )


def test_which_requests_are_made_and_which_downloaded(behavior):
    assert behavior._is_generated_global({"resolution": "7.3km"})
    assert not behavior._is_generated_global({"resolution": "10km"})
    assert not behavior._is_generated_global({})  # 120km by default
    assert not behavior._is_generated_global(
        {"resolution": "7.3km", "regional": {"project_hexes": {}}}
    )


def test_the_identity_of_a_mesh_that_is_made(behavior):
    assert behavior._mesh_identity({"resolution": "7.3km"}) == {
        "resolution": "7.3km",
        "regional": None,
        "division": PARTS,
        "jigsaw_version": behavior.jigsaw_version,
        "mkgrid_version": "feature/nesting",
        "mesh_tools_version": behavior.mesh_tools_version,
    }
    # A downloaded mesh's is what it was before meshes could be made
    assert behavior._mesh_identity({"resolution": "10km"}) == {
        "resolution": "10km",
        "regional": None,
    }


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("mkgrid_ranks", 0),
        ("mkgrid_ranks", -2),
        ("mkgrid_ranks", 2.5),
        ("mkgrid_ranks", True),
        ("mkgrid_ranks", "4"),
        ("mkgrid_ranks_per_node", 0),
        ("mkgrid_ranks_per_node", False),
    ],
)
def test_invalid_ranks_are_refused(tmp_path, keyword, value):
    with pytest.raises(ValueError, match=f"{keyword} must be a positive integer"):
        _MeshBehavior(str(tmp_path / "work"), **{keyword: value})


def test_more_ranks_per_node_than_ranks_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must not be more than mkgrid_ranks"):
        _MeshBehavior(str(tmp_path / "work"), mkgrid_ranks=2, mkgrid_ranks_per_node=4)


def test_ranks_default_to_one_node(tmp_path):
    agent = _MeshBehavior(str(tmp_path / "work"))
    assert (agent.mkgrid_ranks, agent.mkgrid_ranks_per_node) == (4, 4)
    agent = _MeshBehavior(str(tmp_path / "work"), mkgrid_ranks=6)
    assert (agent.mkgrid_ranks, agent.mkgrid_ranks_per_node) == (6, 6)


# The tasks


@pytest.fixture
def script_agent(tmp_path):
    """An agent whose tools are stand-ins that record how they are run."""
    agent = _MeshBehavior(str(tmp_path / "work dir"))
    for prefix, name in (
        (agent._sphtri_subdiv_prefix, "sphtri_subdiv"),
        (agent._jigsaw_prefix, "jigsaw"),
        (agent._mkgrid_prefix, "mkgrid"),
    ):
        program = prefix / "bin" / name
        program.parent.mkdir(parents=True)
        program.write_text(f'#!/bin/sh\necho "{name} $PWD $*" >> "$RECORD"\n')
        program.chmod(0o755)
    return agent


def run_script(script, tmp_path, **env):
    return subprocess.run(
        ["bash", "-c", script],
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "RECORD": str(tmp_path / "record"), **env},
        capture_output=True,
        text=True,
    )


def test_sphtri_subdiv_is_run_in_the_mesh_directory(script_agent, tmp_path):
    mesh_dir = tmp_path / "a mesh"
    mesh_dir.mkdir()
    script = _MeshBehavior._run_sphtri_subdiv.func(script_agent, str(mesh_dir), 1051)
    done = run_script(script, tmp_path)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "record").read_text() == (
        f"sphtri_subdiv {mesh_dir} ICOS.msh INIT.msh 1051\n"
    )


def test_jigsaw_is_run_with_its_configuration(script_agent, tmp_path):
    mesh_dir = tmp_path / "a mesh"
    mesh_dir.mkdir()
    script = _MeshBehavior._run_jigsaw.func(script_agent, str(mesh_dir))
    done = run_script(script, tmp_path)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "record").read_text() == f"jigsaw {mesh_dir} MESH.jig\n"


def test_mkgrid_is_launched_by_the_mpi_prefix_with_the_spacing_in_meters(
    script_agent, tmp_path
):
    mesh_dir = tmp_path / "a mesh"
    mesh_dir.mkdir()
    script = _MeshBehavior._run_mkgrid.func(script_agent, str(mesh_dir), 7303.8)
    assert "$PARSL_MPI_PREFIX" in script
    done = run_script(script, tmp_path, PARSL_MPI_PREFIX="env LAUNCHED=1")
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "record").read_text() == f"mkgrid {mesh_dir} 7303.8\n"


def test_a_failing_program_fails_its_script(script_agent, tmp_path):
    (tmp_path / "mesh").mkdir()
    script_agent._jigsaw.write_text("#!/bin/sh\nexit 255\n")
    script = _MeshBehavior._run_jigsaw.func(script_agent, str(tmp_path / "mesh"))
    done = run_script(script, tmp_path)
    assert done.returncode == 255 and "Completed jigsaw" not in done.stdout


def test_the_inputs_of_mkgrid_are_made_from_jigsaws_mesh(behavior, tmp_path):
    (tmp_path / "MESH.msh").write_text(TETRAHEDRON)
    result = _MeshBehavior._prepare_mkgrid_inputs.func(
        behavior, str(tmp_path), 7.3, PARTS, 7680 / PARTS, COMMITS
    )
    assert result == {"points": 4, "triangles": 4}
    assert (tmp_path / "SaveDensity").read_text() == "1.0\n" * 4
    assert (tmp_path / "SaveVertices").read_text().startswith("1.5 0.25 -3.125\n")
    assert (tmp_path / "SaveTriangles").read_text() == "0 1 2\n0 2 3\n0 3 1\n1 3 2\n"
    config = yaml.safe_load((tmp_path / "SaveConfig").read_text())
    assert config["mesh"] == {"resolution": 7680 / PARTS}
    assert config["chiltepin"]["division"] == PARTS
    assert config["chiltepin"]["jigsaw"] == COMMITS["jigsaw"]
    assert "hfun_min = " in (tmp_path / "SaveCode").read_text()


def test_a_mesh_that_is_not_a_sphere_comes_back_as_an_error(behavior, tmp_path):
    (tmp_path / "MESH.msh").write_text("POINT=1\n1;2;3;0\n")
    result = _MeshBehavior._prepare_mkgrid_inputs.func(
        behavior, str(tmp_path), 7.3, PARTS, 7680 / PARTS, COMMITS
    )
    assert "has no TRIA3 section" in result["error"]
    assert not (tmp_path / "SaveDensity").exists()


def test_the_tools_commits_are_what_the_markers_hold(behavior):
    assert behavior._installed_commit(behavior._jigsaw_prefix) == COMMITS["jigsaw"]
    shutil.rmtree(behavior._jigsaw_prefix)
    assert behavior._installed_commit(behavior._jigsaw_prefix) is None


async def test_a_tool_with_no_recorded_commit_is_an_error(
    behavior, pipeline, tmp_path, monkeypatch
):
    original = behavior._installed_commit
    monkeypatch.setattr(
        behavior,
        "_installed_commit",
        lambda prefix: None if prefix == behavior._mkgrid_prefix else original(prefix),
    )
    with pytest.raises(RuntimeError, match="install of mkgrid has no commit recorded"):
        await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    # Nothing was copied or run, and the name is free
    assert pipeline.calls == []
    assert not (tmp_path / "meshes" / "mesh_7.3km").exists()


async def test_a_jigsaw_mesh_of_another_size_is_made_and_reported(
    behavior, pipeline, tmp_path, monkeypatch
):
    pipeline.prepared = {"points": 11067000, "triangles": 22133996}
    logger = Mock()
    monkeypatch.setattr(mesh_module, "logger", logger)
    result = await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert result["generated"]["cells"] == 11067000
    assert result["generated"]["expected_cells"] == 11067042
    assert "JIGSAW made %d points" in logger.warning.call_args.args[0]
    assert logger.warning.call_args.args[1:] == (11067000, 11067042)


async def test_the_cells_are_the_ones_the_division_gives(behavior, pipeline, tmp_path):
    result = await behavior.generate_mesh(CONFIG, str(tmp_path / "meshes"))
    assert result["generated"]["cells"] == result["generated"]["expected_cells"]


# A resolution written another way than a downloaded mesh's name


@pytest.mark.parametrize("written", ["120.0km", "120.000km", "0120km"])
async def test_a_ucar_resolution_is_downloaded_however_it_is_written(
    behavior, pipeline, tmp_path, monkeypatch, written
):
    assert not behavior._is_generated_global({"resolution": written})
    behavior._validate_mesh_config({"resolution": written})
    download = AsyncMock(return_value={"static": "/s", "graph": "/g"})
    monkeypatch.setattr(behavior, "download_global_mesh", download)
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: None)
    result = await behavior.generate_mesh({"resolution": written}, str(tmp_path / "m"))
    # Under the name of the mesh, as 120km is asked for
    assert result["static"].endswith("/mesh_120km/mesh_120km.static.nc")
    download.assert_awaited_once_with("120km", str(tmp_path / "m"))
    assert pipeline.calls == []


def test_a_ucar_resolution_written_another_way_can_be_cut_too(behavior):
    behavior._validate_mesh_config(
        {
            "resolution": "10.0km",
            "regional": {
                "create_region": {
                    "circle": {"center_lat": 39, "center_lon": -105, "radius_km": 400}
                }
            },
        }
    )


def test_what_is_not_a_ucar_resolution_is_left_as_it_is(behavior):
    for resolution in ("7.3km", "abc", 5, None, ["10km"], "120"):
        assert behavior._canonical_resolution(resolution) == resolution


HEXES = {
    "project_hexes": {
        "center_lat": 40.0,
        "center_lon": -100.0,
        "extent_x_km": 500,
        "extent_y_km": 500,
    }
}


def test_a_made_mesh_differs_from_a_downloaded_one(behavior):
    made = {"resolution": "10km", "regional": None}
    requested = behavior._mesh_identity({"resolution": "7.3km"})
    differences = behavior._mesh_differences(made, requested)
    # The resolutions are compared, since the downloaded mesh has no division
    assert "it is 10km, not 7.3km" in differences
    assert any("it was made with JIGSAW" in difference for difference in differences)


def test_a_downloaded_mesh_differs_from_a_made_one_the_other_way(behavior):
    made = behavior._mesh_identity({"resolution": "7.3km"})
    requested = {"resolution": "10km", "regional": None}
    assert behavior._mesh_differences(made, requested) == ["it is 7.3km, not 10km"]


def test_a_made_mesh_differs_from_a_regional_one(behavior):
    made = behavior._mesh_identity({"resolution": "7.3km"})
    requested = behavior._mesh_identity({"resolution": "7.3km", "regional": HEXES})
    assert "it is a global mesh, not a project_hexes mesh" in (
        behavior._mesh_differences(made, requested)
    )


def test_the_division_decides_between_made_meshes(behavior):
    made = behavior._mesh_identity({"resolution": "7.3km"})
    same = behavior._mesh_identity({"resolution": "7.301km"})
    other = behavior._mesh_identity({"resolution": "7.4km"})
    assert behavior._mesh_differences(made, same) == []
    assert behavior._mesh_differences(made, other) == [
        f"it has {PARTS} divisions, not 1038"
    ]


async def test_a_project_hexes_mesh_keeps_its_resolution_as_written(
    behavior, pipeline, tmp_path, monkeypatch
):
    """Only a mesh that is downloaded is named by the downloaded mesh's name."""
    made = AsyncMock(return_value={"static": "/s", "graph": "/g"})
    monkeypatch.setattr(behavior, "_generate_project_hexes_mesh", made)
    monkeypatch.setattr(behavior, "install_mpas_tools", AsyncMock())
    config = {"resolution": "15.0km", "regional": HEXES}
    await behavior.generate_mesh(config, str(tmp_path / "meshes"))
    # The name and the record are as they were before meshes could be made
    assert (tmp_path / "meshes" / "mesh_15.0km").is_dir()
    assert made.await_args.args[2] == "15.0km"
