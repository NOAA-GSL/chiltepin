# SPDX-License-Identifier: Apache-2.0
"""Real installs of the tools that make global meshes, through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs internet access to GitHub, and its ``compute`` resource needs CMake, C and
C++ compilers, MPI and the PnetCDF library, which JIGSAW, mkgrid and
sphtri_subdiv are built with.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import re
import shlex
import subprocess

import pytest

import chiltepin.configure

COMMIT = re.compile(r"[0-9a-f]{40}")


def runner(config_file):
    """Return a function that runs a program in the environment of ``compute``.

    The programs need the libraries the resources load, such as PnetCDF's.
    """
    platform = chiltepin.configure.parse_file(config_file)
    environment = "\n".join(platform["compute"].get("environment", []))

    def run(program, *arguments):
        command = " ".join(shlex.quote(str(word)) for word in (program, *arguments))
        return subprocess.run(
            ["bash", "-c", f"{environment}\n{command}"], capture_output=True, text=True
        )

    return run


@pytest.mark.asyncio
async def test_real_installs(mesh_test_directory, launch_mesh_agent, config_file):
    """Install the three tools, use them, and reuse them after a restart."""
    run = runner(config_file)
    work = mesh_test_directory / "work"
    jigsaw = work / "jigsaw"
    mkgrid = work / "mkgrid" / "feature%2Fnesting"
    subdiv = work / "mesh-tools"
    async with launch_mesh_agent("first") as agent:
        await agent.install_global_mesh_tools()

    # Each install records the commit it was built from
    markers = {
        "jigsaw": next(jigsaw.glob("*/.installed")),
        "mkgrid": mkgrid / ".installed",
        "sphtri_subdiv": next(subdiv.glob("*/.installed")),
    }
    for name, marker in markers.items():
        assert COMMIT.fullmatch(marker.read_text().strip()), name
    assert markers["jigsaw"].parent.name == "3777fe7f00815295a94065cdcaca27bcb3c2106e"
    assert markers["sphtri_subdiv"].read_text().strip() == (
        "4967f0fbb278584c11b3057c3eb4396d199f030b"
    )

    # The programs run. JIGSAW and mkgrid say what they need when given nothing.
    nothing = run(markers["jigsaw"].parent / "bin" / "jigsaw")
    assert "*.jig file not found" in nothing.stdout + nothing.stderr
    nothing = run(mkgrid / "bin" / "mkgrid")
    assert "Usage: mkgrid" in nothing.stdout + nothing.stderr
    # sphtri_subdiv divides the icosahedron that comes with mkgrid: each of the 20
    # triangles into 4, so 80 triangles and 42 points
    divided = mesh_test_directory / "divided.msh"
    done = run(
        markers["sphtri_subdiv"].parent / "bin" / "sphtri_subdiv",
        str(mkgrid / "share" / "ICOS.msh"),
        str(divided),
        "1",
    )
    assert done.returncode == 0, done.stderr
    text = divided.read_text()
    assert "POINT=42" in text and "TRIA3=80" in text

    # Logs of each stage were written
    logs = work / "logs"
    for tool in ("jigsaw", "mkgrid", "sphtri_subdiv"):
        for stage in ("download", "build"):
            assert (logs / f"{tool}_{stage}.stderr").is_file()

    # A new agent finds the installs on disk and builds nothing
    before = {name: marker.stat().st_mtime_ns for name, marker in markers.items()}
    async with launch_mesh_agent("restart") as agent:
        await agent.install_global_mesh_tools()
        await agent.install_mkgrid()
    after = {name: marker.stat().st_mtime_ns for name, marker in markers.items()}
    assert after == before
