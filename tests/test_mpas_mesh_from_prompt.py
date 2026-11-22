# SPDX-License-Identifier: Apache-2.0
"""Real meshes from prompts, read by a real LLM, through the MPAS mesh agent.

Runs only with ``--config=<platform config>``, and only when
``CHILTEPIN_TEST_LLM_ENV`` names a file the ``service`` resource reads before
its tasks, holding what the model needs, such as credentials or a server's
address. ``CHILTEPIN_TEST_LLM_MODEL`` names the model, by default Claude Sonnet
5.5 on Bedrock, and ``CHILTEPIN_TEST_LLM_REASONING`` its reasoning effort, if
any. For example, with a Bedrock API key::

    $ cat ~/.config/chiltepin/bedrock.env
    export AWS_BEARER_TOKEN_BEDROCK="..."
    export AWS_REGION_NAME="us-east-1"
    $ CHILTEPIN_TEST_LLM_ENV=~/.config/chiltepin/bedrock.env pytest \\
        tests/test_mpas_mesh_from_prompt.py --config=tests/configs/docker.yaml

CI tests with qwen3:8b in Ollama, which needs no credentials, as
``.github/scripts/start-ollama.sh`` starts it. The prompts name what they ask
for, and the checks accept any sound reading of them, so a small model passes.

The ``service`` resource needs internet access to Natural Earth, and both
resources need chiltepin with the ``mpas`` and ``llm`` extras, which the test
extra includes. Making the mesh also installs MPAS-Limited-Area and plotting,
as their own tests describe.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import os
import re
import secrets
from pathlib import Path

import pytest

from chiltepin.agents.mpas._region_shapes import _unit, aeqd

SETTINGS = os.environ.get("CHILTEPIN_TEST_LLM_ENV")
MODEL = os.environ.get(
    "CHILTEPIN_TEST_LLM_MODEL", "bedrock/us.anthropic.claude-sonnet-5-5"
)
REASONING = os.environ.get("CHILTEPIN_TEST_LLM_REASONING") or None

pytestmark = pytest.mark.skipif(
    not SETTINGS, reason="CHILTEPIN_TEST_LLM_ENV names no file of model settings"
)


def distance_km(lat1, lon1, lat2, lon2):
    """Return the distance between two points on the sphere, in km."""
    x, y = aeqd(_unit([lon2], [lat2]), _unit(lon1, lat1))
    return float((x[0] ** 2 + y[0] ** 2) ** 0.5)


def assert_covers_the_contiguous_states(config):
    """Check a request covers the 48 contiguous states, without Alaska.

    A model can make it a project_hexes rectangle or a create_region shape,
    since the prompt names neither.
    """
    regional = config["regional"]
    if "project_hexes" in regional:
        hexes = regional["project_hexes"]
        # Their middle is near 38N 97W, and they are about 4600 by 2900 km
        assert distance_km(38.0, -97.0, hexes["center_lat"], hexes["center_lon"]) < 700
        sides = sorted([hexes["extent_x_km"], hexes["extent_y_km"]])
        assert 2000 < sides[0] < 4500
        assert 4000 < sides[1] < 7000
        return
    ((shape, block),) = regional["create_region"].items()
    if shape == "polygon":
        # Every corner is near the 48 states: none in Alaska or far overseas
        for lat, lon in block["vertices"]:
            assert 15 < lat < 55 and -135 < lon < -55, (lat, lon)
        return
    assert shape in ("circle", "ellipse"), shape
    assert distance_km(41.9, -96.1, block["center_lat"], block["center_lon"]) < 700
    assert block.get("radius_km", block.get("semi_major_axis_km")) < 4000


@pytest.mark.asyncio
async def test_real_meshes_from_prompts(mesh_test_directory, launch_mesh_agent):
    """Read prompts with a real model, and make a small mesh from one."""
    directory = mesh_test_directory
    mesh_data_dir = str(directory / "meshes")
    settings = str(Path(SETTINGS).expanduser())
    assert Path(settings).is_file()
    options = {"reasoning_effort": REASONING}
    # A credential of the run's own, so the check below for leaked ones has
    # one to find with settings that hold none, as Ollama's do. It is read
    # from a file, as credentials are, so it is not in the configuration.
    canary = f"chiltepin-canary-{secrets.token_hex(8)}"
    canary_file = directory / "canary.env"
    canary_file.write_text(f"export CHILTEPIN_TEST_TOKEN={canary}\n")
    environment = [f"source {settings}", f"source {canary_file}"]
    async with launch_mesh_agent("prompt", service_environment=environment) as agent:
        # A resolution that cannot be downloaded, over the 48 states
        conus = await agent.mesh_config_from_prompt(
            "A 3 km mesh of the contiguous United States", MODEL, **options
        )
        assert conus["resolution"] == "3km"
        assert_covers_the_contiguous_states(conus)

        # A small mesh cut from the 480km global mesh, and drawn. A region
        # must reach at least two cells from its center, as this one does.
        made = await agent.create_mesh_from_prompt(
            "a 480 km limited area circle mesh around Colorado, Utah, Wyoming, "
            "Nevada, Arizona and New Mexico",
            mesh_data_dir,
            MODEL,
            mesh_name="west",
            plot=True,
            **options,
        )
        circle = made["mesh_config"]["regional"]["create_region"]["circle"]
        assert made["mesh_config"]["resolution"] == "480km"
        assert made["mesh_config"]["name"] == "west"
        # The six states reach about 940 km from their middle, near 38.0N
        # 110.3W, and the margin adds a tenth of their diameter
        middle = distance_km(38.0, -110.3, circle["center_lat"], circle["center_lon"])
        assert middle < 250
        assert 960 <= circle["radius_km"] < 1350
        result = made["mesh_result"]
        mesh_dir = Path(mesh_data_dir) / "west"
        assert result["static"] == str(mesh_dir / "west.static.nc")
        assert Path(result["static"]).stat().st_size > 0
        assert Path(result["plot"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        # A global mesh at a resolution that cannot be downloaded is refused,
        # saying which can, and so is a region too small for its cells
        with pytest.raises(ValueError, match="Global meshes are available at"):
            await agent.mesh_config_from_prompt("a global 7 km mesh", MODEL, **options)
        with pytest.raises(ValueError, match="Ask for smaller cells"):
            await agent.mesh_config_from_prompt(
                "a 480 km limited area circle mesh around Colorado", MODEL, **options
            )

    work = directory / "work"
    assert (work / "geo" / ".installed").is_file()
    assert (work / "logs" / "prompt_check.stdout").is_file()
    # No file the run wrote holds a credential from the settings file
    leaks = [canary]
    for line in Path(settings).read_text().splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if re.search("KEY|TOKEN|SECRET|PASSWORD", name) and len(value) >= 16:
            leaks.append(value.strip().strip("\"'"))
    for path in directory.rglob("*"):
        if path.is_file() and path != canary_file and path.stat().st_size < 10_000_000:
            text = path.read_bytes()
            for leak in leaks:
                assert leak.encode() not in text, path
