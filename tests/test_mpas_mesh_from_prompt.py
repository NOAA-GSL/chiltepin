# SPDX-License-Identifier: Apache-2.0
"""Real meshes from prompts, read by a real LLM, through the MPAS mesh agent.

Runs only with ``--config=<platform config>``, and only when
``CHILTEPIN_TEST_LLM_ENV`` names a file of the model's credentials, which the
``service`` resource reads before its tasks. For example, for a Bedrock API
key::

    $ cat ~/.config/chiltepin/bedrock.env
    export AWS_BEARER_TOKEN_BEDROCK="..."
    export AWS_REGION_NAME="us-east-1"
    $ CHILTEPIN_TEST_LLM_ENV=~/.config/chiltepin/bedrock.env pytest \\
        tests/test_mpas_mesh_from_prompt.py --config=tests/configs/docker.yaml

``CHILTEPIN_TEST_LLM_MODEL`` chooses the model, by default Claude Sonnet 5.5 on
Bedrock. The ``service`` resource needs internet access to the model and to
Natural Earth, and both resources need chiltepin with the ``mpas`` and ``llm``
extras, which the test extra includes. Making the mesh also installs
MPAS-Limited-Area and plotting, as their own tests describe.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import os
from pathlib import Path

import pytest

from chiltepin.agents.mpas._region_shapes import _unit, aeqd

CREDENTIALS = os.environ.get("CHILTEPIN_TEST_LLM_ENV")
MODEL = os.environ.get(
    "CHILTEPIN_TEST_LLM_MODEL", "bedrock/us.anthropic.claude-sonnet-5-5"
)

pytestmark = pytest.mark.skipif(
    not CREDENTIALS, reason="CHILTEPIN_TEST_LLM_ENV names no credentials file"
)


def distance_km(lat1, lon1, lat2, lon2):
    """Return the distance between two points on the sphere, in km."""
    x, y = aeqd(_unit([lon2], [lat2]), _unit(lon1, lat1))
    return float((x[0] ** 2 + y[0] ** 2) ** 0.5)


@pytest.mark.asyncio
async def test_real_meshes_from_prompts(mesh_test_directory, launch_mesh_agent):
    """Read prompts with a real model, and make a small mesh from one."""
    directory = mesh_test_directory
    mesh_data_dir = str(directory / "meshes")
    credentials = str(Path(CREDENTIALS).expanduser())
    assert Path(credentials).is_file()
    async with launch_mesh_agent(
        "prompt", service_environment=[f"source {credentials}"]
    ) as agent:
        # A rectangle at a resolution that cannot be downloaded, over the 48
        # contiguous states, which only project_hexes makes
        conus = await agent.mesh_config_from_prompt(
            "A 3 km mesh of the contiguous United States", MODEL
        )
        hexes = conus["regional"]["project_hexes"]
        assert conus["resolution"] == "3km"
        assert 30 < hexes["center_lat"] < 45
        assert -105 < hexes["center_lon"] < -90
        # About 4600 by 2900 km, with the margin, and without Alaska
        assert 4000 < max(hexes["extent_x_km"], hexes["extent_y_km"]) < 7000
        assert 2000 < min(hexes["extent_x_km"], hexes["extent_y_km"]) < 4500

        # A small mesh cut from the 480km global mesh, and drawn. A region
        # must reach at least two cells from its center, as this one does.
        made = await agent.create_mesh_from_prompt(
            "a 480 km limited area circle mesh around the contiguous United States",
            mesh_data_dir,
            MODEL,
            mesh_name="conus",
            plot=True,
        )
        circle = made["mesh_config"]["regional"]["create_region"]["circle"]
        assert made["mesh_config"]["resolution"] == "480km"
        assert made["mesh_config"]["name"] == "conus"
        # The smallest circle holding the contiguous states is centered near
        # 41.9N 96.1W, reaching both Florida and the northwest, with a radius
        # of about 2360 km, and the margin adds a tenth of their diameter
        middle = distance_km(41.9, -96.1, circle["center_lat"], circle["center_lon"])
        assert middle < 300
        assert 2500 < circle["radius_km"] < 3600
        result = made["mesh_result"]
        mesh_dir = Path(mesh_data_dir) / "conus"
        assert result["static"] == str(mesh_dir / "conus.static.nc")
        assert Path(result["static"]).stat().st_size > 0
        assert Path(result["plot"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        # A global mesh at a resolution that cannot be downloaded is refused,
        # saying which can, and so is a region too small for its cells
        with pytest.raises(ValueError, match="Global meshes are available at"):
            await agent.mesh_config_from_prompt("a global 7 km mesh", MODEL)
        with pytest.raises(ValueError, match="Ask for smaller cells"):
            await agent.mesh_config_from_prompt(
                "a 480 km limited area circle mesh around Colorado", MODEL
            )

    work = directory / "work"
    assert (work / "geo" / ".installed").is_file()
    assert (work / "logs" / "prompt_check.stdout").is_file()
    # Nothing a model's credentials could be in keeps them
    secrets = [
        line.split("=", 1)[1].strip().strip("\"'")
        for line in Path(credentials).read_text().splitlines()
        if line.startswith("export ") and "=" in line
    ]
    secrets = [secret for secret in secrets if len(secret) >= 16]
    for path in directory.rglob("*"):
        if path.is_file() and path.stat().st_size < 10_000_000:
            text = path.read_bytes()
            for secret in secrets:
                assert secret.encode() not in text, path
