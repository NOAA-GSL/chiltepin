# SPDX-License-Identifier: Apache-2.0
"""Real plots of MPAS meshes through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs internet access to UCAR, GitHub and Natural Earth, and its
``compute`` resource needs chiltepin with the ``mpas`` extra, which the test
extra includes. The regional meshes also build MPAS-Tools, which needs make, a
Fortran compiler and netCDF on ``compute``, and install MPAS-Limited-Area.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

# Versions the tests install, passed to the agent so the test and the agent agree
VERSIONS = {"mpas_tools_version": "2.0.0", "limited_area_version": "2.2"}

# A sphere's image size, and the height of a flat map as tall as allowed
SPHERE_SIZE = (1500, 1590)

GLOBAL = {"resolution": "480km", "name": "globe", "plot": True}
HEXES = {
    "resolution": "50km",
    "name": "hexes",
    "plot": True,
    "regional": {
        "project_hexes": {
            "center_lat": 40.0,
            "center_lon": -105.0,
            "extent_x_km": 1000,
            "extent_y_km": 800,
        }
    },
}
CUT = {
    "resolution": "480km",
    "name": "cut",
    "plot": True,
    "regional": {
        "create_region": {
            "circle": {"center_lat": 39.0, "center_lon": -105.5, "radius_km": 1500}
        }
    },
}


def image_size(path):
    """Return the size of a PNG, checking that it is one."""
    assert Path(path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(path) as image:
        return image.size


@pytest.mark.asyncio
async def test_real_plots(mesh_test_directory, launch_mesh_agent):
    """Prepare plotting and plot each kind of mesh, then reuse it after a restart."""
    directory = mesh_test_directory
    mesh_data_dir = str(directory / "meshes")
    work = directory / "work"
    marker = work / "plot" / ".installed"
    download_log = work / "logs" / "plot_download.stdout"

    async with launch_mesh_agent("first", **VERSIONS) as agent:
        # Concurrent requests that all need plotting, and other tools too
        globe, hexes, cut = await asyncio.gather(
            agent.generate_mesh(GLOBAL, mesh_data_dir),
            agent.generate_mesh(HEXES, mesh_data_dir),
            agent.generate_mesh(CUT, mesh_data_dir),
        )
        for result, name in ((globe, "globe"), (hexes, "hexes"), (cut, "cut")):
            assert "plot_error" not in result, result
            assert result["plot"] == str(Path(mesh_data_dir) / name / f"{name}.png")
        # The global mesh is a sphere, and the regional ones flat maps shaped
        # like their regions
        assert image_size(globe["plot"]) == SPHERE_SIZE
        for result in (hexes, cut):
            width, height = image_size(result["plot"])
            assert (width, height) != SPHERE_SIZE
            assert 1000 < width <= 2124 and 400 < height <= 1368
        # The map layers were downloaded to work_dir, and plotting was checked
        assert marker.is_file()
        layers = list((work / "plot" / "natural_earth").rglob("*.shp"))
        assert len(layers) == 4 and all(layer.stat().st_size > 0 for layer in layers)

        # A zoomed view of the global mesh replaces its image
        before = Path(globe["plot"]).read_bytes()
        zoomed = await agent.plot_mesh(
            globe["static"], title="Zoomed", zoom=4, center_lat=50, center_lon=10
        )
        assert zoomed == globe["plot"]
        assert image_size(zoomed) == SPHERE_SIZE
        assert Path(zoomed).read_bytes() != before

        # A view of a regional mesh is refused, and its image is kept
        kept = Path(cut["plot"]).read_bytes()
        with pytest.raises(ValueError, match="apply only to global meshes"):
            await agent.plot_mesh(cut["static"], zoom=2)
        assert Path(cut["plot"]).read_bytes() == kept

        installed = (marker.stat().st_mtime_ns, download_log.stat().st_mtime_ns)
        await agent.install_plotting()

    # A new agent has no memory of the install, and finds it on disk
    async with launch_mesh_agent("restart", **VERSIONS) as agent:
        again = await agent.plot_mesh(hexes["static"], title="Again")
        assert again == hexes["plot"]
    assert (marker.stat().st_mtime_ns, download_log.stat().st_mtime_ns) == installed

    # Every plot wrote its own logs: three requests, the zoomed view, the
    # refused view, and the plot after the restart
    logs = work / "logs"
    assert len(list(logs.glob("plot_mesh.*.stderr"))) == 6
    assert (logs / "plot_check.stdout").is_file()
