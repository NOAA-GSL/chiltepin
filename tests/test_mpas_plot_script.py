# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh plotting script that need no plotting packages.

The script runs with the Python of the plotting environment, so these tests
cover its arguments and the decisions it makes from them. The drawing itself
is exercised by the integration test, which runs the script for real.
"""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from chiltepin.agents.mpas import _plot_mesh as plot

SCRIPT = Path(plot.__file__)


def test_script_imports_only_the_standard_library():
    """The agent imports the script's exit codes, without plotting packages."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(SCRIPT)!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "heavy = ('numpy', 'netCDF4', 'matplotlib', 'cartopy')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


def test_script_help_runs_without_plotting_packages():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert done.returncode == 0
    assert "--map-data" in done.stdout


def test_parse_args():
    args = plot.parse_args(
        ["m.nc", "out.png", "--map-data", "/d", "--title", "T", "--zoom", "2.5"]
        + ["--center", "50", "-10.5"]
    )
    assert args.mesh == Path("m.nc")
    assert args.output == Path("out.png")
    assert args.map_data == Path("/d")
    assert args.title == "T"
    assert args.zoom == 2.5
    assert args.center == [50.0, -10.5]
    defaults = plot.parse_args(["m.nc", "out.png", "--map-data", "/d"])
    assert (defaults.title, defaults.zoom, defaults.center) == (None, None, None)


def test_parse_args_requires_the_map_data():
    with pytest.raises(SystemExit):
        plot.parse_args(["m.nc", "out.png"])


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("conus.static.nc", "MPAS mesh: conus"),
        ("x1.2562.static.nc", "MPAS mesh: x1.2562"),
        ("x1.2562.grid.nc", "MPAS mesh: x1.2562.grid"),
        ("mesh", "MPAS mesh: mesh"),
    ],
)
def test_default_title(name, title):
    assert plot.default_title(Path("/a") / name) == title


def test_layer_path():
    assert plot.layer_path(Path("/d"), plot.COASTLINE) == Path(
        "/d/shapefiles/natural_earth/physical/ne_10m_coastline.shp"
    )
    assert plot.layer_path(Path("/d"), plot.GLOBE_COASTLINE).name == (
        "ne_110m_coastline.shp"
    )


@pytest.mark.parametrize(
    ("cells", "km"), [(2562, 480), (40962, 120), (163842, 60), (65536002, 3)]
)
def test_global_cell_spacing_matches_the_mesh_names(cells, km):
    assert plot.global_cell_spacing_km(cells) == pytest.approx(km, rel=0.01)


def test_cell_spacing_of_a_hexagon():
    # A hexagon 10 km between centers has an area of sqrt(3)/2 * 100 km2
    assert plot.cell_spacing_km(math.sqrt(3) / 2 * 100) == pytest.approx(10)


@pytest.mark.parametrize(
    ("cells", "zoom", "edges"),
    [
        (2562, plot.DEFAULT_ZOOM, True),
        (40962, plot.DEFAULT_ZOOM, True),
        (163842, plot.DEFAULT_ZOOM, False),
        (163842, 4, True),
        (65536002, plot.DEFAULT_ZOOM, False),
        (65536002, 50, True),
    ],
)
def test_sphere_edges_appear_when_cells_are_large_enough(cells, zoom, edges):
    pixels = plot.sphere_cell_pixels(plot.global_cell_spacing_km(cells), zoom)
    assert (pixels >= plot.MIN_CELL_PIXELS) == edges


def test_sphere_scale():
    assert plot.sphere_pixels_per_unit(1) == plot.SPHERE_PIXELS_PER_UNIT
    assert plot.sphere_pixels_per_unit(2) == 2 * plot.SPHERE_PIXELS_PER_UNIT
    # The default zoom makes the globe about 95% of the image's width
    width = plot.SPHERE_FIGURE_INCHES * plot.SPHERE_DPI
    diameter = 2 * plot.sphere_pixels_per_unit(plot.DEFAULT_ZOOM)
    assert diameter / width == pytest.approx(0.95, abs=0.01)
    assert plot.sphere_cell_pixels(480, plot.DEFAULT_ZOOM) == pytest.approx(53.7, 0.01)


@pytest.mark.parametrize("zoom", [0.5, 1, plot.DEFAULT_ZOOM, 2])
def test_whole_hemisphere_is_in_view_when_zoomed_out(zoom):
    assert plot.visible_dot(zoom) == 0.0


def test_only_cells_near_the_center_are_in_view_when_zoomed_in():
    # At zoom 4 the image's corner is 0.595 sphere radii from the center
    corner = 750 / (plot.SPHERE_PIXELS_PER_UNIT * 4) * math.sqrt(2)
    assert plot.visible_dot(4) == pytest.approx(math.cos(math.asin(corner)))
    assert 0 < plot.visible_dot(4) < plot.visible_dot(20) < 1


def test_sphere_coastline_is_detailed_when_zoomed_in():
    assert plot.sphere_coastline(plot.DEFAULT_ZOOM) == plot.GLOBE_COASTLINE
    assert plot.sphere_coastline(plot.DETAILED_COASTLINE_ZOOM) == plot.GLOBE_COASTLINE
    assert plot.sphere_coastline(4) == plot.COASTLINE


def test_view_center():
    assert plot.view_center(None) == (0.0, 0.0)
    assert plot.view_center([50, -10]) == (50.0, -10.0)


@pytest.mark.parametrize(
    ("spans", "size"),
    [
        ((10, 5), (plot.FLAT_MAP_WIDTH, 5.0)),
        ((10, 10), (plot.FLAT_MAP_HEIGHT, plot.FLAT_MAP_HEIGHT)),
        ((10, 20), (plot.FLAT_MAP_HEIGHT / 2, plot.FLAT_MAP_HEIGHT)),
        ((360, 30), (plot.FLAT_MAP_WIDTH, plot.FLAT_MIN_MAP_HEIGHT)),
    ],
)
def test_flat_map_keeps_the_shape_of_the_region(spans, size):
    assert plot.flat_map_inches(*spans) == pytest.approx(size)


def test_flat_maps_fit_the_image_limits():
    for spans in [(10, 5), (5, 10), (360, 30), (1, 1)]:
        width, height = plot.flat_map_inches(*spans)
        assert width <= plot.FLAT_MAP_WIDTH
        assert height <= plot.FLAT_MAP_HEIGHT


def test_flat_cell_pixels():
    # A 111.2 km cell is about a degree, and a map 10 degrees across 10 inches
    # gives each degree FLAT_DPI pixels
    degree_km = math.pi * plot.SPHERE_RADIUS_KM / 180
    assert plot.flat_cell_pixels(degree_km, 10, 10) == pytest.approx(plot.FLAT_DPI)
    assert plot.flat_cell_pixels(10, 9.9, 10) > plot.MIN_CELL_PIXELS
    assert plot.flat_cell_pixels(60, 360, 10) < plot.MIN_CELL_PIXELS


@pytest.mark.parametrize(
    ("spans", "center", "states"),
    [
        ((9, 6), (39, -105), True),
        ((50, 30), (39, -105), True),
        ((61, 30), (39, -105), False),
        ((50, 30), (50, 10), False),
        ((29, 20), (50, 10), True),
        ((360, 40), (0, 0), False),
    ],
)
def test_shows_states(spans, center, states):
    assert plot.shows_states(*spans, *center) == states


@pytest.mark.parametrize(
    ("lon", "center", "wrapped"),
    [(10, 0, 10), (190, 0, -170), (-170, 170, 20), (179, -179, -2), (0, 180, -180)],
)
def test_wrap_longitude(lon, center, wrapped):
    assert plot.wrap_longitude(lon, center) == pytest.approx(wrapped)
