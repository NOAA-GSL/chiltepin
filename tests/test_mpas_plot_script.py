# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh plotting script.

The agent runs the script with the Python of its plotting environment. The test
extra installs the same plotting packages, so these tests also draw real
meshes in-process: the 480km global mesh, and regional meshes made from it.
The integration test runs the script in the agent's own environment.
"""

import io
import math
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import shapefile
from netCDF4 import Dataset
from PIL import Image

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


# Drawing


def rgb(color):
    """Return a "#rrggbb" color as an (r, g, b) tuple."""
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def count_color(image, color, tolerance=12):
    """Return how many pixels of a PNG are within ``tolerance`` of a color."""
    pixels = np.asarray(Image.open(image).convert("RGB")).reshape(-1, 3).astype(int)
    return int(np.sum(np.abs(pixels - rgb(color)).max(axis=1) <= tolerance))


def count_light_gray(image):
    """Return how many pixels of a PNG are light gray, as thin gray edges are."""
    pixels = np.asarray(Image.open(image).convert("RGB")).astype(int)
    neutral = np.ptp(pixels, axis=2) <= 6
    value = pixels[..., 0]
    return int(np.sum(neutral & (value >= 140) & (value <= 235)))


def has_color(image, color, tolerance=12):
    """Whether any pixel of a PNG is within ``tolerance`` of a color."""
    return count_color(image, color, tolerance) > 0


@pytest.fixture(scope="module")
def global_mesh(tmp_path_factory):
    """Download the 480km global static mesh once for this module."""
    directory = tmp_path_factory.mktemp("plot-meshes")
    url = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes/x1.2562_static.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extract("x1.2562.static.nc", path=directory, filter="data")
    return directory / "x1.2562.static.nc"


def regional_copy(global_mesh, path, keep):
    """Write a regional mesh with the global mesh's cells for which ``keep`` is true.

    Cells are colored by boundary layer, from 1 at the edge of the kept cells to
    7, as a cut from a global mesh would be. Vertices are kept whole, since the
    plot only looks up the corners of its cells.
    """
    with Dataset(global_mesh) as source:
        lat = np.degrees(source["latCell"][:])
        lon = np.degrees(source["lonCell"][:])
        cells = np.where(keep(lat, lon))[0]
        layers = np.zeros(len(cells), dtype="i4")
        layers[: min(len(cells), 7 * 40)] = np.repeat(np.arange(1, 8), 40)[
            : min(len(cells), 280)
        ]
        with Dataset(path, "w") as target:
            target.sphere_radius = source.sphere_radius
            target.createDimension("nCells", len(cells))
            target.createDimension("nVertices", source.dimensions["nVertices"].size)
            target.createDimension("maxEdges", source.dimensions["maxEdges"].size)
            for name in ("latVertex", "lonVertex"):
                target.createVariable(name, "f8", ("nVertices",))[:] = source[name][:]
            for name in ("latCell", "lonCell", "areaCell"):
                target.createVariable(name, "f8", ("nCells",))[:] = source[name][cells]
            target.createVariable("nEdgesOnCell", "i4", ("nCells",))[:] = source[
                "nEdgesOnCell"
            ][cells]
            target.createVariable("verticesOnCell", "i4", ("nCells", "maxEdges"))[:] = (
                source["verticesOnCell"][cells]
            )
            target.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = layers
    return path


@pytest.fixture(scope="module")
def regional_mesh(global_mesh, tmp_path_factory):
    """A regional mesh of the 480km cells within about 2000 km of Colorado."""
    directory = tmp_path_factory.mktemp("plot-regional")

    def near_colorado(lat, lon):
        return (np.abs(lat - 39) < 18) & (np.abs(lon + 105) < 22)

    return regional_copy(global_mesh, directory / "co.static.nc", near_colorado)


@pytest.fixture(scope="module")
def band_mesh(global_mesh, tmp_path_factory):
    """A regional mesh of the 480km cells in a band around the globe."""
    directory = tmp_path_factory.mktemp("plot-band")
    return regional_copy(
        global_mesh, directory / "band.static.nc", lambda lat, lon: np.abs(lat) < 30
    )


@pytest.fixture(scope="module")
def map_data(tmp_path_factory):
    """Write small stand-ins for the four Natural Earth map layers."""
    directory = tmp_path_factory.mktemp("plot-map")
    lines = {
        plot.COASTLINE: [[[-125, 30], [-80, 30], [-80, 48]], [[-120, 10], [-100, 15]]],
        plot.BORDERS: [[[-110, 32], [-100, 32]]],
        plot.STATES: [[[-109, 37], [-102, 37], [-102, 41], [-109, 41], [-109, 37]]],
        plot.GLOBE_COASTLINE: [[[-30, -40], [30, -40], [30, 40]], [[0, 0], [10, 10]]],
    }
    for layer, parts in lines.items():
        path = plot.layer_path(directory, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        with shapefile.Writer(str(path), shapeType=shapefile.POLYLINE) as writer:
            writer.field("name", "C")
            # One shape with several parts and one with a single part
            writer.line(parts)
            writer.record("multi")
            writer.line(parts[:1])
            writer.record("single")
    return directory


def run_main(mesh, output, map_dir, *options):
    return plot.main([str(mesh), str(output), "--map-data", str(map_dir), *options])


def test_global_mesh_is_not_regional(global_mesh, regional_mesh, band_mesh):
    assert not plot.is_regional(global_mesh)
    assert plot.is_regional(regional_mesh)
    assert plot.is_regional(band_mesh)


def test_mesh_without_boundary_layers_is_global(tmp_path):
    path = tmp_path / "bare.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("nCells", 3)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = [0, 1, 2]
    assert not plot.is_regional(path)


def test_draw_global_mesh_on_a_sphere(tmp_path, global_mesh, map_data):
    output = tmp_path / "g.png"
    assert run_main(global_mesh, output, map_data, "--title", "Global") == 0
    with Image.open(output) as image:
        assert image.size == (1500, 1590)
    # 480km cells are large enough for edges, which leave about 110000 light gray
    # pixels, against about 6000 from the outline and coastlines alone. The
    # coastline is drawn too.
    assert count_light_gray(output) > 50_000
    assert has_color(output, "#00008b", tolerance=30)


def test_small_cells_on_a_sphere_have_no_edges(tmp_path, global_mesh, map_data):
    near, far = tmp_path / "near.png", tmp_path / "far.png"
    assert run_main(global_mesh, near, map_data) == 0
    # Zoomed far out, the cells are a few pixels across, and only the globe's
    # outline and coastlines leave some gray pixels where they blend with white.
    # The edges of the near globe leave about a hundred times more.
    assert run_main(global_mesh, far, map_data, "--zoom", "0.15") == 0
    assert count_light_gray(far) < count_light_gray(near) / 20


@pytest.mark.parametrize(
    "view",
    [
        ["--zoom", "4", "--center", "45", "-100"],
        # Above a pole, the globe's outline is found another way
        ["--center", "90", "0"],
        # So far zoomed in that no cell center is in view
        ["--zoom", "5000", "--center", "10", "10"],
    ],
)
def test_draw_global_mesh_with_a_view(tmp_path, global_mesh, map_data, view):
    output = tmp_path / "view.png"
    assert run_main(global_mesh, output, map_data, *view) == 0
    with Image.open(output) as image:
        assert image.size == (1500, 1590)


def test_draw_regional_mesh_on_a_flat_map(tmp_path, regional_mesh, map_data):
    output = tmp_path / "co.png"
    assert run_main(regional_mesh, output, map_data) == 0
    with Image.open(output) as image:
        width, height = image.size
    # This region is tall for its width, so its map is as tall as allowed and
    # narrower than the widest
    assert height == round(
        (plot.FLAT_MARGIN_INCHES + plot.FLAT_MAP_HEIGHT + plot.FLAT_TITLE_INCHES)
        * plot.FLAT_DPI
    )
    assert width < round(
        (plot.FLAT_MARGIN_INCHES + plot.FLAT_MAP_WIDTH + plot.FLAT_BAR_INCHES)
        * plot.FLAT_DPI
    )
    # Edges, which leave about 4500 light gray pixels against about 900 without
    # them, and the boundary layers on their fixed colors
    assert count_light_gray(output) > 2_500
    for color in plot.LAYER_COLORS[1:]:
        assert has_color(output, color, tolerance=8), color


def test_draw_band_around_the_globe(tmp_path, band_mesh, map_data):
    output = tmp_path / "band.png"
    assert run_main(band_mesh, output, map_data, "--title", "Band") == 0
    with Image.open(output) as image:
        width, height = image.size
    # A wide band gets the widest map, and a short one
    assert width == round(
        (plot.FLAT_MARGIN_INCHES + plot.FLAT_MAP_WIDTH + plot.FLAT_BAR_INCHES)
        * plot.FLAT_DPI
    )
    assert height < width / 2


def test_missing_map_layers_are_reported(tmp_path, global_mesh, regional_mesh, capsys):
    empty = tmp_path / "empty"
    output = tmp_path / "g.png"
    assert run_main(global_mesh, output, empty) == plot.EXIT_MISSING_MAP
    assert output.is_file()
    assert "ne_110m_coastline.shp" in capsys.readouterr().err
    assert run_main(regional_mesh, tmp_path / "r.png", empty) == plot.EXIT_MISSING_MAP
    message = capsys.readouterr().err
    assert "ne_10m_coastline.shp" in message
    assert "ne_10m_admin_1_states_provinces_lines.shp" in message


def test_empty_map_layer_counts_as_missing(tmp_path, global_mesh, map_data):
    broken = tmp_path / "broken"
    path = plot.layer_path(broken, plot.GLOBE_COASTLINE)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    assert run_main(global_mesh, tmp_path / "g.png", broken) == plot.EXIT_MISSING_MAP


@pytest.mark.parametrize(
    "view", [["--zoom", "2"], ["--center", "39", "-105"]], ids=["zoom", "center"]
)
def test_view_of_a_regional_mesh_is_refused(
    tmp_path, regional_mesh, map_data, view, capsys
):
    output = tmp_path / "r.png"
    assert run_main(regional_mesh, output, map_data, *view) == (
        plot.EXIT_VIEW_OF_REGIONAL_MESH
    )
    assert not output.exists()
    assert "apply only to global meshes" in capsys.readouterr().err


def test_script_runs_as_a_program(tmp_path, global_mesh, map_data):
    output = tmp_path / "g.png"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), str(global_mesh), str(output)]
        + ["--map-data", str(map_data)],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")},
    )
    assert done.returncode == 0, done.stderr
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
