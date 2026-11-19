# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh plotting script.

The agent runs the script with the Python of its plotting environment. The test
extra installs the same plotting packages, so these tests also draw real
meshes in-process: the 480km global mesh, and regional meshes made from it.
The integration test runs the script in the agent's own environment.
"""

import http.client
import io
import math
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
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
        ["m.nc", "out.png", "--map-data", "/d", "--title=T", "--zoom=2.5"]
        + ["--center-lat=50.0", "--center-lon=-10.5"]
    )
    assert args.mesh == Path("m.nc")
    assert args.output == Path("out.png")
    assert args.map_data == Path("/d")
    assert args.title == "T"
    assert args.zoom == 2.5
    assert (args.center_lat, args.center_lon) == (50.0, -10.5)
    defaults = plot.parse_args(["m.nc", "out.png", "--map-data", "/d"])
    assert (defaults.title, defaults.zoom, defaults.center_lat) == (None, None, None)


def test_parse_args_reads_values_starting_with_a_dash():
    # As the agent passes them, which argparse could otherwise take for options
    args = plot.parse_args(
        ["m.nc", "out.png", "--map-data", "/d", "--title=-x --zoom 3"]
        + [f"--center-lat={-1e-05!r}", f"--center-lon={-1e-05!r}"]
    )
    assert args.title == "-x --zoom 3"
    assert (args.center_lat, args.center_lon) == (-1e-05, -1e-05)


@pytest.mark.parametrize("option", ["--center-lat=1", "--center-lon=1"])
def test_parse_args_needs_both_center_coordinates(option, capsys):
    with pytest.raises(SystemExit) as error:
        plot.parse_args(["m.nc", "out.png", "--map-data", "/d", option])
    assert error.value.code == 2
    assert "must be given together" in capsys.readouterr().err


def test_exit_codes_differ_from_those_of_python_and_argparse():
    codes = {plot.EXIT_MISSING_MAP, plot.EXIT_VIEW_OF_REGIONAL_MESH}
    assert len(codes) == 2
    assert not codes & {0, 1, 2}


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
    assert plot.view_center(None, None) == (0.0, 0.0)
    assert plot.view_center(50, -10) == (50.0, -10.0)


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


# Downloading the map layers


def layer_zip(layer, extra=()):
    """Return a zip file like Natural Earth's for a layer, with extra members."""
    stem = f"ne_{layer[0]}_{layer[2]}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for suffix in plot.SHAPEFILE_PARTS:
            archive.writestr(f"{stem}{suffix}", f"{stem}{suffix} data")
        archive.writestr(f"{stem}.README.html", "readme")
        for name in extra:
            archive.writestr(name, "unwanted")
    return buffer.getvalue()


def fake_natural_earth(monkeypatch, failures=0, bodies=None, error=None):
    """Serve the layers' zip files to urlopen, failing ``failures`` times first.

    A failure raises ``error``, by default an unreachable server.
    """
    calls = []
    by_url = {plot.layer_url(layer): layer_zip(layer) for layer in plot.MAP_LAYERS}
    by_url.update(bodies or {})

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) <= failures:
            raise error or urllib.error.URLError("unreachable")
        return io.BytesIO(by_url[url])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(plot.time, "sleep", lambda seconds: None)
    return calls


def test_layer_url():
    assert plot.layer_url(plot.GLOBE_COASTLINE) == (
        "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip"
    )
    assert plot.layer_url(plot.STATES) == (
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_1_states_provinces_lines.zip"
    )


def test_download_map_data(monkeypatch, tmp_path):
    calls = fake_natural_earth(monkeypatch)
    plot.download_map_data(tmp_path)
    assert calls == [plot.layer_url(layer) for layer in plot.MAP_LAYERS]
    for layer in plot.MAP_LAYERS:
        shp = plot.layer_path(tmp_path, layer)
        assert shp.read_text() == f"{shp.name} data"
        # Only the shapefile's parts, and no partial files
        assert sorted(path.name for path in shp.parent.iterdir()) == sorted(
            path.name
            for path in shp.parent.iterdir()
            if path.suffix in plot.SHAPEFILE_PARTS
        )
    names = {path.name for path in tmp_path.rglob("*")}
    assert not any(name.endswith(".partial") or "README" in name for name in names)


def test_download_writes_each_shapefile_last(monkeypatch, tmp_path):
    fake_natural_earth(monkeypatch)
    order = []
    replace = Path.replace

    def record(self, target):
        order.append(Path(target).name)
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", record)
    plot.download_map_data(tmp_path)
    for layer in plot.MAP_LAYERS:
        stem = f"ne_{layer[0]}_{layer[2]}"
        mine = [name for name in order if name.startswith(stem + ".")]
        assert mine[-1] == f"{stem}.shp"


def test_download_ignores_other_members(monkeypatch, tmp_path):
    layer = plot.GLOBE_COASTLINE
    body = layer_zip(layer, extra=["../escape.shp", "sub/ne_110m_coastline.shp"])
    fake_natural_earth(monkeypatch, bodies={plot.layer_url(layer): body})
    plot.download_map_data(tmp_path / "data")
    assert not (tmp_path / "escape.shp").exists()
    assert not any(path.name == "sub" for path in (tmp_path / "data").rglob("*"))


def test_download_retries(monkeypatch, tmp_path):
    calls = fake_natural_earth(monkeypatch, failures=2)
    plot.download_map_data(tmp_path)
    assert len(calls) == len(plot.MAP_LAYERS) + 2


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "error", {}, None)


@pytest.mark.parametrize(
    "error",
    [http.client.IncompleteRead(b"part", 10), http_error(503), TimeoutError()],
    ids=["incomplete", "503", "timeout"],
)
def test_download_retries_transient_failures(monkeypatch, tmp_path, error):
    calls = fake_natural_earth(monkeypatch, failures=2, error=error)
    plot.download_map_data(tmp_path)
    assert len(calls) == len(plot.MAP_LAYERS) + 2


def test_download_does_not_retry_a_wrong_request(monkeypatch, tmp_path):
    calls = fake_natural_earth(monkeypatch, failures=1, error=http_error(404))
    with pytest.raises(urllib.error.HTTPError):
        plot.download_map_data(tmp_path)
    assert len(calls) == 1


def test_download_gives_up_on_a_server_error(monkeypatch, tmp_path):
    calls = fake_natural_earth(monkeypatch, failures=3, error=http_error(500))
    with pytest.raises(urllib.error.HTTPError):
        plot.download_map_data(tmp_path)
    assert len(calls) == 3


def test_download_keeps_layers_already_there(monkeypatch, tmp_path):
    kept = plot.layer_path(tmp_path, plot.COASTLINE)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"kept")
    # An empty layer is downloaded again
    empty = plot.layer_path(tmp_path, plot.STATES)
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    calls = fake_natural_earth(monkeypatch)
    plot.download_map_data(tmp_path)
    assert plot.layer_url(plot.COASTLINE) not in calls
    assert plot.layer_url(plot.STATES) in calls
    assert kept.read_bytes() == b"kept"
    assert empty.read_text() == f"{empty.name} data"


def test_download_gives_up_after_its_attempts(monkeypatch, tmp_path):
    calls = fake_natural_earth(monkeypatch, failures=3)
    with pytest.raises(urllib.error.URLError):
        plot.download_map_data(tmp_path)
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("body", "message"),
    [(b"not a zip", "is not a zip file"), (None, "holds no ne_110m_coastline.shp")],
)
def test_download_rejects_a_wrong_file(monkeypatch, tmp_path, body, message):
    if body is None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("other.shp", "x")
        body = buffer.getvalue()
    fake_natural_earth(monkeypatch, bodies={plot.layer_url(plot.GLOBE_COASTLINE): body})
    with pytest.raises(ValueError, match=message):
        plot.download_map_data(tmp_path)


def test_main_downloads_the_map_data(monkeypatch, tmp_path):
    fake_natural_earth(monkeypatch)
    assert plot.main(["--download-map-data", str(tmp_path)]) == 0
    assert all(plot.layer_path(tmp_path, layer).is_file() for layer in plot.MAP_LAYERS)


@pytest.mark.parametrize(
    "argv", [["--download-map-data"], ["--download-map-data", "a", "b"]]
)
def test_main_needs_one_map_data_directory(argv, capsys):
    assert plot.main(argv) == 1
    assert "Usage" in capsys.readouterr().err


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


def regional_copy(
    global_mesh,
    path,
    keep,
    layer=None,
    mask=True,
    area_scale=1.0,
    radius=True,
    padding=None,
):
    """Write a regional mesh with the global mesh's cells for which ``keep`` is true.

    Cells are in boundary layers 1 to 7, 40 cells each, and the rest are
    interior, unless ``layer`` puts every cell in one. ``mask=False`` leaves out
    ``bdyMaskCell``, ``radius=False`` leaves out ``sphere_radius``, and
    ``area_scale`` scales the cell areas. ``padding`` replaces the unused
    entries of ``verticesOnCell``. Vertices are kept whole, since the plot only
    looks up the corners of its cells.
    """
    with Dataset(global_mesh) as source:
        lat = np.degrees(source["latCell"][:])
        lon = np.degrees(source["lonCell"][:])
        cells = np.where(keep(lat, lon))[0]
        layers = np.zeros(len(cells), dtype="i4")
        layers[: min(len(cells), 7 * 40)] = np.repeat(np.arange(1, 8), 40)[
            : min(len(cells), 280)
        ]
        if layer is not None:
            layers[:] = layer
        on_cell = source["verticesOnCell"][cells]
        if padding is not None:
            unused = (
                np.arange(on_cell.shape[1]) >= source["nEdgesOnCell"][cells][:, None]
            )
            on_cell[unused] = padding
        with Dataset(path, "w") as target:
            if radius:
                target.sphere_radius = source.sphere_radius
            target.createDimension("nCells", len(cells))
            target.createDimension("nVertices", source.dimensions["nVertices"].size)
            target.createDimension("maxEdges", source.dimensions["maxEdges"].size)
            for name in ("latVertex", "lonVertex"):
                target.createVariable(name, "f8", ("nVertices",))[:] = source[name][:]
            for name in ("latCell", "lonCell", "xCell", "yCell", "zCell"):
                target.createVariable(name, "f8", ("nCells",))[:] = source[name][cells]
            target.createVariable("areaCell", "f8", ("nCells",))[:] = (
                source["areaCell"][cells] * area_scale
            )
            target.createVariable("nEdgesOnCell", "i4", ("nCells",))[:] = source[
                "nEdgesOnCell"
            ][cells]
            target.createVariable("verticesOnCell", "i4", ("nCells", "maxEdges"))[:] = (
                on_cell
            )
            if mask:
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


def test_mesh_without_boundary_layers_or_areas_is_global(tmp_path):
    path = tmp_path / "bare.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("nCells", 3)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = [0, 1, 2]
    assert not plot.is_regional(path)


@pytest.mark.parametrize("radius", [True, False], ids=["radius", "no-radius"])
def test_region_without_boundary_layers_is_regional(tmp_path, global_mesh, radius):
    # By the part of the sphere its cells cover, with the sphere's radius from
    # the file or from the distance of its cells from the center
    region = regional_copy(
        global_mesh,
        tmp_path / "r.nc",
        lambda lat, lon: lat > -60,
        mask=False,
        radius=radius,
    )
    assert plot.is_regional(region)
    whole = regional_copy(
        global_mesh,
        tmp_path / "g.nc",
        lambda lat, lon: lat > -91,
        mask=False,
        radius=radius,
    )
    assert not plot.is_regional(whole)


def test_sphere_radius(tmp_path, global_mesh):
    with Dataset(global_mesh) as ds:
        assert plot.sphere_radius(ds) == ds.sphere_radius == 6371229.0
    bare = regional_copy(
        global_mesh, tmp_path / "r.nc", lambda a, b: a > 0, radius=False
    )
    with Dataset(bare) as ds:
        assert plot.sphere_radius(ds) == pytest.approx(6371229.0)


def test_draw_global_mesh_on_a_sphere(tmp_path, global_mesh, map_data):
    output = tmp_path / "g.png"
    assert run_main(global_mesh, output, map_data, "--title=Global") == 0
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
    assert run_main(global_mesh, far, map_data, "--zoom=0.15") == 0
    assert count_light_gray(far) < count_light_gray(near) / 20


@pytest.mark.parametrize(
    "view",
    [
        ["--zoom=4", "--center-lat=45", "--center-lon=-100"],
        # Above a pole, the globe's outline is found another way
        ["--center-lat=90", "--center-lon=0"],
        ["--center-lat=-90", "--center-lon=0"],
    ],
)
def test_draw_global_mesh_with_a_view(tmp_path, global_mesh, map_data, view):
    output = tmp_path / "view.png"
    assert run_main(global_mesh, output, map_data, *view) == 0
    with Image.open(output) as image:
        assert image.size == (1500, 1590)
    # The cells' edges are drawn, leaving 40000 or more light gray pixels,
    # against a few thousand from a globe without them
    assert count_light_gray(output) > 20_000


def test_draw_global_mesh_with_no_cell_in_view(tmp_path, global_mesh, map_data):
    output = tmp_path / "view.png"
    view = ["--zoom=5000", "--center-lat=10", "--center-lon=10"]
    assert run_main(global_mesh, output, map_data, *view) == 0
    with Image.open(output) as image:
        assert image.size == (1500, 1590)


@pytest.mark.parametrize("zoom", [1.0, plot.DEFAULT_ZOOM])
def test_sphere_has_the_measured_size(tmp_path, global_mesh, zoom):
    output = tmp_path / "g.png"
    run_main(global_mesh, output, tmp_path / "none", f"--zoom={zoom!r}")
    pixels = np.asarray(Image.open(output).convert("RGB")).astype(int)
    # The globe's center is half the square image below the title
    title = pixels.shape[0] - pixels.shape[1]
    row = pixels[title + pixels.shape[1] // 2]
    drawn = np.where(row.min(axis=1) < 200)[0]
    diameter = drawn.max() - drawn.min()
    assert diameter == pytest.approx(2 * plot.sphere_pixels_per_unit(zoom), abs=4)


def test_title_is_drawn_as_written(tmp_path, global_mesh, regional_mesh, map_data):
    # Dollar signs would otherwise start a formula, which this one breaks
    title = "--title=a$\\bad$"
    assert run_main(global_mesh, tmp_path / "g.png", map_data, title) == 0
    assert run_main(regional_mesh, tmp_path / "r.png", map_data, title) == 0


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
    assert run_main(band_mesh, output, map_data, "--title=Band") == 0
    with Image.open(output) as image:
        width, height = image.size
    # A wide band gets the widest map, and a short one
    assert width == round(
        (plot.FLAT_MARGIN_INCHES + plot.FLAT_MAP_WIDTH + plot.FLAT_BAR_INCHES)
        * plot.FLAT_DPI
    )
    assert height < width / 2


def map_pixels(image):
    """Return the pixels of a flat map inside its black frame, as an array.

    The frame's top edge is the first row with a long dark run, which the
    color bar's narrow outline does not have.
    """
    pixels = np.asarray(Image.open(image).convert("RGB")).astype(int)
    dark = (pixels < 60).all(axis=-1)
    rows = np.where(dark.sum(axis=1) > 200)[0]
    top, bottom = rows.min(), rows.max()
    columns = np.where(dark[top])[0]
    runs = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
    frame = max(runs, key=len)
    return pixels[top + 3 : bottom - 2, frame.min() + 3 : frame.max() - 2]


def blank(pixels):
    """Return which pixels are white, where no cell was drawn."""
    return (pixels >= 250).all(axis=-1)


def test_flat_map_of_small_cells_has_no_edges(tmp_path, global_mesh, map_data):
    def near_colorado(lat, lon):
        return (np.abs(lat - 39) < 18) & (np.abs(lon + 105) < 22)

    large = regional_copy(global_mesh, tmp_path / "large.nc", near_colorado, layer=0)
    # Cells a 400th the area are 24 km across, and a pixel or two on the map
    small = regional_copy(
        global_mesh, tmp_path / "small.nc", near_colorado, layer=0, area_scale=1 / 400
    )
    assert run_main(large, tmp_path / "large.png", map_data) == 0
    assert run_main(small, tmp_path / "small.png", map_data) == 0
    assert count_light_gray(tmp_path / "small.png") < (
        count_light_gray(tmp_path / "large.png") / 3
    )


def test_flat_map_ignores_unused_corners(tmp_path, global_mesh, map_data):
    def near_pentagon(lat, lon):
        # A pentagon of the icosahedral mesh is at 26.57N 36E
        return (np.abs(lat - 26.57) < 15) & (np.abs(lon - 36) < 15)

    # The unused corners of a pentagon point to another vertex far away
    plain = regional_copy(global_mesh, tmp_path / "plain.nc", near_pentagon)
    padded = regional_copy(
        global_mesh, tmp_path / "padded.nc", near_pentagon, padding=1
    )
    with Dataset(padded) as ds:
        assert 5 in ds["nEdgesOnCell"][:]
    assert run_main(plain, tmp_path / "plain.png", map_data, "--title=T") == 0
    assert run_main(padded, tmp_path / "padded.png", map_data, "--title=T") == 0
    assert np.array_equal(
        np.asarray(Image.open(tmp_path / "plain.png")),
        np.asarray(Image.open(tmp_path / "padded.png")),
    )


def test_region_across_the_180th_meridian(tmp_path, global_mesh):
    def pacific(lat, lon):
        return (np.abs(lat) < 20) & (np.abs(lon) > 150)

    region = regional_copy(global_mesh, tmp_path / "p.nc", pacific, layer=3)
    output = tmp_path / "p.png"
    assert run_main(region, output, tmp_path / "none") == plot.EXIT_MISSING_MAP
    # The region is about 60 degrees by 40, so it is drawn in one piece on a
    # map of that shape, not as two ends of a map of the whole globe
    pixels = map_pixels(output)
    assert pixels.shape[1] / pixels.shape[0] == pytest.approx(1.5, rel=0.1)
    middle = pixels[:, pixels.shape[1] // 10 : -pixels.shape[1] // 10]
    assert not blank(middle[middle.shape[0] // 4 : -middle.shape[0] // 4]).any()


def test_region_around_a_pole(tmp_path, global_mesh):
    region = regional_copy(global_mesh, tmp_path / "n.nc", lambda a, b: a > 55, layer=3)
    output = tmp_path / "n.png"
    assert run_main(region, output, tmp_path / "none") == plot.EXIT_MISSING_MAP
    # The cell around the pole fills the top of the map, up to the pole
    pixels = map_pixels(output)
    assert not blank(pixels[:10]).any()
    assert has_color(output, plot.LAYER_COLORS[3], tolerance=8)


def test_band_has_no_gap_at_the_seam(tmp_path, global_mesh):
    band = regional_copy(
        global_mesh, tmp_path / "b.nc", lambda a, b: np.abs(a) < 30, layer=3
    )
    output = tmp_path / "b.png"
    assert run_main(band, output, tmp_path / "none") == plot.EXIT_MISSING_MAP
    # Cells cut by the map's left and right edges are drawn on both, as
    # cartopy wraps the parts past the seam
    pixels = map_pixels(output)
    rows = pixels[pixels.shape[0] // 4 : -pixels.shape[0] // 4]
    assert not blank(rows[:, :5]).any()
    assert not blank(rows[:, -5:]).any()


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
    "view",
    [["--zoom=2"], ["--center-lat=39", "--center-lon=-105"]],
    ids=["zoom", "center"],
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
