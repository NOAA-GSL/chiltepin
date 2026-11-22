# SPDX-License-Identifier: Apache-2.0
"""Draw a PNG image of an MPAS mesh.

This is a standalone script, which the MPAS mesh agent runs with the Python of
its tasks on the ``compute`` resource. Drawing needs numpy, netCDF4, matplotlib
and cartopy, which the ``chiltepin[mpas]`` extra installs. The agent imports this
module for its exit codes and map layers without them, so it imports only the
standard library at the top, and the plotting packages inside the functions that
draw.

A mesh is regional if any cell has a nonzero ``bdyMaskCell``, or its cells cover
less of the sphere than a global mesh does, and global otherwise. A regional
mesh is drawn on a flat map, with its cells colored by boundary layer, or all
as interior if it has no ``bdyMaskCell``. A global mesh is drawn on a sphere,
seen from above the view center. Cell edges are drawn only when cells come out
at least ``MIN_CELL_PIXELS`` across in the image, since smaller ones blur together.

Usage::

    python _plot_mesh.py MESH OUTPUT --map-data DIR [--title=TITLE]
        [--zoom=ZOOM] [--center-lat=LAT --center-lon=LON]
    python _plot_mesh.py --download-map-data DIR

The second form downloads the Natural Earth map layers the plots draw to DIR,
where cartopy finds them, skipping complete layers already there. It needs only
the standard library, so it runs on the ``service`` resource, which can reach
Natural Earth, without the extra.

Exit codes:

- 0: the image was written
- 4: the image was written, but without one or more map layers, which are named
  on standard error
- 5: a zoom or view center was given for a regional mesh, and nothing was
  written
- anything else: the plot failed. argparse exits with 2, and Python with 1.

Options are passed as ``--option=VALUE``, so a value starting with "-", such as
a title or a negative number, is not read as an option.
"""

import argparse
import http.client
import io
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Unlike 1 and 2, which Python and argparse exit with
EXIT_MISSING_MAP = 4
EXIT_VIEW_OF_REGIONAL_MESH = 5

# Radius of the sphere MPAS meshes are defined on, in km
SPHERE_RADIUS_KM = 6371.229

# A mesh whose cells cover less than this part of the sphere is regional
GLOBAL_AREA_FRACTION = 0.999

# Cells smaller than this many pixels across are drawn without edges
MIN_CELL_PIXELS = 10.0

# The sphere is drawn on a square figure, with an orthographic projection. At
# zoom 1 a unit sphere is SPHERE_PIXELS_PER_UNIT pixels in radius, as measured,
# and zoom scales that. DEFAULT_ZOOM makes the globe fill 95% of the width.
SPHERE_FIGURE_INCHES = 10.0
SPHERE_DPI = 150
SPHERE_PIXELS_PER_UNIT = 445.5
DEFAULT_ZOOM = 1.6

# Above this zoom the sphere's coastlines come from the detailed 10m layer
DETAILED_COASTLINE_ZOOM = 2.0

# A flat map is at most FLAT_MAP_WIDTH by FLAT_MAP_HEIGHT inches, keeps the
# shape of the region, and has room around it for the title and color bar
FLAT_DPI = 180
FLAT_MAP_WIDTH = 10.0
FLAT_MAP_HEIGHT = 6.8
FLAT_MIN_MAP_HEIGHT = 2.0
FLAT_BAR_INCHES = 1.6
FLAT_TITLE_INCHES = 0.6
FLAT_MARGIN_INCHES = 0.2

# Colors of boundary layers 0 to 7. Layer 0 is the interior.
LAYER_COLORS = (
    "#ffffff",
    "#d0e8ff",
    "#80c8ff",
    "#40a0e0",
    "#ffcc66",
    "#ff8844",
    "#e05050",
    "#aa3090",
)
EDGE_COLOR = "#999999"

# Natural Earth map layers, as (resolution, category, name)
COASTLINE = ("10m", "physical", "coastline")
BORDERS = ("10m", "cultural", "admin_0_boundary_lines_land")
STATES = ("10m", "cultural", "admin_1_states_provinces_lines")
GLOBE_COASTLINE = ("110m", "physical", "coastline")
MAP_LAYERS = (COASTLINE, BORDERS, STATES, GLOBE_COASTLINE)

# Where Natural Earth layers are downloaded from, as cartopy does
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/{resolution}_{category}/"
    "ne_{resolution}_{name}.zip"
)

# Parts of a shapefile kept from a downloaded layer, and those a layer needs
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
REQUIRED_PARTS = (".shp", ".shx", ".dbf")

# HTTP statuses of failed downloads worth trying again, besides 5xx
RETRY_STATUSES = (408, 429)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Return the command-line arguments."""
    parser = argparse.ArgumentParser(description="Draw a PNG image of an MPAS mesh")
    parser.add_argument("mesh", type=Path, help="MPAS static or grid file")
    parser.add_argument("output", type=Path, help="PNG file to write")
    parser.add_argument(
        "--map-data", type=Path, required=True, help="cartopy data directory"
    )
    parser.add_argument("--title", help="title of the image")
    parser.add_argument("--zoom", type=float, help="zoom of a global mesh")
    parser.add_argument(
        "--center-lat", type=float, help="view center latitude of a global mesh"
    )
    parser.add_argument(
        "--center-lon", type=float, help="view center longitude of a global mesh"
    )
    args = parser.parse_args(argv)
    if (args.center_lat is None) != (args.center_lon is None):
        parser.error("--center-lat and --center-lon must be given together")
    return args


def default_title(mesh: Path) -> str:
    """Return a title naming the mesh, such as "MPAS mesh: conus"."""
    return f"MPAS mesh: {mesh.name.removesuffix('.nc').removesuffix('.static')}"


def layer_path(map_data: Path, layer: Tuple[str, str, str]) -> Path:
    """Return the path cartopy stores a Natural Earth layer at."""
    resolution, category, name = layer
    return (
        map_data
        / "shapefiles"
        / "natural_earth"
        / category
        / f"ne_{resolution}_{name}.shp"
    )


def layer_complete(path: Path) -> bool:
    """Whether a layer's ``.shp`` file and the other parts it needs are there.

    Each part must be a non-empty file.
    """
    return all(
        part.is_file() and part.stat().st_size > 0
        for part in (path.with_suffix(suffix) for suffix in REQUIRED_PARTS)
    )


def layer_url(layer: Tuple[str, str, str]) -> str:
    """Return the address a Natural Earth layer is downloaded from."""
    resolution, category, name = layer
    return NATURAL_EARTH_URL.format(resolution=resolution, category=category, name=name)


def download_map_data(
    map_data: Path,
    attempts: int = 3,
    timeout: float = 60,
    layers: Sequence[Tuple[str, str, str]] = MAP_LAYERS,
) -> None:
    """Download Natural Earth layers to where cartopy finds them in ``map_data``.

    ``layers`` are the plots' map layers unless given.

    Each layer's shapefile parts are extracted from its zip file, and the
    ``.shp`` file is written last, so a plot never reads a layer whose other
    parts are still being written. A layer already there with all its
    ``REQUIRED_PARTS`` is kept, so plots drawing it are not disturbed, and
    any other is downloaded again. Failed downloads are
    tried again, except when the server answers that the request itself is
    wrong (a 4xx status other than 408 and 429).

    Raises
    ------
    OSError
        If a layer cannot be downloaded after ``attempts`` tries.
    http.client.HTTPException
        If a layer's download keeps ending early.
    ValueError
        If a downloaded file is not a zip file holding the layer's required
        shapefile parts.
    """
    import urllib.error
    import urllib.request

    for layer in layers:
        url = layer_url(layer)
        target = layer_path(map_data, layer)
        if layer_complete(target):
            print(f"Keeping {target}", flush=True)
            continue
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    data = response.read()
                break
            except urllib.error.HTTPError as e:
                final = 400 <= e.code < 500 and e.code not in RETRY_STATUSES
                if final or attempt == attempts:
                    raise
            except (OSError, http.client.HTTPException):
                if attempt == attempts:
                    raise
            time.sleep(2**attempt)
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ValueError(f"{url} is not a zip file") from e
        stem = target.stem
        names = set(archive.namelist())
        absent = [f"{stem}{suffix}" for suffix in REQUIRED_PARTS]
        absent = [name for name in absent if name not in names]
        if absent:
            raise ValueError(f"{url} holds no {', '.join(absent)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only the shapefile's own parts, by name, so nothing lands elsewhere
        parts = [f"{stem}{suffix}" for suffix in SHAPEFILE_PARTS]
        for part in sorted(parts, key=lambda name: name.endswith(".shp")):
            if part in names:
                partial = target.with_name(f".{part}.partial")
                partial.write_bytes(archive.read(part))
                partial.replace(target.with_name(part))
        print(f"Downloaded {url} to {target}", flush=True)


def global_cell_spacing_km(n_cells: int) -> float:
    """Return the mean distance between cell centers of a global hexagonal mesh."""
    return cell_spacing_km(4 * math.pi * SPHERE_RADIUS_KM**2 / n_cells)


def cell_spacing_km(mean_area_km2: float) -> float:
    """Return the distance between centers of hexagons of a given mean area.

    A hexagon whose centers are ``d`` apart has an area of ``sqrt(3)/2 d**2``.
    """
    return math.sqrt(2 * mean_area_km2 / math.sqrt(3))


def sphere_pixels_per_unit(zoom: float) -> float:
    """Return how many pixels one sphere radius spans at a zoom."""
    return SPHERE_PIXELS_PER_UNIT * zoom


def sphere_cell_pixels(spacing_km: float, zoom: float) -> float:
    """Return how many pixels across a cell is at the view center of a sphere."""
    return spacing_km / SPHERE_RADIUS_KM * sphere_pixels_per_unit(zoom)


def visible_dot(zoom: float) -> float:
    """Return the smallest dot product with the view direction of a cell in view.

    The image shows a square of half-width ``(SPHERE_FIGURE_INCHES * SPHERE_DPI
    / 2) / sphere_pixels_per_unit(zoom)`` sphere radii, so cells farther than
    its corner from the view center are out of view. Cells on the far side of
    the globe are never in view.
    """
    half_width = SPHERE_FIGURE_INCHES * SPHERE_DPI / 2 / sphere_pixels_per_unit(zoom)
    corner = half_width * math.sqrt(2)
    if corner >= 1:
        return 0.0
    return math.cos(math.asin(corner))


def sphere_coastline(zoom: float) -> Tuple[str, str, str]:
    """Return the coastline layer for a sphere at a zoom, finer when zoomed in."""
    return COASTLINE if zoom > DETAILED_COASTLINE_ZOOM else GLOBE_COASTLINE


def view_center(lat: Optional[float], lon: Optional[float]) -> Tuple[float, float]:
    """Return the view center of a global mesh, by default 0N 0E."""
    return (0.0, 0.0) if lat is None else (float(lat), float(lon))


def flat_map_inches(lon_span: float, lat_span: float) -> Tuple[float, float]:
    """Return the size of a flat map, which keeps the shape of the region.

    A degree of latitude is drawn as large as a degree of longitude. The map is
    as wide as allowed unless that would make it too tall, and never shorter
    than ``FLAT_MIN_MAP_HEIGHT``, which only a band around most of the globe
    would need.
    """
    width = FLAT_MAP_WIDTH
    height = width * lat_span / lon_span
    if height > FLAT_MAP_HEIGHT:
        width *= FLAT_MAP_HEIGHT / height
        height = FLAT_MAP_HEIGHT
    return width, max(height, FLAT_MIN_MAP_HEIGHT)


def flat_cell_pixels(spacing_km: float, lon_span: float, map_width: float) -> float:
    """Return how many pixels across a cell is on a flat map.

    The map shows ``lon_span`` degrees across ``map_width`` inches. A cell is
    ``spacing_km`` tall, which is the smaller of its two sizes in degrees.
    """
    degrees = spacing_km / (math.pi * SPHERE_RADIUS_KM / 180)
    return degrees * map_width * FLAT_DPI / lon_span


def shows_states(lon_span: float, lat_span: float, lat: float, lon: float) -> bool:
    """Whether a flat map is small enough to draw state and province lines.

    Over North America, where states are large, they are drawn on maps up to 60
    degrees across, and elsewhere up to 30 degrees.
    """
    limit = 60.0 if -130 <= lon <= -60 and 20 <= lat <= 55 else 30.0
    return max(lon_span, lat_span) < limit


def wrap_longitude(lon, center: float):
    """Return longitudes relative to a center, from -180 to 180 degrees.

    ``lon`` can be a number or a numpy array.
    """
    return (lon - center + 180) % 360 - 180


def main(argv: Sequence[str]) -> int:
    """Draw the image or download the map layers, and return the exit code."""
    if argv[:1] == ["--download-map-data"]:
        if len(argv) != 2:
            print("Usage: _plot_mesh.py --download-map-data DIR", file=sys.stderr)
            return 1
        download_map_data(Path(argv[1]))
        return 0
    args = parse_args(argv)
    title = args.title if args.title is not None else default_title(args.mesh)
    regional = is_regional(args.mesh)
    if regional and (args.zoom is not None or args.center_lat is not None):
        print(
            "zoom and the view center apply only to global meshes, and "
            f"{args.mesh} is regional",
            file=sys.stderr,
        )
        return EXIT_VIEW_OF_REGIONAL_MESH
    if regional:
        missing = draw_flat(args.mesh, args.output, title, args.map_data)
    else:
        zoom = args.zoom if args.zoom is not None else DEFAULT_ZOOM
        missing = draw_sphere(
            args.mesh,
            args.output,
            title,
            args.map_data,
            zoom,
            view_center(args.center_lat, args.center_lon),
        )
    if missing:
        print(
            "Plotted without these map layers, which are missing: "
            + ", ".join(str(path) for path in missing),
            file=sys.stderr,
        )
        return EXIT_MISSING_MAP
    return 0


# Cells and vertices are read this many at a time, to bound memory
READ_STEP = 1_000_000


def sphere_radius(ds) -> float:
    """Return the radius of the sphere a mesh is defined on, in its units.

    It is the ``sphere_radius`` attribute, or else the distance of the first
    cell center from the center of the sphere, or else the Earth's radius in
    meters, the unit of MPAS meshes.
    """
    import numpy as np

    if "sphere_radius" in ds.ncattrs():
        return float(ds.sphere_radius)
    if all(f"{axis}Cell" in ds.variables for axis in "xyz"):
        center = [float(ds.variables[f"{axis}Cell"][0]) for axis in "xyz"]
        return float(np.linalg.norm(center))
    return SPHERE_RADIUS_KM * 1000


def is_regional(mesh: Path) -> bool:
    """Whether the mesh is regional.

    It is if any cell is in a boundary layer, or its cells cover less than
    ``GLOBAL_AREA_FRACTION`` of the sphere, as a region cut without boundary
    layers does. A mesh with neither ``bdyMaskCell`` nor ``areaCell`` is global.
    """
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(mesh) as ds:
        n_cells = ds.dimensions["nCells"].size
        pieces = range(0, n_cells, READ_STEP)
        if "bdyMaskCell" in ds.variables:
            mask = ds.variables["bdyMaskCell"]
            if any(np.any(mask[start : start + READ_STEP] != 0) for start in pieces):
                return True
        if "areaCell" not in ds.variables:
            return False
        area = ds.variables["areaCell"]
        total = sum(float(np.sum(area[start : start + READ_STEP])) for start in pieces)
        return total < GLOBAL_AREA_FRACTION * 4 * math.pi * sphere_radius(ds) ** 2


def map_layer(
    map_data: Path, layer: Tuple[str, str, str], missing: List[Path]
) -> Optional[Path]:
    """Return a map layer's path, or record it as missing and return None."""
    path = layer_path(map_data, layer)
    if layer_complete(path):
        return path
    missing.append(path)
    return None


def draw_flat(mesh: Path, output: Path, title: str, map_data: Path) -> List[Path]:
    """Draw a regional mesh on a flat map, and return the missing map layers.

    The whole region is read and drawn at once, which takes about 2.5 GB of
    memory for each million cells.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib
    import numpy as np
    from netCDF4 import Dataset

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap

    with Dataset(mesh) as ds:
        lat_vertex = np.degrees(ds.variables["latVertex"][:])
        lon_vertex = np.degrees(ds.variables["lonVertex"][:])
        lat_cell = np.degrees(ds.variables["latCell"][:])
        lon_cell = np.degrees(ds.variables["lonCell"][:])
        n_edges = ds.variables["nEdgesOnCell"][:]
        on_cell = ds.variables["verticesOnCell"][:] - 1
        # A region without boundary layers is all interior
        if "bdyMaskCell" in ds.variables:
            layer = np.asarray(ds.variables["bdyMaskCell"][:], dtype=float)
        else:
            layer = np.zeros(len(lat_cell))
        # Areas are on the sphere the mesh is defined on, scaled to the Earth
        mean_area_km2 = (
            float(np.mean(ds.variables["areaCell"][:]))
            * (SPHERE_RADIUS_KM / sphere_radius(ds)) ** 2
        )

    # Longitudes relative to the region's center, so a region across the
    # antimeridian is drawn in one piece
    angles = np.radians(lon_cell)
    center_lon = float(
        np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    )
    x_cell = wrap_longitude(lon_cell, center_lon)
    # Cells with fewer corners than the most repeat their last corner
    corners = np.arange(on_cell.shape[1])
    last = np.minimum(corners[None, :], n_edges[:, None] - 1)
    ids = np.take_along_axis(on_cell, last, axis=1)
    x_corner = wrap_longitude(lon_vertex[ids], center_lon)
    # A cell cut by the seam opposite the center keeps its corners together
    offset = x_corner - x_cell[:, None]
    x_corner = np.where(offset > 180, x_corner - 360, x_corner)
    x_corner = np.where(offset < -180, x_corner + 360, x_corner)
    polygons = list(np.stack((x_corner, lat_vertex[ids]), axis=-1))

    # A cell around a pole has corners all the way around it, which on the map
    # is a band along the top or bottom, closed at the pole. cartopy draws the
    # part of a cell past the seam on the other side of the map.
    lon_steps = wrap_longitude(
        np.diff(lon_vertex[ids], axis=1, append=lon_vertex[ids][:, :1]), 0
    )
    for cell in np.where(np.abs(lon_steps.sum(axis=1)) > 180)[0]:
        corner_ids = ids[cell, : int(n_edges[cell])]
        lons = lon_vertex[corner_ids]
        steps = wrap_longitude(np.diff(lons, append=lons[0]), 0)
        start = wrap_longitude(lons[0], center_lon)
        x = start + np.concatenate(([0], np.cumsum(steps)))
        y = np.append(lat_vertex[corner_ids], lat_vertex[corner_ids[0]])
        pole = 90.0 if lat_cell[cell] > 0 else -90.0
        polygons[cell] = np.column_stack(
            (np.append(x, [x[-1], x[0]]), np.append(y, [pole, pole]))
        )

    x_pad = max(float(np.ptp(x_cell)) * 0.05, 0.1)
    y_pad = max(float(np.ptp(lat_cell)) * 0.05, 0.1)
    x_min, x_max = max(x_cell.min() - x_pad, -180), min(x_cell.max() + x_pad, 180)
    y_min, y_max = max(lat_cell.min() - y_pad, -90), min(lat_cell.max() + y_pad, 90)
    lon_span, lat_span = x_max - x_min, y_max - y_min
    map_width, map_height = flat_map_inches(lon_span, lat_span)
    edges = (
        flat_cell_pixels(cell_spacing_km(mean_area_km2), lon_span, map_width)
        >= MIN_CELL_PIXELS
    )

    margin = FLAT_MARGIN_INCHES
    fig_width = margin + map_width + FLAT_BAR_INCHES
    fig_height = margin + map_height + FLAT_TITLE_INCHES
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FLAT_DPI)
    projection = ccrs.PlateCarree(central_longitude=center_lon)
    ax = fig.add_axes(
        [
            margin / fig_width,
            margin / fig_height,
            map_width / fig_width,
            map_height / fig_height,
        ],
        projection=projection,
    )
    cells = PolyCollection(
        polygons,
        array=layer,
        cmap=ListedColormap(LAYER_COLORS),
        norm=BoundaryNorm(np.arange(-0.5, len(LAYER_COLORS)), len(LAYER_COLORS)),
        edgecolors=EDGE_COLOR if edges else "face",
        linewidths=0.3 if edges else 0.1,
        transform=projection,
    )
    ax.add_collection(cells)
    ax.set_extent([x_min, x_max, y_min, y_max], crs=projection)

    missing: List[Path] = []
    lines = [(COASTLINE, "black", 1.0), (BORDERS, "#222222", 1.2)]
    center_lat = float((lat_cell.min() + lat_cell.max()) / 2)
    if shows_states(lon_span, lat_span, center_lat, center_lon):
        lines.append((STATES, "#666666", 0.5))
    for map_layer_name, color, width in lines:
        path = map_layer(map_data, map_layer_name, missing)
        if path is not None:
            ax.add_feature(
                cfeature.ShapelyFeature(
                    shpreader.Reader(str(path)).geometries(),
                    ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor=color,
                    linewidth=width,
                )
            )

    bar = fig.add_axes(
        [
            (margin + map_width + 0.35) / fig_width,
            margin / fig_height,
            0.22 / fig_width,
            map_height / fig_height,
        ]
    )
    fig.colorbar(cells, cax=bar, ticks=range(len(LAYER_COLORS)), label="Boundary layer")
    fig.text(
        0.5,
        1 - FLAT_TITLE_INCHES / 2 / fig_height,
        title,
        ha="center",
        va="center",
        fontsize=14,
        parse_math=False,
    )
    fig.savefig(output, dpi=FLAT_DPI)
    plt.close(fig)
    return missing


def draw_sphere(
    mesh: Path,
    output: Path,
    title: str,
    map_data: Path,
    zoom: float,
    center: Tuple[float, float],
) -> List[Path]:
    """Draw a global mesh on a sphere, and return the missing map layers."""
    import cartopy.io.shapereader as shpreader
    import matplotlib
    import numpy as np
    from netCDF4 import Dataset

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def xyz(lat, lon, radius=1.0):
        lat, lon = np.radians(lat), np.radians(lon)
        return np.stack(
            (
                radius * np.cos(lon) * np.cos(lat),
                radius * np.sin(lon) * np.cos(lat),
                radius * np.sin(lat),
            ),
            axis=-1,
        )

    view = xyz(*center)
    # The image is square, with room above it for the title
    title_inches = 0.6
    fig_height = SPHERE_FIGURE_INCHES + title_inches
    fig = plt.figure(figsize=(SPHERE_FIGURE_INCHES, fig_height), dpi=SPHERE_DPI)
    ax = fig.add_axes([0, 0, 1, SPHERE_FIGURE_INCHES / fig_height], projection="3d")
    ax.set_proj_type("ortho")
    ax.computed_zorder = False
    fig.text(
        0.5,
        1 - title_inches / 2 / fig_height,
        title,
        ha="center",
        va="center",
        fontsize=16,
        parse_math=False,
    )

    with Dataset(mesh) as ds:
        n_cells = ds.dimensions["nCells"].size
        if sphere_cell_pixels(global_cell_spacing_km(n_cells), zoom) >= MIN_CELL_PIXELS:
            # Coordinates are normalized to the unit sphere, whatever radius
            # the file uses. Edges are drawn only when cells are large in the
            # image, so few are in view. Cells are read in pieces, keeping
            # those in view, and then only the vertices they need.
            least = max(visible_dot(zoom), 0.05)
            kept_edges, kept_on_cell = [], []
            for start in range(0, n_cells, READ_STEP):
                piece = slice(start, start + READ_STEP)
                cells = np.stack(
                    [ds.variables[f"{axis}Cell"][piece] for axis in "xyz"], axis=-1
                )
                cells /= np.linalg.norm(cells, axis=-1, keepdims=True)
                keep = np.where(cells @ view > least)[0]
                kept_edges.append(ds.variables["nEdgesOnCell"][piece][keep])
                kept_on_cell.append(ds.variables["verticesOnCell"][piece][keep] - 1)
            n_edges = np.concatenate(kept_edges)
            on_cell = np.concatenate(kept_on_cell)
            corner = np.arange(on_cell.shape[1])[None, :] < n_edges[:, None]
            needed = np.unique(on_cell[corner])
            vertices = np.empty((needed.size, 3))
            for start in range(0, ds.dimensions["nVertices"].size, READ_STEP):
                low, high = np.searchsorted(needed, [start, start + READ_STEP])
                if low < high:
                    rows = needed[low:high] - start
                    for column, axis in enumerate("xyz"):
                        values = ds.variables[f"{axis}Vertex"][
                            start : start + READ_STEP
                        ]
                        vertices[low:high, column] = values[rows]
            vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)
            rows = np.searchsorted(needed, on_cell)
            polygons = [vertices[row[:count]] for count, row in zip(n_edges, rows)]
            ax.add_collection3d(
                Poly3DCollection(
                    polygons,
                    edgecolors=EDGE_COLOR,
                    facecolors="white",
                    linewidths=0.25,
                    zorder=1,
                )
            )

    # The edge of the globe, which is the circle around the view direction
    up = np.array([0.0, 0.0, 1.0]) if abs(view[2]) < 0.9 else np.array([1.0, 0, 0])
    side = np.cross(view, up)
    side /= np.linalg.norm(side)
    up = np.cross(side, view)
    angle = np.linspace(0, 2 * np.pi, 361)[:, None]
    limb = np.cos(angle) * side + np.sin(angle) * up
    ax.plot3D(*limb.T, color="#666666", linewidth=1.0, zorder=5)

    missing: List[Path] = []
    path = map_layer(map_data, sphere_coastline(zoom), missing)
    if path is not None:
        for geometry in shpreader.Reader(str(path)).geometries():
            multi = geometry.geom_type.startswith("Multi")
            for part in geometry.geoms if multi else [geometry]:
                coords = np.asarray(part.coords)
                line = xyz(coords[:, 1], coords[:, 0], radius=1.001)
                # Draw only the pieces on the side of the globe facing the view
                front = line @ view > 0
                breaks = np.where(np.diff(front.astype(int)) != 0)[0] + 1
                for piece in np.split(np.arange(len(line)), breaks):
                    if len(piece) > 1 and front[piece[0]]:
                        ax.plot3D(
                            *line[piece].T, color="darkblue", linewidth=1.8, zorder=10
                        )

    # SPHERE_PIXELS_PER_UNIT was measured with these limits
    limit = 1.0 / zoom
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.view_init(
        elev=float(np.degrees(np.arcsin(view[2]))),
        azim=float(np.degrees(np.arctan2(view[1], view[0]))),
    )
    fig.savefig(output, dpi=SPHERE_DPI)
    plt.close(fig)
    return missing


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
