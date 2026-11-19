# SPDX-License-Identifier: Apache-2.0
"""Draw a PNG image of an MPAS mesh.

This is a standalone script, which the MPAS mesh agent runs with the Python of
its tasks on the ``compute`` resource. Drawing needs numpy, netCDF4, matplotlib
and cartopy, which the ``chiltepin[mpas]`` extra installs. The agent imports this
module for its exit codes and map layers without them, so it imports only the
standard library at the top, and the plotting packages inside the functions that
draw.

A mesh is regional if any cell has a nonzero ``bdyMaskCell``, and global
otherwise. A regional mesh is drawn on a flat map, with its cells colored by
boundary layer. A global mesh is drawn on a sphere, seen from above the view
center. Cell edges are drawn only when cells come out at least
``MIN_CELL_PIXELS`` across in the image, since smaller ones blur together.

Usage::

    python _plot_mesh.py MESH OUTPUT --map-data DIR [--title TITLE]
        [--zoom ZOOM] [--center LAT LON]
    python _plot_mesh.py --download-map-data DIR

The second form downloads the Natural Earth map layers the plots draw to DIR,
where cartopy finds them. It needs only the standard library, so it runs on the
``service`` resource, which can reach Natural Earth, without the extra.

Exit codes:

- 0: the image was written
- 2: the image was written, but without one or more map layers, which are named
  on standard error
- 3: a zoom or view center was given for a regional mesh, and nothing was
  written
- anything else: the plot failed
"""

import argparse
import io
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

EXIT_MISSING_MAP = 2
EXIT_VIEW_OF_REGIONAL_MESH = 3

# Radius of the sphere MPAS meshes are defined on, in km
SPHERE_RADIUS_KM = 6371.229

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

# Parts of a shapefile kept from a downloaded layer
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


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
        "--center",
        type=float,
        nargs=2,
        metavar=("LAT", "LON"),
        help="view center of a global mesh, in degrees",
    )
    return parser.parse_args(argv)


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


def layer_url(layer: Tuple[str, str, str]) -> str:
    """Return the address a Natural Earth layer is downloaded from."""
    resolution, category, name = layer
    return NATURAL_EARTH_URL.format(resolution=resolution, category=category, name=name)


def download_map_data(map_data: Path, attempts: int = 3, timeout: float = 60) -> None:
    """Download the map layers to where cartopy finds them in ``map_data``.

    Each layer's shapefile parts are extracted from its zip file, and the
    ``.shp`` file is written last, so a layer is complete once it exists.

    Raises
    ------
    OSError
        If a layer cannot be downloaded after ``attempts`` tries.
    ValueError
        If a downloaded file is not a zip file holding the layer's shapefile.
    """
    import urllib.request

    for layer in MAP_LAYERS:
        url = layer_url(layer)
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == attempts:
                    raise
                time.sleep(2**attempt)
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ValueError(f"{url} is not a zip file") from e
        target = layer_path(map_data, layer)
        stem = target.stem
        names = set(archive.namelist())
        if f"{stem}.shp" not in names:
            raise ValueError(f"{url} holds no {stem}.shp")
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


def view_center(center: Optional[Sequence[float]]) -> Tuple[float, float]:
    """Return the view center of a global mesh, by default 0N 0E."""
    return (0.0, 0.0) if center is None else (float(center[0]), float(center[1]))


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
    if regional and (args.zoom is not None or args.center is not None):
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
            args.mesh, args.output, title, args.map_data, zoom, view_center(args.center)
        )
    if missing:
        print(
            "Plotted without these map layers, which are missing: "
            + ", ".join(str(path) for path in missing),
            file=sys.stderr,
        )
        return EXIT_MISSING_MAP
    return 0


def is_regional(mesh: Path) -> bool:
    """Whether any cell of the mesh is in a boundary layer."""
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(mesh) as ds:
        if "bdyMaskCell" not in ds.variables:
            return False
        mask = ds.variables["bdyMaskCell"]
        step = 1_000_000
        return any(
            bool(np.any(mask[start : start + step] != 0))
            for start in range(0, mask.shape[0], step)
        )


def map_layer(
    map_data: Path, layer: Tuple[str, str, str], missing: List[Path]
) -> Optional[Path]:
    """Return a map layer's path, or record it as missing and return None."""
    path = layer_path(map_data, layer)
    if path.is_file() and path.stat().st_size > 0:
        return path
    missing.append(path)
    return None


def draw_flat(mesh: Path, output: Path, title: str, map_data: Path) -> List[Path]:
    """Draw a regional mesh on a flat map, and return the missing map layers."""
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
        layer = np.asarray(ds.variables["bdyMaskCell"][:], dtype=float)
        # Areas are in square meters on the sphere the mesh is defined on
        radius_km = float(getattr(ds, "sphere_radius", SPHERE_RADIUS_KM * 1000)) / 1000
        mean_area_km2 = (
            float(np.mean(ds.variables["areaCell"][:]))
            * (SPHERE_RADIUS_KM / radius_km / 1000) ** 2
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
    polygons = np.stack((x_corner, lat_vertex[ids]), axis=-1)

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
    )

    with Dataset(mesh) as ds:
        n_cells = ds.dimensions["nCells"].size
        if sphere_cell_pixels(global_cell_spacing_km(n_cells), zoom) >= MIN_CELL_PIXELS:
            # Coordinates are normalized to the unit sphere, whatever radius
            # the file uses, and cells are read in pieces, keeping those in view
            least = max(visible_dot(zoom), 0.05)
            vertices = np.stack(
                [ds.variables[f"{axis}Vertex"][:] for axis in "xyz"], axis=-1
            )
            vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)
            polygons = []
            step = 1_000_000
            for start in range(0, n_cells, step):
                piece = slice(start, start + step)
                cells = np.stack(
                    [ds.variables[f"{axis}Cell"][piece] for axis in "xyz"], axis=-1
                )
                cells /= np.linalg.norm(cells, axis=-1, keepdims=True)
                keep = np.where(cells @ view > least)[0]
                if keep.size == 0:
                    continue
                n_edges = ds.variables["nEdgesOnCell"][piece][keep]
                on_cell = ds.variables["verticesOnCell"][piece][keep]
                for count, ids in zip(n_edges, on_cell):
                    polygons.append(vertices[ids[:count] - 1])
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
