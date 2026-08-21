# SPDX-License-Identifier: Apache-2.0

"""Geographic region lookup with tiered data sources.

Resolves region names to Shapely geometries for MPAS mesh generation.
Tries local Natural Earth shapefiles first (countries, states/provinces),
then falls back to OpenStreetMap Nominatim for finer-grained entities
(cities, counties, neighborhoods).
"""

import hashlib
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

import numpy as np
import requests
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Natural Earth configuration
# ---------------------------------------------------------------------------

_NE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "natural_earth"
_NE_BASE_URL = "https://naciscdn.org/naturalearth"

# ---------------------------------------------------------------------------
# OSM Nominatim configuration
# ---------------------------------------------------------------------------

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "chiltepin-mpas-mesh-agent"
_CONTACT_EMAIL = "harrop@colorado.edu"
_POLYGON_THRESHOLD = 0.01  # ~1.1 km at equator
_REQUEST_INTERVAL = 1.1  # seconds between Nominatim requests
_NOMINATIM_CACHE_DIR = Path(__file__).parent / ".nominatim_cache"

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

_PROXIMITY_THRESHOLD_KM = 1000.0

# ---------------------------------------------------------------------------
# Natural Earth helpers
# ---------------------------------------------------------------------------


def _ensure_shapefile(data_dir: Path, dataset: str, scale: str) -> Path:
    """Download and extract a Natural Earth shapefile if not already present."""
    shp = data_dir / dataset / f"{dataset}.shp"
    if shp.exists():
        return shp
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_NE_BASE_URL}/{scale}/{dataset}.zip"
    zip_path = data_dir / f"{dataset}.zip"
    if not zip_path.exists():
        urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir / dataset)
    return shp


def _load_countries(data_dir: Path = _NE_DATA_DIR):
    import geopandas as gpd

    shp = _ensure_shapefile(data_dir, "ne_110m_admin_0_countries", "110m/cultural")
    return gpd.read_file(shp)


def _load_states(data_dir: Path = _NE_DATA_DIR):
    import geopandas as gpd

    shp = _ensure_shapefile(data_dir, "ne_10m_admin_1_states_provinces", "10m/cultural")
    return gpd.read_file(shp)


# ---------------------------------------------------------------------------
# OSM Nominatim helpers
# ---------------------------------------------------------------------------

_last_request_time = 0.0


def _rate_limit():
    """Sleep if necessary to respect Nominatim's 1 req/sec policy."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _cache_key(name: str, feature_type: Optional[str], polygon_threshold: float) -> str:
    raw = f"{name}|{feature_type}|{polygon_threshold}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _query_nominatim(
    name: str,
    feature_type: Optional[str] = None,
    polygon_threshold: float = _POLYGON_THRESHOLD,
) -> Optional[dict]:
    """Query Nominatim for a place name; results are cached on disk."""
    _NOMINATIM_CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(name, feature_type, polygon_threshold)
    cache_file = _NOMINATIM_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data.get("feature")

    _rate_limit()
    params = {
        "q": name,
        "format": "geojson",
        "polygon_geojson": 1,
        "polygon_threshold": polygon_threshold,
        "limit": 1,
        "email": _CONTACT_EMAIL,
    }
    if feature_type:
        params["featureType"] = feature_type

    for attempt in range(4):
        resp = requests.get(
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        if resp.status_code != 429 or attempt == 3:
            resp.raise_for_status()
            break
        time.sleep(2**attempt)
    data = resp.json()
    feature = data["features"][0] if data.get("features") else None
    cache_file.write_text(json.dumps({"feature": feature}))
    return feature


def _feature_to_geometry(feature: dict) -> Optional[Polygon]:
    geom = shape(feature["geometry"])
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return None


# ---------------------------------------------------------------------------
# Shared geometry utilities
# ---------------------------------------------------------------------------


def _filter_proximate_polygons(
    geom, threshold_km: float = _PROXIMITY_THRESHOLD_KM
) -> Polygon:
    """Keep only polygons within threshold_km of the largest polygon."""
    if isinstance(geom, Polygon):
        return geom

    polys = list(geom.geoms)
    if len(polys) == 1:
        return polys[0]

    areas = [p.area for p in polys]
    anchor_idx = int(np.argmax(areas))
    anchor = polys[anchor_idx]

    threshold_deg = threshold_km / 111.0
    kept = [anchor]
    for i, poly in enumerate(polys):
        if i == anchor_idx:
            continue
        if anchor.distance(poly) < threshold_deg:
            kept.append(poly)

    if len(kept) == 1:
        return kept[0]
    return unary_union(kept)


# ---------------------------------------------------------------------------
# Tier 1: Natural Earth lookup (local, no network)
# ---------------------------------------------------------------------------


def _lookup_region_ne(
    names: List[str],
    proximity_threshold_km: float = _PROXIMITY_THRESHOLD_KM,
    exclude_names: Optional[List[str]] = None,
) -> Optional[Polygon]:
    """Look up regions from Natural Earth shapefiles (countries, states)."""
    found_geoms = []
    exclude_set = {n.lower() for n in (exclude_names or [])}

    countries_gdf = _load_countries()
    states_gdf = None

    for name in names:
        if "CONTINENT" in countries_gdf.columns:
            continent_match = countries_gdf[
                countries_gdf["CONTINENT"].str.lower() == name.lower()
            ]
            if not continent_match.empty:
                for _, row in continent_match.iterrows():
                    if row["NAME"].lower() in exclude_set:
                        continue
                    found_geoms.append(
                        _filter_proximate_polygons(row.geometry, proximity_threshold_km)
                    )
                continue

        if name.lower() in exclude_set:
            continue
        match = countries_gdf[countries_gdf["NAME"].str.lower() == name.lower()]
        if match.empty:
            match = countries_gdf[countries_gdf["ADMIN"].str.lower() == name.lower()]
        if match.empty:
            mask = countries_gdf["NAME"].str.lower().str.contains(
                name.lower(), regex=False
            ) | countries_gdf["ADMIN"].str.lower().str.contains(
                name.lower(), regex=False
            )
            match = countries_gdf[mask]

        if not match.empty:
            for geom in match.geometry:
                found_geoms.append(
                    _filter_proximate_polygons(geom, proximity_threshold_km)
                )
            continue

        if states_gdf is None:
            states_gdf = _load_states()

        match = states_gdf[states_gdf["name"].str.lower() == name.lower()]
        if match.empty and "name_alt" in states_gdf.columns:
            mask = (
                states_gdf["name_alt"]
                .fillna("")
                .str.lower()
                .str.contains(name.lower(), regex=False)
            )
            match = states_gdf[mask]

        if not match.empty:
            for geom in match.geometry:
                found_geoms.append(geom)

    if not found_geoms:
        return None

    unified = unary_union(found_geoms)
    return _filter_proximate_polygons(unified, proximity_threshold_km)


# ---------------------------------------------------------------------------
# Tier 2: OSM Nominatim lookup (network, cached)
# ---------------------------------------------------------------------------


def _lookup_region_osm(
    names: List[str],
    proximity_threshold_km: float = _PROXIMITY_THRESHOLD_KM,
    exclude_names: Optional[List[str]] = None,
    polygon_threshold: float = _POLYGON_THRESHOLD,
) -> Optional[Polygon]:
    """Look up regions via Nominatim (cities, counties, neighborhoods, etc.)."""
    found_geoms = []
    exclude_set = {n.lower() for n in (exclude_names or [])}

    for name in names:
        if name.lower() in exclude_set:
            continue

        geom = None

        feature = _query_nominatim(
            name, feature_type="country", polygon_threshold=polygon_threshold
        )
        if feature:
            geom = _feature_to_geometry(feature)

        if geom is None:
            feature = _query_nominatim(
                name, feature_type="state", polygon_threshold=polygon_threshold
            )
            if feature:
                geom = _feature_to_geometry(feature)

        if geom is None:
            feature = _query_nominatim(
                name, feature_type="city", polygon_threshold=polygon_threshold
            )
            if feature:
                geom = _feature_to_geometry(feature)

        if geom is None:
            feature = _query_nominatim(name, polygon_threshold=polygon_threshold)
            if feature:
                geom = _feature_to_geometry(feature)

        if geom is not None:
            found_geoms.append(_filter_proximate_polygons(geom, proximity_threshold_km))

    if not found_geoms:
        return None

    unified = unary_union(found_geoms)
    return _filter_proximate_polygons(unified, proximity_threshold_km)


# ---------------------------------------------------------------------------
# Public API: tiered lookup
# ---------------------------------------------------------------------------


def lookup_region(
    names: List[str],
    proximity_threshold_km: float = _PROXIMITY_THRESHOLD_KM,
    exclude_names: Optional[List[str]] = None,
) -> Optional[Polygon]:
    """Look up geographic entities by name and return their unified geometry.

    Tries Natural Earth (local shapefiles) first for countries and
    states/provinces.  Falls back to OSM Nominatim for finer-grained
    entities (cities, counties, neighborhoods).
    """
    geom = _lookup_region_ne(names, proximity_threshold_km, exclude_names)
    if geom is None:
        geom = _lookup_region_osm(names, proximity_threshold_km, exclude_names)
    return geom


# ---------------------------------------------------------------------------
# Geometry-to-mesh-config conversion functions
# ---------------------------------------------------------------------------


def geometry_to_ellipse(geom, buffer_km: float = 50.0) -> Dict[str, Any]:
    """Convert a geometry to minimum enclosing ellipse parameters.

    Returns dict with center_lat, center_lon, semi_major_m, semi_minor_m,
    and orientation_deg (clockwise from north).
    """
    hull = geom.convex_hull

    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0

    coords = np.array(hull.exterior.coords)
    cos_ref = math.cos(math.radians(center_lat))
    x = (coords[:, 0] - center_lon) * meters_per_deg * cos_ref
    y = (coords[:, 1] - center_lat) * meters_per_deg

    buffer_m = buffer_km * 1000.0
    projected = MultiPoint(list(zip(x, y))).buffer(buffer_m)
    rect = projected.minimum_rotated_rectangle

    rect_coords = np.array(rect.exterior.coords[:-1])
    edges = np.diff(np.vstack([rect_coords, rect_coords[0:1]]), axis=0)
    edge_lengths = np.sqrt((edges**2).sum(axis=1))

    idx = int(np.argmax(edge_lengths[:2]))
    long_edge = edges[idx]
    major_len = float(edge_lengths[idx])
    minor_len = float(edge_lengths[1 - idx])

    angle_from_east = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    orientation = 90.0 - angle_from_east
    orientation = orientation % 360.0
    if orientation > 180.0:
        orientation -= 180.0

    centroid = rect.centroid
    final_lat = center_lat + float(centroid.y) / meters_per_deg
    final_lon = center_lon + float(centroid.x) / (
        meters_per_deg * math.cos(math.radians(final_lat))
    )

    semi_major = major_len / 2.0
    semi_minor = minor_len / 2.0

    # Scale semi-axes so the ellipse contains all buffered hull points.
    theta = math.radians(angle_from_east)
    cx, cy = float(centroid.x), float(centroid.y)
    hull_coords = np.array(projected.convex_hull.exterior.coords)
    dx = hull_coords[:, 0] - cx
    dy = hull_coords[:, 1] - cy
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    ellipse_vals = (u / semi_major) ** 2 + (v / semi_minor) ** 2
    max_val = float(ellipse_vals.max())
    if max_val > 1.0:
        scale = math.sqrt(max_val)
        semi_major *= scale
        semi_minor *= scale

    return {
        "center_lat": round(final_lat, 4),
        "center_lon": round(final_lon, 4),
        "semi_major_m": int(semi_major),
        "semi_minor_m": int(semi_minor),
        "orientation_deg": int(round(orientation)),
    }


def geometry_to_circle(geom, buffer_km: float = 50.0) -> Dict[str, Any]:
    """Convert a geometry to minimum enclosing circle parameters.

    Returns dict with center_lat, center_lon, and radius_m.
    """
    hull = geom.convex_hull

    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0

    coords = np.array(hull.exterior.coords)
    cos_ref = math.cos(math.radians(center_lat))
    x = (coords[:, 0] - center_lon) * meters_per_deg * cos_ref
    y = (coords[:, 1] - center_lat) * meters_per_deg

    buffer_m = buffer_km * 1000.0
    projected = MultiPoint(list(zip(x, y))).buffer(buffer_m)
    buffered_hull = projected.convex_hull

    cx, cy = float(buffered_hull.centroid.x), float(buffered_hull.centroid.y)
    hull_coords = np.array(buffered_hull.exterior.coords)
    dx = hull_coords[:, 0] - cx
    dy = hull_coords[:, 1] - cy
    dists = np.sqrt(dx**2 + dy**2)
    radius = float(dists.max())

    final_lat = center_lat + cy / meters_per_deg
    final_lon = center_lon + cx / (meters_per_deg * math.cos(math.radians(final_lat)))

    return {
        "center_lat": round(final_lat, 4),
        "center_lon": round(final_lon, 4),
        "radius_m": int(radius),
    }


def geometry_to_polygon(geom, buffer_km: float = 50.0) -> Dict[str, Any]:
    """Convert a geometry to a convex polygon specification.

    Returns dict with point_lat, point_lon (interior point) and
    vertices (list of (lat, lon) tuples forming the convex hull).
    """
    hull = geom.convex_hull

    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0

    coords = np.array(hull.exterior.coords)
    x = (coords[:, 0] - center_lon) * meters_per_deg * np.cos(np.radians(coords[:, 1]))
    y = (coords[:, 1] - center_lat) * meters_per_deg

    buffer_m = buffer_km * 1000.0
    projected = MultiPoint(list(zip(x, y))).buffer(buffer_m)
    buffered_hull = projected.convex_hull

    simplified = buffered_hull.simplify(buffer_m * 0.1)
    if not isinstance(simplified, Polygon) or simplified.is_empty:
        simplified = buffered_hull

    proj_coords = np.array(simplified.exterior.coords[:-1])
    lats = center_lat + proj_coords[:, 1] / meters_per_deg
    lons = center_lon + proj_coords[:, 0] / (meters_per_deg * np.cos(np.radians(lats)))

    vertices = list(zip(np.round(lats, 6).tolist(), np.round(lons, 6).tolist()))

    cx, cy = float(buffered_hull.centroid.x), float(buffered_hull.centroid.y)
    point_lat = center_lat + cy / meters_per_deg
    point_lon = center_lon + cx / (meters_per_deg * math.cos(math.radians(point_lat)))

    return {
        "point_lat": round(point_lat, 4),
        "point_lon": round(point_lon, 4),
        "vertices": vertices,
    }


def geometry_to_rectangle(geom, buffer_km: float = 50.0) -> Dict[str, Any]:
    """Convert a geometry to a bounding rectangle for project_hexes.

    Determines rotation from the unbuffered hull so elongation is
    preserved, then adds the buffer to the rotated extents.

    Returns dict with center_lat, center_lon, extent_x_km, extent_y_km,
    and rotation_degrees (0.0 when no rotation helps).
    """
    hull = geom.convex_hull

    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0

    coords = np.array(hull.exterior.coords)
    cos_ref = math.cos(math.radians(center_lat))
    x = (coords[:, 0] - center_lon) * meters_per_deg * cos_ref
    y = (coords[:, 1] - center_lat) * meters_per_deg

    unbuffered = MultiPoint(list(zip(x, y))).convex_hull
    buffer_m = buffer_km * 1000.0

    ub_bounds = unbuffered.bounds
    aa_width = ub_bounds[2] - ub_bounds[0]
    aa_height = ub_bounds[3] - ub_bounds[1]
    aa_area = aa_width * aa_height

    rect = unbuffered.minimum_rotated_rectangle
    rect_coords = np.array(rect.exterior.coords[:-1])
    edges = np.diff(np.vstack([rect_coords, rect_coords[0:1]]), axis=0)
    edge_lengths = np.sqrt((edges**2).sum(axis=1))

    idx = int(np.argmax(edge_lengths[:2]))
    long_len = float(edge_lengths[idx])
    short_len = float(edge_lengths[1 - idx])
    rotated_area = long_len * short_len

    if aa_area > 0 and rotated_area < aa_area:
        long_edge = edges[idx]
        angle_from_east = math.degrees(math.atan2(long_edge[1], long_edge[0]))
        rotation = (angle_from_east + 90.0) % 180.0 - 90.0

        centroid = rect.centroid
        cx, cy = float(centroid.x), float(centroid.y)
        extent_x_m = long_len + 2.0 * buffer_m
        extent_y_m = short_len + 2.0 * buffer_m
    else:
        rotation = 0.0
        cx = (ub_bounds[0] + ub_bounds[2]) / 2.0
        cy = (ub_bounds[1] + ub_bounds[3]) / 2.0
        extent_x_m = aa_width + 2.0 * buffer_m
        extent_y_m = aa_height + 2.0 * buffer_m

    final_lat = center_lat + cy / meters_per_deg
    final_lon = center_lon + cx / (meters_per_deg * math.cos(math.radians(final_lat)))

    # LC scale correction: tangent LC at center_lat undercovers at edges
    half_y_deg = extent_y_m / (2.0 * meters_per_deg)
    edge_lat = max(abs(final_lat - half_y_deg), abs(final_lat + half_y_deg))
    dlat_rad = math.radians(abs(edge_lat - abs(final_lat)))
    tan_ref = math.tan(math.radians(abs(final_lat))) if abs(final_lat) > 1 else 0.0
    lc_scale = 1.0 + dlat_rad**2 * tan_ref**2
    extent_x_m *= lc_scale
    extent_y_m *= lc_scale

    return {
        "center_lat": round(final_lat, 4),
        "center_lon": round(final_lon, 4),
        "extent_x_km": round(extent_x_m / 1000.0, 1),
        "extent_y_km": round(extent_y_m / 1000.0, 1),
        "rotation_degrees": round(rotation, 1),
    }
