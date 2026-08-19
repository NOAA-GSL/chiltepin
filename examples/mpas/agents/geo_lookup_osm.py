# SPDX-License-Identifier: Apache-2.0

"""Geographic region lookup using OpenStreetMap Nominatim API.

Drop-in replacement for geo_lookup.py that queries OSM Nominatim instead of
local Natural Earth shapefiles. Provides much finer granularity (cities,
boroughs, neighborhoods) at the cost of requiring internet access and
respecting rate limits (max 1 request/sec on the public instance).
"""

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_USER_AGENT = "chiltepin-mpas-mesh-agent"
_CONTACT_EMAIL = "harrop@colorado.edu"

# Simplification tolerance in degrees (0.01 deg ~ 1.1 km at the equator)
_POLYGON_THRESHOLD = 0.01

# Minimum seconds between Nominatim requests (policy: max 1 req/sec)
_REQUEST_INTERVAL = 1.1

_PROXIMITY_THRESHOLD_KM = 1000.0

_last_request_time = 0.0


def _rate_limit():
    """Sleep if necessary to respect Nominatim's 1 req/sec policy."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _query_nominatim(
    name: str,
    feature_type: Optional[str] = None,
    polygon_threshold: float = _POLYGON_THRESHOLD,
) -> Optional[dict]:
    """Query Nominatim for a single place name and return the first GeoJSON feature."""
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

    resp = requests.get(
        _NOMINATIM_URL,
        params=params,
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("features"):
        return data["features"][0]
    return None


def _feature_to_geometry(feature: dict) -> Optional[Polygon]:
    """Extract a Polygon/MultiPolygon from a GeoJSON feature, or None."""
    geom = shape(feature["geometry"])
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return None


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


def lookup_region(
    names: List[str],
    proximity_threshold_km: float = _PROXIMITY_THRESHOLD_KM,
    exclude_names: Optional[List[str]] = None,
    polygon_threshold: float = _POLYGON_THRESHOLD,
) -> Optional[Polygon]:
    """Look up geographic entities by name via Nominatim and return their unified geometry.

    Tries featureType=country first, then state, then untyped search.
    Filters out distant territories using proximity clustering.

    Note: Continents (Europe, Asia, etc.) do NOT return polygon geometry from
    Nominatim. The caller should expand continent names into country lists
    before calling this function.
    """
    found_geoms = []
    exclude_set = {n.lower() for n in (exclude_names or [])}

    for name in names:
        if name.lower() in exclude_set:
            continue

        geom = None

        # Try country-level first
        feature = _query_nominatim(name, feature_type="country", polygon_threshold=polygon_threshold)
        if feature:
            geom = _feature_to_geometry(feature)

        # Try state-level
        if geom is None:
            feature = _query_nominatim(name, feature_type="state", polygon_threshold=polygon_threshold)
            if feature:
                geom = _feature_to_geometry(feature)

        # Try city-level
        if geom is None:
            feature = _query_nominatim(name, feature_type="city", polygon_threshold=polygon_threshold)
            if feature:
                geom = _feature_to_geometry(feature)

        # Untyped fallback (counties, boroughs, neighborhoods, etc.)
        if geom is None:
            feature = _query_nominatim(name, polygon_threshold=polygon_threshold)
            if feature:
                geom = _feature_to_geometry(feature)

        if geom is not None:
            found_geoms.append(
                _filter_proximate_polygons(geom, proximity_threshold_km)
            )

    if not found_geoms:
        return None

    unified = unary_union(found_geoms)
    return _filter_proximate_polygons(unified, proximity_threshold_km)


def geometry_to_ellipse(
    geom, buffer_km: float = 50.0
) -> Dict[str, Any]:
    """Convert a geometry to minimum enclosing ellipse parameters.

    Returns dict with center_lat, center_lon, semi_major_m, semi_minor_m,
    and orientation_deg (clockwise from north).
    """
    hull = geom.convex_hull

    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0
    cos_lat = math.cos(math.radians(center_lat))

    coords = np.array(hull.exterior.coords)
    x = (coords[:, 0] - center_lon) * meters_per_deg * cos_lat
    y = (coords[:, 1] - center_lat) * meters_per_deg

    buffer_m = buffer_km * 1000.0
    projected = MultiPoint(list(zip(x, y))).buffer(buffer_m)
    rect = projected.minimum_rotated_rectangle

    rect_coords = np.array(rect.exterior.coords[:-1])
    edges = np.diff(np.vstack([rect_coords, rect_coords[0:1]]), axis=0)
    edge_lengths = np.sqrt((edges ** 2).sum(axis=1))

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
    final_lon = center_lon + float(centroid.x) / (meters_per_deg * cos_lat)
    final_lat = center_lat + float(centroid.y) / meters_per_deg

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
