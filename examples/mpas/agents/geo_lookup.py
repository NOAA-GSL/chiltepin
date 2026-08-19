# SPDX-License-Identifier: Apache-2.0

"""Geographic region lookup using Natural Earth shapefiles.

Resolves region names (countries, states/provinces) to geometries and computes
minimum enclosing ellipse parameters for MPAS mesh generation.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

# Default data directory relative to this file
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "natural_earth"

# Proximity threshold for filtering disconnected territories (in km)
_PROXIMITY_THRESHOLD_KM = 1000.0


def _load_countries(data_dir: Path = _DATA_DIR) -> gpd.GeoDataFrame:
    shp = data_dir / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
    if not shp.exists():
        raise FileNotFoundError(f"Natural Earth countries shapefile not found: {shp}")
    return gpd.read_file(shp)


def _load_states(data_dir: Path = _DATA_DIR) -> gpd.GeoDataFrame:
    shp = data_dir / "ne_10m_admin_1_states_provinces" / "ne_10m_admin_1_states_provinces.shp"
    if not shp.exists():
        raise FileNotFoundError(f"Natural Earth states shapefile not found: {shp}")
    return gpd.read_file(shp)


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
    data_dir: Path = _DATA_DIR,
    proximity_threshold_km: float = _PROXIMITY_THRESHOLD_KM,
    exclude_names: Optional[List[str]] = None,
) -> Optional[Polygon]:
    """Look up geographic entities by name and return their unified geometry.

    Searches continents, countries, then states/provinces. Filters out distant
    territories using proximity clustering. Excludes any entities in exclude_names.
    """
    found_geoms = []
    exclude_set = {n.lower() for n in (exclude_names or [])}

    countries_gdf = _load_countries(data_dir)
    states_gdf = None  # lazy load

    for name in names:
        # Try continent match first
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

        # Try country match (case-insensitive)
        if name.lower() in exclude_set:
            continue
        match = countries_gdf[countries_gdf["NAME"].str.lower() == name.lower()]
        if match.empty:
            match = countries_gdf[countries_gdf["ADMIN"].str.lower() == name.lower()]

        if not match.empty:
            for geom in match.geometry:
                found_geoms.append(
                    _filter_proximate_polygons(geom, proximity_threshold_km)
                )
            continue

        # Try state/province match
        if states_gdf is None:
            states_gdf = _load_states(data_dir)

        match = states_gdf[states_gdf["name"].str.lower() == name.lower()]
        if match.empty and "name_alt" in states_gdf.columns:
            # Try alternate names
            mask = states_gdf["name_alt"].fillna("").str.lower().str.contains(
                name.lower(), regex=False
            )
            match = states_gdf[mask]

        if not match.empty:
            for geom in match.geometry:
                found_geoms.append(geom)

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

    # Project to local tangent plane centered on the hull centroid
    center_lat = hull.centroid.y
    center_lon = hull.centroid.x
    meters_per_deg = 111_320.0
    cos_lat = math.cos(math.radians(center_lat))

    coords = np.array(hull.exterior.coords)
    x = (coords[:, 0] - center_lon) * meters_per_deg * cos_lat
    y = (coords[:, 1] - center_lat) * meters_per_deg

    # Apply buffer in projected space
    buffer_m = buffer_km * 1000.0
    projected = MultiPoint(list(zip(x, y))).buffer(buffer_m)
    rect = projected.minimum_rotated_rectangle

    # Extract rectangle dimensions and orientation
    rect_coords = np.array(rect.exterior.coords[:-1])
    edges = np.diff(np.vstack([rect_coords, rect_coords[0:1]]), axis=0)
    edge_lengths = np.sqrt((edges ** 2).sum(axis=1))

    idx = int(np.argmax(edge_lengths[:2]))
    long_edge = edges[idx]
    major_len = float(edge_lengths[idx])
    minor_len = float(edge_lengths[1 - idx])

    # Orientation: angle of long edge from north (y-axis), clockwise
    angle_from_east = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    orientation = 90.0 - angle_from_east
    orientation = orientation % 360.0
    if orientation > 180.0:
        orientation -= 180.0

    # Use rectangle centroid for final center
    centroid = rect.centroid
    final_lon = center_lon + float(centroid.x) / (meters_per_deg * cos_lat)
    final_lat = center_lat + float(centroid.y) / meters_per_deg

    semi_major = major_len / 2.0
    semi_minor = minor_len / 2.0

    # Scale semi-axes so the ellipse actually contains all buffered hull points.
    # The rectangle's corners are outside an inscribed ellipse, so we must expand.
    theta = math.radians(angle_from_east)
    cx, cy = float(centroid.x), float(centroid.y)
    hull_coords = np.array(projected.convex_hull.exterior.coords)
    dx = hull_coords[:, 0] - cx
    dy = hull_coords[:, 1] - cy
    # Rotate points into ellipse-aligned frame
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    # Find maximum ellipse parameter value across all points
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


def region_to_mesh_config(
    names: List[str],
    resolution: str,
    mesh_name: str,
    buffer_km: float = 50.0,
    data_dir: Path = _DATA_DIR,
    exclude_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full pipeline: region names → MPAS mesh config."""
    geom = lookup_region(names, data_dir, exclude_names=exclude_names)
    if geom is None:
        raise ValueError(f"Could not resolve any of: {names}")

    ellipse = geometry_to_ellipse(geom, buffer_km=buffer_km)

    return {
        "resolution": resolution,
        "name": mesh_name,
        "regional": {
            "create_region": {
                "ellipse": {
                    "point": f"{ellipse['center_lat']}, {ellipse['center_lon']}",
                    "semi-major-axis": ellipse["semi_major_m"],
                    "semi-minor-axis": ellipse["semi_minor_m"],
                    "orientation-angle": ellipse["orientation_deg"],
                }
            }
        },
    }
