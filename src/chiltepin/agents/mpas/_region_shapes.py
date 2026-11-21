# SPDX-License-Identifier: Apache-2.0
"""Fit a region to the shapes the MPAS mesh tools make meshes of.

The MPAS mesh agent looks a prompt's places up as a region outline, and these
functions turn the outline into the ``regional`` block of a mesh request, in
kilometers, as ``generate_mesh`` takes it. Each shape is fitted in the frame
its tool draws it in, so a shape holds the whole region and its margin:

- ``create_region`` draws a circle and an ellipse around their center in an
  azimuthal equidistant frame, where distances from the center are distances
  on the sphere. An ellipse's orientation is the azimuth of its major axis,
  clockwise from north.
- ``create_region`` joins a polygon's vertices by great circles, which are
  straight lines in a gnomonic frame about the polygon's center.
- ``project_hexes`` draws its rectangle in a Lambert conformal frame tangent
  at its center, and ``grid_rotate`` turns the mesh counter-clockwise about
  its center.

Distances are on a sphere of ``EARTH_RADIUS_KM``, which both tools use. The
module imports only the standard library at the top. Fitting needs numpy and
shapely, which the ``chiltepin[mpas]`` extra installs.
"""

import math
from typing import Any, Dict, Tuple

EARTH_RADIUS_KM = 6371.229

# Boundary layers project_hexes adds outside its rectangle, as create_region
# adds its own outside its shape
PROJECT_HEXES_BOUNDARY_LAYERS = 7

# The margin around a region is at least this part of its diameter
MARGIN_FRACTION = 0.1

# A shape reaches at least this many cells from its center in every direction.
# create_region fills the whole globe, and project_hexes fails, for shapes
# about one cell across or less.
MIN_HALF_CELLS = 2.0

# Points this many directions around each corner of the region carry its margin
MARGIN_DIRECTIONS = 16

# Polygons and project_hexes rectangles reach at most this far from their
# center, in degrees of arc. Their frames grow without limit toward 90.
MAX_FLAT_RADIUS_DEGREES = 80.0

# project_hexes rectangles are rotated only if that makes them this much smaller
MIN_ROTATION_GAIN = 0.05

# The search for the smallest project_hexes rectangle moves its center by
# steps from this many km, and turns it by steps from this many degrees, until
# the steps are this many km
RECTANGLE_STEP_KM = 500.0
RECTANGLE_STEP_DEGREES = 10.0
RECTANGLE_TOLERANCE_KM = 1.0

# Times a shape's center is moved toward the middle of the region
CENTER_STEPS = 4

# Edges of a region's outline are split into pieces at most this many degrees
# long, since an edge along a parallel can be long and bulge between its ends
MAX_EDGE_DEGREES = 1.0

# A region whose outline averages less than this far from the center of the
# sphere surrounds the globe, and has no center
MIN_CENTER_NORM = 0.01

# A project_hexes rectangle's center keeps this far from the poles, where its
# frame has no size
MAX_CENTER_LAT = 89.9

# project_hexes divides by the sine of its standard parallels, which the agent
# puts at the center. A center nearer the equator than this, where the cone's
# apex is so far that its frame loses all precision, gets parallels this far
# from the equator, on the center's side.
EQUATOR_PARALLEL_DEGREES = 1e-4

SHAPES = ("circle", "ellipse", "polygon", "rectangle")


class RegionTooLarge(ValueError):
    """The region does not fit the shape asked for."""


class RegionTooSmall(ValueError):
    """The shape is too small for the mesh's cells."""


# Points on the sphere


def _unit(lon, lat):
    """Return unit vectors of longitudes and latitudes in degrees."""
    import numpy as np

    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
    )


def _lonlat(vector) -> Tuple[float, float]:
    """Return the longitude, from -180 to 180, and latitude of a unit vector."""
    x, y, z = (float(value) for value in vector)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    lon = math.degrees(math.atan2(y, x))
    return (lon + 180.0) % 360.0 - 180.0, lat


def _frame(center):
    """Return unit vectors pointing east and north at a point."""
    import numpy as np

    east = np.cross([0.0, 0.0, 1.0], center)
    if np.linalg.norm(east) < 1e-9:
        # At a pole, any direction can be east
        east = np.array([0.0, 1.0, 0.0])
    east = east / np.linalg.norm(east)
    return east, np.cross(center, east)


def _destination(center, azimuth, distance):
    """Return the points ``distance`` radians from a center at the azimuths."""
    import numpy as np

    east, north = _frame(center)
    azimuth, distance = np.asarray(azimuth), np.asarray(distance)
    direction = np.sin(azimuth)[..., None] * east + np.cos(azimuth)[..., None] * north
    return (
        np.cos(distance)[..., None] * center + np.sin(distance)[..., None] * direction
    )


def _angle(points, center):
    """Return the angles in radians between points and a center."""
    import numpy as np

    return np.arccos(np.clip(points @ center, -1.0, 1.0))


def _rotate(points, axis, angle: float):
    """Rotate points counter-clockwise about an axis, seen from outside."""
    import numpy as np

    cos, sin = math.cos(angle), math.sin(angle)
    return (
        points * cos
        + np.cross(axis, points) * sin
        + np.outer(points @ axis, axis) * (1 - cos)
    )


def _with_margin(points, margin_km: float):
    """Return points with others ``margin_km`` around each of them."""
    import numpy as np

    if margin_km <= 0:
        return points
    azimuths = np.linspace(0, 2 * np.pi, MARGIN_DIRECTIONS, endpoint=False)
    around = [points]
    for point in points:
        around.append(_destination(point, azimuths, margin_km / EARTH_RADIUS_KM))
    return np.concatenate(around)


# Frames


def aeqd(points, center):
    """Return azimuthal equidistant x (east) and y (north) of points, in km."""
    import numpy as np

    east, north = _frame(center)
    distance = _angle(points, center) * EARTH_RADIUS_KM
    azimuth = np.arctan2(points @ east, points @ north)
    return distance * np.sin(azimuth), distance * np.cos(azimuth)


def gnomonic(points, center):
    """Return gnomonic x (east) and y (north) of points, on the unit sphere."""
    east, north = _frame(center)
    depth = points @ center
    return (points @ east) / depth, (points @ north) / depth


def gnomonic_inverse(x, y, center):
    """Return the points at gnomonic x and y."""
    import numpy as np

    east, north = _frame(center)
    points = center + np.asarray(x)[..., None] * east + np.asarray(y)[..., None] * north
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


def standard_parallel(center_lat: float) -> float:
    """Return the latitude project_hexes' cone touches the sphere at."""
    if abs(center_lat) >= EQUATOR_PARALLEL_DEGREES:
        return center_lat
    return math.copysign(EQUATOR_PARALLEL_DEGREES, center_lat)


def lambert_apex_km(center_lat: float) -> float:
    """Return how far north or south of the center project_hexes can reach, in km.

    The apex of its cone is the pole, and project_hexes finds a point's angle
    about it with ``atan``, so its rectangle must stay on the center's side of
    the apex, about 3.6 billion km from a center on the equator.
    """
    parallel = standard_parallel(center_lat)
    return abs(EARTH_RADIUS_KM / math.tan(math.radians(parallel)))


def project_hexes_reach_km(extent_y_km: float, resolution_km: float) -> float:
    """Return how far north or south of its center a project_hexes mesh reaches.

    It reaches half its height, and then its boundary layers, and a cell more
    for the offset rows of its hexagons.
    """
    return extent_y_km / 2 + (PROJECT_HEXES_BOUNDARY_LAYERS + 1) * resolution_km


def lambert(points, center_lat: float, center_lon: float):
    """Return x (east) and y (north) of points in km, as project_hexes draws them.

    The frame is Lambert conformal with both standard parallels at
    ``standard_parallel(center_lat)``, and the center at the origin.
    """
    import numpy as np

    phi1 = math.radians(standard_parallel(center_lat))
    phi0 = math.radians(center_lat)
    n = math.sin(phi1)
    factor = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n
    rho0 = EARTH_RADIUS_KM * factor / math.tan(math.pi / 4 + phi0 / 2) ** n
    lat = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
    lon = np.arctan2(points[:, 1], points[:, 0])
    dlon = (lon - math.radians(center_lon) + np.pi) % (2 * np.pi) - np.pi
    rho = EARTH_RADIUS_KM * factor / np.tan(np.pi / 4 + lat / 2) ** n
    theta = n * dlon
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


# Fitting


def _points(region):
    """Return the unit vectors of a region's outline."""
    import numpy as np
    from shapely import segmentize
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(region, Polygon):
        polygons = [region]
    elif isinstance(region, MultiPolygon):
        polygons = list(region.geoms)
    else:
        raise ValueError(f"A region must be polygons, got {region.geom_type}")
    if not polygons or region.is_empty:
        raise ValueError("The region is empty")
    # Edges are straight in longitude and latitude, as Natural Earth's are
    coords = np.concatenate(
        [np.asarray(segmentize(p.exterior, MAX_EDGE_DEGREES).coords) for p in polygons]
    )
    return _unit(coords[:, 0], coords[:, 1])


def _hull(x, y):
    """Return the indices of the points on the convex hull of x and y."""
    import numpy as np
    from shapely import MultiPoint

    xy = np.column_stack((x, y))
    hull = MultiPoint(xy).convex_hull
    corners = np.asarray(
        hull.exterior.coords if hull.geom_type == "Polygon" else hull.coords
    )
    # Map the corners back to the points they came from
    nearest = [int(np.argmin(np.sum((xy - corner) ** 2, axis=1))) for corner in corners]
    return np.unique(nearest)


def _circle_center(points):
    """Return the center of the smallest cap holding the points, approximately."""
    import numpy as np
    from shapely import MultiPoint, minimum_bounding_circle

    center = points.mean(axis=0)
    if np.linalg.norm(center) < MIN_CENTER_NORM:
        raise RegionTooLarge("The region surrounds the globe and has no center")
    center = center / np.linalg.norm(center)
    for _ in range(CENTER_STEPS):
        x, y = aeqd(points, center)
        middle = minimum_bounding_circle(MultiPoint(np.column_stack((x, y)))).centroid
        center = _destination(
            center,
            math.atan2(middle.x, middle.y),
            math.hypot(middle.x, middle.y) / EARTH_RADIUS_KM,
        )
    return center


def _round_up(value: float) -> float:
    """Round a size up to tenths of a kilometer, so a shape never shrinks."""
    return math.ceil(value * 10) / 10


def _rounded(center):
    """Return a center rounded as it is written, and its unit vector.

    Shapes are sized about the rounded center, so rounding never cuts the
    region.
    """
    lon, lat = _lonlat(center)
    lon, lat = round(lon, 4), round(lat, 4)
    return {"center_lat": lat, "center_lon": lon}, _unit(lon, lat)


def diameter_km(region) -> float:
    """Return the region's diameter: that of the smallest circle holding it."""
    points = _points(region)
    return 2 * float(_angle(points, _circle_center(points)).max()) * EARTH_RADIUS_KM


def fit_circle(region, margin_km: float) -> Dict[str, Any]:
    """Return the ``create_region`` circle holding a region and its margin."""
    points = _points(region)
    written, center = _rounded(_circle_center(points))
    radius = float(_angle(points, center).max()) * EARTH_RADIUS_KM + margin_km
    if radius >= math.pi * EARTH_RADIUS_KM:
        raise RegionTooLarge("The region and its margin reach around the globe")
    return {"circle": {**written, "radius_km": _round_up(radius)}}


def _enclosing_ellipse(xy, tolerance: float = 1e-5, steps: int = 2000):
    """Return the center and matrix of the smallest ellipse holding points.

    A point p is inside when ``(p - center) @ matrix @ (p - center) <= 1``.
    This is Khachiyan's algorithm.
    """
    import numpy as np

    count = len(xy)
    lifted = np.column_stack((xy, np.ones(count))).T
    weights = np.full(count, 1.0 / count)
    for _ in range(steps):
        moment = lifted @ np.diag(weights) @ lifted.T
        reach = np.einsum("ij,ji->i", lifted.T @ np.linalg.inv(moment), lifted)
        far = int(np.argmax(reach))
        step = (reach[far] - 3) / (3 * (reach[far] - 1))
        updated = (1 - step) * weights
        updated[far] += step
        done = np.linalg.norm(updated - weights) < tolerance
        weights = updated
        if done:
            break
    center = xy.T @ weights
    spread = xy.T @ np.diag(weights) @ xy - np.outer(center, center)
    return center, np.linalg.inv(spread) / 2


def fit_ellipse(region, margin_km: float) -> Dict[str, Any]:
    """Return the ``create_region`` ellipse holding a region and its margin."""
    import numpy as np

    points = _points(region)
    center = _circle_center(points)
    # Distances in the ellipse's frame go the short way around, so a margin
    # reaching around the globe is caught here
    reach = float(_angle(points, center).max()) * EARTH_RADIUS_KM + margin_km
    if reach >= math.pi * EARTH_RADIUS_KM:
        raise RegionTooLarge("The region and its margin reach around the globe")
    for step in range(CENTER_STEPS + 1):
        x, y = aeqd(points, center)
        corners = _with_margin(points[_hull(x, y)], margin_km)
        x, y = aeqd(corners, center)
        corners = corners[_hull(x, y)]
        x, y = aeqd(corners, center)
        middle, matrix = _enclosing_ellipse(np.column_stack((x, y)))
        if step < CENTER_STEPS:
            center = _destination(
                center,
                math.atan2(middle[0], middle[1]),
                math.hypot(middle[0], middle[1]) / EARTH_RADIUS_KM,
            )
    # The ellipse is centered on the center, with the shape that fits best, and
    # grown until it holds every point, about the center and orientation as
    # they are written
    values, vectors = np.linalg.eigh(matrix)
    minor_axis, major_axis = 1 / np.sqrt(values[1]), 1 / np.sqrt(values[0])
    direction = vectors[:, 0]
    orientation = round(math.degrees(math.atan2(direction[0], direction[1])) % 180.0, 1)
    written, center = _rounded(center)
    x, y = aeqd(corners, center)
    theta = math.radians(orientation)
    u = x * math.sin(theta) + y * math.cos(theta)
    v = x * math.cos(theta) - y * math.sin(theta)
    grow = float(np.sqrt((u / major_axis) ** 2 + (v / minor_axis) ** 2).max())
    major, minor = major_axis * grow, minor_axis * grow
    if major >= math.pi * EARTH_RADIUS_KM:
        raise RegionTooLarge("The region and its margin reach around the globe")
    return {
        "ellipse": {
            **written,
            "semi_major_axis_km": _round_up(major),
            "semi_minor_axis_km": _round_up(minor),
            "orientation_degrees": orientation,
        }
    }


def _flat_center(points, what: str):
    """Return the circle center, checking the region fits a flat frame there."""
    center = _circle_center(points)
    reach = math.degrees(float(_angle(points, center).max()))
    if reach > MAX_FLAT_RADIUS_DEGREES:
        raise RegionTooLarge(
            f"The region reaches {reach:.0f} degrees from its center, and a {what} "
            f"can reach at most {MAX_FLAT_RADIUS_DEGREES:.0f}. Use a circle or an "
            "ellipse."
        )
    return center


def _polygon(points, center, margin_km: float, rectangle: bool) -> Dict[str, Any]:
    """Return the polygon holding points and their margin, in the gnomonic frame."""
    import numpy as np
    from shapely import MultiPoint
    from shapely.geometry.polygon import orient

    x, y = gnomonic(points, center)
    corners = _with_margin(points[_hull(x, y)], margin_km)
    if float((corners @ center).min()) <= math.cos(math.radians(89.0)):
        raise RegionTooLarge("The region and its margin reach too far for a polygon")
    x, y = gnomonic(corners, center)
    hull = MultiPoint(np.column_stack((x, y))).convex_hull
    if rectangle:
        hull = hull.minimum_rotated_rectangle
    # Counter-clockwise, seen from above, and without the repeated first vertex
    ring = np.asarray(orient(hull, sign=1.0).exterior.coords)[:-1]
    vertices = gnomonic_inverse(ring[:, 0], ring[:, 1], center)
    lon, lat = _lonlat(center)
    return {
        "polygon": {
            "vertices": [
                [round(vlat, 6), round(vlon, 6)]
                for vlon, vlat in (_lonlat(vertex) for vertex in vertices)
            ],
            "inside_lat": round(lat, 6),
            "inside_lon": round(lon, 6),
        }
    }


def fit_polygon(region, margin_km: float) -> Dict[str, Any]:
    """Return the ``create_region`` polygon holding a region and its margin.

    It is the region's convex hull, grown by the margin.
    """
    points = _points(region)
    center = _flat_center(points, "polygon")
    return _polygon(points, center, margin_km, rectangle=False)


def fit_rectangle(region, margin_km: float) -> Dict[str, Any]:
    """Return the ``create_region`` polygon of the smallest rectangle holding a
    region and its margin, with great circles for sides."""
    points = _points(region)
    center = _flat_center(points, "rectangle")
    return _polygon(points, center, margin_km, rectangle=True)


def _clamp_lat(center):
    """Keep a center latitude away from the poles."""
    lon, lat = _lonlat(center)
    return _unit(lon, max(-MAX_CENTER_LAT, min(MAX_CENTER_LAT, lat)))


def _lambert_frame(points, center, rotation: float):
    """Return x, y and the cone angle of points in project_hexes' frame.

    Points are turned back by ``rotation`` about the center first, as
    grid_rotate turns the mesh by it.
    """
    import numpy as np

    lon, lat = _lonlat(center)
    turned = _rotate(points, center, -rotation)
    x, y = lambert(turned, lat, lon)
    lons = np.degrees(np.arctan2(turned[:, 1], turned[:, 0]))
    dlon = (lons - lon + 180.0) % 360.0 - 180.0
    return x, y, math.sin(math.radians(standard_parallel(lat))) * dlon


def _half_extents(points, center, rotation: float) -> Tuple[float, float]:
    """Return the half width and height of the rectangle, about its center."""
    import numpy as np

    x, y, _ = _lambert_frame(points, center, rotation)
    return float(np.abs(x).max()), float(np.abs(y).max())


def _rectangle_size(points, center, rotation: float) -> float:
    """Return the area of the rectangle about a center and rotation, in km2."""
    half_x, half_y = _half_extents(points, center, rotation)
    return 4 * half_x * half_y


def _coarse_rotation(points, center) -> float:
    """Return the rotation, in radians, to the degree, of the smallest rectangle."""
    import numpy as np

    rotations = np.radians(np.arange(-90.0, 90.0, 1.0))
    sizes = [_rectangle_size(points, center, rotation) for rotation in rotations]
    return float(rotations[int(np.argmin(sizes))])


def _smallest_rectangle(points, center, rotation: float, turn: bool):
    """Return the center, rotation and area of the smallest rectangle nearby.

    The center moves east, west, north and south, and with ``turn`` the
    rotation turns, by steps that halve whenever no move makes the rectangle
    smaller, down to about a kilometer.
    """
    step_km, step_turn = RECTANGLE_STEP_KM, math.radians(RECTANGLE_STEP_DEGREES)
    size = _rectangle_size(points, center, rotation)
    while step_km > RECTANGLE_TOLERANCE_KM:
        moves = [(step_km, 0.0, 0.0), (-step_km, 0.0, 0.0), (0.0, step_km, 0.0)]
        moves += [(0.0, -step_km, 0.0)]
        if turn:
            moves += [(0.0, 0.0, step_turn), (0.0, 0.0, -step_turn)]
        for east, north, angle in moves:
            moved = center
            if east or north:
                moved = _clamp_lat(
                    _destination(
                        center,
                        math.atan2(east, north),
                        math.hypot(east, north) / EARTH_RADIUS_KM,
                    )
                )
            candidate = _rectangle_size(points, moved, rotation + angle)
            if candidate < size * (1 - 1e-9):
                center, rotation, size = moved, rotation + angle, candidate
                break
        else:
            step_km, step_turn = step_km / 2, step_turn / 2
    return center, rotation, size


def fit_project_hexes(
    region, margin_km: float, resolution_km: float = 0.0
) -> Dict[str, Any]:
    """Return the ``project_hexes`` rectangle holding a region and its margin.

    ``resolution_km`` is the cell spacing, which sets how far the mesh's
    boundary layers reach past the rectangle.

    The rectangle is rotated, with ``rotation_degrees``, only if that makes it
    ``MIN_ROTATION_GAIN`` smaller. Its center and rotation are searched from
    the middle of the region for the smallest rectangle, with the rotation
    free and with it fixed at 0.

    Raises
    ------
    RegionTooLarge
        If project_hexes cannot draw the rectangle: a region near 90 degrees
        from its center, around a pole, or reaching past the pole on its cone.
    """
    import numpy as np

    points = _points(region)
    center = _clamp_lat(_flat_center(points, "project_hexes rectangle"))
    lon, lat = _lonlat(center)
    x, y = lambert(points, lat, lon)
    points = _with_margin(points[_hull(x, y)], margin_km)

    turned = _smallest_rectangle(
        points, center, _coarse_rotation(points, center), turn=True
    )
    straight = _smallest_rectangle(points, center, 0.0, turn=False)
    if turned[2] <= (1 - MIN_ROTATION_GAIN) * straight[2]:
        center, rotation, _ = turned
    else:
        center, rotation, _ = straight

    # The rectangle is sized about the center and rotation as they are written
    rotation_degrees = round((math.degrees(rotation) + 90.0) % 180.0 - 90.0, 1)
    written, center = _rounded(_clamp_lat(center))
    rotation = math.radians(rotation_degrees)
    x, y, angle = _lambert_frame(points, center, rotation)
    half_x, half_y = float(np.abs(x).max()), float(np.abs(y).max())
    lat = written["center_lat"]
    apex = lambert_apex_km(lat)
    reach = project_hexes_reach_km(2 * half_y, resolution_km)
    if float(np.abs(angle).max()) >= 90.0 or reach >= apex:
        raise RegionTooLarge(
            f"The region reaches around or past the pole on project_hexes' cone at "
            f"{lat:.1f} degrees, where a mesh can reach less than {apex:.0f} km "
            "north or south of its center. Use create_region."
        )
    block = {
        **written,
        "extent_x_km": _round_up(2 * half_x),
        "extent_y_km": _round_up(2 * half_y),
    }
    if rotation_degrees:
        block["rotation_degrees"] = rotation_degrees
    return block


def regional_config(
    region, method: str, shape: str, resolution_km: float, buffer_km: float
) -> Dict[str, Any]:
    """Return the ``regional`` block of a mesh request for a region.

    Parameters
    ----------
    region : shapely geometry
        The region's outline, from the geo lookup.
    method : str
        ``"project_hexes"`` or ``"create_region"``.
    shape : str
        One of ``SHAPES``. ``project_hexes`` takes only ``"rectangle"``.
    resolution_km : float
        Cell spacing of the mesh.
    buffer_km : float
        Margin around the region. It is at least ``MARGIN_FRACTION`` of the
        region's diameter. Both tools add their boundary layers outside the
        shape, so the margin needs no room for them.

    Raises
    ------
    RegionTooLarge
        If the region does not fit the shape.
    RegionTooSmall
        If the shape reaches less than ``MIN_HALF_CELLS`` cells from its
        center, which the tools cannot make a mesh of.
    ValueError
        If the method or shape is unknown, or they do not go together.
    """
    if method not in ("project_hexes", "create_region"):
        raise ValueError(f"Unknown method {method!r}")
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. Allowed: {', '.join(SHAPES)}")
    margin = max(float(buffer_km), MARGIN_FRACTION * diameter_km(region))
    if method == "project_hexes":
        if shape != "rectangle":
            raise ValueError(f"project_hexes makes only rectangles, not a {shape}")
        regional = {
            "project_hexes": fit_project_hexes(region, margin, float(resolution_km))
        }
    else:
        fit = {
            "circle": fit_circle,
            "ellipse": fit_ellipse,
            "polygon": fit_polygon,
            "rectangle": fit_rectangle,
        }[shape]
        regional = {"create_region": fit(region, margin)}
    half = half_size_km(regional)
    least = MIN_HALF_CELLS * float(resolution_km)
    if half < least:
        raise RegionTooSmall(
            f"The {shape} reaches only {half:.0f} km from its center, and with "
            f"{float(resolution_km):g} km cells a mesh must reach at least "
            f"{least:.0f} km. Ask for smaller cells or a larger region."
        )
    return regional


def half_size_km(regional: Dict[str, Any]) -> float:
    """Return how far a regional block's shape reaches from its center, at least.

    It is a circle's radius, an ellipse's semi-minor axis, half a rectangle's
    shorter side, or the distance from a polygon's inside point to its nearest
    side.
    """
    import numpy as np
    from shapely.geometry import Point, Polygon

    ((method, block),) = regional.items()
    if method == "project_hexes":
        return min(block["extent_x_km"], block["extent_y_km"]) / 2
    ((shape, values),) = block.items()
    if shape == "circle":
        return values["radius_km"]
    if shape == "ellipse":
        return values["semi_minor_axis_km"]
    center = _unit(values["inside_lon"], values["inside_lat"])
    vertices = np.asarray(values["vertices"], dtype=float)
    x, y = gnomonic(_unit(vertices[:, 1], vertices[:, 0]), center)
    # The gnomonic frame is the sphere's own scale at its center
    return Polygon(zip(x, y)).exterior.distance(Point(0, 0)) * EARTH_RADIUS_KM
