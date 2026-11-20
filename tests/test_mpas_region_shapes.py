# SPDX-License-Identifier: Apache-2.0
"""Tests of fitting regions to the shapes the MPAS mesh tools make.

Regions are built here on the sphere, as shapely polygons of longitude and
latitude. Each fitted shape is checked in its tool's own frame: create_region's
circle and ellipse in the azimuthal equidistant frame its code draws them in,
its polygon in the gnomonic frame where its great-circle sides are straight,
and project_hexes' rectangle with the inverse of its own projection, copied
from MPAS-Tools.
"""

import math
import subprocess
import sys

import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Point, Polygon

from chiltepin.agents.mpas import _region_shapes as shapes
from chiltepin.agents.mpas.mesh import _MeshBehavior

R = shapes.EARTH_RADIUS_KM


def region_from(points):
    """Return the region whose outline is the given unit vectors."""
    lonlat = [shapes._lonlat(point) for point in points]
    # Longitudes stay continuous around the region, as the geo lookup's do
    lons = np.unwrap(np.radians([lon for lon, _ in lonlat]))
    return Polygon(zip(np.degrees(lons), [lat for _, lat in lonlat]))


def cap(lon, lat, radius_km, count=90):
    """Return a circle on the sphere, as create_region draws one."""
    center = shapes._unit(lon, lat)
    azimuths = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return region_from(shapes._destination(center, azimuths, radius_km / R))


def ellipse(lon, lat, major_km, minor_km, orientation, count=180):
    """Return an ellipse in the azimuthal equidistant frame, as create_region's."""
    center = shapes._unit(lon, lat)
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    u, v = major_km * np.cos(t), minor_km * np.sin(t)
    # u is along the azimuth of the major axis, clockwise from north
    theta = math.radians(orientation)
    x = u * math.sin(theta) + v * math.cos(theta)
    y = u * math.cos(theta) - v * math.sin(theta)
    return region_from(
        shapes._destination(center, np.arctan2(x, y), np.hypot(x, y) / R)
    )


def lambert_inverse(x, y, center_lat, center_lon):
    """Return longitudes and latitudes of plane points, as project_hexes does.

    This is ``xy_to_lambert_conformal_projection`` from MPAS-Tools 2.0.0, with
    the reference latitude at the center, and both standard parallels where
    the agent writes them.
    """
    phi0 = math.radians(center_lat)
    phi1 = math.radians(shapes.standard_parallel(center_lat))
    n = math.sin(phi1)
    factor = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n
    rho0 = factor * R / math.tan(math.pi / 4 + phi0 / 2) ** n
    rho = np.sign(n) * np.sqrt(x**2 + (rho0 - y) ** 2)
    theta = np.arctan(x / (rho0 - y))
    lon = math.radians(center_lon) + theta / n
    lat = 2 * np.arctan((R * factor / rho) ** (1 / n)) - math.pi / 2
    return np.degrees(lon), np.degrees(lat)


def rectangle(lon, lat, width_km, height_km, rotation=0.0, count=40):
    """Return the outline of a project_hexes rectangle, rotated by grid_rotate."""
    side = np.linspace(-0.5, 0.5, count, endpoint=False)
    x = np.concatenate([side, np.full(count, 0.5), -side, np.full(count, -0.5)])
    y = np.concatenate([np.full(count, -0.5), side, np.full(count, 0.5), -side])
    lons, lats = lambert_inverse(x * width_km, y * height_km, lat, lon)
    center = shapes._unit(lon, lat)
    points = shapes._rotate(shapes._unit(lons, lats), center, math.radians(rotation))
    return region_from(points)


def outline(region):
    return shapes._points(region)


def margin_ring(region, margin_km):
    """Return points a little less than ``margin_km`` around the outline."""
    points = outline(region)
    azimuths = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    return np.concatenate(
        [shapes._destination(p, azimuths, 0.95 * margin_km / R) for p in points]
    )


def inside_circle(block, points):
    center = shapes._unit(block["center_lon"], block["center_lat"])
    return shapes._angle(points, center) * R <= block["radius_km"]


def inside_ellipse(block, points):
    center = shapes._unit(block["center_lon"], block["center_lat"])
    x, y = shapes.aeqd(points, center)
    theta = math.radians(block["orientation_degrees"])
    u = x * math.sin(theta) + y * math.cos(theta)
    v = x * math.cos(theta) - y * math.sin(theta)
    a, b = block["semi_major_axis_km"], block["semi_minor_axis_km"]
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def inside_polygon(block, points):
    center = shapes._unit(block["inside_lon"], block["inside_lat"])
    vertices = np.asarray(block["vertices"])
    vx, vy = shapes.gnomonic(shapes._unit(vertices[:, 1], vertices[:, 0]), center)
    x, y = shapes.gnomonic(points, center)
    # Vertices are written to a millionth of a degree, about 0.1 m
    return contains_xy(Polygon(zip(vx, vy)).buffer(1e-8), x, y)


def inside_rectangle(block, points):
    center = shapes._unit(block["center_lon"], block["center_lat"])
    rotation = math.radians(block.get("rotation_degrees", 0.0))
    unrotated = shapes._rotate(points, center, -rotation)
    x, y = shapes.lambert(unrotated, block["center_lat"], block["center_lon"])
    return (np.abs(x) <= block["extent_x_km"] / 2) & (
        np.abs(y) <= block["extent_y_km"] / 2
    )


def test_module_imports_only_the_standard_library():
    code = (
        "import sys\n"
        "import chiltepin.agents.mpas._region_shapes\n"
        "print(sorted(m for m in ('numpy', 'shapely') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


# Frames


def test_destination_and_azimuthal_equidistant_frame():
    center = shapes._unit(-105, 40)
    points = shapes._destination(center, np.radians([0, 90, 200]), [0.1, 0.2, 0.3])
    x, y = shapes.aeqd(points, center)
    assert np.hypot(x, y) == pytest.approx(np.array([0.1, 0.2, 0.3]) * R)
    # North is +y and east is +x
    assert (x[0], y[0]) == pytest.approx((0, 0.1 * R), abs=1e-6)
    assert (x[1], y[1]) == pytest.approx((0.2 * R, 0), abs=1e-6)


def test_frame_at_a_pole():
    east, north = shapes._frame(np.array([0.0, 0.0, 1.0]))
    assert east @ north == pytest.approx(0)
    assert np.linalg.norm(east) == pytest.approx(1)


def test_lonlat_wraps_longitudes():
    assert shapes._lonlat(shapes._unit(190, 10)) == pytest.approx((-170, 10))


def test_gnomonic_frame_makes_great_circles_straight():
    center = shapes._unit(10, 50)
    a, b = shapes._unit(0, 40), shapes._unit(30, 60)
    # Points along the great circle from a to b
    middle = [(a * (1 - t) + b * t) for t in np.linspace(0, 1, 9)]
    middle = np.array([m / np.linalg.norm(m) for m in middle])
    x, y = shapes.gnomonic(middle, center)
    slopes = np.diff(y) / np.diff(x)
    assert slopes == pytest.approx(np.full(len(slopes), slopes[0]))
    back = shapes.gnomonic_inverse(x, y, center)
    assert back == pytest.approx(middle)


@pytest.mark.parametrize(
    ("lat", "lon", "rotation"),
    [(40, -100, 0), (-35, 150, 30), (70, 10, -60), (0, 20, 0)],
)
def test_lambert_matches_project_hexes(lat, lon, rotation):
    # Points project_hexes puts on the sphere go back to its plane
    x = np.array([-900.0, 0.0, 450.0, 700.0])
    y = np.array([300.0, -800.0, 0.0, 650.0])
    lons, lats = lambert_inverse(x, y, lat, lon)
    fx, fy = shapes.lambert(shapes._unit(lons, lats), lat, lon)
    assert fx == pytest.approx(x, abs=1e-6)
    assert fy == pytest.approx(y, abs=1e-6)


def test_rotation_is_counter_clockwise_from_above():
    center = shapes._unit(0, 0)
    east = shapes._destination(center, math.pi / 2, 0.1)
    turned = shapes._rotate(east[None, :], center, math.pi / 2)[0]
    x, y = shapes.aeqd(turned[None, :], center)
    # East turned a quarter counter-clockwise is north
    assert (x[0], y[0]) == pytest.approx((0, 0.1 * R), abs=1e-6)


def test_lambert_apex():
    # For a cone tangent at 40 degrees, the apex is R cot(40) north
    assert shapes.lambert_apex_km(40) == pytest.approx(R / math.tan(math.radians(40)))
    assert shapes.lambert_apex_km(0) == math.inf


def test_standard_parallel():
    assert shapes.standard_parallel(40) == 40
    assert shapes.standard_parallel(0) == shapes.EQUATOR_PARALLEL_DEGREES


# Shapes


@pytest.mark.parametrize(("lon", "lat"), [(-105, 39), (179, -10), (0, 85)])
def test_circle_of_a_circle(lon, lat):
    region = cap(lon, lat, 600)
    block = shapes.fit_circle(region, 100)["circle"]
    assert block["radius_km"] == pytest.approx(700, rel=0.01)
    center = shapes._unit(block["center_lon"], block["center_lat"])
    assert shapes._angle(center[None, :], shapes._unit(lon, lat))[0] * R < 10
    assert inside_circle(block, outline(region)).all()
    assert inside_circle(block, margin_ring(region, 100)).all()


@pytest.mark.parametrize(
    ("lon", "lat", "orientation"), [(-100, 40, 30), (179.5, 0, 120), (20, 70, 90)]
)
def test_ellipse_of_an_ellipse(lon, lat, orientation):
    region = ellipse(lon, lat, 1500, 600, orientation)
    block = shapes.fit_ellipse(region, 0)["ellipse"]
    assert block["semi_major_axis_km"] == pytest.approx(1500, rel=0.02)
    assert block["semi_minor_axis_km"] == pytest.approx(600, rel=0.02)
    turn = (block["orientation_degrees"] - orientation + 90) % 180 - 90
    assert abs(turn) < 2
    assert inside_ellipse(block, outline(region)).all()


def test_ellipse_holds_the_margin():
    region = ellipse(-100, 40, 1500, 600, 30)
    block = shapes.fit_ellipse(region, 200)["ellipse"]
    assert inside_ellipse(block, outline(region)).all()
    assert inside_ellipse(block, margin_ring(region, 200)).all()
    assert block["semi_minor_axis_km"] == pytest.approx(800, rel=0.05)


def test_polygon_of_a_region():
    # An L shape, whose hull cuts its inside corner
    region = Polygon(
        [(-110, 30), (-90, 30), (-90, 35), (-105, 35), (-105, 45), (-110, 45)]
    )
    block = shapes.fit_polygon(region, 100)["polygon"]
    assert inside_polygon(block, outline(region)).all()
    assert inside_polygon(block, margin_ring(region, 100)).all()
    inside = shapes._unit(block["inside_lon"], block["inside_lat"])
    assert inside_polygon(block, inside[None, :]).all()
    # Counter-clockwise seen from above, without the first vertex repeated
    center = inside
    vertices = np.asarray(block["vertices"])
    x, y = shapes.gnomonic(shapes._unit(vertices[:, 1], vertices[:, 0]), center)
    assert Polygon(zip(x, y)).exterior.is_ccw
    assert block["vertices"][0] != block["vertices"][-1]


def test_rectangle_is_a_polygon_of_four_vertices():
    region = ellipse(140, 38, 1200, 300, 40)
    block = shapes.fit_rectangle(region, 50)["polygon"]
    assert len(block["vertices"]) == 4
    assert inside_polygon(block, outline(region)).all()
    assert inside_polygon(block, margin_ring(region, 50)).all()


def test_polygon_across_the_180th_meridian():
    region = Polygon([(170, -20), (190, -20), (190, -5), (170, -5)])
    block = shapes.fit_polygon(region, 0)["polygon"]
    assert inside_polygon(block, outline(region)).all()
    assert all(-180 <= lon < 180 for _, lon in block["vertices"])


@pytest.mark.parametrize(
    ("lon", "lat", "rotation"),
    [(-100, 40, 0), (140, 38, 35), (-70, -35, -20), (179, 15, 10)],
)
def test_project_hexes_of_a_rectangle(lon, lat, rotation):
    region = rectangle(lon, lat, 1800, 700, rotation)
    block = shapes.fit_project_hexes(region, 0)
    assert block["center_lat"] == pytest.approx(lat, abs=0.2)
    assert ((block["center_lon"] - lon + 180) % 360 - 180) == pytest.approx(0, abs=0.2)
    assert block["extent_x_km"] == pytest.approx(1800, rel=0.02)
    assert block["extent_y_km"] == pytest.approx(700, rel=0.02)
    assert block.get("rotation_degrees", 0.0) == pytest.approx(rotation, abs=1)
    assert inside_rectangle(block, outline(region)).all()


def test_project_hexes_is_not_rotated_for_a_small_gain():
    # A circle is as small at any rotation
    block = shapes.fit_project_hexes(cap(-100, 40, 500), 0)
    assert "rotation_degrees" not in block


def test_project_hexes_holds_the_margin():
    region = rectangle(-100, 40, 1000, 600, 25)
    block = shapes.fit_project_hexes(region, 150)
    assert inside_rectangle(block, outline(region)).all()
    assert inside_rectangle(block, margin_ring(region, 150)).all()


def test_project_hexes_on_the_equator():
    # The agent tangents the cone just off the equator, so the center can be on it
    region = Polygon([(-80, -3), (-76, -3), (-76, 3), (-80, 3)])
    block = shapes.fit_project_hexes(region, 0)
    assert block["center_lat"] == pytest.approx(0, abs=1e-3)
    assert inside_rectangle(block, outline(region)).all()
    on_it = {**block, "center_lat": 0.0}
    _MeshBehavior._validate_project_hexes(on_it)
    assert inside_rectangle(on_it, outline(region)).all()


def test_project_hexes_near_a_pole():
    region = cap(30, 80, 400)
    block = shapes.fit_project_hexes(region, 0)
    assert inside_rectangle(block, outline(region)).all()


@pytest.mark.parametrize(
    "region",
    [
        # Around the pole, which is the apex of every project_hexes cone
        cap(30, 88, 400),
        # Wide at a high latitude, as Russia is, so it reaches around the cone
        Polygon([(20, 45), (190, 45), (190, 75), (20, 75)]),
    ],
    ids=["pole", "wide"],
)
def test_project_hexes_cannot_reach_past_the_pole(region):
    with pytest.raises(shapes.RegionTooLarge, match="Use create_region"):
        shapes.fit_project_hexes(region, 0)


def test_project_hexes_rectangle_reaching_past_the_apex(monkeypatch):
    # Every point is in front of the cone, but the rectangle is as tall as its
    # farthest point the other way, which reaches past the apex
    monkeypatch.setattr(shapes, "lambert_apex_km", lambda lat: 100.0)
    with pytest.raises(shapes.RegionTooLarge, match="less than 100 km"):
        shapes.fit_project_hexes(cap(-100, 40, 300), 0)


def test_project_hexes_boundary_layers_count_toward_the_pole():
    # The pole is 670 km from the center. A 15km mesh reaches about 370 km,
    # and a 60km mesh's boundary layers reach about 730 km.
    region = cap(30, 84, 250)
    shapes.fit_project_hexes(region, 0, resolution_km=15)
    with pytest.raises(shapes.RegionTooLarge, match="Use create_region"):
        shapes.fit_project_hexes(region, 0, resolution_km=60)


def test_project_hexes_reach():
    assert shapes.project_hexes_reach_km(1000, 15) == 500 + 8 * 15


@pytest.mark.parametrize("fit", [shapes.fit_polygon, shapes.fit_rectangle])
def test_large_regions_are_not_flat(fit):
    region = cap(0, 0, 9500)
    with pytest.raises(shapes.RegionTooLarge, match="circle or an ellipse"):
        fit(region, 0)
    with pytest.raises(shapes.RegionTooLarge, match="circle or an ellipse"):
        shapes.fit_project_hexes(region, 0)


def test_polygon_margin_reaching_too_far():
    region = cap(0, 0, 8800)
    with pytest.raises(shapes.RegionTooLarge, match="too far for a polygon"):
        shapes.fit_polygon(region, 1300)


def test_circle_and_ellipse_of_a_large_region():
    region = cap(0, 0, 9500)
    assert shapes.fit_circle(region, 0)["circle"]["radius_km"] > 9000
    assert shapes.fit_ellipse(region, 0)["ellipse"]["semi_major_axis_km"] > 9000


@pytest.mark.parametrize("fit", [shapes.fit_circle, shapes.fit_ellipse])
def test_margin_reaching_around_the_globe(fit):
    with pytest.raises(shapes.RegionTooLarge, match="around the globe"):
        fit(cap(0, 0, 9500), 11000)


def test_region_around_the_globe_has_no_center():
    # Places spread evenly around the equator
    around = MultiPolygon(
        [
            Polygon([(lon, -5), (lon + 10, -5), (lon + 10, 5), (lon, 5)])
            for lon in (-5, 85, 175, 265)
        ]
    )
    with pytest.raises(shapes.RegionTooLarge, match="no center"):
        shapes.fit_circle(around, 0)


def test_ellipse_grown_around_the_globe(monkeypatch):
    # An ellipse whose fitted shape is so narrow that growing it to hold the
    # region makes it reach around the globe
    narrow = np.diag([1 / 20000.0**2, 1 / 1.0**2])
    monkeypatch.setattr(shapes, "_enclosing_ellipse", lambda xy: (np.zeros(2), narrow))
    with pytest.raises(shapes.RegionTooLarge, match="around the globe"):
        shapes.fit_ellipse(cap(0, 0, 2000), 0)


def test_region_of_several_polygons():
    region = MultiPolygon([cap(10, 50, 200), cap(20, 52, 200)])
    block = shapes.fit_circle(region, 0)["circle"]
    assert inside_circle(block, outline(region)).all()


@pytest.mark.parametrize(
    ("region", "message"),
    [(Point(0, 0), "must be polygons"), (Polygon(), "empty")],
)
def test_region_must_be_polygons(region, message):
    with pytest.raises(ValueError, match=message):
        shapes.fit_circle(region, 0)


def test_hull_of_points_in_a_line():
    assert list(shapes._hull(np.array([0.0, 1.0, 2.0]), np.zeros(3))) == [0, 2]


def test_diameter():
    assert shapes.diameter_km(cap(-100, 40, 500)) == pytest.approx(1000, rel=0.01)


# The regional block


@pytest.mark.parametrize(
    ("method", "shape", "kind"),
    [
        ("create_region", "circle", "circle"),
        ("create_region", "ellipse", "ellipse"),
        ("create_region", "polygon", "polygon"),
        ("create_region", "rectangle", "polygon"),
        ("project_hexes", "rectangle", None),
    ],
)
def test_regional_config_is_a_valid_request(method, shape, kind):
    region = ellipse(140, 38, 1000, 400, 40)
    regional = shapes.regional_config(region, method, shape, 15, 50)
    assert list(regional) == [method]
    _MeshBehavior._validate_regional(regional)
    if kind:
        assert list(regional[method]) == [kind]


def test_margin_is_a_part_of_the_diameter():
    region = cap(-100, 40, 1000)
    # A buffer smaller than a tenth of the 2000 km diameter gives 200 km
    small = shapes.regional_config(region, "create_region", "circle", 15, 10)
    assert small["create_region"]["circle"]["radius_km"] == pytest.approx(
        1200, rel=0.01
    )
    large = shapes.regional_config(region, "create_region", "circle", 15, 300)
    assert large["create_region"]["circle"]["radius_km"] == pytest.approx(
        1300, rel=0.01
    )


def test_project_hexes_gets_the_margin_and_the_cell_spacing(monkeypatch):
    # Its boundary layers are outside its rectangle, so the margin has no room
    # for them, and the cell spacing sets how far they reach
    calls = []
    monkeypatch.setattr(
        shapes,
        "fit_project_hexes",
        lambda region, margin, resolution: (
            calls.append((margin, resolution)),
            {"extent_x_km": 500, "extent_y_km": 400},
        )[1],
    )
    region = cap(-100, 40, 100)
    shapes.regional_config(region, "project_hexes", "rectangle", 15, 50)
    assert calls == [(50.0, 15.0)]


@pytest.mark.parametrize(
    ("method", "shape", "message"),
    [
        ("grid", "circle", "Unknown method 'grid'"),
        ("create_region", "hexagon", "Unknown shape 'hexagon'"),
        ("project_hexes", "circle", "only rectangles, not a circle"),
    ],
)
def test_regional_config_rejects(method, shape, message):
    with pytest.raises(ValueError, match=message):
        shapes.regional_config(cap(0, 40, 100), method, shape, 15, 50)


@pytest.mark.parametrize(
    ("method", "shape"),
    [
        ("create_region", "circle"),
        ("create_region", "ellipse"),
        ("create_region", "polygon"),
        ("create_region", "rectangle"),
        ("project_hexes", "rectangle"),
    ],
)
def test_shape_too_small_for_its_cells(method, shape):
    # Colorado, about 750 km across, reaches about 450 km from its center with
    # its margin, less than two 480 km cells, and more than two 60 km cells
    region = Polygon([(-109.05, 37), (-102.05, 37), (-102.05, 41), (-109.05, 41)])
    with pytest.raises(shapes.RegionTooSmall, match="Ask for smaller cells"):
        shapes.regional_config(region, method, shape, 480, 50)
    shapes.regional_config(region, method, shape, 60, 50)


def test_half_size():
    assert shapes.half_size_km({"create_region": {"circle": {"radius_km": 7}}}) == 7
    ellipse = {"semi_major_axis_km": 9, "semi_minor_axis_km": 4}
    assert shapes.half_size_km({"create_region": {"ellipse": ellipse}}) == 4
    hexes = {"extent_x_km": 300, "extent_y_km": 100}
    assert shapes.half_size_km({"project_hexes": hexes}) == 50
    # A square 2 degrees on a side at the equator reaches a degree from its
    # middle, about 111 km
    square = {
        "vertices": [[-1, -1], [-1, 1], [1, 1], [1, -1]],
        "inside_lat": 0,
        "inside_lon": 0,
    }
    half = shapes.half_size_km({"create_region": {"polygon": square}})
    assert half == pytest.approx(math.pi / 180 * R, rel=1e-3)
