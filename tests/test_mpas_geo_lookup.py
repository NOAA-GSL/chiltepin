# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh agent's geo lookup.

Most tests look names up in small Natural Earth stand-ins written here, and
Nominatim is replaced by a fake. One test downloads Natural Earth's real
countries.
"""

import io
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

import pytest
import shapefile

from chiltepin.agents.mpas import _geo_lookup as geo
from chiltepin.agents.mpas import _plot_mesh as plot


def square(lon, lat, size=1.0):
    """Return a square's ring, from its southwest corner."""
    return [
        [lon, lat],
        [lon, lat + size],
        [lon + size, lat + size],
        [lon + size, lat],
        [lon, lat],
    ]


COUNTRIES = [
    # ADMIN, NAME, NAME_LONG, NAME_EN, CONTINENT, parts
    ("Alpha", "Alpha", "Republic of Alpha", "Alpha", "Testland", [square(0, 0, 4)]),
    # Its second part is 40 degrees away, far overseas
    ("Beta", "Beta", "Beta", "Béta", "Testland", [square(10, 0, 4), square(50, 0)]),
    # Split at the 180th meridian, as Natural Earth splits Russia and Fiji
    (
        "Gamma",
        "Gamma",
        "Gamma",
        "Gamma",
        "Farland",
        [square(178, 10, 2), square(-180, 10, 2)],
    ),
    ("Georgia", "Georgia", "Georgia", "Georgia", "Farland", [square(40, 40)]),
]

STATES = [
    # name, name_en, gn_name, woe_name, name_alt, admin, ring
    ("Rio", "Rio", "", "", "", "Alpha", square(0, 0)),
    ("Punjab", "Punjab", "", "", "", "Alpha", square(1, 1)),
    ("Punjab", "Punjab", "", "", "", "Beta", square(11, 1)),
    ("Georgia", "Georgia", "", "", "", "Beta", square(12, 2)),
    ("Québec", "Quebec", "", "", "Province of Quebec|Kebek", "Beta", square(13, 1)),
]


def write_boundaries(directory, countries=COUNTRIES, states=STATES):
    """Write Natural Earth stand-ins with the fields the lookup reads."""
    path = plot.layer_path(directory, geo.COUNTRIES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        for field in ("ADMIN", "NAME", "NAME_LONG", "NAME_EN", "CONTINENT"):
            writer.field(field, "C")
        for *fields, parts in countries:
            writer.poly(parts)
            writer.record(*fields)
    if states is not None:
        path = plot.layer_path(directory, geo.STATES)
        with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
            for field in ("name", "name_en", "gn_name", "woe_name", "name_alt"):
                writer.field(field, "C")
            writer.field("admin", "C")
            for *fields, ring in states:
                writer.poly([ring])
                writer.record(*fields)
    return directory


@pytest.fixture
def boundaries(tmp_path):
    return write_boundaries(tmp_path / "ne")


def bounds(geometry):
    return [round(value, 6) for value in geometry.bounds]


def test_module_imports_only_the_standard_library():
    """The agent imports the lookup without the chiltepin[mpas] extra."""
    code = (
        "import sys\n"
        "import chiltepin.agents.mpas._geo_lookup\n"
        "heavy = ('numpy', 'shapely', 'shapefile', 'netCDF4', 'matplotlib')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("name", "normal"),
    [("Québec", "quebec"), ("  New   York ", "new york"), ("SÃO PAULO", "sao paulo")],
)
def test_normalize_name(name, normal):
    assert geo.normalize_name(name) == normal


def test_user_agent_names_chiltepin():
    agent = geo.user_agent()
    assert agent.startswith("chiltepin/")
    assert geo.PROJECT_URL in agent


def test_download_boundaries(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        geo, "download_map_data", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    geo.download_boundaries(str(tmp_path))
    assert calls == [((tmp_path,), {"layers": geo.BOUNDARY_LAYERS})]


# Natural Earth names


@pytest.mark.parametrize(
    "name", ["Alpha", "Republic of Alpha", "  alpha ", "ALPHA"], ids=str.strip
)
def test_country_by_any_of_its_names(boundaries, name):
    assert bounds(geo.lookup_region([name], boundaries)) == [0, 0, 4, 4]


def test_country_by_its_name_with_accents(boundaries):
    # NAME_EN is "Béta", which matches without its accent too
    assert bounds(geo.lookup_region(["beta"], boundaries)) == [10, 0, 14, 4]
    assert bounds(geo.lookup_region(["Béta"], boundaries)) == [10, 0, 14, 4]


def test_overseas_parts_are_left_out(boundaries):
    # Beta's part 36 degrees from its mainland is left out, and kept when the
    # limit reaches it
    assert bounds(geo.lookup_region(["Beta"], boundaries)) == [10, 0, 14, 4]
    far = geo.lookup_region(["Beta"], boundaries, proximity_km=5000)
    assert bounds(far) == [10, 0, 51, 4]


def test_continent_stands_for_its_countries(boundaries):
    # Each country keeps its own parts, so neither is dropped for being far
    # from the other
    region = geo.lookup_region(["Testland"], boundaries, proximity_km=500)
    assert bounds(region) == [0, 0, 14, 4]


def test_continent_without_excluded_countries(boundaries):
    region = geo.lookup_region(
        ["Testland"], boundaries, exclude_names=["Republic of Alpha"]
    )
    assert bounds(region) == [10, 0, 14, 4]


@pytest.mark.parametrize("name", ["Québec", "quebec", "Province of Quebec", "Kebek"])
def test_state_by_its_names(boundaries, name):
    assert bounds(geo.lookup_region([name], boundaries)) == [13, 1, 14, 2]


def test_country_comes_before_a_state_of_the_same_name(boundaries):
    assert bounds(geo.lookup_region(["Georgia"], boundaries)) == [40, 40, 41, 41]
    state = geo.lookup_region(["Georgia, Beta"], boundaries)
    assert bounds(state) == [12, 2, 13, 3]


def test_state_used_in_several_countries_is_ambiguous(boundaries):
    with pytest.raises(geo.RegionNotFound, match=r"'Name, Country': Punjab \(Alpha"):
        geo.lookup_region(["Punjab"], boundaries)


@pytest.mark.parametrize("name", ["Punjab, Alpha", "punjab , Republic of Alpha"])
def test_state_with_its_country(boundaries, name):
    assert bounds(geo.lookup_region([name], boundaries)) == [1, 1, 2, 2]


@pytest.mark.parametrize("name", ["Rio, Beta", "Rio, Nowhere"])
def test_state_with_another_country_is_missing(boundaries, name):
    with pytest.raises(geo.RegionNotFound, match=repr(name)):
        geo.lookup_region([name], boundaries)


def test_several_names_are_joined(boundaries):
    region = geo.lookup_region(["Rio", "Québec"], boundaries)
    assert bounds(region) == [0, 0, 14, 2]


def test_missing_name_is_not_dropped(boundaries):
    # Without Nominatim, a name Natural Earth lacks fails the lookup, even when
    # the others are found
    with pytest.raises(geo.RegionNotFound) as error:
        geo.lookup_region(["Alpha", "Springfield", "Nowhere"], boundaries)
    message = str(error.value)
    assert "'Springfield', 'Nowhere'" in message
    assert "without a Nominatim contact" in message


def test_missing_and_ambiguous_names_are_both_reported(boundaries):
    with pytest.raises(geo.RegionNotFound) as error:
        geo.lookup_region(["Punjab", "Nowhere"], boundaries)
    assert "'Nowhere'" in str(error.value)
    assert "Punjab (Alpha, Beta)" in str(error.value)


def test_excluded_names_are_skipped(boundaries):
    region = geo.lookup_region(["Alpha", "Rio", "Nowhere"], boundaries, ["nowhere"])
    assert bounds(region) == [0, 0, 4, 4]


def test_region_of_only_excluded_names(boundaries):
    with pytest.raises(geo.RegionNotFound, match="Every name is excluded"):
        geo.lookup_region(["Alpha"], boundaries, exclude_names=["alpha"])


def test_continent_of_only_excluded_countries(boundaries):
    with pytest.raises(geo.RegionNotFound, match="'Farland'"):
        geo.lookup_region(["Farland"], boundaries, exclude_names=["Gamma", "Georgia"])


def test_country_across_the_180th_meridian_is_one_piece(boundaries):
    # Gamma's part west of the meridian is put east of it, next to the other
    region = geo.lookup_region(["Gamma"], boundaries)
    assert region.geom_type == "Polygon"
    assert bounds(region) == [178, 10, 182, 12]


def test_places_are_put_near_the_largest(tmp_path):
    countries = COUNTRIES + [
        ("Delta", "Delta", "Delta", "Delta", "Farland", [square(-179, 20, 1)]),
    ]
    data = write_boundaries(tmp_path / "ne", countries=countries)
    region = geo.lookup_region(["Gamma", "Delta"], data)
    assert bounds(region) == [178, 10, 182, 21]


# Nominatim


def feature_collection(geometry):
    features = [{"type": "Feature", "geometry": geometry}] if geometry else []
    return {"type": "FeatureCollection", "features": features}


def polygon(lon, lat, size=1.0):
    return {"type": "Polygon", "coordinates": [square(lon, lat, size)]}


@pytest.fixture
def nominatim(monkeypatch):
    """Replace Nominatim with a fake, which answers from ``places``.

    Set ``failures`` to a list of errors raised, in turn, before answering.
    """

    class Fake:
        places = {
            "Springfield": polygon(100, 30),
            "Center": {"type": "Point", "coordinates": [5, 5]},
        }
        failures = []
        requests = []
        sleeps = []

        def urlopen(self, request, timeout):
            self.requests.append(request)
            if self.failures:
                raise self.failures.pop(0)
            query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
            answer = feature_collection(self.places.get(query["q"][0]))
            return io.BytesIO(json.dumps(answer).encode())

    fake = Fake()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(geo.time, "sleep", fake.sleeps.append)
    monkeypatch.setattr(geo, "_last_nominatim_request", 0.0)
    return fake


def http_error(code):
    return urllib.error.HTTPError(geo.NOMINATIM_URL, code, "error", {}, None)


def test_nominatim_request(nominatim, tmp_path):
    geometry = geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert geometry == polygon(100, 30)
    (request,) = nominatim.requests
    url = urllib.parse.urlparse(request.full_url)
    assert f"{url.scheme}://{url.netloc}{url.path}" == geo.NOMINATIM_URL
    query = urllib.parse.parse_qs(url.query)
    assert query["q"] == ["Springfield"]
    assert query["email"] == ["me@example.org"]
    assert query["format"] == ["geojson"]
    assert query["polygon_geojson"] == ["1"]
    assert query["limit"] == ["1"]
    assert request.get_header("User-agent") == geo.user_agent()


def test_nominatim_answers_are_cached(nominatim, tmp_path):
    cache = tmp_path / "cache"
    assert geo.query_nominatim("Springfield", "me@example.org", cache)
    # Names that differ only in case or spaces, and places not found, are
    # asked for once
    assert geo.query_nominatim(" springfield ", "me@example.org", cache)
    assert geo.query_nominatim("Nowhere", "me@example.org", cache) is None
    assert geo.query_nominatim("nowhere", "me@example.org", cache) is None
    assert len(nominatim.requests) == 2
    assert len(list(cache.glob("*.json"))) == 2
    assert not list(cache.glob(".*"))


def test_nominatim_requests_are_spaced(nominatim, tmp_path, monkeypatch):
    # The first request starts at 100.0 and ends at 100.2, and the second is
    # ready at 100.3
    clock = iter([100.0, 100.2, 100.3, 101.4])
    monkeypatch.setattr(geo.time, "monotonic", lambda: next(clock))
    geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    geo.query_nominatim("Elsewhere", "me@example.org", tmp_path)
    # So it waits until the interval after the first one ends
    assert nominatim.sleeps == [pytest.approx(100.2 + geo.NOMINATIM_INTERVAL - 100.3)]


@pytest.mark.parametrize(
    "error",
    [http_error(429), http_error(503), urllib.error.URLError("down"), TimeoutError()],
    ids=["429", "503", "unreachable", "timeout"],
)
def test_nominatim_retries_transient_failures(nominatim, tmp_path, error):
    nominatim.failures = [error, error]
    assert geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert len(nominatim.requests) == 3
    # Backing off 2 and 4 seconds between tries
    assert [s for s in nominatim.sleeps if s >= 2] == [2, 4]


@pytest.mark.parametrize(
    "error", [http_error(503), urllib.error.URLError("down")], ids=["503", "down"]
)
def test_nominatim_gives_up_after_its_attempts(nominatim, tmp_path, error):
    nominatim.failures = [error] * 3
    with pytest.raises(type(error)):
        geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert len(nominatim.requests) == 3
    assert not list(tmp_path.iterdir())


def test_nominatim_does_not_retry_a_refused_request(nominatim, tmp_path):
    nominatim.failures = [http_error(403)]
    with pytest.raises(urllib.error.HTTPError):
        geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert len(nominatim.requests) == 1


def test_lookup_asks_nominatim_only_for_names_natural_earth_lacks(
    nominatim, boundaries, tmp_path
):
    region = geo.lookup_region(
        ["Alpha", "Springfield"],
        boundaries,
        nominatim_contact="me@example.org",
        cache_dir=tmp_path / "cache",
        proximity_km=50_000,
    )
    assert bounds(region) == [0, 0, 101, 31]
    assert len(nominatim.requests) == 1


def test_place_without_an_outline_is_missing(nominatim, boundaries, tmp_path):
    with pytest.raises(geo.RegionNotFound) as error:
        geo.lookup_region(
            ["Center", "Nowhere"],
            boundaries,
            nominatim_contact="me@example.org",
            cache_dir=tmp_path,
        )
    assert "'Center', 'Nowhere'" in str(error.value)
    assert "Nominatim contact" not in str(error.value)


def test_nominatim_contact_needs_a_cache(boundaries):
    with pytest.raises(ValueError, match="cache_dir is needed"):
        geo.lookup_region(["Alpha"], boundaries, nominatim_contact="me@example.org")


# Natural Earth's real countries


@pytest.fixture(scope="module")
def real_countries(tmp_path_factory):
    """Download Natural Earth's countries, with a stand-in for its states."""
    directory = tmp_path_factory.mktemp("natural-earth")
    plot.download_map_data(directory, layers=[geo.COUNTRIES])
    path = plot.layer_path(directory, geo.STATES)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        for field in ("name", "name_en", "gn_name", "woe_name", "name_alt", "admin"):
            writer.field(field, "C")
        writer.poly([square(-109.05, 37, 7)])
        writer.record("Colorado", "Colorado", "", "", "", "United States of America")
    return directory


def test_real_countries(real_countries):
    # France without French Guiana
    assert geo.lookup_region(["France"], real_countries).bounds[0] > -10
    # Russia in one piece across the 180th meridian
    russia = geo.lookup_region(["Russia"], real_countries)
    assert 180 < russia.bounds[2] < 200
    # Europe without Russia keeps Iceland and Portugal
    europe = geo.lookup_region(["Europe"], real_countries, exclude_names=["Russia"])
    assert europe.bounds[0] < -20
    assert europe.bounds[2] < 50
    colorado = geo.lookup_region(["Colorado"], real_countries)
    assert bounds(colorado) == [-109.05, 37, -102.05, 44]
