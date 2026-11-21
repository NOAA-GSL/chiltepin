# SPDX-License-Identifier: Apache-2.0
"""Tests of the MPAS mesh agent's geo lookup.

Most tests look names up in small Natural Earth stand-ins written here, and
Nominatim is replaced by a fake. One test downloads Natural Earth's real
countries.
"""

import io
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

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
    # Far overseas, and smaller than Beta's other states
    ("Faraway", "Faraway", "", "", "", "Beta", square(60, 0, 0.5)),
]


def code(country):
    """Return the code joining a stand-in country to its states."""
    return country[:3].upper()


def write_boundaries(directory, countries=COUNTRIES, states=STATES):
    """Write Natural Earth stand-ins with the fields the lookup reads."""
    path = plot.layer_path(directory, geo.COUNTRIES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        for field in ("ADMIN", "NAME", "NAME_LONG", "NAME_EN", "CONTINENT", "ADM0_A3"):
            writer.field(field, "C")
        for *fields, parts in countries:
            writer.poly(parts)
            writer.record(*fields, code(fields[0]))
    if states is not None:
        path = plot.layer_path(directory, geo.STATES)
        with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
            for field in ("name", "name_en", "gn_name", "woe_name", "name_alt"):
                writer.field(field, "C")
            writer.field("admin", "C")
            writer.field("adm0_a3", "C")
            for *fields, ring in states:
                writer.poly([ring])
                writer.record(*fields, code(fields[-1]))
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
    region = geo.lookup_region(["Alpha", "Quebec", "Nowhere"], boundaries, ["nowhere"])
    assert bounds(region) == [0, 0, 14, 4]


def test_region_of_only_excluded_names(boundaries):
    with pytest.raises(geo.RegionNotFound, match="Every place is left out"):
        geo.lookup_region(["Alpha"], boundaries, exclude_names=["alpha"])


def test_continent_of_only_excluded_countries(boundaries):
    with pytest.raises(geo.RegionNotFound, match="Every place is left out"):
        geo.lookup_region(["Farland"], boundaries, exclude_names=["Gamma", "Georgia"])


def test_country_without_a_state_is_drawn_from_its_other_states(boundaries):
    # Beta's states are Punjab, Georgia, Quebec and Faraway, which is far
    # overseas and left out as Beta's own overseas part is
    region = geo.lookup_region(["Beta"], boundaries, exclude_names=["Quebec"])
    assert bounds(region) == [11, 1, 13, 3]


def test_continent_without_a_state_of_one_country(boundaries):
    # Alpha is drawn from its states without Rio, and Beta stays whole
    region = geo.lookup_region(["Testland"], boundaries, exclude_names=["Rio"])
    assert bounds(region) == [1, 0, 14, 4]
    assert region.difference(geo.lookup_region(["Rio"], boundaries)).equals(region)


def test_state_left_out_of_a_list_of_states(boundaries):
    region = geo.lookup_region(
        ["Rio", "Quebec", "Punjab, Alpha"], boundaries, exclude_names=["Quebec"]
    )
    assert bounds(region) == [0, 0, 2, 2]


def test_exclusion_outside_the_region_is_an_error(boundaries):
    # Leaving out a place the region does not have may mean it was not read
    # as meant, so it is not ignored
    with pytest.raises(geo.RegionNotFound, match="not in the region: 'Quebec'"):
        geo.lookup_region(["Alpha"], boundaries, exclude_names=["Quebec"])


def test_unknown_exclusion_is_an_error(boundaries):
    with pytest.raises(
        geo.RegionNotFound, match="to leave out is named 'Springfield'"
    ) as error:
        geo.lookup_region(["Alpha"], boundaries, exclude_names=["Springfield"])
    assert "not in the region" not in str(error.value)


def test_ambiguous_exclusion_is_an_error(boundaries):
    with pytest.raises(geo.RegionNotFound, match=r"Punjab \(Alpha, Beta\)") as error:
        geo.lookup_region(["Testland"], boundaries, exclude_names=["Punjab"])
    assert "not in the region" not in str(error.value)


def test_exclusion_settled_by_the_region(boundaries):
    # Of the two Punjabs, only Alpha's is in a region of Alpha
    region = geo.lookup_region(["Alpha"], boundaries, exclude_names=["Punjab"])
    assert bounds(region) == [0, 0, 1, 1]


def test_country_named_with_its_states_is_not_in_the_region(boundaries):
    region = geo.lookup_region(["Rio", "Republic of Alpha"], boundaries)
    assert bounds(region) == [0, 0, 1, 1]
    # A continent named with states of one of its countries stays whole
    continent = geo.lookup_region(["Testland", "Rio"], boundaries)
    assert bounds(continent) == [0, 0, 14, 4]


def test_names_settled_by_the_other_names(boundaries):
    # Next to Quebec, a state of Beta, Punjab and Georgia are Beta's states
    region = geo.lookup_region(["Quebec", "Punjab", "Georgia"], boundaries)
    assert bounds(region) == [11, 1, 14, 3]
    # Georgia alone, or next to a country with no state of that name, is the
    # country
    assert bounds(geo.lookup_region(["Georgia", "Alpha"], boundaries)) == [0, 0, 41, 41]


def test_state_by_another_name_only_if_none_has_it_as_its_own(tmp_path):
    states = STATES + [
        # Rio's woe_name is "Big", and Big is another state's own name
        ("Rio Big", "Rio Big", "", "Big", "", "Alpha", square(2, 2)),
        ("Big", "Big", "", "", "", "Alpha", square(3, 3)),
        ("Small", "Small", "", "", "Sm", "Alpha", square(2, 3)),
    ]
    data = write_boundaries(tmp_path / "ne", states=states)
    assert bounds(geo.lookup_region(["Big"], data)) == [3, 3, 4, 4]
    assert bounds(geo.lookup_region(["Sm"], data)) == [2, 3, 3, 4]


def test_name_with_a_comma_that_names_no_country(boundaries):
    with pytest.raises(geo.RegionNotFound, match="'Rio, Nowhere'"):
        geo.lookup_region(["Rio, Nowhere"], boundaries)


def test_ambiguous_exclusion_with_its_country(boundaries):
    region = geo.lookup_region(["Alpha"], boundaries, exclude_names=["Punjab, Alpha"])
    assert bounds(region) == [0, 0, 1, 1]


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
        # Seconds since the epoch, which advances only when the test says
        now = 1000.0

        def urlopen(self, request, timeout):
            self.requests.append(request)
            if self.failures:
                failure = self.failures.pop(0)
                if isinstance(failure, bytes):
                    return io.BytesIO(failure)
                raise failure
            query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
            answer = feature_collection(self.places.get(query["q"][0]))
            return io.BytesIO(json.dumps(answer).encode())

    fake = Fake()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(geo.time, "sleep", fake.sleeps.append)
    monkeypatch.setattr(geo.time, "time", lambda: fake.now)
    return fake


def http_error(code, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(geo.NOMINATIM_URL, code, "error", headers, None)


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
    # Only the lock is left besides the answers, and no partial file
    assert [path.name for path in cache.glob(".*")] == [".lock"]


def test_nominatim_requests_are_spaced(nominatim, tmp_path):
    geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    # The second request comes 0.3 s after the first, so it waits the rest
    nominatim.now += 0.3
    geo.query_nominatim("Elsewhere", "me@example.org", tmp_path)
    assert nominatim.sleeps == [pytest.approx(geo.NOMINATIM_INTERVAL - 0.3)]
    # The time of the last request is in the lock's file, which every process
    # sharing the cache reads
    assert float((tmp_path / ".lock").read_text()) == nominatim.now


def test_nominatim_spacing_is_shared_by_processes(nominatim, tmp_path):
    # Another process wrote the time of its request a moment ago
    (tmp_path / ".lock").write_text(repr(nominatim.now - 0.5))
    geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert nominatim.sleeps == [pytest.approx(geo.NOMINATIM_INTERVAL - 0.5)]


def test_nominatim_lock_with_an_unreadable_time(nominatim, tmp_path):
    (tmp_path / ".lock").write_text("not a time")
    geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert nominatim.sleeps == []


def test_nominatim_lock_is_exclusive():
    import fcntl
    import tempfile

    # On a local disk, since some shared ones, such as Docker Desktop's host
    # mounts, ignore locks
    directory = Path(tempfile.mkdtemp())
    lock = geo._wait_for_nominatim(directory)
    other = open(directory / ".lock", "a+")
    with pytest.raises(BlockingIOError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    geo._mark_nominatim_request(lock)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    other.close()
    shutil.rmtree(directory)


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


def test_nominatim_waits_as_retry_after_asks(nominatim, tmp_path):
    nominatim.failures = [http_error(429, "30"), http_error(503, "600")]
    assert geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    # 30 s, more than the 2 s backoff, and then at most a minute
    assert [s for s in nominatim.sleeps if s >= 2] == [30.0, geo.MAX_RETRY_AFTER]


def test_nominatim_ignores_an_unreadable_retry_after(nominatim, tmp_path):
    nominatim.failures = [http_error(429, "soon")]
    assert geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert [s for s in nominatim.sleeps if s >= 2] == [2.0]


def test_nominatim_answer_that_is_not_json_is_tried_again(nominatim, tmp_path):
    nominatim.failures = [b"<html>Busy</html>"]
    assert geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert len(nominatim.requests) == 2


def test_nominatim_answers_that_stay_not_json(nominatim, tmp_path):
    nominatim.failures = [b"<html>Busy</html>"] * 3
    with pytest.raises(OSError, match="something other than JSON"):
        geo.query_nominatim("Springfield", "me@example.org", tmp_path)


@pytest.mark.parametrize(
    "error", [http_error(503), urllib.error.URLError("down")], ids=["503", "down"]
)
def test_nominatim_gives_up_after_its_attempts(nominatim, tmp_path, error):
    nominatim.failures = [error] * 3
    with pytest.raises(type(error)):
        geo.query_nominatim("Springfield", "me@example.org", tmp_path)
    assert len(nominatim.requests) == 3
    # Nothing is cached, so the name is asked for again later
    assert not list(tmp_path.glob("*.json"))


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


# Natural Earth's real boundaries


@pytest.fixture(scope="module")
def real_boundaries(tmp_path_factory):
    """Download Natural Earth's real countries and states, about 45 MB."""
    directory = tmp_path_factory.mktemp("natural-earth")
    geo.download_boundaries(directory)
    return directory


def lookup(names, data, **kwargs):
    return geo.lookup_region(names, data, **kwargs)


def test_real_countries(real_boundaries):
    # France without French Guiana
    assert lookup(["France"], real_boundaries).bounds[0] > -10
    # Russia in one piece across the 180th meridian
    assert 180 < lookup(["Russia"], real_boundaries).bounds[2] < 200
    # The detailed layer keeps small islands and small countries: Bornholm,
    # the Dodecanese, the Ryukyus, and Singapore, Bahrain and Hong Kong
    assert lookup(["Denmark"], real_boundaries).bounds[2] > 15
    assert lookup(["Greece"], real_boundaries).bounds[2] > 28
    assert lookup(["Japan"], real_boundaries).bounds[0] < 124
    for name in ("Singapore", "Bahrain", "Hong Kong", "Maldives"):
        assert lookup([name], real_boundaries).area > 0


def test_real_europe_leaves_russia_out_unless_named(real_boundaries):
    europe = lookup(["Europe"], real_boundaries)
    # Iceland and Portugal in the west, and nothing past the Urals
    assert europe.bounds[0] < -20
    assert europe.bounds[2] < 50
    with_russia = lookup(["Europe", "Russia"], real_boundaries)
    assert with_russia.bounds[2] > 180
    # Leaving Russia out, as a prompt may, is no error
    assert lookup(["Europe"], real_boundaries, exclude_names=["Russia"]).equals(europe)


def test_real_state_names(real_boundaries):
    # Washington is the state, though another name of the District of
    # Columbia is "Washington"
    washington = lookup(["Washington"], real_boundaries)
    assert [round(b) for b in washington.bounds] == [-125, 46, -117, 49]
    # Florida is a US state and a department of Uruguay, so it alone is
    # ambiguous, and settled by the US states next to it
    with pytest.raises(geo.RegionNotFound, match="Florida .*Uruguay"):
        lookup(["Florida"], real_boundaries)
    south = lookup(["Florida", "Alabama"], real_boundaries)
    assert -88.5 < south.bounds[0] and south.bounds[2] < -79.5
    # Georgia alone is the country, and next to Alabama the US state
    assert lookup(["Georgia"], real_boundaries).bounds[0] > 39
    assert lookup(["Georgia", "Alabama"], real_boundaries).bounds[0] < -84


def test_real_states_of_countries_with_other_names(real_boundaries):
    # The state layer names Czechia "Czech Republic", and the layers are joined
    # by code, so Prague can be left out of it
    czechia = lookup(["Czechia"], real_boundaries)
    without = lookup(["Czechia"], real_boundaries, exclude_names=["Prague"])
    assert without.area < czechia.area


def test_real_country_named_with_its_states(real_boundaries):
    # As small models answer "a mesh over Washington": the country only says
    # which Washington, and is not part of the region
    both = lookup(["Washington", "United States of America"], real_boundaries)
    assert both.equals(lookup(["Washington"], real_boundaries))
    # A country with states of another country stays
    mixed = lookup(["Ontario", "United States of America"], real_boundaries)
    assert mixed.bounds[0] < -150


def test_real_contiguous_states(real_boundaries):
    conus = lookup(
        ["United States of America"],
        real_boundaries,
        exclude_names=["Alaska", "Hawaii"],
    )
    assert [round(b) for b in conus.bounds] == [-125, 25, -67, 49]
    # A region's own names stand for the same
    for name in ("Contiguous United States", "CONUS", "the Lower 48"):
        assert lookup([name], real_boundaries).equals(conus), name
    # With the places it leaves out left out again, or another place
    again = lookup(["CONUS"], real_boundaries, exclude_names=["Alaska"])
    assert again.equals(conus)
    with_ontario = lookup(["CONUS", "Ontario"], real_boundaries)
    assert with_ontario.bounds[3] > 55
