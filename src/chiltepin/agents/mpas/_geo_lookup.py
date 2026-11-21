# SPDX-License-Identifier: Apache-2.0
"""Look up the outline of a region from the names of the places it covers.

The MPAS mesh agent turns a prompt's place names into a region, and the region
into a mesh. Names are looked up in Natural Earth's countries and its states
and provinces, which are downloaded once to a data directory. A name Natural
Earth lacks, such as a city, is looked up with OpenStreetMap's Nominatim
service, but only if a contact for Nominatim is given, since Nominatim limits
and blocks heavy use. Its answers are kept in a cache directory, so a name is
never asked for twice.

A name matches exactly, ignoring case, accents and extra spaces. It can be:

- a continent, such as "Africa", which stands for its countries. "Europe"
  stands for its countries without Russia, which reaches the Pacific, unless
  Russia is named too.
- a country, by any of its Natural Earth names, such as "France" or
  "United States of America"
- a state or province, such as "Colorado", by its own name, or else by one
  of its other names, such as "WA". A name of states in several countries,
  such as "Punjab", or of a country and a state, such as "Georgia", stands
  for the one in the country the other names point to, as "Georgia" does for
  the US state next to "Alabama", or else for the country. Naming the country
  settles it, as "Punjab, India" does. A country named with some of its own
  states, as models sometimes name "United States of America" with "Texas",
  only settles which states they are, and is not part of the region.
- a region named in ``REGIONS``, such as "Contiguous United States", which
  stands for its places, without those it leaves out
- with Nominatim, any place with an outline, such as "Denver"

The region is the union of the outlines, without the places the exclusions
name. A country of a continent, or a state or province of a country, can be
left out, such as Spain from "Europe" or Alaska from "United States of
America". A country losing a state is drawn from its other states. An
exclusion that leaves nothing out of the region is an error, so none is lost.
Parts of a place are kept if they are within ``PROXIMITY_KM`` of its largest
part, or of another part kept, so a chain of islands stays, and parts far from
all the rest, as Hawaii is, are left out.
Longitudes are made continuous around the region, so a region across the
180th meridian has longitudes beyond 180 or -180.

The module imports only the standard library at the top, so the agent can
import it without the ``chiltepin[mpas]`` extra. Looking up needs shapely and
pyshp, which the extra installs.
"""

import fcntl
import hashlib
import http.client
import json
import os
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chiltepin.agents.mpas._plot_mesh import download_map_data, layer_path

# Natural Earth layers names are looked up in, as (resolution, category, name).
# The detailed countries keep small islands and small countries.
COUNTRIES = ("10m", "cultural", "admin_0_countries")
STATES = ("10m", "cultural", "admin_1_states_provinces")
BOUNDARY_LAYERS = (COUNTRIES, STATES)

# Fields of each layer that hold names a place can be looked up by. States
# are found by the tiers of their names in turn: their name, then their
# English name, which for the District of Columbia is "Washington", and then
# their other names, only if no state has the name in an earlier tier.
COUNTRY_FIELDS = ("NAME", "NAME_LONG", "ADMIN", "NAME_EN")
STATE_NAME_TIERS = (("name",), ("name_en",), ("gn_name", "woe_name"))
# Field of alternative state names, separated by "|", in the last tier
STATE_ALT_FIELD = "name_alt"
# Fields of a country's continent, of the code joining countries to their
# states in both layers, and of a state's country's name
CONTINENT_FIELD = "CONTINENT"
COUNTRY_CODE_FIELD = "ADM0_A3"
STATE_CODE_FIELD = "adm0_a3"
STATE_COUNTRY_FIELD = "admin"

# Countries a continent stands for only when they are named too, with the
# continent's name and their codes. Russia reaches the Pacific.
CONTINENTS_WITHOUT = {"europe": {"RUS"}}

# Regions known by a name of their own, as the places they cover and those
# they leave out
_CONTIGUOUS_STATES = (("United States of America",), ("Alaska", "Hawaii"))
REGIONS = {
    "contiguous united states": _CONTIGUOUS_STATES,
    "contiguous united states of america": _CONTIGUOUS_STATES,
    "contiguous us": _CONTIGUOUS_STATES,
    "conterminous united states": _CONTIGUOUS_STATES,
    "conus": _CONTIGUOUS_STATES,
    "lower 48": _CONTIGUOUS_STATES,
    "lower 48 states": _CONTIGUOUS_STATES,
}

# Parts of a place farther than this from all its other parts kept, starting
# from its largest part, are left out
PROXIMITY_KM = 1000.0
KM_PER_DEGREE = 111.32

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PROJECT_URL = "https://github.com/NOAA-GSL/chiltepin"
# Nominatim's usage policy allows at most one request a second, from all the
# processes sharing a cache directory
NOMINATIM_INTERVAL = 1.1
# Longest wait a Retry-After answer is honored for, in seconds
MAX_RETRY_AFTER = 60.0
# Outlines from Nominatim are simplified to about this many degrees
NOMINATIM_SIMPLIFY_DEGREES = 0.01


class RegionNotFound(ValueError):
    """Some names match no place, or more than one."""


def download_boundaries(data_dir: Path) -> None:
    """Download the Natural Earth layers names are looked up in to ``data_dir``.

    Layers already there are kept.
    """
    download_map_data(Path(data_dir), layers=BOUNDARY_LAYERS)


def normalize_name(name: str) -> str:
    """Return a name as it is matched: lower case, without accents or extra spaces."""
    decomposed = unicodedata.normalize("NFKD", name)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.casefold().split())


def user_agent() -> str:
    """Return the User-Agent naming chiltepin, as Nominatim's policy asks."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("chiltepin")
    except PackageNotFoundError:  # pragma: no cover - chiltepin is installed
        release = "unknown"
    return f"chiltepin/{release} (+{PROJECT_URL})"


class _Boundaries:
    """The Natural Earth places of a data directory, indexed by name."""

    def __init__(self, data_dir: Path):
        import shapefile

        self.countries = shapefile.Reader(str(layer_path(data_dir, COUNTRIES)))
        self.states = shapefile.Reader(str(layer_path(data_dir, STATES)))
        self.country_index: Dict[str, List[int]] = {}
        self.continent_index: Dict[str, List[int]] = {}
        self.code: Dict[int, str] = {}
        for i, record in enumerate(self.countries.iterRecords()):
            self.code[i] = record[COUNTRY_CODE_FIELD]
            for name in {
                normalize_name(record[f]) for f in COUNTRY_FIELDS if record[f]
            }:
                self.country_index.setdefault(name, []).append(i)
            continent = normalize_name(record[CONTINENT_FIELD])
            self.continent_index.setdefault(continent, []).append(i)
        # Each name's states by country code, in each tier of names, and each
        # country's states
        self.state_tiers: List[Dict[str, Dict[str, List[int]]]] = [
            {} for _ in STATE_NAME_TIERS
        ]
        self.states_of: Dict[str, List[int]] = {}
        self.state_code: Dict[int, str] = {}
        self.country_names: Dict[str, str] = {}
        for i, record in enumerate(self.states.iterRecords()):
            code = record[STATE_CODE_FIELD]
            self.state_code[i] = code
            self.country_names.setdefault(code, record[STATE_COUNTRY_FIELD])
            self.states_of.setdefault(code, []).append(i)
            for tier, fields in zip(self.state_tiers, STATE_NAME_TIERS):
                names = {normalize_name(record[f]) for f in fields if record[f]}
                if tier is self.state_tiers[-1]:
                    names |= {
                        normalize_name(alt)
                        for alt in (record[STATE_ALT_FIELD] or "").split("|")
                        if alt.strip()
                    }
                for name in names:
                    tier.setdefault(name, {}).setdefault(code, []).append(i)

    def country(self, i: int):
        from shapely.geometry import shape

        return shape(self.countries.shape(i).__geo_interface__)

    def state(self, i: int):
        from shapely.geometry import shape

        return shape(self.states.shape(i).__geo_interface__)

    def states_named(self, key: str) -> Dict[str, List[int]]:
        """Return the states a name stands for, by country code.

        The first tier of names that has it settles which states it names.
        """
        for tier in self.state_tiers:
            if key in tier:
                return tier[key]
        return {}

    def candidates(self, name: str) -> Tuple[List[int], Dict[str, List[int]]]:
        """Return the countries and the states, by code, a name could stand for.

        A continent stands for its countries. "Name, Country" stands for the
        states of that name in that country.
        """
        key = normalize_name(name)
        if key in self.continent_index:
            return list(self.continent_index[key]), {}
        if "," in key and key not in self.country_index:
            place, _, country = (part.strip() for part in key.rpartition(","))
            codes = {self.code[i] for i in self.country_index.get(country, [])}
            states = self.states_named(place)
            return [], {code: states[code] for code in codes if code in states}
        return list(self.country_index.get(key, [])), self.states_named(key)

    def resolve(self, name: str, context: set) -> Tuple[str, Any]:
        """Return what a name stands for, as a kind and its records.

        ``context`` holds the codes of the countries the request's other names
        point to. The kind is "countries" or "states", with their records,
        "ambiguous", with the names of the countries it could be in, or
        "missing".
        """
        countries, states = self.candidates(name)
        near = {code: records for code, records in states.items() if code in context}
        if countries and not (states and len(near) == 1):
            return "countries", countries
        if len(states) == 1:
            ((_, records),) = states.items()
            return "states", list(records)
        if len(near) == 1:
            ((_, records),) = near.items()
            return "states", list(records)
        if states:
            return "ambiguous", sorted(self.country_names[code] for code in states)
        return "missing", None

    def codes_of(self, name: str) -> set:
        """Return the country codes a name surely points to, for context.

        A name that could be several places points to none.
        """
        countries, states = self.candidates(name)
        if countries and not states:
            return {self.code[i] for i in countries}
        if len(states) == 1 and not countries:
            return set(states)
        return set()


def _wait_for_nominatim(cache_dir: Path) -> "Any":
    """Take the lock shared by all processes using a cache directory, and wait.

    The lock is held until the returned file is closed. Requests are at least
    ``NOMINATIM_INTERVAL`` seconds apart, by the time of the last one, which
    the lock's file holds.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = open(cache_dir / ".lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    lock.seek(0)
    try:
        last = float(lock.read().strip() or 0)
    except ValueError:
        last = 0.0
    wait = last + NOMINATIM_INTERVAL - time.time()
    if wait > 0:
        time.sleep(min(wait, NOMINATIM_INTERVAL))
    return lock


def _mark_nominatim_request(lock) -> None:
    """Record the time of a request in the lock's file, and release the lock."""
    lock.seek(0)
    lock.truncate()
    lock.write(repr(time.time()))
    lock.flush()
    lock.close()


def _retry_after(error: urllib.error.HTTPError) -> float:
    """Return how long an answer asks to wait, in seconds, at most a minute."""
    try:
        seconds = float(error.headers.get("Retry-After", 0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(seconds, MAX_RETRY_AFTER))


def query_nominatim(
    name: str,
    contact: str,
    cache_dir: Path,
    attempts: int = 3,
    timeout: float = 30,
) -> Optional[Dict[str, Any]]:
    """Return Nominatim's outline of a place as GeoJSON, or None if it has none.

    Answers, found or not, are kept in ``cache_dir`` and never asked for again.
    Requests from all processes sharing ``cache_dir`` are at least
    ``NOMINATIM_INTERVAL`` seconds apart. Requests answered with 429 or a 5xx
    status, or with something other than JSON, and requests that fail on the
    way, are tried again, after the wait a ``Retry-After`` asks for.

    Raises
    ------
    OSError
        If Nominatim cannot be reached after ``attempts`` tries, or answers
        with another error, or with something other than JSON.
    """
    cache_dir = Path(cache_dir)
    key = normalize_name(name)
    cached = cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
    if cached.is_file():
        return json.loads(cached.read_text())["geometry"]

    params = urllib.parse.urlencode(
        {
            "q": name,
            "format": "geojson",
            "polygon_geojson": 1,
            "polygon_threshold": NOMINATIM_SIMPLIFY_DEGREES,
            "limit": 1,
            "email": contact,
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}", headers={"User-Agent": user_agent()}
    )
    for attempt in range(1, attempts + 1):
        backoff = float(2**attempt)
        lock = _wait_for_nominatim(cache_dir)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            answer = json.loads(body)
            break
        except urllib.error.HTTPError as e:
            if not (e.code == 429 or e.code >= 500) or attempt == attempts:
                raise
            backoff = max(backoff, _retry_after(e))
        except json.JSONDecodeError as e:
            if attempt == attempts:
                raise OSError(f"Nominatim answered with something other than JSON: {e}")
        except (OSError, http.client.HTTPException):
            if attempt == attempts:
                raise
        finally:
            _mark_nominatim_request(lock)
        time.sleep(backoff)
    features = answer.get("features") or []
    geometry = features[0].get("geometry") if features else None
    partial = cached.with_name(f".{cached.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps({"name": name, "geometry": geometry}))
    partial.replace(cached)
    return geometry


def _polygons(geometry) -> List[Any]:
    """Return the polygons of a geometry, which may have none."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [part for g in geometry.geoms for part in _polygons(g)]
    return []


def _shift(polygon, degrees: float):
    from shapely.affinity import translate

    return translate(polygon, xoff=degrees) if degrees else polygon


def _unwrap(polygons: Sequence[Any], reference: float) -> List[Any]:
    """Shift each polygon by 360 degrees as needed to be near ``reference``."""
    shifted = []
    for polygon in polygons:
        lon = polygon.centroid.x
        shift = 360.0 * round((reference - lon) / 360.0)
        shifted.append(_shift(polygon, shift))
    return shifted


def _near_largest(polygons: Sequence[Any], proximity_km: float) -> List[Any]:
    """Return the polygons reached from the largest in steps of ``proximity_km``.

    A chain of islands stays with its mainland, as the Ryukyus do with Japan,
    while parts far from every other, as Hawaii is, are left out.
    """
    from shapely import STRtree

    polygons = list(polygons)
    tree = STRtree(polygons)
    limit = proximity_km / KM_PER_DEGREE
    largest = max(range(len(polygons)), key=lambda i: polygons[i].area)
    kept, frontier = {largest}, [largest]
    while frontier:
        near = tree.query(
            [polygons[i] for i in frontier], predicate="dwithin", distance=limit
        )[1]
        frontier = [int(i) for i in set(near.tolist()) - kept]
        kept.update(frontier)
    return [polygons[i] for i in sorted(kept)]


def lookup_region(
    names: Sequence[str],
    data_dir: Path,
    exclude_names: Sequence[str] = (),
    nominatim_contact: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    proximity_km: float = PROXIMITY_KM,
):
    """Return the outline of the region the names cover, as a shapely geometry.

    Parameters
    ----------
    names : sequence of str
        Names of the places the region covers.
    data_dir : Path
        Directory the Natural Earth layers were downloaded to.
    exclude_names : sequence of str, optional
        Names of places left out: continents, countries, states or provinces,
        such as a country of a continent or a state of a country.
    nominatim_contact : str, optional
        Email address sent to Nominatim with each request. Without it, names
        Natural Earth lacks are not looked up.
    cache_dir : Path, optional
        Directory of Nominatim's answers, needed with ``nominatim_contact``.
    proximity_km : float, optional
        Parts farther than this from every other part kept are left out.

    Raises
    ------
    RegionNotFound
        If any name matches no place or more than one, an exclusion leaves
        nothing out of the region, or everything is left out. The message
        names them.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if nominatim_contact and cache_dir is None:
        raise ValueError("cache_dir is needed with nominatim_contact")
    # A region's name stands for its places, without those it leaves out
    named_names, exclude_names = list(names), list(exclude_names)
    names = []
    for name in named_names:
        key = normalize_name(name).removeprefix("the ")
        if key in REGIONS:
            places, left_out_places = REGIONS[key]
            names += [place for place in places if place not in names]
            exclude_names += [
                place for place in left_out_places if place not in exclude_names
            ]
        else:
            names.append(name)
    boundaries = _Boundaries(Path(data_dir))

    # The countries the names surely point to settle names that could be
    # several places
    context = set().union(*(boundaries.codes_of(name) for name in names))
    named = {
        boundaries.code[i]
        for name in names
        if normalize_name(name) not in boundaries.continent_index
        for i in boundaries.candidates(name)[0]
    }

    # The records each exclusion names, and the exclusions naming each record
    left_out: Dict[Tuple[str, int], set] = {}
    excluded_states: Dict[str, set] = {}
    unknown_exclusions: List[str] = []
    ambiguous: List[str] = []
    unresolved: set = set()
    for name in exclude_names:
        kind, records = boundaries.resolve(name, context)
        if kind == "ambiguous":
            ambiguous.append(f"{name} ({', '.join(records)})")
            unresolved.add(name)
        elif kind == "missing":
            unknown_exclusions.append(name)
            unresolved.add(name)
        else:
            for i in records:
                left_out.setdefault((kind, i), set()).add(name)
                if kind == "states":
                    code = boundaries.state_code[i]
                    excluded_states.setdefault(code, set()).add(i)
    excluded_keys = {normalize_name(name): name for name in exclude_names}
    used: set = set()

    def outlines_of(kind: str, records: List[int], continent: str) -> List[Any]:
        """Return the outlines of records, without the excluded places."""
        outlines = []
        for i in records:
            if (kind, i) in left_out:
                used.update(left_out[kind, i])
            elif kind == "states":
                outlines.append(boundaries.state(i))
            elif boundaries.code[i] in CONTINENTS_WITHOUT.get(continent, set()) and (
                boundaries.code[i] not in named
            ):
                # Left out of the continent unless named too
                continue
            elif boundaries.code[i] in excluded_states:
                # A country losing a state is drawn from its other states
                cut = excluded_states[boundaries.code[i]]
                used.update(*(left_out["states", j] for j in cut))
                states = boundaries.states_of[boundaries.code[i]]
                kept = [j for j in states if j not in cut]
                if kept:
                    outlines.append(unary_union([boundaries.state(j) for j in kept]))
            else:
                outlines.append(boundaries.country(i))
        return outlines

    # A country a continent leaves out is left out, as an exclusion asks
    for continent, codes in CONTINENTS_WITHOUT.items():
        if continent in {normalize_name(name) for name in names}:
            for (kind, i), exclusions in left_out.items():
                if kind == "countries" and boundaries.code[i] in codes - named:
                    used.update(exclusions)

    # A country named with some of its own states only says which states
    resolved = {name: boundaries.resolve(name, context) for name in names}
    with_states = {
        boundaries.state_code[i]
        for kind, records in resolved.values()
        if kind == "states"
        for i in records
    }

    places: List[List[Any]] = []
    missing: List[str] = []
    for name in names:
        if normalize_name(name) in excluded_keys:
            used.add(excluded_keys[normalize_name(name)])
            continue
        kind, records = resolved[name]
        if (
            kind == "countries"
            and normalize_name(name) not in boundaries.continent_index
            and {boundaries.code[i] for i in records} <= with_states
        ):
            continue
        if kind == "ambiguous":
            ambiguous.append(f"{name} ({', '.join(records)})")
            continue
        if kind == "missing":
            geometry = None
            if nominatim_contact:
                geometry = query_nominatim(name, nominatim_contact, Path(cache_dir))
            found = [_polygons(shape(geometry))] if geometry else []
            if not any(found):
                missing.append(name)
            places.extend(polygons for polygons in found if polygons)
            continue
        outlines = outlines_of(kind, records, normalize_name(name))
        found = [_polygons(outline) for outline in outlines]
        places.extend(polygons for polygons in found if polygons)

    problems = []
    if missing:
        problems.append(
            "No place with an outline is named " + ", ".join(map(repr, missing))
        )
        if not nominatim_contact:
            problems[-1] += (
                ". Only countries, states and provinces are looked up without a "
                "Nominatim contact"
            )
    if ambiguous:
        problems.append(
            "These names are used in several countries, so name the one meant, "
            "as 'Name, Country': " + "; ".join(ambiguous)
        )
    # An exclusion that is also a name asked for leaves that name out
    unknown_exclusions = [name for name in unknown_exclusions if name not in used]
    if unknown_exclusions:
        problems.append(
            "No continent, country, state or province to leave out is named "
            + ", ".join(map(repr, unknown_exclusions))
        )
    unused = [name for name in exclude_names if name not in used | unresolved]
    if unused:
        problems.append(
            "These places to leave out are not in the region: "
            + ", ".join(map(repr, unused))
        )
    if problems:
        raise RegionNotFound(". ".join(problems))
    if not places:
        raise RegionNotFound("Every place is left out, so there is no region")

    # Each place keeps its parts near its largest part, as one piece around
    # the 180th meridian, and the places are put near the largest place
    kept = []
    for polygons in places:
        largest = max(polygons, key=lambda polygon: polygon.area)
        unwrapped = _unwrap(polygons, largest.centroid.x)
        kept.append(_near_largest(unwrapped, proximity_km))
    largest = max((p for polygons in kept for p in polygons), key=lambda p: p.area)
    return unary_union(
        [p for polygons in kept for p in _unwrap(polygons, largest.centroid.x)]
    )
