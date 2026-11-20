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

- a continent, such as "Europe", which stands for its countries
- a country, by any of its Natural Earth names, such as "France" or
  "United States of America"
- a state or province, such as "Colorado". A name used in more than one
  country, such as "Punjab", must name its country, as "Punjab, India".
- with Nominatim, any place with an outline, such as "Denver"

The region is the union of the outlines, without the places the exclusions
name. A country of a continent, or a state or province of a country, can be
left out, such as Russia from "Europe" or Alaska from "United States of
America". A country losing a state is drawn from its other states. An
exclusion that leaves nothing out of the region is an error, so none is lost.
Overseas parts, more than ``PROXIMITY_KM`` from a place's largest part, are
left out.
Longitudes are made continuous around the region, so a region across the
180th meridian has longitudes beyond 180 or -180.

The module imports only the standard library at the top, so the agent can
import it without the ``chiltepin[mpas]`` extra. Looking up needs shapely and
pyshp, which the extra installs.
"""

import hashlib
import http.client
import json
import os
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chiltepin.agents.mpas._plot_mesh import download_map_data, layer_path

# Natural Earth layers names are looked up in, as (resolution, category, name)
COUNTRIES = ("110m", "cultural", "admin_0_countries")
STATES = ("10m", "cultural", "admin_1_states_provinces")
BOUNDARY_LAYERS = (COUNTRIES, STATES)

# Fields of each layer that hold names a place can be looked up by
COUNTRY_FIELDS = ("NAME", "NAME_LONG", "ADMIN", "NAME_EN")
STATE_FIELDS = ("name", "name_en", "gn_name", "woe_name")
# Field of alternative state names, separated by "|"
STATE_ALT_FIELD = "name_alt"
# Fields of a country's continent, and of a state's country
CONTINENT_FIELD = "CONTINENT"
STATE_COUNTRY_FIELD = "admin"

# Parts of a place farther than this from its largest part are left out
PROXIMITY_KM = 1000.0
KM_PER_DEGREE = 111.32

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PROJECT_URL = "https://github.com/NOAA-GSL/chiltepin"
# Nominatim's usage policy allows at most one request a second
NOMINATIM_INTERVAL = 1.1
# Outlines from Nominatim are simplified to about this many degrees
NOMINATIM_SIMPLIFY_DEGREES = 0.01

_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


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
        self.country_names: Dict[int, set] = {}
        for i, record in enumerate(self.countries.iterRecords()):
            names = {normalize_name(record[f]) for f in COUNTRY_FIELDS if record[f]}
            self.country_names[i] = names
            for name in names:
                self.country_index.setdefault(name, []).append(i)
            continent = normalize_name(record[CONTINENT_FIELD])
            self.continent_index.setdefault(continent, []).append(i)
        # name -> {country: [state records]}, and country -> [state records]
        self.state_index: Dict[str, Dict[str, List[int]]] = {}
        self.states_of: Dict[str, List[int]] = {}
        for i, record in enumerate(self.states.iterRecords()):
            names = {normalize_name(record[f]) for f in STATE_FIELDS if record[f]}
            names |= {
                normalize_name(alt)
                for alt in (record[STATE_ALT_FIELD] or "").split("|")
                if alt.strip()
            }
            country = normalize_name(record[STATE_COUNTRY_FIELD])
            self.states_of.setdefault(country, []).append(i)
            for name in names:
                self.state_index.setdefault(name, {}).setdefault(country, []).append(i)

    def country(self, i: int):
        from shapely.geometry import shape

        return shape(self.countries.shape(i).__geo_interface__)

    def state(self, i: int):
        from shapely.geometry import shape

        return shape(self.states.shape(i).__geo_interface__)

    def admin(self, i: int) -> str:
        """Return the name states use for a country's record."""
        return normalize_name(self.countries.record(i)["ADMIN"])

    def country_of(self, name: str) -> Optional[str]:
        """Return the name states use for the country a name stands for."""
        records = self.country_index.get(name)
        return self.admin(records[0]) if records else None

    def resolve(self, name: str) -> Tuple[str, Any]:
        """Return what a name stands for, as a kind and its records.

        The kind is "countries" or "states", with their records, "ambiguous",
        with the countries of the states it names, or "missing". A continent
        stands for its countries.
        """
        key = normalize_name(name)
        if key in self.continent_index:
            return "countries", list(self.continent_index[key])
        if key in self.country_index:
            return "countries", list(self.country_index[key])
        if key in self.state_index:
            by_country = self.state_index[key]
            if len(by_country) > 1:
                return "ambiguous", sorted(
                    self.states.record(records[0])[STATE_COUNTRY_FIELD]
                    for records in by_country.values()
                )
            (records,) = by_country.values()
            return "states", list(records)
        if "," in key:
            place, _, country = (part.strip() for part in key.rpartition(","))
            country = self.country_of(country) or country
            records = self.state_index.get(place, {}).get(country)
            if records:
                return "states", list(records)
        return "missing", None


def query_nominatim(
    name: str,
    contact: str,
    cache_dir: Path,
    attempts: int = 3,
    timeout: float = 30,
) -> Optional[Dict[str, Any]]:
    """Return Nominatim's outline of a place as GeoJSON, or None if it has none.

    Answers, found or not, are kept in ``cache_dir`` and never asked for again.
    Requests are at least ``NOMINATIM_INTERVAL`` seconds apart in a process.
    Requests answered with 429 or a 5xx status, and requests that fail on the
    way, are tried again.

    Raises
    ------
    OSError
        If Nominatim cannot be reached after ``attempts`` tries, or answers
        with another error.
    """
    global _last_nominatim_request

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
        with _nominatim_lock:
            wait = _last_nominatim_request + NOMINATIM_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    answer = json.loads(response.read())
                break
            except urllib.error.HTTPError as e:
                if not (e.code == 429 or e.code >= 500) or attempt == attempts:
                    raise
            except (OSError, http.client.HTTPException):
                if attempt == attempts:
                    raise
            finally:
                _last_nominatim_request = time.monotonic()
        time.sleep(2**attempt)
    features = answer.get("features") or []
    geometry = features[0].get("geometry") if features else None
    cache_dir.mkdir(parents=True, exist_ok=True)
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
    """Return the polygons within ``proximity_km`` of the largest one."""
    largest = max(polygons, key=lambda polygon: polygon.area)
    limit = proximity_km / KM_PER_DEGREE
    return [p for p in polygons if p is largest or largest.distance(p) < limit]


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
        Parts farther than this from a place's largest part are left out.

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
    boundaries = _Boundaries(Path(data_dir))

    # The records each exclusion names, and the exclusions naming each record
    left_out: Dict[Tuple[str, int], set] = {}
    excluded_states: Dict[str, set] = {}
    unknown_exclusions: List[str] = []
    ambiguous: List[str] = []
    unresolved: set = set()
    for name in exclude_names:
        kind, records = boundaries.resolve(name)
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
                    country = normalize_name(
                        boundaries.states.record(i)[STATE_COUNTRY_FIELD]
                    )
                    excluded_states.setdefault(country, set()).add(i)
    excluded_keys = {normalize_name(name): name for name in exclude_names}
    used: set = set()

    def outlines_of(kind: str, records: List[int]) -> List[Any]:
        """Return the outlines of records, without the excluded places."""
        outlines = []
        for i in records:
            if (kind, i) in left_out:
                used.update(left_out[kind, i])
            elif kind == "states":
                outlines.append(boundaries.state(i))
            elif boundaries.admin(i) in excluded_states:
                # A country losing a state is drawn from its other states
                cut = excluded_states[boundaries.admin(i)]
                used.update(*(left_out["states", j] for j in cut))
                states = boundaries.states_of[boundaries.admin(i)]
                kept = [j for j in states if j not in cut]
                if kept:
                    outlines.append(unary_union([boundaries.state(j) for j in kept]))
            else:
                outlines.append(boundaries.country(i))
        return outlines

    places: List[List[Any]] = []
    missing: List[str] = []
    for name in names:
        if normalize_name(name) in excluded_keys:
            used.add(excluded_keys[normalize_name(name)])
            continue
        kind, records = boundaries.resolve(name)
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
        found = [_polygons(outline) for outline in outlines_of(kind, records)]
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
            "These names are used in several countries, so name one, as "
            "'Name, Country': " + "; ".join(ambiguous)
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
