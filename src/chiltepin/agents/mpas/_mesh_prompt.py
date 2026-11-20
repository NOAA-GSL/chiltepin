# SPDX-License-Identifier: Apache-2.0
"""Ask an LLM what mesh a prompt describes.

The model answers in the fields of ``MeshPromptResponse``: a resolution, a
name, the places the mesh covers, and optionally a shape, a method, and the
latitudes of a channel. instructor makes it answer with a tool call, and its
validators normalize what a model writes in different ways, such as "15 km" or
"limited area". A value they cannot normalize is sent back to the model with
the reason, for it to try again.

The model never writes a mesh request. The agent builds one from these fields,
so a model's answer can only name places and choices, which are checked.

Credentials come from the environment of the process that calls the model, as
litellm reads them, for example ``AWS_BEARER_TOKEN_BEDROCK`` for Bedrock. They
are never arguments, and are removed from error messages.

This module needs the ``chiltepin[llm]`` extra.
"""

import math
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

METHODS = {
    "project_hexes": (
        "project_hexes",
        "project hexes",
        "hex_projection",
        "hex projection",
        "hexagonal projection",
        "hexagon projection",
    ),
    "create_region": (
        "create_region",
        "create region",
        "limited_area",
        "limited area",
        "mpas_limited_area",
        "mpas limited area",
        "mpas-limited-area",
        "lam",
        "limited area model",
        "regional extraction",
        "cutout",
        "cut out",
    ),
}

SHAPES = {
    "rectangle": ("rectangle", "rectangular", "square", "box", "bounding box", "rect"),
    "circle": ("circle", "circular", "round", "disk", "disc"),
    "ellipse": ("ellipse", "elliptical", "elliptic", "oval"),
    "polygon": ("polygon", "convex hull", "hull", "outline"),
    "channel": ("channel", "band", "latitude band", "zonal band", "belt", "strip"),
}

# Longest mesh name the agent accepts, in bytes
MAX_NAME_BYTES = 200

# Most places one prompt can name
MAX_NAMES = 300

# Times a request is tried again when the provider's content filter wrongly
# refuses it, which Bedrock sometimes does
FILTER_ATTEMPTS = 4

# Times instructor sends an invalid answer back to the model
VALIDATION_RETRIES = 3

_RESOLUTION_RE = re.compile(
    r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:km|kms|kilometers?|kilometres?)?\s*",
    re.IGNORECASE,
)


def _lookup(value: str, table, what: str) -> str:
    key = " ".join(value.casefold().replace("_", " ").replace("-", " ").split())
    for canonical, names in table.items():
        if key in {
            " ".join(n.replace("_", " ").replace("-", " ").split()) for n in names
        }:
            return canonical
    raise ValueError(
        f"Unknown {what} {value!r}. Use one of: {', '.join(table)}, or null"
    )


def normalize_method(value: Optional[str]) -> Optional[str]:
    """Return the method a name stands for, such as "limited area"."""
    if value is None or not str(value).strip():
        return None
    return _lookup(str(value), METHODS, "method")


def normalize_shape(value: Optional[str]) -> Optional[str]:
    """Return the shape a name stands for, such as "oval"."""
    if value is None or not str(value).strip():
        return None
    return _lookup(str(value), SHAPES, "shape")


def normalize_resolution(value: Any) -> str:
    """Return a resolution as the agent writes it, such as "15km" for "15 km"."""
    if isinstance(value, bool):
        raise ValueError(f"Resolution must be a number of km, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _RESOLUTION_RE.fullmatch(str(value))
        if not match:
            raise ValueError(
                f"Resolution must be a number of km, such as '15km', got {value!r}"
            )
        number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Resolution must be greater than 0 km, got {value!r}")
    return f"{number:g}km" if number < 1e6 else f"{number:.0f}km"


def normalize_name(value: Any) -> str:
    """Return a mesh name made safe for a file name, such as "japan_15km".

    Letters, digits, ".", "_" and "-" are kept, accents are dropped, and other
    characters become "_".
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not text:
        raise ValueError(f"Name {value!r} has no letters or digits")
    if len(text.encode()) > MAX_NAME_BYTES:
        raise ValueError(f"Name must be at most {MAX_NAME_BYTES} characters")
    return text


def _names(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    seen, names = set(), []
    for value in values:
        name = " ".join(str(value).split())
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    if len(names) > MAX_NAMES:
        raise ValueError(f"At most {MAX_NAMES} places can be named")
    return names


class MeshPromptResponse(BaseModel):
    """The mesh a prompt describes."""

    resolution: str = Field(description="Mesh resolution in km, such as '15km'")
    name: str = Field(description="Short mesh name, such as 'japan_15km'")
    method: Optional[str] = Field(
        default=None,
        description=(
            "'project_hexes' or 'create_region' if the user names a method, or null"
        ),
    )
    shape: Optional[str] = Field(
        default=None,
        description=(
            "'rectangle', 'circle', 'ellipse', 'polygon' or 'channel' if the user "
            "names a shape, or null"
        ),
    )
    region_names: List[str] = Field(
        default_factory=list,
        description="Names of the places the mesh covers. Empty for a global mesh.",
    )
    exclude_names: List[str] = Field(
        default_factory=list, description="Names of places to leave out"
    )
    upper_lat: Optional[float] = Field(
        default=None, description="Northern latitude of a channel, in degrees"
    )
    lower_lat: Optional[float] = Field(
        default=None, description="Southern latitude of a channel, in degrees"
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def check_resolution(cls, value: Any) -> str:
        return normalize_resolution(value)

    @field_validator("name", mode="before")
    @classmethod
    def check_name(cls, value: Any) -> str:
        return normalize_name(value)

    @field_validator("method", mode="before")
    @classmethod
    def check_method(cls, value: Any) -> Optional[str]:
        return normalize_method(value)

    @field_validator("shape", mode="before")
    @classmethod
    def check_shape(cls, value: Any) -> Optional[str]:
        return normalize_shape(value)

    @field_validator("region_names", "exclude_names", mode="before")
    @classmethod
    def check_names(cls, value: Any) -> List[str]:
        return _names(value)

    @field_validator("upper_lat", "lower_lat")
    @classmethod
    def check_latitude(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not -90 <= value <= 90:
            raise ValueError(f"A latitude must be from -90 to 90, got {value}")
        return value

    @model_validator(mode="after")
    def check_channel(self) -> "MeshPromptResponse":
        if self.shape != "channel":
            # Latitudes only describe a channel
            self.upper_lat = self.lower_lat = None
            return self
        if self.upper_lat is None or self.lower_lat is None:
            raise ValueError("A channel needs both upper_lat and lower_lat")
        if self.upper_lat == self.lower_lat:
            raise ValueError("A channel's upper_lat and lower_lat must differ")
        if self.upper_lat < self.lower_lat:
            self.upper_lat, self.lower_lat = self.lower_lat, self.upper_lat
        # A channel goes around the globe, so it covers no named places
        self.region_names, self.exclude_names = [], []
        return self

    @property
    def is_global(self) -> bool:
        """Whether the prompt asks for a global mesh."""
        return not self.region_names and self.shape != "channel"


def system_prompt(resolutions: Sequence[str]) -> str:
    """Return the instructions the model gets, with the global resolutions."""
    return (
        "You turn a request for an MPAS weather model mesh into the fields of the "
        "MeshPromptResponse tool. Answer only by calling that tool.\n"
        f"Global meshes are available at these resolutions: {', '.join(resolutions)}. "
        "A regional mesh can have any resolution the user asks for.\n"
        "- resolution: the resolution the user asks for, such as '15km'. If they "
        "give none, use '60km'.\n"
        "- name: a short name for the mesh, made of letters, digits and "
        "underscores, such as 'japan_15km'. Use the name the user gives, if any.\n"
        "- region_names: the places the mesh covers, as English names of "
        "continents, countries, or states and provinces, as Natural Earth names "
        "them, such as 'Europe', 'Japan', 'United States of America', 'Texas' or "
        "'Quebec'. A state or province whose name is used in several countries "
        "gets its country after a comma, such as 'Punjab, India'. For a region "
        "known by another name, such as 'CONUS', 'Tornado Alley' or 'the "
        "Rockies', list the countries or states it covers. Name a city or county "
        "only if the user asks for one. For a global mesh, leave it empty.\n"
        "- exclude_names: places to leave out of a larger region, such as Russia "
        "for 'Europe without Russia', or Alaska and Hawaii for the contiguous "
        "United States. Only continents, countries, states and provinces can be "
        "left out. Otherwise empty.\n"
        "- shape: 'rectangle', 'circle', 'ellipse', 'polygon' or 'channel', only "
        "if the user names a shape. Otherwise null. A channel is a band of "
        "latitude around the whole globe, such as the tropics.\n"
        "- method: only if the user names how to make the mesh, otherwise null. "
        "'project_hexes' (also called hex projection) makes a new rectangular "
        "mesh. 'create_region' (also called limited area, MPAS limited area, LAM, "
        "or a cutout) cuts a region from a global mesh. These words name methods, "
        "not places.\n"
        "- upper_lat and lower_lat: for a channel only, its northern and "
        "southern latitudes in degrees. The tropics are 23.43676 to -23.43676.\n"
    )


def _secrets() -> List[str]:
    """Return the values of environment variables that look like credentials."""
    pattern = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
    return [
        value
        for name, value in os.environ.items()
        if pattern.search(name) and len(value) >= 8
    ]


def redact(text: str) -> str:
    """Return text with the values of credential variables replaced."""
    for secret in sorted(_secrets(), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    return text


def ask(
    prompt: str,
    model: str,
    resolutions: Sequence[str],
    timeout: float = 120,
) -> MeshPromptResponse:
    """Return the mesh the model reads in a prompt.

    ``model`` is a litellm model string, such as
    ``bedrock/us.anthropic.claude-sonnet-5-5`` or ``ollama_chat/qwen2.5:3b``.
    Models that cannot be made to call a tool, as some Bedrock models cannot,
    are asked to, which ``drop_params`` lets litellm do.

    Raises
    ------
    RuntimeError
        If the model cannot be reached or gives no valid answer. The message
        holds no credentials.
    """
    import instructor
    import litellm

    # litellm otherwise prints help to standard output with each error
    litellm.suppress_debug_info = True

    def completion(**kwargs):
        return litellm.completion(drop_params=True, **kwargs)

    client = instructor.from_litellm(completion)
    messages = [
        {"role": "system", "content": system_prompt(resolutions)},
        {"role": "user", "content": prompt},
    ]
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=MeshPromptResponse,
                max_retries=VALIDATION_RETRIES,
                timeout=timeout,
            )
        except Exception as e:
            message = redact(str(e))
            if "content filtering policy" in message.lower() and (
                attempt < FILTER_ATTEMPTS
            ):
                time.sleep(2**attempt)
                continue
            # The original error can hold the request, so it is not chained
            raise RuntimeError(
                f"The LLM request failed: {type(e).__name__}: {message}"
            ) from None


def choose_method(
    answer: MeshPromptResponse, downloadable: Sequence[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Return the method and shape to make a mesh with, or None and None.

    A global mesh is downloaded, so it has neither. A circle, an ellipse, a
    polygon and a channel are cut from a global mesh with ``create_region``,
    so their resolution must be one of ``downloadable``. A rectangle, or a
    region of no shape, is made with ``project_hexes`` unless the prompt asks
    for ``create_region``, which cuts a polygon around it.

    Raises
    ------
    ValueError
        If the method, shape and resolution do not go together. The message
        says what to ask for instead.
    """
    available = ", ".join(downloadable)
    if answer.is_global:
        if answer.shape or answer.method:
            raise ValueError(
                "A shape or method is for a regional mesh, so name the places it covers"
            )
        if answer.resolution not in downloadable:
            raise ValueError(
                f"Global meshes are available at {available}, not "
                f"{answer.resolution}. Name a region to make a mesh at another "
                "resolution."
            )
        return None, None
    method, shape = answer.method, answer.shape
    if shape in ("circle", "ellipse", "polygon", "channel"):
        if method == "project_hexes":
            raise ValueError(
                f"project_hexes makes only rectangles, not a {shape}. Ask for a "
                "rectangle, or for create_region."
            )
        method = "create_region"
    elif method is None:
        method = "project_hexes"
    if method == "create_region" and answer.resolution not in downloadable:
        raise ValueError(
            f"create_region cuts a downloaded global mesh, which is available at "
            f"{available}, not {answer.resolution}. Ask for one of those, or for "
            "a rectangle made with project_hexes."
        )
    if shape is None:
        shape = "rectangle" if method == "project_hexes" else "polygon"
    return method, shape


def mesh_config(
    answer: MeshPromptResponse,
    downloadable: Sequence[str],
    data_dir,
    buffer_km: float,
    nominatim_contact: Optional[str] = None,
    cache_dir=None,
) -> Dict[str, Any]:
    """Return the mesh request an answer describes, as ``generate_mesh`` takes.

    The places are looked up in ``data_dir``, and Nominatim with a contact,
    and the region is fitted to the shape with a margin of ``buffer_km``.

    Raises
    ------
    ValueError
        If the method, shape and resolution do not go together, the places
        are not found, or the region does not fit the shape.
    """
    from chiltepin.agents.mpas._geo_lookup import lookup_region
    from chiltepin.agents.mpas._region_shapes import regional_config

    method, shape = choose_method(answer, downloadable)
    config: Dict[str, Any] = {"resolution": answer.resolution, "name": answer.name}
    if method is None:
        return config
    if shape == "channel":
        block = {"upper_lat": answer.upper_lat, "lower_lat": answer.lower_lat}
        config["regional"] = {"create_region": {"channel": block}}
        return config
    region = lookup_region(
        answer.region_names,
        data_dir,
        exclude_names=answer.exclude_names,
        nominatim_contact=nominatim_contact,
        cache_dir=cache_dir,
    )
    resolution_km = float(answer.resolution[:-2])
    config["regional"] = regional_config(
        region, method, shape, resolution_km, buffer_km
    )
    return config
