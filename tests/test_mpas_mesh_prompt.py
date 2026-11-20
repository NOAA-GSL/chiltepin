# SPDX-License-Identifier: Apache-2.0
"""Tests of reading a mesh request from a prompt with an LLM.

litellm's completion is replaced by a fake that answers with a tool call, as a
model does, so instructor's own parsing and retries run without a model.
"""

import json
import math

import litellm
import pytest
from pydantic import ValidationError

from chiltepin.agents.mpas import _mesh_prompt as prompt
from chiltepin.agents.mpas._mesh_prompt import MeshPromptResponse

RESOLUTIONS = ["480km", "120km", "60km", "15km"]


def response(**fields):
    """Return a valid response, with some fields changed."""
    values = {"resolution": "15km", "name": "texas", "region_names": ["Texas"]}
    return MeshPromptResponse(**{**values, **fields})


# Normalizing


@pytest.mark.parametrize(
    ("value", "method"),
    [
        ("project_hexes", "project_hexes"),
        ("Project Hexes", "project_hexes"),
        ("hex-projection", "project_hexes"),
        ("create_region", "create_region"),
        ("Limited-Area", "create_region"),
        ("MPAS limited area", "create_region"),
        ("LAM", "create_region"),
        ("cut out", "create_region"),
        (None, None),
        ("  ", None),
    ],
)
def test_normalize_method(value, method):
    assert prompt.normalize_method(value) == method


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ("Square", "rectangle"),
        ("bounding box", "rectangle"),
        ("round", "circle"),
        ("Oval", "ellipse"),
        ("convex-hull", "polygon"),
        ("latitude band", "channel"),
        (None, None),
    ],
)
def test_normalize_shape(value, shape):
    assert prompt.normalize_shape(value) == shape


@pytest.mark.parametrize("normalize", [prompt.normalize_method, prompt.normalize_shape])
def test_unknown_names_list_the_choices(normalize):
    with pytest.raises(ValueError, match="Use one of: .*, or null"):
        normalize("triangle")


@pytest.mark.parametrize(
    ("value", "resolution"),
    [
        ("15km", "15km"),
        ("15 km", "15km"),
        ("15 KM", "15km"),
        ("7.5 kilometers", "7.5km"),
        ("3 kilometre", "3km"),
        ("15", "15km"),
        (15, "15km"),
        (15.0, "15km"),
        (7.5, "7.5km"),
        ("0.25km", "0.25km"),
        (12345678, "12345678km"),
    ],
)
def test_normalize_resolution(value, resolution):
    assert prompt.normalize_resolution(value) == resolution


@pytest.mark.parametrize(
    "value", ["fifteen", "15 miles", "", "-5km", "0km", 0, -1, math.inf, math.nan, True]
)
def test_invalid_resolutions(value):
    with pytest.raises(ValueError, match="Resolution must be"):
        prompt.normalize_resolution(value)


@pytest.mark.parametrize(
    ("value", "name"),
    [
        ("japan_15km", "japan_15km"),
        ("Japan 15 km!", "Japan_15_km"),
        ("São Paulo", "Sao_Paulo"),
        ("../etc/passwd", "etc_passwd"),
        ("a$(touch x)b", "a_touch_x_b"),
        ("x1.2562", "x1.2562"),
    ],
)
def test_normalize_name(value, name):
    assert prompt.normalize_name(value) == name


@pytest.mark.parametrize("value", ["", "...", "!!", "中文"])
def test_name_without_letters_or_digits(value):
    with pytest.raises(ValueError, match="no letters or digits"):
        prompt.normalize_name(value)


def test_name_too_long():
    prompt.normalize_name("a" * prompt.MAX_NAME_BYTES)
    with pytest.raises(ValueError, match="at most 200"):
        prompt.normalize_name("a" * (prompt.MAX_NAME_BYTES + 1))


# The response


def test_response_normalizes_its_fields():
    answer = response(
        resolution="15 KM",
        name="Japan 15 km",
        method="Limited Area",
        shape="oval",
        region_names=["Japan", " japan ", "  South   Korea "],
        exclude_names="Russia",
    )
    assert answer.resolution == "15km"
    assert answer.name == "Japan_15_km"
    assert answer.method == "create_region"
    assert answer.shape == "ellipse"
    assert answer.region_names == ["Japan", "South Korea"]
    assert answer.exclude_names == ["Russia"]
    assert not answer.is_global


def test_global_response():
    answer = response(region_names=[], name="global")
    assert answer.is_global
    assert response(region_names=None).region_names == []


def test_too_many_names():
    with pytest.raises(ValidationError, match="At most 300 places"):
        response(region_names=[f"place {i}" for i in range(prompt.MAX_NAMES + 1)])


def test_channel_latitudes_are_put_in_order():
    answer = response(shape="channel", upper_lat=-23.4, lower_lat=23.4)
    assert (answer.upper_lat, answer.lower_lat) == (23.4, -23.4)
    # A channel goes around the globe, so it covers no named places, and is
    # not a global mesh
    assert answer.region_names == []
    assert not answer.is_global


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"upper_lat": 10}, "needs both upper_lat and lower_lat"),
        ({"upper_lat": 10, "lower_lat": 10}, "must differ"),
        ({"upper_lat": 91, "lower_lat": 0}, "from -90 to 90"),
    ],
)
def test_invalid_channels(fields, message):
    with pytest.raises(ValidationError, match=message):
        response(shape="channel", **fields)


def test_latitudes_without_a_channel_are_dropped():
    answer = response(upper_lat=10, lower_lat=-10)
    assert (answer.upper_lat, answer.lower_lat) == (None, None)


def test_system_prompt_lists_the_resolutions():
    text = prompt.system_prompt(RESOLUTIONS)
    assert "480km, 120km, 60km, 15km" in text
    assert "MeshPromptResponse" in text


# Credentials


def test_redact(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-123456")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdefgh")
    # Short values, and variables that are not credentials, stay
    monkeypatch.setenv("SOME_TOKEN", "short")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1-region")
    text = "key=bedrock-api-key-123456 openai sk-abcdefgh short us-east-1-region"
    assert prompt.redact(text) == (
        "key=<redacted> openai <redacted> short us-east-1-region"
    )


# Asking the model


def tool_call(arguments):
    """Return a litellm response calling the MeshPromptResponse tool."""
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "MeshPromptResponse",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    )


@pytest.fixture
def model(monkeypatch):
    """Replace litellm's completion with a model that gives ``answers`` in turn.

    An answer is the tool call's arguments, or an exception to raise.
    """

    class Fake:
        answers = []
        calls = []
        sleeps = []

        def completion(self, **kwargs):
            self.calls.append(kwargs)
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return tool_call(answer)

    fake = Fake()
    monkeypatch.setattr(litellm, "completion", fake.completion)
    monkeypatch.setattr(prompt.time, "sleep", fake.sleeps.append)
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    return fake


def test_ask(model):
    model.answers = [
        {"resolution": "15 km", "name": "Texas", "region_names": ["Texas"]}
    ]
    answer = prompt.ask("A 15 km mesh over Texas", "bedrock/model", RESOLUTIONS, 30)
    # instructor keeps the raw response with the answer, so compare the fields
    assert answer.model_dump() == response(name="Texas").model_dump()
    (call,) = model.calls
    assert call["model"] == "bedrock/model"
    assert call["timeout"] == 30
    # litellm turns the forced tool call into one the model chooses, for
    # models that cannot be forced
    assert call["drop_params"] is True
    assert call["tool_choice"]["function"]["name"] == "MeshPromptResponse"
    system, user = call["messages"]
    assert system == {"role": "system", "content": prompt.system_prompt(RESOLUTIONS)}
    assert user == {"role": "user", "content": "A 15 km mesh over Texas"}
    assert litellm.suppress_debug_info is True


def test_invalid_answer_is_sent_back(model):
    model.answers = [
        {"resolution": "fifteen", "name": "texas", "region_names": ["Texas"]},
        {"resolution": "15km", "name": "texas", "region_names": ["Texas"]},
    ]
    answer = prompt.ask("Texas", "m", RESOLUTIONS)
    assert answer.model_dump() == response().model_dump()
    # The second request tells the model what was wrong
    retry = json.dumps(model.calls[1]["messages"])
    assert "Resolution must be a number of km" in retry


def test_answers_that_stay_invalid(model):
    invalid = {"resolution": "fifteen", "name": "texas"}
    model.answers = [invalid] * (prompt.VALIDATION_RETRIES + 1)
    with pytest.raises(RuntimeError, match="The LLM request failed"):
        prompt.ask("Texas", "m", RESOLUTIONS)


def test_content_filter_refusals_are_tried_again(model):
    refused = litellm.BadRequestError(
        "Blocked by the content filtering policy", "bedrock", "m"
    )
    model.answers = [refused, refused, {"resolution": "15km", "name": "texas"}]
    assert prompt.ask("Texas", "m", RESOLUTIONS).name == "texas"
    assert model.sleeps == [2, 4]


def test_content_filter_refusals_give_up(model):
    refused = litellm.BadRequestError(
        "Blocked by the content filtering policy", "bedrock", "m"
    )
    model.answers = [refused] * prompt.FILTER_ATTEMPTS
    with pytest.raises(RuntimeError, match="content filtering policy"):
        prompt.ask("Texas", "m", RESOLUTIONS)
    assert len(model.calls) == prompt.FILTER_ATTEMPTS


def test_errors_hold_no_credentials(model, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-123456")
    model.answers = [
        litellm.AuthenticationError(
            "Bearer bedrock-api-key-123456 was refused", "bedrock", "m"
        )
    ]
    with pytest.raises(RuntimeError) as error:
        prompt.ask("Texas", "m", RESOLUTIONS)
    assert "bedrock-api-key-123456" not in str(error.value)
    assert "<redacted>" in str(error.value)
    # The original error, which can hold the request, is not kept
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


# Choosing the method


DOWNLOADABLE = ["480km", "120km", "60km", "15km"]


@pytest.mark.parametrize(
    ("fields", "choice"),
    [
        # A global mesh is downloaded
        ({"region_names": []}, (None, None)),
        # A region of no shape is a project_hexes rectangle, at any resolution
        ({}, ("project_hexes", "rectangle")),
        ({"resolution": "7km"}, ("project_hexes", "rectangle")),
        ({"shape": "rectangle"}, ("project_hexes", "rectangle")),
        # Other shapes are cut from a global mesh
        ({"shape": "circle"}, ("create_region", "circle")),
        ({"shape": "ellipse"}, ("create_region", "ellipse")),
        ({"shape": "polygon", "method": "create_region"}, ("create_region", "polygon")),
        (
            {"shape": "channel", "upper_lat": 10, "lower_lat": -10},
            ("create_region", "channel"),
        ),
        # create_region without a shape cuts a polygon around the region
        ({"method": "create_region"}, ("create_region", "polygon")),
        (
            {"method": "create_region", "shape": "rectangle"},
            ("create_region", "rectangle"),
        ),
    ],
)
def test_choose_method(fields, choice):
    assert prompt.choose_method(response(**fields), DOWNLOADABLE) == choice


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"region_names": [], "resolution": "7km"}, "Global meshes are available at"),
        ({"region_names": [], "shape": "circle"}, "name the places it covers"),
        (
            {"region_names": [], "method": "project_hexes"},
            "name the places it covers",
        ),
        (
            {"shape": "circle", "method": "project_hexes"},
            "project_hexes makes only rectangles, not a circle",
        ),
        ({"shape": "ellipse", "resolution": "7km"}, "create_region cuts a downloaded"),
        ({"method": "create_region", "resolution": "7km"}, "or for a rectangle"),
    ],
)
def test_choose_method_rejects(fields, message):
    with pytest.raises(ValueError, match=message):
        prompt.choose_method(response(**fields), DOWNLOADABLE)


# Building the request


@pytest.fixture
def region_tools(monkeypatch):
    """Replace the lookup and the fit, recording what they are asked."""
    from chiltepin.agents.mpas import _geo_lookup, _region_shapes

    calls = {}

    def lookup(names, data_dir, exclude_names, nominatim_contact, cache_dir):
        calls["lookup"] = (names, data_dir, exclude_names, nominatim_contact, cache_dir)
        return "the region"

    def fit(region, method, shape, resolution_km, buffer_km):
        calls["fit"] = (region, method, shape, resolution_km, buffer_km)
        return {method: {"fitted": shape}}

    monkeypatch.setattr(_geo_lookup, "lookup_region", lookup)
    monkeypatch.setattr(_region_shapes, "regional_config", fit)
    return calls


def test_mesh_config_of_a_global_mesh(region_tools):
    answer = response(region_names=[], resolution="120km", name="global")
    config = prompt.mesh_config(answer, DOWNLOADABLE, "/data", 50)
    assert config == {"resolution": "120km", "name": "global"}
    assert region_tools == {}


def test_mesh_config_of_a_channel(region_tools):
    answer = response(shape="channel", upper_lat=-20, lower_lat=20, name="band")
    config = prompt.mesh_config(answer, DOWNLOADABLE, "/data", 50)
    assert config == {
        "resolution": "15km",
        "name": "band",
        "regional": {
            "create_region": {"channel": {"upper_lat": 20.0, "lower_lat": -20.0}}
        },
    }
    assert region_tools == {}


def test_mesh_config_of_a_region(region_tools):
    answer = response(
        resolution="7.5km",
        region_names=["United States of America"],
        exclude_names=["Alaska"],
    )
    config = prompt.mesh_config(
        answer, DOWNLOADABLE, "/data", 80, "me@example.org", "/cache"
    )
    assert config == {
        "resolution": "7.5km",
        "name": "texas",
        "regional": {"project_hexes": {"fitted": "rectangle"}},
    }
    assert region_tools["lookup"] == (
        ["United States of America"],
        "/data",
        ["Alaska"],
        "me@example.org",
        "/cache",
    )
    assert region_tools["fit"] == ("the region", "project_hexes", "rectangle", 7.5, 80)
