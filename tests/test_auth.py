# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Chiltepin authentication and non-interactive consumers."""

import json
from unittest import mock

import pytest
import requests
from globus_sdk import ClientApp, TransferAPIError, UserApp
from globus_sdk.token_storage import MemoryTokenStorage

import chiltepin.cli as cli
import chiltepin.data as data
import chiltepin.endpoint as endpoint
from chiltepin.agent_runtime import AgentRuntime


@pytest.mark.parametrize("app_class, allowed", [(UserApp, False), (ClientApp, True)])
def test_noninteractive_login(app_class, allowed):
    app = mock.Mock(spec=app_class)
    app.login_required.return_value = True
    with (
        mock.patch.object(endpoint, "get_chiltepin_app", return_value=app),
        mock.patch.object(endpoint, "Client"),
        mock.patch.object(endpoint, "TransferClient"),
    ):
        if allowed:
            endpoint.login(interactive=False)
            app.login.assert_called_once()
        else:
            with pytest.raises(RuntimeError, match="Chiltepin login is required"):
                endpoint.login(interactive=False)
            app.login.assert_not_called()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("service_identity", [False, True])
def test_one_login_requests_all_service_scopes(monkeypatch, cached, service_identity):
    monkeypatch.delenv("GLOBUS_COMPUTE_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("GLOBUS_COMPUTE_CLIENT_ID", endpoint.CHILTEPIN_CLIENT_UUID)
    if service_identity:
        monkeypatch.setenv("GLOBUS_COMPUTE_CLIENT_SECRET", "test-secret")
    for name in (
        "GLOBUS_CLI_CLIENT_ID",
        "GLOBUS_CLI_CLIENT_SECRET",
        "ACADEMY_GLOBUS_CLIENT_ID",
        "ACADEMY_GLOBUS_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    with mock.patch.object(
        endpoint, "get_token_storage", return_value=MemoryTokenStorage()
    ):
        app = endpoint.get_chiltepin_app()
    assert isinstance(app, ClientApp if service_identity else UserApp)
    assert app.config.request_refresh_tokens is True
    assert app.config.token_validation_error_handler is None
    assert app.config.auto_redrive_gares is False
    requested_scopes = []

    def capture_flow(auth_params=None):
        requested_scopes.extend(
            app._auth_params_with_required_scopes(auth_params).required_scopes
        )

    with (
        mock.patch.object(endpoint, "get_chiltepin_app", return_value=app),
        mock.patch.object(app, "login_required", return_value=not cached),
        mock.patch.object(app, "_run_login_flow", side_effect=capture_flow) as flow,
        mock.patch.object(endpoint.Client, "version_check"),
        mock.patch.object(endpoint.TransferClient, "operation_ls") as listing,
    ):
        clients = endpoint.login(interactive=not service_identity)
    listing.assert_not_called()
    assert clients["compute"].app is app
    assert clients["transfer"]._app is app
    assert flow.call_count == (0 if cached else 1)
    if not cached:
        assert str(endpoint.ComputeClientV2.scopes.all) in requested_scopes
        assert str(endpoint.TransferClient.scopes.all) in requested_scopes
        assert str(endpoint.AcademyExchangeScopes.academy_exchange) in requested_scopes
        assert {
            str(scope)
            for scope in endpoint.ComputeAuthClient.default_scope_requirements
        }.issubset(requested_scopes)


def test_login_authenticates_before_constructing_clients():
    app = mock.Mock(spec=UserApp)
    app.login_required.return_value = True
    calls = mock.Mock()
    calls.attach_mock(app.login, "login")
    with (
        mock.patch.object(endpoint, "get_chiltepin_app", return_value=app),
        mock.patch.object(endpoint, "Client") as compute,
        mock.patch.object(endpoint, "TransferClient") as transfer,
    ):
        calls.attach_mock(compute, "compute")
        calls.attach_mock(transfer, "transfer")
        clients = endpoint.login()
    assert calls.mock_calls == [
        mock.call.login(),
        mock.call.compute(app=app),
        mock.call.transfer(app=app),
    ]
    assert clients == {
        "compute": compute.return_value,
        "transfer": transfer.return_value,
    }


def test_shared_app_is_checked_and_logged_out_once():
    app = mock.Mock()
    app.login_required.return_value = False
    with mock.patch.object(endpoint, "get_chiltepin_app", return_value=app):
        assert endpoint.login_required() is False
        endpoint.logout()
    app.login_required.assert_called_once()
    app.logout.assert_called_once()


async def test_custom_unauthenticated_academy_exchange_does_not_load_globus_app():
    runtime = AgentRuntime(
        mock.Mock(), exchange_address="http://localhost:8000", auth_method=None
    )
    runtime._executors = {}
    with (
        mock.patch("chiltepin.agent_runtime.get_chiltepin_app") as get_app,
        mock.patch("chiltepin.agent_runtime.HttpExchangeFactory") as factory,
        mock.patch(
            "chiltepin.agent_runtime.Manager.from_exchange_factory",
            new_callable=mock.AsyncMock,
        ),
    ):
        await runtime.manager()
    get_app.assert_not_called()
    factory.assert_called_once_with(auth_method=None, url="http://localhost:8000")


@pytest.mark.parametrize("needs_login", [False, True])
async def test_academy_uses_shared_token_without_own_login(needs_login):
    runtime = AgentRuntime(mock.Mock())
    runtime._executors = {}
    app = mock.Mock()
    app.login_required.return_value = needs_login
    app.get_authorizer.return_value.get_authorization_header.return_value = (
        "test-header"
    )
    with (
        mock.patch("chiltepin.agent_runtime.get_chiltepin_app", return_value=app),
        mock.patch("chiltepin.agent_runtime.HttpExchangeFactory") as factory,
        mock.patch(
            "chiltepin.agent_runtime.Manager.from_exchange_factory",
            new_callable=mock.AsyncMock,
        ),
    ):
        if needs_login:
            with pytest.raises(RuntimeError, match="Chiltepin login is required"):
                await runtime.manager()
            factory.assert_not_called()
        else:
            await runtime.manager()
            factory.assert_called_once_with(
                auth_method=None, additional_headers={"Authorization": "test-header"}
            )
            app.get_authorizer.assert_called_once_with(
                endpoint.AcademyExchangeScopes.resource_server
            )
    app.login.assert_not_called()


async def test_academy_missing_header_fails_explicitly():
    runtime = AgentRuntime(mock.Mock())
    runtime._executors = {}
    app = mock.Mock()
    app.login_required.return_value = False
    app.get_authorizer.return_value.get_authorization_header.return_value = None
    with mock.patch("chiltepin.agent_runtime.get_chiltepin_app", return_value=app):
        with pytest.raises(RuntimeError, match="did not provide a bearer token"):
            await runtime.manager()


def test_cli_authentication_options():
    args = vars(cli.root_parser.parse_args(["login", "--non-interactive"]))
    assert args["interactive"] is False
    assert args["func"].__module__ == endpoint.__name__
    assert args["func"].__name__ == "login"


def test_cli_rejects_removed_collection_configuration():
    with pytest.raises(SystemExit):
        cli.root_parser.parse_args(["login", "--transfer-config", "collections.yaml"])


@pytest.mark.parametrize("operation", ["transfer", "delete"])
@pytest.mark.parametrize("with_client", [True, False])
def test_data_operations_never_reauthenticate(operation, with_client):
    response = requests.Response()
    response.status_code = 403
    response.headers["Content-Type"] = "application/json"
    response.request = requests.Request(
        "POST", "https://transfer.api.globus.org/v0.10/transfer"
    ).prepare()
    response._content = json.dumps(
        {
            "code": "PermissionDenied",
            "message": "Session reauthentication required",
            "authorization_parameters": {
                "session_required_identities": ["test-identity"]
            },
        }
    ).encode()
    error = TransferAPIError(response)
    client = mock.Mock()
    client.endpoint_search.return_value = [{"id": "test", "display_name": "test"}]
    client.submit_transfer.side_effect = error
    client.submit_delete.side_effect = error
    kwargs = {"client": client} if with_client else {}
    with mock.patch.object(
        endpoint, "login", return_value={"transfer": client}
    ) as login:
        with pytest.raises(RuntimeError, match="chiltepin logout") as exc:
            if operation == "transfer":
                data.transfer("test", "test", "source", "dest", **kwargs)
            else:
                data.delete("test", "dest", **kwargs)
    assert exc.value.__cause__ is error
    if with_client:
        login.assert_not_called()
    else:
        login.assert_called_once_with(interactive=False)
    client.app.login.assert_not_called()
