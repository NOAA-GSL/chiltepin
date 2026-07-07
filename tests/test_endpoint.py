# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import pathlib
import platform
import shutil
import subprocess
from unittest.mock import MagicMock, mock_open, patch
from uuid import UUID

import pytest
import yaml

import chiltepin.endpoint as endpoint

# =============================================================================
# Integration Tests - These test the full endpoint lifecycle and must run in order
# =============================================================================


@pytest.mark.skipif(
    platform.system() != "Linux" or not endpoint.ENDPOINT_MANAGEMENT_AVAILABLE,
    reason="Endpoint management requires Linux and globus-compute-endpoint",
)
class TestEndpointIntegration:
    """Integration tests for endpoint lifecycle: configure -> start -> stop -> delete.

    Tests depend on state from previous tests and must run in the order they appear
    in this file. Each test is split by config_dir scenario so that failures are
    isolated to the specific scenario that failed.
    """

    def test_show_empty(self):
        """Test listing endpoints when none exist (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        config_dir_test.mkdir(parents=True, exist_ok=True)

        # Start from a clean state
        if os.path.exists(f"{config_dir_test}"):
            shutil.rmtree(f"{config_dir_test}")

        ep_list = endpoint.show(config_dir=f"{config_dir_test}")
        assert ep_list == {}

    def test_is_running_nonexistent(self):
        """Test is_running returns False for nonexistent endpoint."""
        assert endpoint.is_running("nonexistent_endpoint_xyz_123") is False

    def test_exists_nonexistent(self):
        """Test exists returns False for nonexistent endpoint."""
        assert endpoint.exists("nonexistent_endpoint_xyz_123") is False

    def test_configure_default_config_dir(self):
        """Test configuring endpoint with default config_dir."""
        config_dir_default = pathlib.Path.home() / ".globus_compute"

        # Start from a clean state
        if os.path.exists(f"{config_dir_default}/foo"):
            shutil.rmtree(f"{config_dir_default}/foo")

        # Configure an endpoint without a config_dir
        endpoint.configure("foo")
        assert os.path.exists(f"{config_dir_default}/foo/config.yaml")

    def test_configure_custom_config_dir(self):
        """Test configuring endpoint with custom config_dir."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        config_dir_test.mkdir(parents=True, exist_ok=True)

        # Start from a clean state
        if os.path.exists(f"{config_dir_test}/bar"):
            shutil.rmtree(f"{config_dir_test}/bar")

        # Configure an endpoint with a config_dir
        endpoint.configure("bar", config_dir=f"{config_dir_test}")
        assert os.path.exists(f"{config_dir_test}/bar/config.yaml")

    def test_show_initialized_default_config_dir(self):
        """Test listing endpoint in Initialized state (default config_dir)."""
        ep_list = endpoint.show()
        assert ep_list["foo"] == {"id": None, "status": "Initialized"}

    def test_show_initialized_custom_config_dir(self):
        """Test listing endpoint in Initialized state (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"

        ep_list = endpoint.show(config_dir=f"{config_dir_test}")
        assert ep_list["bar"] == {"id": None, "status": "Initialized"}

    def test_exists_default_config_dir(self):
        """Test exists returns True for configured endpoint (default config_dir)."""
        assert endpoint.exists("foo") is True

    def test_exists_custom_config_dir(self):
        """Test exists returns True for configured endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        assert endpoint.exists("bar", config_dir=f"{config_dir_test}") is True

    def test_start_default_config_dir(self):
        """Test starting endpoint (default config_dir)."""
        endpoint.start("foo", timeout=30)

        # Verify it is running
        ep_list = endpoint.show()
        assert ep_list["foo"]["status"] == "Running"
        # Verify the endpoint ID is a valid UUID
        assert UUID(ep_list["foo"]["id"]) is not None

    def test_is_running_default_config_dir(self):
        """Test is_running returns True for running endpoint (default config_dir)."""
        assert endpoint.is_running("foo") is True

    def test_start_custom_config_dir(self):
        """Test starting endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"

        endpoint.start("bar", config_dir=f"{config_dir_test}", timeout=30)

        # Verify it is running
        ep_list = endpoint.show(config_dir=f"{config_dir_test}")
        assert ep_list["bar"]["status"] == "Running"
        # Verify the endpoint ID is a valid UUID
        assert UUID(ep_list["bar"]["id"]) is not None

    def test_is_running_custom_config_dir(self):
        """Test is_running returns True for running endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        assert endpoint.is_running("bar", config_dir=f"{config_dir_test}") is True

    def test_stop_default_config_dir(self):
        """Test stopping endpoint (default config_dir)."""
        endpoint.stop("foo", timeout=30)

        # Verify it is stopped
        ep_list = endpoint.show()
        assert ep_list["foo"]["status"] == "Stopped"
        # Verify the endpoint ID is still a valid UUID
        assert UUID(ep_list["foo"]["id"]) is not None

    def test_is_not_running_default_config_dir(self):
        """Test is_running returns False for stopped endpoint (default config_dir)."""
        assert endpoint.is_running("foo") is False

    def test_stop_custom_config_dir(self):
        """Test stopping endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"

        endpoint.stop("bar", config_dir=f"{config_dir_test}", timeout=30)

        # Verify it is stopped
        ep_list = endpoint.show(config_dir=f"{config_dir_test}")
        assert ep_list["bar"]["status"] == "Stopped"
        # Verify the endpoint ID is still a valid UUID
        assert UUID(ep_list["bar"]["id"]) is not None

    def test_is_not_running_custom_config_dir(self):
        """Test is_running returns False for stopped endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        assert endpoint.is_running("bar", config_dir=f"{config_dir_test}") is False

    def test_delete_default_config_dir(self):
        """Test deleting endpoint (default config_dir)."""
        endpoint.delete("foo", timeout=30)

        # Verify it is deleted
        ep_list = endpoint.show()
        assert ep_list.get("foo", None) is None
        assert not os.path.exists(f"{pathlib.Path.home()}/.globus_compute/foo")

    def test_delete_custom_config_dir(self):
        """Test deleting endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"

        endpoint.delete("bar", config_dir=f"{config_dir_test}", timeout=30)

        # Verify it is deleted
        ep_list = endpoint.show(config_dir=f"{config_dir_test}")
        assert ep_list.get("bar", None) is None
        assert not os.path.exists(f"{config_dir_test}/bar")

    def test_not_exists_default_config_dir(self):
        """Test exists returns False for deleted endpoint (default config_dir)."""
        assert endpoint.exists("foo") is False

    def test_not_exists_custom_config_dir(self):
        """Test exists returns False for deleted endpoint (custom config_dir)."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute"
        assert endpoint.exists("bar", config_dir=f"{config_dir_test}") is False


# =============================================================================
# Unit Tests - Organized by function in endpoint.py order
# =============================================================================


class TestGetChiltepinApps:
    """Tests for get_chiltepin_apps() function."""

    def test_with_client_secret_but_no_id(self):
        """Test that RuntimeError is raised when SECRET is set but ID is not."""
        # Save original env vars
        orig_id = os.environ.pop("GLOBUS_COMPUTE_CLIENT_ID", None)
        orig_secret = os.environ.get("GLOBUS_COMPUTE_CLIENT_SECRET")

        try:
            # Set up scenario: SECRET set but ID not set
            os.environ["GLOBUS_COMPUTE_CLIENT_SECRET"] = "test_secret"

            with pytest.raises(
                RuntimeError,
                match=r"\$GLOBUS_COMPUTE_CLIENT_SECRET is set but \$GLOBUS_COMPUTE_CLIENT_ID is not",
            ):
                endpoint.get_chiltepin_apps()
        finally:
            # Restore original env vars
            os.environ.pop("GLOBUS_COMPUTE_CLIENT_SECRET", None)
            if orig_id:
                os.environ["GLOBUS_COMPUTE_CLIENT_ID"] = orig_id
            if orig_secret:
                os.environ["GLOBUS_COMPUTE_CLIENT_SECRET"] = orig_secret

    def test_with_client_credentials(self):
        """Test that ClientApp is created when client_secret is provided."""
        with patch.dict(
            os.environ,
            {
                "GLOBUS_COMPUTE_CLIENT_ID": "test_client_id",
                "GLOBUS_COMPUTE_CLIENT_SECRET": "test_secret",
            },
            clear=False,
        ):
            with patch(
                "chiltepin.endpoint.get_globus_compute_app"
            ) as mock_get_compute_app:
                with patch(
                    "chiltepin.endpoint.get_globus_academy_app"
                ) as mock_get_academy_app:
                    with patch("chiltepin.endpoint.ClientApp") as mock_client_app:
                        with patch("chiltepin.endpoint.UserApp"):
                            mock_compute_app = MagicMock()
                            mock_academy_app = MagicMock()
                            mock_get_compute_app.return_value = mock_compute_app
                            mock_get_academy_app.return_value = mock_academy_app

                            compute_app, transfer_app, academy_app = (
                                endpoint.get_chiltepin_apps()
                            )

                            # Verify GLOBUS_CLI_* env vars were set
                            assert (
                                os.environ["GLOBUS_CLI_CLIENT_ID"] == "test_client_id"
                            )
                            assert (
                                os.environ["GLOBUS_CLI_CLIENT_SECRET"] == "test_secret"
                            )
                            assert (
                                os.environ["ACADEMY_GLOBUS_CLIENT_ID"]
                                == "test_client_id"
                            )
                            assert (
                                os.environ["ACADEMY_GLOBUS_CLIENT_SECRET"]
                                == "test_secret"
                            )

                            # Verify ClientApp was called for transfer client
                            mock_client_app.assert_called_once_with(
                                "chiltepin",
                                client_id="test_client_id",
                                client_secret="test_secret",
                            )

    def test_without_client_credentials(self):
        """Test that UserApp is created when no client_secret is provided."""
        # Save original env vars
        orig_id = os.environ.pop("GLOBUS_COMPUTE_CLIENT_ID", None)
        orig_secret = os.environ.pop("GLOBUS_COMPUTE_CLIENT_SECRET", None)

        try:
            with patch(
                "chiltepin.endpoint.get_globus_compute_app"
            ) as mock_get_compute_app:
                with patch(
                    "chiltepin.endpoint.get_globus_academy_app"
                ) as mock_get_academy_app:
                    with patch("chiltepin.endpoint.UserApp") as mock_user_app:
                        mock_compute_app = MagicMock()
                        mock_academy_app = MagicMock()
                        mock_get_compute_app.return_value = mock_compute_app
                        mock_get_academy_app.return_value = mock_academy_app

                        compute_app, transfer_app, academy_app = (
                            endpoint.get_chiltepin_apps()
                        )

                        # Verify UserApp was called for transfer client
                        mock_user_app.assert_called_once_with(
                            "chiltepin",
                            client_id=endpoint.CHILTEPIN_CLIENT_UUID,
                        )
        finally:
            # Restore original env vars
            if orig_id:
                os.environ["GLOBUS_COMPUTE_CLIENT_ID"] = orig_id
            if orig_secret:
                os.environ["GLOBUS_COMPUTE_CLIENT_SECRET"] = orig_secret


class TestLogin:
    """Tests for login() function."""

    def test_with_login_required(self):
        """Test login when both apps require login."""
        with patch("chiltepin.endpoint.get_chiltepin_apps") as mock_get_apps:
            with patch("chiltepin.endpoint.Client"):
                with patch("chiltepin.endpoint.TransferClient"):
                    # Setup mocks
                    mock_compute_app = MagicMock()
                    mock_transfer_app = MagicMock()
                    mock_academy_app = MagicMock()
                    mock_compute_app.login_required.return_value = True
                    mock_transfer_app.login_required.return_value = True
                    mock_academy_app.login_required.return_value = True

                    mock_get_apps.return_value = (
                        mock_compute_app,
                        mock_transfer_app,
                        mock_academy_app,
                    )

                    # Call login
                    clients = endpoint.login()

                    # Verify login was called on all apps
                    mock_compute_app.login.assert_called_once()
                    mock_transfer_app.login.assert_called_once()
                    mock_academy_app.login.assert_called_once()

                    # Verify clients were created
                    assert "compute" in clients
                    assert "transfer" in clients
                    # Academy client is not needed and is not returned by login(), so we don't check for it here

    def test_without_login_required(self):
        """Test login when apps don't require login."""
        with patch("chiltepin.endpoint.get_chiltepin_apps") as mock_get_apps:
            with patch("chiltepin.endpoint.Client"):
                with patch("chiltepin.endpoint.TransferClient"):
                    # Setup mocks
                    mock_compute_app = MagicMock()
                    mock_transfer_app = MagicMock()
                    mock_academy_app = MagicMock()
                    mock_compute_app.login_required.return_value = False
                    mock_transfer_app.login_required.return_value = False
                    mock_academy_app.login_required.return_value = False

                    mock_get_apps.return_value = (
                        mock_compute_app,
                        mock_transfer_app,
                        mock_academy_app,
                    )

                    # Call login
                    endpoint.login()

                    # Verify login was NOT called
                    mock_compute_app.login.assert_not_called()
                    mock_transfer_app.login.assert_not_called()
                    mock_academy_app.login.assert_not_called()


class TestLoginRequired:
    """Tests for login_required() function."""

    def test_returns_true_when_login_needed(self):
        """Test login_required returns True if any app requires login."""
        with patch("chiltepin.endpoint.get_chiltepin_apps") as mock_get_apps:
            mock_compute_app = MagicMock()
            mock_transfer_app = MagicMock()
            mock_academy_app = MagicMock()
            mock_compute_app.login_required.return_value = True
            mock_transfer_app.login_required.return_value = False
            mock_academy_app.login_required.return_value = False

            mock_get_apps.return_value = (
                mock_compute_app,
                mock_transfer_app,
                mock_academy_app,
            )

            # Should return True if either app requires login
            assert endpoint.login_required() is True


class TestLogout:
    """Tests for logout() function."""

    def test_logout_calls_all_apps(self):
        """Test logout calls logout on all apps."""
        with patch("chiltepin.endpoint.get_chiltepin_apps") as mock_get_apps:
            mock_compute_app = MagicMock()
            mock_transfer_app = MagicMock()
            mock_academy_app = MagicMock()
            mock_get_apps.return_value = (
                mock_compute_app,
                mock_transfer_app,
                mock_academy_app,
            )

            endpoint.logout()

            # Verify logout was called on all apps
            mock_compute_app.logout.assert_called_once()
            mock_transfer_app.logout.assert_called_once()
            mock_academy_app.logout.assert_called_once()


class TestPlatformChecks:
    """Tests for platform checks on endpoint management functions.

    These tests verify that NotImplementedError is raised on non-Linux platforms.
    They don't require globus-compute-endpoint to be installed since the platform
    check happens before the library availability check.
    """

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_configure_not_supported(self, mock_system, platform_name):
        """Test that configure raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.configure("test_endpoint")

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_start_not_supported(self, mock_system, platform_name):
        """Test that start raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.start("test_endpoint")

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_stop_not_supported(self, mock_system, platform_name):
        """Test that stop raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.stop("test_endpoint")

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_delete_not_supported(self, mock_system, platform_name):
        """Test that delete raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.delete("test_endpoint")

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_show_not_supported(self, mock_system, platform_name):
        """Test that show raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.show()

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_exists_not_supported(self, mock_system, platform_name):
        """Test that exists raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.exists("test_endpoint")

    @patch("chiltepin.endpoint.platform.system")
    @pytest.mark.parametrize("platform_name", ["Windows", "Darwin"])
    def test_is_running_not_supported(self, mock_system, platform_name):
        """Test that is_running raises NotImplementedError on Windows and macOS."""
        mock_system.return_value = platform_name
        with pytest.raises(
            NotImplementedError,
            match="Endpoint management is only supported on Linux",
        ):
            endpoint.is_running("test_endpoint")


@pytest.mark.skipif(
    platform.system() != "Linux" or not endpoint.ENDPOINT_MANAGEMENT_AVAILABLE,
    reason="Endpoint management requires Linux and globus-compute-endpoint",
)
class TestConfigure:
    """Tests for configure() function."""

    def test_timeout(self):
        """Test that configure raises TimeoutError when subprocess times out."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_timeout"
        config_dir_test.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.Popen") as mock_popen:
            import subprocess

            mock_process = MagicMock()
            mock_process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 0.1)
            mock_popen.return_value = mock_process

            with pytest.raises(TimeoutError, match="configure command timed out"):
                endpoint.configure(
                    "timeout_test", config_dir=str(config_dir_test), timeout=0.1
                )

    def test_command_failure(self):
        """Test that configure raises RuntimeError when command fails."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_fail"
        config_dir_test.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("error output", "")
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            with pytest.raises(RuntimeError, match="Failed to configure endpoint"):
                endpoint.configure("fail_test", config_dir=str(config_dir_test))

    def test_yaml_read_error(self):
        """Test that configure returns False when YAML read fails."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_yaml_read"
        config_dir_test.mkdir(parents=True, exist_ok=True)
        endpoint_dir = config_dir_test / "yaml_read_test"
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("success", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            with patch("builtins.open", mock_open(read_data="invalid: yaml: [{}")):
                with patch(
                    "yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")
                ):
                    result = endpoint.configure(
                        "yaml_read_test", config_dir=str(config_dir_test)
                    )
                    assert result is False

    def test_yaml_write_error(self):
        """Test that configure returns False when YAML write fails."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_yaml_write"
        config_dir_test.mkdir(parents=True, exist_ok=True)
        endpoint_dir = config_dir_test / "yaml_write_test"
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.return_value = ("success", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            # Mock the file operations
            config_content = yaml.dump({"display_name": "test", "debug": False})

            # First open for reading succeeds, second for writing fails
            m = mock_open(read_data=config_content)
            with patch("builtins.open", m):
                with patch("yaml.dump", side_effect=yaml.YAMLError("Write error")):
                    result = endpoint.configure(
                        "yaml_write_test", config_dir=str(config_dir_test)
                    )
                    # The function should return False when yaml.dump fails
                    assert result is False

    def test_path_capture_timeout(self):
        """Test configure when PATH capture subprocess times out."""
        import subprocess

        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_path_timeout"
        config_dir_test.mkdir(parents=True, exist_ok=True)
        endpoint_dir = config_dir_test / "path_timeout_test"
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        call_count = [0]

        def mock_popen_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_process = MagicMock()

            if call_count[0] == 1:
                # First call (configure command) succeeds
                mock_process.communicate.return_value = ("success", "")
                mock_process.returncode = 0
            else:
                # Second call (PATH capture) times out
                mock_process.communicate.side_effect = subprocess.TimeoutExpired(
                    "cmd", 0.5
                )

            return mock_process

        with patch("subprocess.Popen", side_effect=mock_popen_side_effect):
            # Create a minimal config file
            (endpoint_dir / "config.yaml").write_text(
                yaml.dump({"display_name": "test", "debug": False})
            )

            with pytest.raises(TimeoutError, match="PATH capture command timed out"):
                endpoint.configure(
                    "path_timeout_test", config_dir=str(config_dir_test), timeout=1
                )

    def test_path_capture_failure(self):
        """Test configure when PATH capture command returns non-zero."""
        pwd = pathlib.Path(__file__).parent.resolve()
        config_dir_test = pwd / "test_output" / ".globus_compute_path_fail"
        config_dir_test.mkdir(parents=True, exist_ok=True)
        endpoint_dir = config_dir_test / "path_fail_test"
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        call_count = [0]

        def mock_popen_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_process = MagicMock()

            if call_count[0] == 1:
                # First call (configure command) succeeds
                mock_process.communicate.return_value = ("success", "")
                mock_process.returncode = 0
            else:
                # Second call (PATH capture) fails
                mock_process.communicate.return_value = ("", "PATH error")
                mock_process.returncode = 1

            return mock_process

        with patch("subprocess.Popen", side_effect=mock_popen_side_effect):
            # Create a minimal config file
            (endpoint_dir / "config.yaml").write_text(
                yaml.dump({"display_name": "test", "debug": False})
            )

            with pytest.raises(RuntimeError, match="Failed to capture system PATH"):
                endpoint.configure("path_fail_test", config_dir=str(config_dir_test))


@pytest.mark.skipif(
    platform.system() != "Linux" or not endpoint.ENDPOINT_MANAGEMENT_AVAILABLE,
    reason="Endpoint management requires Linux and globus-compute-endpoint",
)
class TestStart:
    """Tests for start() function."""

    def test_login_required(self):
        """Test that start raises RuntimeError when login is required."""
        with patch("chiltepin.endpoint.login_required", return_value=True):
            with pytest.raises(RuntimeError, match="Chiltepin login is required"):
                endpoint.start("test_endpoint")

    def test_communicate_timeout(self):
        """Test that start raises TimeoutError when communicate() times out."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint._link_token_store"):
                # Create a mock process that times out during communicate()
                mock_process = MagicMock()
                mock_process.communicate.side_effect = subprocess.TimeoutExpired(
                    cmd="globus-compute-endpoint", timeout=5
                )
                with patch("subprocess.Popen", return_value=mock_process):
                    with pytest.raises(
                        TimeoutError,
                        match="globus-compute-endpoint start command timed out",
                    ):
                        endpoint.start("test_endpoint", timeout=5)
                    # Verify cleanup was called
                    mock_process.kill.assert_called_once()
                    mock_process.wait.assert_called_once()

    def test_start_command_failure(self):
        """Test that start raises RuntimeError when start command fails."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint._link_token_store"):
                # Create a mock process that fails with non-zero exit code
                mock_process = MagicMock()
                mock_process.communicate.return_value = (
                    "Error: endpoint configuration not found",
                    None,
                )
                mock_process.returncode = 1
                with patch("subprocess.Popen", return_value=mock_process):
                    with pytest.raises(
                        RuntimeError, match="Failed to start endpoint.*test_endpoint"
                    ):
                        endpoint.start("test_endpoint", timeout=5)

    def test_polling_timeout(self):
        """Test that start raises TimeoutError when polling for Running state times out."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint._link_token_store"):
                # Create a mock process that succeeds
                mock_process = MagicMock()
                mock_process.communicate.return_value = ("", None)
                mock_process.returncode = 0
                with patch("subprocess.Popen", return_value=mock_process):
                    # Mock is_running to always return False (never reaches Running state)
                    with patch("chiltepin.endpoint.is_running", return_value=False):
                        with pytest.raises(
                            TimeoutError,
                            match="Timeout of.*exceeded while waiting for endpoint",
                        ):
                            endpoint.start("test_endpoint", timeout=0.1)


class TestLinkTokenStore:
    """Tests for the _link_token_store() helper function."""

    def test_noop_for_default_config_dir(self, tmp_path, monkeypatch):
        """No link is made when the config dir is the default location."""
        monkeypatch.setenv("HOME", str(tmp_path))
        default_dir = tmp_path / ".globus_compute"
        default_dir.mkdir()
        (default_dir / "storage.db").write_text("tokens")

        # config_dir == default location: should return early untouched
        endpoint._link_token_store(str(default_dir))

        assert (default_dir / "storage.db").is_file()
        assert not (default_dir / "storage.db").is_symlink()

    def test_noop_when_no_default_store(self, tmp_path, monkeypatch):
        """No link is made when there is no default token store to link."""
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".globus_compute").mkdir()  # exists, but has no storage.db
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()

        endpoint._link_token_store(str(custom_dir))

        assert not (custom_dir / "storage.db").exists()
        assert not (custom_dir / "storage.db").is_symlink()

    def test_noop_when_store_already_present(self, tmp_path, monkeypatch):
        """An existing token store in the custom dir is left untouched."""
        monkeypatch.setenv("HOME", str(tmp_path))
        default_dir = tmp_path / ".globus_compute"
        default_dir.mkdir()
        (default_dir / "storage.db").write_text("default tokens")
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "storage.db").write_text("existing tokens")

        endpoint._link_token_store(str(custom_dir))

        assert not (custom_dir / "storage.db").is_symlink()
        assert (custom_dir / "storage.db").read_text() == "existing tokens"

    def test_links_default_store_into_custom_dir(self, tmp_path, monkeypatch):
        """The default token store is symlinked into a custom config dir."""
        monkeypatch.setenv("HOME", str(tmp_path))
        default_dir = tmp_path / ".globus_compute"
        default_dir.mkdir()
        default_store = default_dir / "storage.db"
        default_store.write_text("tokens")
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()

        endpoint._link_token_store(str(custom_dir))

        custom_store = custom_dir / "storage.db"
        assert custom_store.is_symlink()
        assert os.path.realpath(custom_store) == os.path.realpath(default_store)
        assert custom_store.read_text() == "tokens"


@pytest.mark.skipif(
    platform.system() != "Linux" or not endpoint.ENDPOINT_MANAGEMENT_AVAILABLE,
    reason="Endpoint management requires Linux and globus-compute-endpoint",
)
class TestStop:
    """Tests for stop() function."""

    def test_login_required(self):
        """Test that stop raises RuntimeError when login is required."""
        with patch("chiltepin.endpoint.login_required", return_value=True):
            with pytest.raises(RuntimeError, match="Chiltepin login is required"):
                endpoint.stop("test_endpoint")

    def test_timeout(self):
        """Test that stop raises TimeoutError when endpoint doesn't stop in time."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch("chiltepin.endpoint.Endpoint.stop_endpoint"):
                    with patch("chiltepin.endpoint.is_running", return_value=True):
                        with pytest.raises(TimeoutError, match="Timeout of"):
                            endpoint.stop("test_endpoint", timeout=0.1)

    def test_with_psutil_timeout(self):
        """Test that stop tolerates psutil.TimeoutExpired and re-issues the stop.

        globus-compute-endpoint <=4.7 raises psutil.TimeoutExpired when the
        endpoint is slow to shut down. stop() should swallow it and keep trying
        while the endpoint is still running, rather than propagating it.
        """
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch("chiltepin.endpoint.Endpoint.stop_endpoint") as mock_stop:
                    import psutil

                    # First call raises TimeoutExpired, second succeeds
                    mock_stop.side_effect = [psutil.TimeoutExpired(1), None]
                    # Still running right after the timeout, stopped after retry
                    with patch(
                        "chiltepin.endpoint.is_running",
                        side_effect=[True, False],
                    ):
                        # Should not raise an exception
                        endpoint.stop("test_endpoint", timeout=5)
                        # Verify stop_endpoint was retried once the endpoint was
                        # still running after the first (timed-out) attempt
                        assert mock_stop.call_count == 2

    def test_with_system_exit(self):
        """Test that stop tolerates the SystemExit(-1) slow-shutdown signal.

        globus-compute-endpoint >=4.8 no longer raises psutil.TimeoutExpired on a
        slow shutdown; instead stop_endpoint logs a warning and calls
        sys.exit(-1). stop() should treat that as a transient timeout and retry
        while the endpoint is still running, rather than letting SystemExit
        escape into the caller.
        """
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch("chiltepin.endpoint.Endpoint.stop_endpoint") as mock_stop:
                    # First call exits non-zero (slow shutdown), second succeeds
                    mock_stop.side_effect = [SystemExit(-1), None]
                    with patch(
                        "chiltepin.endpoint.is_running",
                        side_effect=[True, False],
                    ):
                        # Should not raise an exception
                        endpoint.stop("test_endpoint", timeout=5)
                        assert mock_stop.call_count == 2

    def test_clean_system_exit_propagates(self):
        """Test that a clean SystemExit from stop_endpoint is not swallowed."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch("chiltepin.endpoint.Endpoint.stop_endpoint") as mock_stop:
                    # A zero/None exit code is an intentional shutdown, not the
                    # slow-shutdown signal, so it must propagate.
                    mock_stop.side_effect = SystemExit(0)
                    with patch("chiltepin.endpoint.is_running", return_value=False):
                        with pytest.raises(SystemExit):
                            endpoint.stop("test_endpoint", timeout=5)


@pytest.mark.skipif(
    platform.system() != "Linux" or not endpoint.ENDPOINT_MANAGEMENT_AVAILABLE,
    reason="Endpoint management requires Linux and globus-compute-endpoint",
)
class TestDelete:
    """Tests for delete() function."""

    def test_login_required(self):
        """Test that delete raises RuntimeError when login is required."""
        with patch("chiltepin.endpoint.login_required", return_value=True):
            with pytest.raises(RuntimeError, match="Chiltepin login is required"):
                endpoint.delete("test_endpoint")

    def test_timeout(self):
        """Test that delete raises TimeoutError when endpoint deletion times out."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch("chiltepin.endpoint.Endpoint.delete_endpoint"):
                    with patch("chiltepin.endpoint.exists", return_value=True):
                        with pytest.raises(TimeoutError, match="Timeout of"):
                            endpoint.delete("test_endpoint", timeout=0.1)

    def test_with_config_error(self):
        """Test that delete uses force=True when get_config raises exception."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch(
                "chiltepin.endpoint.get_config", side_effect=Exception("Config error")
            ):
                with patch(
                    "chiltepin.endpoint.Endpoint.delete_endpoint"
                ) as mock_delete:
                    with patch("chiltepin.endpoint.exists", return_value=False):
                        endpoint.delete("test_endpoint", timeout=5)
                        # Verify delete was called with force=True
                        assert mock_delete.call_args[1]["force"] is True

    def test_with_deletion_error(self):
        """Test that delete raises RuntimeError when Endpoint.delete_endpoint fails."""
        with patch("chiltepin.endpoint.login_required", return_value=False):
            with patch("chiltepin.endpoint.get_config"):
                with patch(
                    "chiltepin.endpoint.Endpoint.delete_endpoint",
                    side_effect=Exception("Delete failed"),
                ):
                    with pytest.raises(RuntimeError, match="Error deleting endpoint"):
                        endpoint.delete("test_endpoint", timeout=5)


class TestEndpointManagementUnavailable:
    """Tests for when globus-compute-endpoint is not installed."""

    def setup_method(self):
        """Store original value."""
        self.original_available = endpoint.ENDPOINT_MANAGEMENT_AVAILABLE
        self.original_error = endpoint._ENDPOINT_IMPORT_ERROR

    def teardown_method(self):
        """Restore original value."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = self.original_available
        endpoint._ENDPOINT_IMPORT_ERROR = self.original_error

    def test_import_error_handling(self):
        """Test that module handles ImportError correctly at import time."""
        import sys

        # Mock the globus_compute_endpoint packages to raise ImportError
        original_modules = {}
        mock_modules = [
            "globus_compute_endpoint",
            "globus_compute_endpoint.endpoint",
            "globus_compute_endpoint.endpoint.config",
            "globus_compute_endpoint.endpoint.config.utils",
        ]

        for mod_name in mock_modules:
            if mod_name in sys.modules:
                original_modules[mod_name] = sys.modules[mod_name]
            sys.modules[mod_name] = None

        try:
            # Reload the existing module object to trigger ImportError handling
            # This avoids creating a new module instance that could cause
            # cross-test contamination
            test_endpoint = importlib.reload(endpoint)

            # Verify the except block executed correctly
            assert test_endpoint.ENDPOINT_MANAGEMENT_AVAILABLE is False
            assert test_endpoint._ENDPOINT_IMPORT_ERROR is not None
            assert test_endpoint.get_config is None
            assert test_endpoint.Endpoint is None
        finally:
            # Restore original module state
            for mod_name in mock_modules:
                if mod_name in original_modules:
                    sys.modules[mod_name] = original_modules[mod_name]
                elif mod_name in sys.modules:
                    del sys.modules[mod_name]

            # Reload the module again to restore working state
            importlib.reload(endpoint)

    @patch("chiltepin.endpoint.platform.system", return_value="Linux")
    def test_configure_not_available(self, mock_system):
        """Test configure raises ImportError when endpoint library not available."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = False
        endpoint._ENDPOINT_IMPORT_ERROR = ImportError("test import error")
        with pytest.raises(
            ImportError, match="Endpoint management requires.*globus-compute-endpoint"
        ):
            endpoint.configure("test_endpoint")

    @patch("chiltepin.endpoint.platform.system", return_value="Linux")
    def test_show_not_available(self, mock_system):
        """Test show raises ImportError when endpoint library not available."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = False
        endpoint._ENDPOINT_IMPORT_ERROR = ImportError("test import error")
        with pytest.raises(
            ImportError, match="Endpoint management requires.*globus-compute-endpoint"
        ):
            endpoint.show()

    @patch("chiltepin.endpoint.platform.system", return_value="Linux")
    def test_start_not_available(self, mock_system):
        """Test start raises ImportError when endpoint library not available."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = False
        endpoint._ENDPOINT_IMPORT_ERROR = ImportError("test import error")
        with pytest.raises(
            ImportError, match="Endpoint management requires.*globus-compute-endpoint"
        ):
            endpoint.start("test_endpoint")

    @patch("chiltepin.endpoint.platform.system", return_value="Linux")
    def test_stop_not_available(self, mock_system):
        """Test stop raises ImportError when endpoint library not available."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = False
        endpoint._ENDPOINT_IMPORT_ERROR = ImportError("test import error")
        with pytest.raises(
            ImportError, match="Endpoint management requires.*globus-compute-endpoint"
        ):
            endpoint.stop("test_endpoint")

    @patch("chiltepin.endpoint.platform.system", return_value="Linux")
    def test_delete_not_available(self, mock_system):
        """Test delete raises ImportError when endpoint library not available."""
        endpoint.ENDPOINT_MANAGEMENT_AVAILABLE = False
        endpoint._ENDPOINT_IMPORT_ERROR = ImportError("test import error")
        with pytest.raises(
            ImportError, match="Endpoint management requires.*globus-compute-endpoint"
        ):
            endpoint.delete("test_endpoint")
