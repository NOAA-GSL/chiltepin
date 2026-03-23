# SPDX-License-Identifier: Apache-2.0

"""Tests for chiltepin.workflow module.

This test suite validates that workflow context managers work correctly
for managing Parsl lifecycle without requiring users to directly interact
with Parsl.
"""

import pathlib
import warnings
from unittest import mock

import parsl
import parsl.errors
import pytest
import yaml

import chiltepin.configure
from chiltepin import Workflow
from chiltepin.tasks import python_task


# Helper function to add PYTHONPATH to config
def add_pythonpath_to_config(config_dict, resource_name):
    """Add project root to PYTHONPATH for a specific resource."""
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    if resource_name in config_dict:
        env = config_dict[resource_name].get("environment", [])
        # Make a copy to avoid modifying shared references
        env = env.copy() if env else []
        env.append(f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}")
        config_dict[resource_name]["environment"] = env
    return config_dict


def _cleanup_parsl_state(phase, ignore_already_cleaned=False):
    """Helper function to cleanup Parsl DFK state.

    Args:
        phase: Description of when cleanup is happening (e.g., "pre-test", "post-test")
        ignore_already_cleaned: If True, suppress warnings about already cleaned DFK
    """
    try:
        dfk = parsl.dfk()
        if dfk:
            try:
                dfk.cleanup()
            except Exception as e:
                # Don't warn about double-cleanup attempts if requested
                if ignore_already_cleaned and "already been cleaned-up" in str(e):
                    return
                warnings.warn(
                    f"DFK cleanup failed in {phase} fixture cleanup: {e}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            try:
                parsl.clear()
            except Exception as e:
                warnings.warn(
                    f"parsl.clear() failed in {phase} fixture cleanup: {e}",
                    RuntimeWarning,
                    stacklevel=3,
                )
    except parsl.errors.NoDataFlowKernelError:
        # Expected when no DFK loaded
        pass
    except Exception as e:
        # Unexpected errors
        warnings.warn(
            f"Unexpected error in {phase} cleanup: {e}",
            RuntimeWarning,
            stacklevel=3,
        )


# Cleanup any existing Parsl state before tests
@pytest.fixture(scope="function", autouse=True)
def cleanup_parsl():
    """Ensure Parsl is cleaned up before and after each test."""
    # Cleanup before test
    _cleanup_parsl_state("pre-test")

    yield

    # Cleanup after test
    _cleanup_parsl_state("post-test", ignore_already_cleaned=True)


class TestWorkflowContextManager:
    """Test Workflow context manager with different configurations."""

    def test_workflow_with_dict_config(self, tmp_path):
        """Test workflow context manager with a dictionary config."""

        @python_task
        def add_numbers(a, b):
            """Simple task for testing."""
            return a + b

        @python_task
        def multiply(x, y):
            """Another simple task for testing."""
            return x * y

        # Get project root for PYTHONPATH
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-executor": {
                "provider": "localhost",
                "cores_per_node": 2,
                "max_workers_per_node": 2,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        with Workflow(config, run_dir=str(tmp_path / "runinfo1")):
            # Submit tasks
            future1 = add_numbers(10, 32, executor=["test-executor"])
            future2 = multiply(6, 7, executor=["test-executor"])

            # Get results
            result1 = future1.result()
            result2 = future2.result()

            assert result1 == 42
            assert result2 == 42

    def test_workflow_with_file_config(self, config_file, tmp_path):
        """Test workflow context manager with a YAML config file."""

        @python_task
        def greet(name):
            """Task that returns a greeting."""
            return f"Hello, {name}!"

        # Load config from file and add PYTHONPATH dynamically
        config_dict = chiltepin.configure.parse_file(config_file)
        add_pythonpath_to_config(config_dict, "service")

        # Write modified config to temp file
        temp_config_file = tmp_path / "temp_config.yaml"
        with open(temp_config_file, "w") as f:
            yaml.dump(config_dict, f)

        # Test with file path argument
        with Workflow(
            str(temp_config_file),
            include=["service"],
            run_dir=str(tmp_path / "runinfo2"),
        ):
            future = greet("Workflow", executor=["service"])
            result = future.result()
            assert result == "Hello, Workflow!"


class TestWorkflowConfigTypes:
    """Test Workflow with dict and file configuration arguments."""

    def test_workflow_from_dict(self, tmp_path):
        """Test Workflow with a dictionary configuration."""

        @python_task
        def add_numbers(a, b):
            return a + b

        # Get project root for PYTHONPATH
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "my-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        with Workflow(config, run_dir=str(tmp_path / "runinfo5")):
            future = add_numbers(100, 200, executor=["my-executor"])
            result = future.result()
            assert result == 300

    def test_workflow_from_file(self, config_file, tmp_path):
        """Test Workflow with a file path configuration."""

        @python_task
        def multiply(x, y):
            return x * y

        # Load config from file and add PYTHONPATH dynamically
        config_dict = chiltepin.configure.parse_file(config_file)
        add_pythonpath_to_config(config_dict, "service")

        # Write modified config to temp file
        temp_config_file = tmp_path / "temp_config.yaml"
        with open(temp_config_file, "w") as f:
            yaml.dump(config_dict, f)

        # Test with file path argument
        with Workflow(
            str(temp_config_file),
            include=["service"],
            run_dir=str(tmp_path / "runinfo6"),
        ):
            future = multiply(21, 2, executor=["service"])
            result = future.result()
            assert result == 42


class TestWorkflowCleanup:
    """Test that workflow context managers properly cleanup resources."""

    def test_sequential_workflows(self, tmp_path):
        """Test that multiple workflows can be created sequentially."""

        @python_task
        def add_numbers(a, b):
            return a + b

        @python_task
        def multiply(x, y):
            return x * y

        # Get project root for PYTHONPATH
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config1 = {
            "exec1": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        config2 = {
            "exec2": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # First workflow
        with Workflow(config1, run_dir=str(tmp_path / "runinfo7")):
            future = add_numbers(1, 2, executor=["exec1"])
            assert future.result() == 3

        # Second workflow should work without conflicts
        with Workflow(config2, run_dir=str(tmp_path / "runinfo8")):
            future = multiply(3, 4, executor=["exec2"])
            assert future.result() == 12

    def test_workflow_cleanup_on_exception(self, tmp_path):
        """Test that workflow cleans up even if an exception occurs."""

        @python_task
        def add_numbers(a, b):
            return a + b

        # Get project root for PYTHONPATH
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Workflow should cleanup even if exception occurs
        with pytest.raises(ValueError):
            with Workflow(config, run_dir=str(tmp_path / "runinfo9")):
                future = add_numbers(5, 5, executor=["test-exec"])
                result = future.result()
                assert result == 10
                raise ValueError("Intentional test error")

        # Should be able to create another workflow after exception
        with Workflow(config, run_dir=str(tmp_path / "runinfo10")):
            future = add_numbers(7, 8, executor=["test-exec"])
            assert future.result() == 15


class TestWorkflowLifecycle:
    """Test workflow lifecycle methods."""

    def test_start_called_twice_raises_error(self, tmp_path):
        """Test that calling start() twice without cleanup raises RuntimeError."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo_lifecycle1"))
        try:
            workflow.start()

            # Attempting to start again should raise RuntimeError
            with pytest.raises(RuntimeError, match="Workflow already started"):
                workflow.start()
        finally:
            # Clean up
            workflow.cleanup()

    def test_start_failure_with_cleanup_failure(self, tmp_path):
        """Test that startup exception is raised even if cleanup also fails."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Mock parsl.load to fail during startup
        # Also mock parsl.clear to fail during cleanup
        with mock.patch("parsl.load") as mock_load:
            with mock.patch("parsl.clear") as mock_clear:
                mock_load.side_effect = RuntimeError("Load failed")
                mock_clear.side_effect = RuntimeError("Clear failed")

                # The startup exception should be raised, not the cleanup exception
                with pytest.raises(RuntimeError, match="Load failed"):
                    with Workflow(config, run_dir=str(tmp_path / "runinfo_lifecycle2")):
                        pass

    def test_explicit_start_cleans_up_on_failure(self, tmp_path):
        """Test that explicit start() cleans up partial state on failure."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Mock parsl.load to fail after logger is set up
        with mock.patch("parsl.load") as mock_load:
            mock_load.side_effect = RuntimeError("Load failed")

            workflow = Workflow(
                config,
                run_dir=str(tmp_path / "runinfo_lifecycle3"),
                log_file=str(tmp_path / "test.log"),
            )

            # Verify start() raises and cleans up partial state
            with pytest.raises(RuntimeError, match="Load failed"):
                workflow.start()

            # Verify state was reset - logger_handler should be None
            assert workflow.logger_handler is None
            assert workflow.dfk is None

    def test_cleanup_preserves_state_on_failure(self, tmp_path):
        """Test that cleanup preserves state when it fails for debugging."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo_lifecycle4"))
        workflow.start()
        dfk_ref = workflow.dfk

        # Mock cleanup to fail
        real_cleanup = parsl.DataFlowKernel.cleanup
        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            mock_cleanup.side_effect = RuntimeError("Cleanup failed")

            # Cleanup should raise exception
            with pytest.raises(RuntimeError, match="Cleanup failed"):
                workflow.cleanup()

            # State should be preserved when cleanup fails (not reset to None)
            assert workflow.dfk is not None
            assert workflow.dfk is dfk_ref

        # Now actually clean up for real
        try:
            real_cleanup(dfk_ref)
            parsl.clear()
        except Exception:
            pass

    def test_cleanup_preserves_state_when_suppressing_exceptions(self, tmp_path):
        """Test that cleanup preserves state even when suppressing exceptions."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo_lifecycle5"))
        workflow.start()
        dfk_ref = workflow.dfk

        # Mock cleanup to fail
        real_cleanup = parsl.DataFlowKernel.cleanup
        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            mock_cleanup.side_effect = RuntimeError("Cleanup failed")

            # Cleanup with suppress_exceptions=True should not raise
            workflow.cleanup(suppress_exceptions=True)

            # State should still be preserved when cleanup fails
            # even though we suppressed the exception
            assert workflow.dfk is not None
            assert workflow.dfk is dfk_ref

        # Now actually clean up for real
        try:
            real_cleanup(dfk_ref)
            parsl.clear()
        except Exception:
            pass


class TestWorkflowExceptionHandling:
    """Test exception handling during workflow cleanup."""

    def test_dfk_cleanup_exception(self, tmp_path):
        """Test that exceptions during dfk.cleanup() are properly raised."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            mock_cleanup.side_effect = RuntimeError("Cleanup failed")

            with pytest.raises(RuntimeError, match="Cleanup failed"):
                with Workflow(config, run_dir=str(tmp_path / "runinfo_exc1")):
                    dfk_ref = parsl.dfk()  # Capture dfk reference
                    pass

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_parsl_clear_exception(self, tmp_path):
        """Test that exceptions during parsl.clear() are properly raised."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        with mock.patch("parsl.clear") as mock_clear:
            mock_clear.side_effect = RuntimeError("Clear failed")

            with pytest.raises(RuntimeError, match="Clear failed"):
                with Workflow(config, run_dir=str(tmp_path / "runinfo_exc2")):
                    pass

    def test_logger_handler_exception(self, tmp_path):
        """Test that exceptions during logger_handler() are properly raised.

        Note: Logger cleanup happens both inside dfk.cleanup() and in our explicit
        logger_handler() call. When it fails during dfk.cleanup(), that exception
        is what gets raised.
        """
        # Ensure clean state before test
        try:
            parsl.clear()
        except Exception:
            pass

        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Mock the logger handler to raise an exception
        def mock_set_file_logger(*args, **kwargs):
            def failing_handler():
                raise RuntimeError("Logger cleanup failed")

            return failing_handler

        with mock.patch("parsl.set_file_logger", side_effect=mock_set_file_logger):
            # The logger cleanup exception will be raised (either from dfk.cleanup or our call)
            with pytest.raises(RuntimeError):
                with Workflow(
                    config,
                    run_dir=str(tmp_path / "runinfo_exc3"),
                    log_file=str(tmp_path / "test.log"),
                ):
                    pass

    def test_chained_exceptions_cleanup_then_clear(self, tmp_path):
        """Test exception chaining when dfk.cleanup() and parsl.clear() both fail."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            with mock.patch("parsl.clear") as mock_clear:
                mock_cleanup.side_effect = RuntimeError("Cleanup failed")
                mock_clear.side_effect = RuntimeError("Clear failed")

                with pytest.raises(RuntimeError) as exc_info:
                    with Workflow(config, run_dir=str(tmp_path / "runinfo_exc4")):
                        dfk_ref = parsl.dfk()  # Capture dfk reference
                        pass

                # The last exception (clear) should be raised
                assert "Clear failed" in str(exc_info.value)
                # And the previous exception (cleanup) should be in the chain
                assert exc_info.value.__cause__ is not None
                assert "Cleanup failed" in str(exc_info.value.__cause__)

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_chained_exceptions_all_three(self, tmp_path):
        """Test exception chaining when all three cleanup operations fail."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        # Mock the logger handler to raise an exception
        def mock_set_file_logger(*args, **kwargs):
            def failing_handler():
                raise RuntimeError("Logger cleanup failed")

            return failing_handler

        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            with mock.patch("parsl.clear") as mock_clear:
                with mock.patch(
                    "parsl.set_file_logger", side_effect=mock_set_file_logger
                ):
                    mock_cleanup.side_effect = RuntimeError("Cleanup failed")
                    mock_clear.side_effect = RuntimeError("Clear failed")

                    with pytest.raises(RuntimeError) as exc_info:
                        with Workflow(
                            config,
                            run_dir=str(tmp_path / "runinfo_exc5"),
                            log_file=str(tmp_path / "test.log"),
                        ):
                            dfk_ref = parsl.dfk()  # Capture dfk reference
                            pass

                    # The last exception (logger) should be raised
                    assert "Logger cleanup failed" in str(exc_info.value)
                    # Check the exception chain
                    assert exc_info.value.__cause__ is not None
                    assert "Clear failed" in str(exc_info.value.__cause__)
                    assert exc_info.value.__cause__.__cause__ is not None
                    assert "Cleanup failed" in str(exc_info.value.__cause__.__cause__)

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_parsl_clear_called_when_dfk_is_none(self, tmp_path):
        """Test that parsl.clear() is called even when dfk is None."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Mock parsl.load to fail, so dfk stays None
        with mock.patch("parsl.load") as mock_load:
            with mock.patch("parsl.clear") as mock_clear:
                mock_load.side_effect = RuntimeError("Load failed")

                with pytest.raises(RuntimeError, match="Load failed"):
                    with Workflow(config, run_dir=str(tmp_path / "runinfo_exc6")):
                        pass

                # parsl.clear() should still be called even though dfk is None
                mock_clear.assert_called_once()

    def test_parsl_clear_exception_without_cleanup_exception(self, tmp_path):
        """Test parsl.clear() exception when dfk.cleanup() succeeds."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # This tests the branch where parsl.clear() fails but dfk.cleanup() succeeds
        # Explicitly ensure dfk.cleanup() succeeds so cleanup_exception stays None
        original_clear = parsl.clear
        dfk_ref = None

        def mock_cleanup_success(self):
            """Mock dfk.cleanup() to succeed without doing anything."""
            # Don't actually clean up - we'll do it manually after

        call_count = [0]

        def mock_clear_selective():
            call_count[0] += 1
            if call_count[0] == 2:
                # Second call (from workflow) - make it fail
                raise RuntimeError("Clear failed")
            else:
                # First and subsequent calls should succeed
                try:
                    original_clear()
                except Exception:
                    pass

        with mock.patch.object(parsl.DataFlowKernel, "cleanup", mock_cleanup_success):
            with mock.patch("parsl.clear", side_effect=mock_clear_selective):
                # Explicitly make the first call to parsl.clear() to "use up" count==1
                try:
                    parsl.clear()
                except Exception:
                    pass

                # Now the workflow's call will be count==2 and will raise
                with pytest.raises(RuntimeError, match="Clear failed"):
                    with Workflow(config, run_dir=str(tmp_path / "runinfo_exc7")):
                        dfk_ref = parsl.dfk()  # Capture for manual cleanup
                        pass

        # Manually clean up since we mocked cleanup
        if dfk_ref:
            try:
                parsl.DataFlowKernel.cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass


@pytest.fixture
def config_file_fixture(tmp_path):
    """Create a temporary config file for testing file-based workflows."""
    import yaml

    project_root = pathlib.Path(__file__).parent.parent.resolve()

    config = {
        "service": {
            "provider": "localhost",
            "cores_per_node": 1,
            "max_workers_per_node": 1,
            "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
        }
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return str(config_file)


class TestWorkflowFromFileCoverage:
    """Tests for Workflow with file path configuration."""

    def test_workflow_from_file_with_local_fixture(self, config_file_fixture, tmp_path):
        """Test Workflow with a config file fixture."""

        @python_task
        def simple_task():
            return "success"

        with Workflow(
            config_file_fixture,
            include=["service"],
            run_dir=str(tmp_path / "runinfo_file"),
        ):
            future = simple_task(executor=["service"])
            result = future.result()
            assert result == "success"


class TestUserExceptionPrecedence:
    """Test that user exceptions are not masked by cleanup exceptions."""

    def test_user_exception_not_masked_by_cleanup_exception(self, tmp_path, caplog):
        """Test that user exceptions take precedence over cleanup exceptions."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        # Mock dfk.cleanup to fail
        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            mock_cleanup.side_effect = RuntimeError("Cleanup failed")

            # User exception should be raised, not cleanup exception
            with pytest.raises(ValueError, match="User error"):
                with Workflow(config, run_dir=str(tmp_path / "runinfo_user1")):
                    dfk_ref = parsl.dfk()  # Capture dfk reference
                    raise ValueError("User error")

            # Cleanup exception should be logged as a warning
            assert any(
                "Exception during dfk.cleanup() (suppressing)" in record.getMessage()
                for record in caplog.records
            )

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_all_cleanup_operations_attempted(self, tmp_path, caplog):
        """Test that all cleanup operations are attempted even when some fail."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        # Mock dfk.cleanup and parsl.clear to fail
        original_clear = parsl.clear
        call_count = [0]

        def mock_clear_selective():
            call_count[0] += 1
            # Fail on first call (our explicit call after dfk.cleanup fails)
            if call_count[0] == 1:
                raise RuntimeError("Clear failed")
            else:
                # Subsequent calls use original
                original_clear()

        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            with mock.patch("parsl.clear", side_effect=mock_clear_selective):
                mock_cleanup.side_effect = RuntimeError("Cleanup failed")

                # User exception should be raised, not any cleanup exception
                with pytest.raises(ValueError, match="User error"):
                    with Workflow(config, run_dir=str(tmp_path / "runinfo_user2")):
                        dfk_ref = parsl.dfk()  # Capture dfk reference
                        raise ValueError("User error")

                # Both cleanup exceptions should be logged as warnings
                assert any(
                    "Exception during dfk.cleanup() (suppressing)"
                    in record.getMessage()
                    for record in caplog.records
                )
                assert any(
                    "Exception during parsl.clear() (suppressing)"
                    in record.getMessage()
                    for record in caplog.records
                )

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_cleanup_exception_raised_when_no_user_exception(self, tmp_path):
        """Test that cleanup exceptions are raised when there's no user exception."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Store reference to real cleanup for later
        real_cleanup = parsl.DataFlowKernel.cleanup
        dfk_ref = None

        # Mock dfk.cleanup to fail
        with mock.patch("parsl.DataFlowKernel.cleanup") as mock_cleanup:
            mock_cleanup.side_effect = RuntimeError("Cleanup failed")

            # Cleanup exception should be raised when there's no user exception
            with pytest.raises(RuntimeError, match="Cleanup failed"):
                with Workflow(config, run_dir=str(tmp_path / "runinfo_user3")):
                    dfk_ref = parsl.dfk()  # Capture dfk reference
                    pass  # No user exception

        # Actually clean up the DFK now that we're done testing
        if dfk_ref:
            try:
                real_cleanup(dfk_ref)
                parsl.clear()
            except Exception:
                pass

    def test_user_exception_with_logger_cleanup_failure(self, tmp_path, caplog):
        """Test that user exceptions take precedence when logger cleanup fails."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        # Mock the logger handler to raise an exception
        def mock_set_file_logger(*args, **kwargs):
            def failing_handler():
                raise RuntimeError("Logger cleanup failed")

            return failing_handler

        with mock.patch("parsl.set_file_logger", side_effect=mock_set_file_logger):
            # User exception should be raised, not logger cleanup exception
            with pytest.raises(ValueError, match="User error"):
                with Workflow(
                    config,
                    run_dir=str(tmp_path / "runinfo_user4"),
                    log_file=str(tmp_path / "test.log"),
                ):
                    raise ValueError("User error")

            # Logger cleanup exception should be logged as a warning
            assert any(
                "Exception during logger cleanup (suppressing)" in record.getMessage()
                for record in caplog.records
            )

    def test_logger_handler_exception_standalone(self, tmp_path):
        """Test logger cleanup failure when dfk.cleanup and parsl.clear succeed."""
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-exec": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        #  Create a handler that succeeds first time (dfk cleanup) but fails second time (our explicit call)
        call_count = [0]

        def mock_set_file_logger(*args, **kwargs):
            def conditional_failing_handler():
                call_count[0] += 1
                if call_count[0] == 2:  # Fail on second call (our explicit cleanup)
                    raise RuntimeError("Logger cleanup failed")

            return conditional_failing_handler

        with mock.patch("parsl.set_file_logger", side_effect=mock_set_file_logger):
            # Logger cleanup exception should be raised
            with pytest.raises(RuntimeError, match="Logger cleanup failed"):
                with Workflow(
                    config,
                    run_dir=str(tmp_path / "runinfo_logger_standalone"),
                    log_file=str(tmp_path / "test.log"),
                ):
                    pass
