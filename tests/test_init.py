# SPDX-License-Identifier: Apache-2.0

"""Tests for the chiltepin package initialization.

This test suite validates:
1. Version string is available
2. Version fallback to 'dev' when package is not installed
3. Submodule access via __getattr__
"""

import importlib
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest


class TestVersionHandling:
    """Test version retrieval and fallback behavior."""

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        import chiltepin

        assert isinstance(chiltepin.__version__, str)
        assert len(chiltepin.__version__) > 0

    def test_version_fallback_when_package_not_found(self):
        """Test that __version__ falls back to 'dev' when package is not installed."""
        # Import chiltepin first to ensure it exists
        import chiltepin

        # Test 1: Mock PackageNotFoundError - should get 'dev'
        with mock.patch(
            "importlib.metadata.version", side_effect=PackageNotFoundError("chiltepin")
        ):
            importlib.reload(chiltepin)
            assert chiltepin.__version__ == "dev"

        # Test 2: Mock normal version lookup - should get the mocked version
        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            importlib.reload(chiltepin)
            assert chiltepin.__version__ == "1.2.3"

        # Restore to whatever the actual state is
        importlib.reload(chiltepin)

    def test_version_raises_unexpected_errors(self):
        """Test that unexpected errors during version retrieval are not caught."""
        # Import chiltepin first to ensure it exists
        import chiltepin

        # Mock the version function to raise an unexpected error
        with mock.patch(
            "importlib.metadata.version", side_effect=RuntimeError("Unexpected error")
        ):
            # Reload should raise the RuntimeError, not catch it
            with pytest.raises(RuntimeError, match="Unexpected error"):
                importlib.reload(chiltepin)

        # Restore normal state by reloading without the patch
        importlib.reload(chiltepin)


class TestSubmoduleAccess:
    """Test submodule access via __getattr__."""

    def test_lazy_loading_workflow_functions(self):
        """Test that workflow functions are lazily loaded via __getattr__."""
        import chiltepin

        # Access lazy-loaded workflow function
        run_workflow = chiltepin.run_workflow
        assert callable(run_workflow)
        assert run_workflow.__module__ == "chiltepin.workflow"

        # Verify other workflow functions are also available
        assert callable(chiltepin.run_workflow_from_file)
        assert callable(chiltepin.run_workflow_from_dict)

    def test_submodule_import_via_attribute_access(self):
        """Test that submodules can be accessed as attributes."""
        import chiltepin

        # Clean up any pre-loaded submodules to ensure __getattr__ gets called
        # (other tests may have imported them directly, which adds them to the namespace)
        if "tasks" in chiltepin.__dict__:
            delattr(chiltepin, "tasks")
        if "configure" in chiltepin.__dict__:
            delattr(chiltepin, "configure")

        # Access submodule via attribute - should trigger __getattr__
        tasks = chiltepin.tasks
        assert tasks.__name__ == "chiltepin.tasks"

        # Access another submodule - should also trigger __getattr__
        configure = chiltepin.configure
        assert configure.__name__ == "chiltepin.configure"

    def test_submodule_with_broken_dependency_propagates_error(self):
        """Test that ModuleNotFoundError from submodule dependencies is propagated."""
        import chiltepin

        # Mock import_module to raise ModuleNotFoundError for a different module
        # This simulates a submodule that exists but has a missing dependency
        def mock_import_module(name, package):
            # Raise error for a dependency, not the submodule itself
            raise ModuleNotFoundError(
                "No module named 'some_dependency'", name="some_dependency"
            )

        with mock.patch("importlib.import_module", side_effect=mock_import_module):
            # This should propagate the ModuleNotFoundError, not convert to AttributeError
            with pytest.raises(ModuleNotFoundError, match="some_dependency"):
                _ = chiltepin.fake_module

    def test_invalid_attribute_raises_attribute_error(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        import chiltepin

        # Mock import_module to raise ModuleNotFoundError for the actual submodule
        # This simulates trying to access a non-existent submodule
        def mock_import_module(name, package):
            # Raise error for the submodule itself (matching the e.name check)
            # The name will be something like "chiltepin.nonexistent"
            full_name = f"{package}{name}" if name.startswith(".") else name
            raise ModuleNotFoundError(f"No module named '{full_name}'", name=full_name)

        with mock.patch("importlib.import_module", side_effect=mock_import_module):
            # This should raise AttributeError since the submodule doesn't exist
            with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
                _ = chiltepin.nonexistent
