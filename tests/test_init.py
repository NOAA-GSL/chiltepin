# SPDX-License-Identifier: Apache-2.0

"""Tests for the chiltepin package initialization.

This test suite validates:
1. Version string is available
2. Version fallback to 'dev' when package is not installed
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

        # Mock the version function to raise PackageNotFoundError
        with mock.patch(
            "importlib.metadata.version", side_effect=PackageNotFoundError("chiltepin")
        ):
            # Reload the module with the patch applied
            importlib.reload(chiltepin)

            # Should have fallback version
            assert chiltepin.__version__ == "dev"

        # Restore normal state by reloading without the patch
        importlib.reload(chiltepin)

        # Verify it's back to normal
        assert chiltepin.__version__ != "dev"

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
