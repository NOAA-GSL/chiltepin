# SPDX-License-Identifier: Apache-2.0

"""Tests for the chiltepin package initialization.

This test suite validates:
1. Version string is available
2. Version fallback to 'dev' when package is not installed
"""

import sys
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
        # Mock the version function to raise PackageNotFoundError
        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            # Remove chiltepin from sys.modules to force re-import
            if "chiltepin" in sys.modules:
                del sys.modules["chiltepin"]

            # Import should succeed with fallback version
            import chiltepin

            assert chiltepin.__version__ == "dev"

        # Clean up and restore normal import
        if "chiltepin" in sys.modules:
            del sys.modules["chiltepin"]
        import chiltepin  # noqa: F401

    def test_version_raises_unexpected_errors(self):
        """Test that unexpected errors during version retrieval are not caught."""
        # Mock the version function to raise an unexpected error
        with mock.patch(
            "importlib.metadata.version", side_effect=RuntimeError("Unexpected error")
        ):
            # Remove chiltepin from sys.modules to force re-import
            if "chiltepin" in sys.modules:
                del sys.modules["chiltepin"]

            # Import should raise the RuntimeError, not catch it
            with pytest.raises(RuntimeError, match="Unexpected error"):
                import chiltepin  # noqa: F401

        # Clean up
        if "chiltepin" in sys.modules:
            del sys.modules["chiltepin"]
        import chiltepin  # noqa: F401
