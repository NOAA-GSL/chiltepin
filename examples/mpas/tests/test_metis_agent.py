# SPDX-License-Identifier: Apache-2.0

"""Tests for MetisAgent."""

import pytest

from agents import MetisAgent


class TestMetisAgent:
    """Test suite for MetisAgent."""

    def test_init(self, mock_install_dir):
        """Test MetisAgent initialization."""
        agent = MetisAgent(install_dir=str(mock_install_dir))

        assert agent.install_dir == mock_install_dir
        assert agent.version == "5.1.0"  # Default version
        assert agent.source_dir is None  # Not downloaded yet
        assert agent.gpmetis_path is None  # Not built yet

    def test_init_custom_version(self, mock_install_dir):
        """Test MetisAgent initialization with custom version."""
        agent = MetisAgent(install_dir=str(mock_install_dir), version="5.2.0")

        assert agent.version == "5.2.0"

    # TODO: Add tests for agent actions
    # These tests will require mocking the actual download, build, and partition operations
    # or setting up test fixtures with sample data
    #
    # Example tests to add:
    # - test_download_metis()
    # - test_build()
    # - test_partition_mesh()
    # - test_partition_mesh_raises_without_build()
