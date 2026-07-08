# SPDX-License-Identifier: Apache-2.0

"""Tests for MPASLimitedAreaAgent."""

import pytest

from agents import MPASLimitedAreaAgent


class TestMPASLimitedAreaAgent:
    """Test suite for MPASLimitedAreaAgent."""

    def test_init(self, mock_install_dir):
        """Test MPASLimitedAreaAgent initialization."""
        agent = MPASLimitedAreaAgent(install_dir=str(mock_install_dir))

        assert agent.install_dir == mock_install_dir
        assert agent.version == "master"  # Default version
        assert agent.source_dir is None
        assert agent.create_region_path is None
        assert agent.mesh_data_dir == mock_install_dir / "mesh_data"

    def test_init_custom_version(self, mock_install_dir):
        """Test MPASLimitedAreaAgent initialization with custom version."""
        agent = MPASLimitedAreaAgent(install_dir=str(mock_install_dir), version="v1.0")

        assert agent.version == "v1.0"

    # TODO: Add tests for agent actions
    # - test_download()
    # - test_build()
    # - test_download_global_mesh()
    # - test_create_conus_mesh()
    # - test_create_conus_mesh_raises_without_build()
