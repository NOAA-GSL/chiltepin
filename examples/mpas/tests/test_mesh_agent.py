# SPDX-License-Identifier: Apache-2.0

"""Tests for MeshAgent."""

import pytest

from agents import MeshAgent


class TestMeshAgent:
    """Test suite for MeshAgent."""

    def test_init(self, mock_install_dir):
        """Test MeshAgent initialization with defaults."""
        agent = MeshAgent(install_dir=str(mock_install_dir))
        behavior = agent._behavior

        assert behavior.install_dir == mock_install_dir
        assert behavior.metis_tag == "5.2.1"
        assert behavior.limited_area_version == "master"
        assert behavior.log_dir == mock_install_dir / "logs"
        assert behavior.metis_downloaded is False
        assert behavior.metis_built is False
        assert behavior.metis_source_dir is None
        assert behavior.gpmetis_path is None
        assert behavior.limited_area_installed is False
        assert behavior.limited_area_source_dir is None
        assert behavior.create_region_path is None
        assert behavior.mesh_data_dir == mock_install_dir / "mesh_data"

    def test_init_custom_versions(self, mock_install_dir):
        """Test MeshAgent initialization with custom versions."""
        agent = MeshAgent(
            install_dir=str(mock_install_dir),
            metis_tag="5.2.0",
            limited_area_version="v1.0",
        )
        behavior = agent._behavior

        assert behavior.metis_tag == "5.2.0"
        assert behavior.limited_area_version == "v1.0"

    # TODO: Add tests for agent actions
    # These tests will require mocking the actual download, build, and partition
    # operations or setting up test fixtures with sample data
    #
    # Example tests to add:
    # - test_install_metis()
    # - test_install_limited_area()
    # - test_install()
    # - test_download_global_mesh()
    # - test_create_regional_mesh()
    # - test_create_regional_mesh_raises_without_install()
    # - test_partition_mesh()
    # - test_partition_mesh_raises_without_install()
