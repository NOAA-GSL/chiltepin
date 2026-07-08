# SPDX-License-Identifier: Apache-2.0

"""Tests for MPASAgent."""

import pytest

from agents import MPASAgent


class TestMPASAgent:
    """Test suite for MPASAgent."""

    def test_init(self, mock_install_dir):
        """Test MPASAgent initialization."""
        agent = MPASAgent(install_dir=str(mock_install_dir))

        assert agent.install_dir == mock_install_dir
        assert agent.version == "v8.2.0"  # Default version
        assert agent.source_dir is None
        assert agent.init_atmosphere_path is None
        assert agent.atmosphere_model_path is None

    def test_init_custom_version(self, mock_install_dir):
        """Test MPASAgent initialization with custom version."""
        agent = MPASAgent(install_dir=str(mock_install_dir), version="v8.1.0")

        assert agent.version == "v8.1.0"

    # TODO: Add tests for agent actions
    # - test_download_mpas()
    # - test_build()
    # - test_initialize_ics()
    # - test_initialize_lbcs()
    # - test_run_forecast()
    # - test_actions_raise_without_build()
