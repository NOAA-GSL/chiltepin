# SPDX-License-Identifier: Apache-2.0

"""Tests for WPSAgent."""

import pytest

from agents import WPSAgent


class TestWPSAgent:
    """Test suite for WPSAgent."""

    def test_init(self, mock_install_dir):
        """Test WPSAgent initialization."""
        agent = WPSAgent(install_dir=str(mock_install_dir))

        assert agent.install_dir == mock_install_dir
        assert agent.version == "4.5"  # Default version
        assert agent.source_dir is None
        assert agent.ungrib_path is None

    def test_init_custom_version(self, mock_install_dir):
        """Test WPSAgent initialization with custom version."""
        agent = WPSAgent(install_dir=str(mock_install_dir), version="4.6")

        assert agent.version == "4.6"

    # TODO: Add tests for agent actions
    # - test_download_wps()
    # - test_build()
    # - test_run_ungrib()
    # - test_run_ungrib_raises_without_build()
