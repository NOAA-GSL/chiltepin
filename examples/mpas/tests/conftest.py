# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for MPAS example tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs.

    Yields
    ------
    Path
        Path to temporary directory

    Notes
    -----
    Directory is automatically cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_install_dir(temp_dir):
    """Create a mock installation directory structure.

    Parameters
    ----------
    temp_dir : Path
        Temporary directory from temp_dir fixture

    Returns
    -------
    Path
        Path to mock installation directory
    """
    install_dir = temp_dir / "installs"
    install_dir.mkdir(parents=True, exist_ok=True)
    return install_dir


@pytest.fixture
def mock_experiment_dir(temp_dir):
    """Create a mock experiment directory structure.

    Parameters
    ----------
    temp_dir : Path
        Temporary directory from temp_dir fixture

    Returns
    -------
    dict
        Dictionary of experiment directory paths
    """
    exp_dir = temp_dir / "experiment"
    paths = {
        "experiment_dir": exp_dir,
        "mesh_dir": exp_dir / "mesh",
        "grib_data_dir": exp_dir / "grib_data",
        "ungrib_dir": exp_dir / "ungrib_output",
        "init_dir": exp_dir / "initialization",
        "forecast_dir": exp_dir / "forecast",
        "runinfo": exp_dir / "runinfo",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


@pytest.fixture
def test_config(mock_experiment_dir):
    """Create a test configuration dictionary.

    Parameters
    ----------
    mock_experiment_dir : dict
        Experiment directory paths from mock_experiment_dir fixture

    Returns
    -------
    dict
        Test configuration dictionary
    """
    return {
        "platform": "test",
        "workflow": {
            "manager-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            },
            "build-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            },
            "compute-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            },
        },
        "agent_workflow": {
            "build-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            },
            "compute-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            },
        },
        "manager_executors": ["manager-executor"],
        "build_executors": ["build-executor"],
        "compute_executors": ["compute-executor"],
        "model": {
            "resolution": "120km",
            "region": "conus",
            "init_ranks": 4,
            "forecast_ranks": 8,
            "forecast_length_hours": 6,
            "output_interval_hours": 1,
        },
        "paths": {
            str(k): str(v) for k, v in mock_experiment_dir.items()
        },
        "versions": {
            "metis": "5.1.0",
            "wps": "4.5",
            "mpas_limited_area": "master",
            "mpas": "v8.2.0",
        },
    }


@pytest.fixture
def sample_mesh_files(mock_experiment_dir):
    """Create sample mesh files for testing.

    Parameters
    ----------
    mock_experiment_dir : dict
        Experiment directory paths

    Returns
    -------
    dict
        Dictionary of sample mesh file paths
    """
    mesh_dir = Path(mock_experiment_dir["mesh_dir"])

    # Create dummy files (empty for now)
    static_file = mesh_dir / "static.nc"
    graph_file = mesh_dir / "graph.info"

    static_file.touch()
    graph_file.touch()

    return {
        "static": str(static_file),
        "graph": str(graph_file),
    }
