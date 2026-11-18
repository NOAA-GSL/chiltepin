# SPDX-License-Identifier: Apache-2.0

"""Real Metis installation and mesh partitioning through the MPAS mesh agent.

Runs only with ``--config=<platform config>``. The config's ``service`` resource
needs git, wget and tar plus internet access to GitHub and UCAR, and its
``compute`` resource needs cmake, make and a C compiler.

This lives in its own module because Parsl allows only one loaded configuration
at a time, and the mocked tests in ``test_mpas_mesh.py`` keep one loaded.
"""

import asyncio
from pathlib import Path

import pytest

# Number of cells in the 480km global mesh
CELLS_480KM = 2562

# Metis version the tests install, passed to the agent so the test and the agent agree
METIS_VERSION = "5.2.1"


def assert_valid_partitions(result, ranks):
    """Check partition files for a 480km mesh from a ``generate_mesh`` result."""
    assert set(result) == {"static", "graph", "partitions"}
    assert set(result["partitions"]) == set(ranks)
    for rank, path in result["partitions"].items():
        assert path == f"{result['graph']}.part.{rank}"
        # One partition number per mesh cell, using every requested part
        parts = [int(value) for value in Path(path).read_text().split()]
        assert len(parts) == CELLS_480KM
        assert set(parts) == set(range(rank))


@pytest.mark.asyncio
async def test_real_metis_install_and_partition(mesh_test_directory, launch_mesh_agent):
    """Install Metis, partition real meshes, and reuse it after a restart."""
    directory = mesh_test_directory
    mesh_data_dir = str(directory / "meshes")
    work = directory / "work"
    gpmetis = work / "metis" / METIS_VERSION / "bin" / "gpmetis"
    download_log = work / "logs" / "metis_download.stdout"

    async with launch_mesh_agent("first", metis_version=METIS_VERSION) as agent:
        # Concurrent requests both get a working installation and use separate logs
        first, second = await asyncio.gather(
            agent.generate_mesh(
                {"resolution": "480km", "name": "p4", "forecast_ranks": 4},
                mesh_data_dir,
            ),
            agent.generate_mesh(
                {
                    "resolution": "480km",
                    "name": "p4-p8",
                    "init_ranks": 4,
                    "forecast_ranks": 8,
                },
                mesh_data_dir,
            ),
        )
        assert_valid_partitions(first, {4})
        assert_valid_partitions(second, {4, 8})
        assert gpmetis.is_file()
        assert (gpmetis.parent.parent / ".installed").is_file()
        # The downloaded sources are removed once the build has finished
        assert not (work / "metis" / "build" / METIS_VERSION).exists()

        # The lock that makes these concurrent requests share one install is
        # covered by test_concurrent_installs_run_once, because the logs of one
        # install cannot be told apart from those of two.
        # An explicit install and a later request do not change the installation
        installed = (gpmetis.stat().st_mtime_ns, download_log.stat().st_mtime_ns)
        await agent.install_metis()
        third = await agent.generate_mesh(
            {"resolution": "480km", "name": "p2", "forecast_ranks": 2}, mesh_data_dir
        )
        assert_valid_partitions(third, {2})
        assert (
            gpmetis.stat().st_mtime_ns,
            download_log.stat().st_mtime_ns,
        ) == installed

    # A new agent has no memory of the install, and finds Metis on disk
    async with launch_mesh_agent("restart", metis_version=METIS_VERSION) as agent:
        fourth = await agent.generate_mesh(
            {"resolution": "480km", "name": "p3", "forecast_ranks": 3}, mesh_data_dir
        )
        assert_valid_partitions(fourth, {3})
        assert (
            gpmetis.stat().st_mtime_ns,
            download_log.stat().st_mtime_ns,
        ) == installed

    # Each rank count in each partition call wrote its own log, including the two
    # calls that both asked for 4 ranks at the same time
    rank_counts_per_call = [[4], [4, 8], [2], [3]]
    logs = list((work / "logs").glob("partition_mesh.*.stderr"))
    assert len(logs) == sum(len(ranks) for ranks in rank_counts_per_call)
    assert (work / "logs" / "metis_build.stdout").is_file()
