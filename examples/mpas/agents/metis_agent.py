# SPDX-License-Identifier: Apache-2.0

"""MetisAgent - Manages Metis graph partitioning library.

This agent handles downloading, building, and running the Metis graph
partitioning library (gpmetis utility) used to partition MPAS mesh files
for MPI parallel execution.
"""

from pathlib import Path
from typing import Optional

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task


@chiltepin_agent(agent_workflow_include=["build-executor"])
class MetisAgent:
    """Agent for managing Metis graph partitioning library.

    This agent manages the complete lifecycle of Metis:
    - Downloads Metis source code from GitHub
    - Builds the gpmetis utility
    - Partitions MPAS mesh files for specified MPI rank counts

    Attributes
    ----------
    install_dir : str
        Base directory for Metis installation
    version : str
        Metis version to install
    source_dir : Path
        Path to Metis source directory (set after download)
    gpmetis_path : Path
        Path to gpmetis executable (set after build)
    """

    def __init__(self, install_dir: str, version: str = "5.1.0"):
        """Initialize MetisAgent.

        Parameters
        ----------
        install_dir : str
            Directory where Metis will be installed
        version : str, optional
            Metis version to install, by default "5.1.0"
        """
        self.install_dir = Path(install_dir)
        self.version = version
        self.source_dir: Optional[Path] = None
        self.gpmetis_path: Optional[Path] = None

    @bash_task
    @agent_action
    async def download_metis(self) -> str:
        """Download Metis source code from GitHub.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads Metis from KarypisLab/METIS GitHub repository.
        """
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.install_dir / f"metis-{self.version}"

        # TODO: Implement download logic
        # Reference: Old MPAS app install_metis step
        # - Clone or download tarball from GitHub
        # - Extract to self.source_dir
        # - Return path as string

        return str(self.source_dir)

    @bash_task
    @agent_action
    async def build(self) -> str:
        """Build gpmetis utility.

        Returns
        -------
        str
            Path to gpmetis executable

        Raises
        ------
        RuntimeError
            If download_metis has not been called first

        Notes
        -----
        Builds Metis using make:
        - Configures build with cmake
        - Compiles gpmetis executable
        - Sets self.gpmetis_path
        """
        if self.source_dir is None:
            raise RuntimeError("Must call download_metis() before build()")

        # TODO: Implement build logic
        # Reference: Old MPAS app install_metis step
        # - Run make config
        # - Run make
        # - Set self.gpmetis_path to build/programs/gpmetis
        # - Return path as string

        self.gpmetis_path = self.source_dir / "build" / "programs" / "gpmetis"
        return str(self.gpmetis_path)

    @bash_task
    @agent_action
    async def partition_mesh(self, mesh_path: str, num_ranks: int) -> str:
        """Partition MPAS mesh file for MPI execution.

        Parameters
        ----------
        mesh_path : str
            Path to MPAS mesh file (graph.info file)
        num_ranks : int
            Number of MPI ranks to partition for

        Returns
        -------
        str
            Path to partitioned mesh file (graph.info.part.{num_ranks})

        Raises
        ------
        RuntimeError
            If build has not been called first
        FileNotFoundError
            If mesh file does not exist

        Notes
        -----
        Runs gpmetis to create partitioned mesh:
        - Input: graph.info file from MPAS mesh
        - Output: graph.info.part.{num_ranks} file
        """
        if self.gpmetis_path is None:
            raise RuntimeError("Must call build() before partition_mesh()")

        mesh_file = Path(mesh_path)
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        output_file = f"{mesh_path}.part.{num_ranks}"

        # TODO: Implement partition logic
        # Reference: Old MPAS app gpmetis step
        # - Run: gpmetis {mesh_path} {num_ranks}
        # - Verify output file created
        # - Return output path

        return output_file
