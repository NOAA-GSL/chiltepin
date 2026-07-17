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


@chiltepin_agent()
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

    def __init__(self, install_dir: str, tag: str = "5.1.0"):
        """Initialize MetisAgent.

        Parameters
        ----------
        install_dir : str
            Directory where Metis will be installed
        tag : str, optional
            Metis version tag to install, by default "5.1.0"
        """
        self.install_dir = Path(install_dir)
        self.tag = tag
        self.source_dir: Optional[Path] = None
        self.gpmetis_path: Optional[Path] = None

    @agent_action
    @bash_task
    def download(self) -> str:
        """Download Metis source code from GitHub.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads Metis from KarypisLab/METIS GitHub repository.
        """
        import textwrap
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.install_dir / f"metis-{self.tag}"

        return textwrap.dedent(
            f"""
            set +e
            echo Started at $(date)
            echo Executing on $(hostname)
            git lfs install --skip-repo
            rm -rf {self.install_dir}/metis/build/{self.tag}
            mkdir -p {self.install_dir}/metis/build/{self.tag}
            cd {self.install_dir}/metis/build/{self.tag}
            # Clone GKlib tools needed by metis
            git clone https://github.com/KarypisLab/GKlib.git
            # Fetch and untar metis tarball
            wget -T 30 -t 3 https://github.com/KarypisLab/METIS/archive/refs/tags/v{self.tag}.tar.gz
            tar -xzf v{self.tag}.tar.gz
            rm -f metis-{self.tag}.tar.gz
            echo Completed at $(date)
            """
        )

    @agent_action
    @bash_task
    def build(self, inputs=None) -> str:
        """Build gpmetis utility.

        Parameters
        ----------
        inputs : list of AppFuture, optional
            List of task futures that must complete before this build starts.
            Used to enforce dependencies (e.g., download must complete first).

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
        import textwrap

        if self.source_dir is None:
            raise RuntimeError("Must call download_metis() before build()")

        return textwrap.dedent(
            f"""
            set +e
            echo Started at $(date)
            echo Executing on $(hostname)
            cd {self.install_dir}/metis/build/{self.tag}/GKlib
            make config prefix={self.install_dir}/metis/{self.tag}
            make install
            cd {self.install_dir}/metis/build/{self.tag}/METIS-{self.tag}
            make config prefix={self.install_dir}/metis/{self.tag}
            make install
            rm -rf {self.install_dir}/metis/build
            echo Completed at $(date)
            """
        )

    @agent_action
    @bash_task
    def partition_mesh(self, mesh_path: str, num_ranks: int, inputs=None) -> str:
        """Partition MPAS mesh file for MPI execution.

        Parameters
        ----------
        mesh_path : str
            Path to MPAS mesh file (graph.info file)
        num_ranks : int
            Number of MPI ranks to partition for
        inputs : list of AppFuture, optional
            List of task futures that must complete before this partition starts.
            Used to enforce dependencies (e.g., build must complete first).

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
