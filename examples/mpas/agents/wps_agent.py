# SPDX-License-Identifier: Apache-2.0

"""WPSAgent - Manages WRF Preprocessing System (WPS).

This agent handles downloading, building, and running the WPS ungrib utility
used to process GRIB data files for MPAS initialization.
"""

from pathlib import Path
from typing import Dict, Optional

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task


@chiltepin_agent(agent_workflow_include=["build-executor"])
class WPSAgent:
    """Agent for managing WRF Preprocessing System (WPS).

    This agent manages the complete lifecycle of WPS:
    - Downloads WPS source code from GitHub
    - Builds ungrib utility (minimal WPS build)
    - Runs ungrib to process GRIB data for MPAS

    Attributes
    ----------
    install_dir : str
        Base directory for WPS installation
    version : str
        WPS version to install
    source_dir : Path
        Path to WPS source directory (set after download)
    ungrib_path : Path
        Path to ungrib executable (set after build)
    """

    def __init__(self, install_dir: str, version: str = "4.5"):
        """Initialize WPSAgent.

        Parameters
        ----------
        install_dir : str
            Directory where WPS will be installed
        version : str, optional
            WPS version to install, by default "4.5"
        """
        self.install_dir = Path(install_dir)
        self.version = version
        self.source_dir: Optional[Path] = None
        self.ungrib_path: Optional[Path] = None

    @bash_task
    @agent_action
    async def download_wps(self) -> str:
        """Download WPS source code from GitHub.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads WPS from wrf-model/WPS GitHub repository.
        """
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.install_dir / f"WPS-{self.version}"

        # TODO: Implement download logic
        # Reference: Old MPAS app install_wps step
        # - Clone or download tarball from GitHub
        # - Extract to self.source_dir
        # - Return path as string

        return str(self.source_dir)

    @bash_task
    @agent_action
    async def build(self) -> str:
        """Build ungrib utility.

        Returns
        -------
        str
            Path to ungrib executable

        Raises
        ------
        RuntimeError
            If download_wps has not been called first

        Notes
        -----
        Builds WPS ungrib utility:
        - Configures WPS (configure script)
        - Compiles only ungrib (not full WPS)
        - Sets self.ungrib_path
        """
        if self.source_dir is None:
            raise RuntimeError("Must call download_wps() before build()")

        # TODO: Implement build logic
        # Reference: Old MPAS app install_wps step
        # - Run ./configure (select compiler)
        # - Edit configure.wps to build only ungrib
        # - Run make (or compile ungrib only)
        # - Set self.ungrib_path
        # - Return path as string

        self.ungrib_path = self.source_dir / "ungrib.exe"
        return str(self.ungrib_path)

    @bash_task
    @agent_action
    async def run_ungrib(
        self,
        grib_data_path: str,
        output_dir: str,
        vtable: str,
        config: Optional[Dict[str, str]] = None,
    ) -> str:
        """Run ungrib to process GRIB data for MPAS.

        Parameters
        ----------
        grib_data_path : str
            Path to directory containing GRIB files or pattern
        output_dir : str
            Directory for ungrib output files
        vtable : str
            Path to Vtable file for GRIB format
        config : dict, optional
            Additional configuration parameters for namelist.wps

        Returns
        -------
        str
            Path to output directory containing processed files

        Raises
        ------
        RuntimeError
            If build has not been called first
        FileNotFoundError
            If GRIB data or Vtable not found

        Notes
        -----
        Runs ungrib to convert GRIB to intermediate format:
        - Links GRIB files
        - Links Vtable
        - Creates namelist.wps
        - Runs ungrib.exe
        - Output files named FILE:YYYY-MM-DD_HH
        """
        if self.ungrib_path is None:
            raise RuntimeError("Must call build() before run_ungrib()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement ungrib execution logic
        # Reference: Old MPAS app ungrib step
        # - Link GRIB files (ln -sf gribfile GRIBFILE.AAA, etc.)
        # - Link Vtable (ln -sf vtable Vtable)
        # - Create namelist.wps with config
        # - Run ungrib.exe
        # - Verify output files created
        # - Return output directory path

        return str(output_path)
