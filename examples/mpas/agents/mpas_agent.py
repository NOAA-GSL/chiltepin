# SPDX-License-Identifier: Apache-2.0

"""MPASAgent - Manages MPAS model initialization and forecasting.

This agent handles downloading, building, and running the MPAS model for
initialization and forecasting.
"""

from pathlib import Path
from typing import Dict, Optional

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task


@chiltepin_agent(agent_workflow_include=["compute-executor"])
class MPASAgent:
    """Agent for managing MPAS model initialization and forecasting.

    This agent manages the complete lifecycle of MPAS:
    - Downloads MPAS-Model source code
    - Builds init_atmosphere_model and atmosphere_model executables
    - Initializes MPAS with processed initial conditions
    - Initializes MPAS with processed lateral boundary conditions
    - Runs MPAS forecast

    Attributes
    ----------
    install_dir : str
        Base directory for MPAS installation
    version : str
        MPAS version to install (branch or tag)
    source_dir : Path
        Path to MPAS source directory (set after download)
    init_atmosphere_path : Path
        Path to init_atmosphere_model executable (set after build)
    atmosphere_model_path : Path
        Path to atmosphere_model executable (set after build)
    """

    def __init__(self, install_dir: str, version: str = "v8.2.0"):
        """Initialize MPASAgent.

        Parameters
        ----------
        install_dir : str
            Directory where MPAS will be installed
        version : str, optional
            MPAS version to install, by default "v8.2.0"
        """
        self.install_dir = Path(install_dir)
        self.version = version
        self.source_dir: Optional[Path] = None
        self.init_atmosphere_path: Optional[Path] = None
        self.atmosphere_model_path: Optional[Path] = None

    @bash_task
    @agent_action
    async def download_mpas(self) -> str:
        """Download MPAS-Model source code.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads from MPAS-Dev/MPAS-Model GitHub repository.
        """
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.install_dir / "MPAS-Model"

        # TODO: Implement download logic
        # Reference: Old MPAS app install_mpas step
        # - Clone from GitHub: MPAS-Dev/MPAS-Model
        # - Checkout specified version
        # - Return path as string

        return str(self.source_dir)

    @bash_task
    @agent_action
    async def build(self) -> dict:
        """Build MPAS executables.

        Returns
        -------
        dict
            Dictionary with paths to executables
            {"init": str, "model": str}

        Raises
        ------
        RuntimeError
            If download_mpas has not been called first

        Notes
        -----
        Builds MPAS atmosphere core executables:
        - init_atmosphere_model: For initialization
        - atmosphere_model: For forecast integration
        """
        if self.source_dir is None:
            raise RuntimeError("Must call download_mpas() before build()")

        # TODO: Implement build logic
        # Reference: Old MPAS app install_mpas step
        # - Run make gfortran CORE=atmosphere (or appropriate compiler)
        # - Build both init_atmosphere_model and atmosphere_model
        # - Set self.init_atmosphere_path and self.atmosphere_model_path
        # - Return paths as dict

        self.init_atmosphere_path = self.source_dir / "init_atmosphere_model"
        self.atmosphere_model_path = self.source_dir / "atmosphere_model"

        return {
            "init": str(self.init_atmosphere_path),
            "model": str(self.atmosphere_model_path),
        }

    @bash_task
    @agent_action
    async def initialize_ics(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS with processed initial conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to initialized file

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process initial conditions:
        - Reads ungrib intermediate files
        - Interpolates to MPAS mesh
        - Writes initialized state
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_ics()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement initialization logic
        # Reference: Old MPAS app mpas_init_ics step
        # - Copy mesh, streams, namelist to output_dir
        # - Link ungrib files
        # - Run init_atmosphere_model
        # - Return path to initialized file

        initialized_file = output_path / "init.nc"
        return str(initialized_file)

    @bash_task
    @agent_action
    async def initialize_lbcs(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS lateral boundary conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to directory containing LBC files

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process lateral boundary conditions:
        - Reads ungrib intermediate files for multiple times
        - Interpolates to MPAS mesh for each time
        - Writes LBC files for each boundary update time
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_lbcs()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement initialization logic
        # Reference: Old MPAS app mpas_init_lbcs step
        # - Copy mesh, streams, namelist to output_dir
        # - Link ungrib files for all boundary times
        # - Run init_atmosphere_model
        # - Return path to LBC directory

        return str(output_path)

    @bash_task
    @agent_action
    async def run_forecast(
        self,
        ics_file: str,
        lbcs_dir: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS forecast.

        Parameters
        ----------
        ics_file : str
            Path to initialized conditions file
        lbcs_dir : str
            Path to directory containing LBC files
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for forecast
        namelist_file : str
            Path to namelist file for forecast
        output_dir : str
            Directory for forecast output

        Returns
        -------
        str
            Path to forecast output directory

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs atmosphere_model for forecast integration:
        - Reads initialized conditions
        - Reads lateral boundary conditions
        - Integrates forward in time
        - Writes forecast output files (history, diagnostics)
        """
        if self.atmosphere_model_path is None:
            raise RuntimeError("Must call build() before run_forecast()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement forecast logic
        # Reference: Old MPAS app mpas_forecast step
        # - Copy mesh, streams, namelist to output_dir
        # - Link ICS file
        # - Link LBC files
        # - Run atmosphere_model with appropriate MPI configuration
        # - Return path to output directory

        return str(output_path)
