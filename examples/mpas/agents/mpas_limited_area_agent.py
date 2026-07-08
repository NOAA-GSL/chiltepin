# SPDX-License-Identifier: Apache-2.0

"""MPASLimitedAreaAgent - Manages MPAS regional mesh generation.

This agent handles downloading, building, and running MPAS-Limited-Area tools
used to generate regional CONUS meshes from global mesh data.
"""

from pathlib import Path
from typing import Optional

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task


@chiltepin_agent(agent_workflow_include=["build-executor"])
class MPASLimitedAreaAgent:
    """Agent for managing MPAS-Limited-Area tools.

    This agent manages the complete lifecycle of MPAS-Limited-Area:
    - Downloads MPAS-Limited-Area source code
    - Builds create_region utility
    - Downloads global mesh files
    - Creates regional CONUS mesh from global mesh

    Attributes
    ----------
    install_dir : str
        Base directory for MPAS-Limited-Area installation
    version : str
        Version to install (branch or tag)
    source_dir : Path
        Path to source directory (set after download)
    create_region_path : Path
        Path to create_region executable (set after build)
    mesh_data_dir : Path
        Directory for downloaded global mesh files
    """

    def __init__(self, install_dir: str, version: str = "master"):
        """Initialize MPASLimitedAreaAgent.

        Parameters
        ----------
        install_dir : str
            Directory where MPAS-Limited-Area will be installed
        version : str, optional
            Branch or tag to use, by default "master"
        """
        self.install_dir = Path(install_dir)
        self.version = version
        self.source_dir: Optional[Path] = None
        self.create_region_path: Optional[Path] = None
        self.mesh_data_dir = self.install_dir / "mesh_data"

    @bash_task
    @agent_action
    async def download(self) -> str:
        """Download MPAS-Limited-Area source code.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads from MPAS-Dev/MPAS-Limited-Area GitHub repository.
        """
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.install_dir / "MPAS-Limited-Area"

        # TODO: Implement download logic
        # Reference: Old MPAS app install_limited_area step
        # - Clone from GitHub: MPAS-Dev/MPAS-Limited-Area
        # - Checkout specified version
        # - Return path as string

        return str(self.source_dir)

    @bash_task
    @agent_action
    async def build(self) -> str:
        """Build create_region utility.

        Returns
        -------
        str
            Path to create_region executable

        Raises
        ------
        RuntimeError
            If download has not been called first

        Notes
        -----
        Builds MPAS-Limited-Area create_region tool:
        - Compiles with make
        - Sets self.create_region_path
        """
        if self.source_dir is None:
            raise RuntimeError("Must call download() before build()")

        # TODO: Implement build logic
        # Reference: Old MPAS app install_limited_area step
        # - Run make in source directory
        # - Set self.create_region_path
        # - Return path as string

        self.create_region_path = self.source_dir / "create_region"
        return str(self.create_region_path)

    @bash_task
    @agent_action
    async def download_global_mesh(self, resolution: str) -> dict:
        """Download global mesh files for specified resolution.

        Parameters
        ----------
        resolution : str
            Mesh resolution (e.g., "120km", "60km", "30km")

        Returns
        -------
        dict
            Dictionary with paths to static.nc and graph.info files
            {"static": str, "graph": str}

        Notes
        -----
        Downloads global mesh files from MPAS mesh repository:
        - static.nc: Static fields (topography, land use, etc.)
        - graph.info: Mesh connectivity information
        """
        self.mesh_data_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Implement download logic
        # Reference: Old MPAS app create_region step
        # - Download from https://mpas-dev.github.io/atmosphere/atmosphere_meshes.html
        # - Get static.nc and graph.info for specified resolution
        # - Return paths as dict

        static_path = self.mesh_data_dir / f"x1.{resolution}.static.nc"
        graph_path = self.mesh_data_dir / f"x1.{resolution}.graph.info"

        return {
            "static": str(static_path),
            "graph": str(graph_path),
        }

    @bash_task
    @agent_action
    async def create_conus_mesh(
        self,
        global_static: str,
        global_graph: str,
        region_spec: str,
        output_dir: str,
    ) -> dict:
        """Create regional CONUS mesh from global mesh.

        Parameters
        ----------
        global_static : str
            Path to global static.nc file
        global_graph : str
            Path to global graph.info file
        region_spec : str
            Region specification (e.g., lat/lon bounds or named region)
        output_dir : str
            Directory for output mesh files

        Returns
        -------
        dict
            Dictionary with paths to regional mesh files
            {"static": str, "graph": str}

        Raises
        ------
        RuntimeError
            If build has not been called first
        FileNotFoundError
            If global mesh files not found

        Notes
        -----
        Uses create_region utility to extract regional mesh:
        - Reads global mesh files
        - Extracts cells within specified region
        - Writes regional static.nc and graph.info
        """
        if self.create_region_path is None:
            raise RuntimeError("Must call build() before create_conus_mesh()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Implement create_region logic
        # Reference: Old MPAS app create_region step
        # - Run create_region with global files and region spec
        # - Output regional static.nc and graph.info
        # - Return paths as dict

        regional_static = output_path / "static.nc"
        regional_graph = output_path / "graph.info"

        return {
            "static": str(regional_static),
            "graph": str(regional_graph),
        }
