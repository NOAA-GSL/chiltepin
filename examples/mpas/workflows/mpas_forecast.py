# SPDX-License-Identifier: Apache-2.0

"""MPASForecastWorkflow - Orchestrates MPAS multi-agent forecast.

This module provides the main workflow orchestration class that coordinates
all MPAS component agents to build, configure, and run MPAS forecasts.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from chiltepin import AgentRuntime, Workflow
from chiltepin.manager import Manager

from agents import MPASAgent, MPASLimitedAreaAgent, MetisAgent, WPSAgent


class MPASForecastWorkflow:
    """Orchestrates MPAS multi-agent forecast workflow.

    This class manages the complete MPAS forecast workflow by coordinating
    multiple specialized agents:
    - MetisAgent: Graph partitioning
    - WPSAgent: Data preprocessing
    - MPASLimitedAreaAgent: Regional mesh generation
    - MPASAgent: Model initialization and forecasting

    The workflow proceeds through phases:
    1. Setup: Launch all agents
    2. Build: Parallel compilation of all components
    3. Mesh: Generate and partition regional mesh
    4. Preprocess: Fetch and process input data
    5. Initialize: Prepare MPAS initial and boundary conditions
    6. Forecast: Run MPAS model
    7. Cleanup: Shutdown agents

    Attributes
    ----------
    config : dict
        Configuration dictionary with workflow and model settings
    workflow : Workflow
        Chiltepin workflow managing executors
    agent_runtime : AgentRuntime
        Runtime for launching and managing agents
    manager : Manager
        Manager for agent communication
    agents : dict
        Dictionary of launched agent handles
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MPAS forecast workflow.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing:
            - workflow: Executor configurations
            - model: Model settings (resolution, domain, etc.)
            - paths: Directory paths for experiment
        """
        self.config = config
        self.workflow: Optional[Workflow] = None
        self.agent_runtime: Optional[AgentRuntime] = None
        self.manager: Optional[Manager] = None
        self.agents: Dict[str, Any] = {}

    async def setup_agents(self) -> None:
        """Launch all component agents.

        Creates Workflow and AgentRuntime, then launches:
        - MetisAgent
        - WPSAgent
        - MPASLimitedAreaAgent
        - MPASAgent

        Raises
        ------
        RuntimeError
            If workflow setup fails
        """
        # Extract workflow configuration
        workflow_config = self.config.get("workflow", {})
        manager_executors = self.config.get("manager_executors", ["manager-executor"])

        # Create workflow with executor configurations
        self.workflow = Workflow(
            workflow_config,
            include=manager_executors,
            run_dir=self.config.get("run_dir"),
        )
        self.workflow.start()

        # Create agent runtime
        self.agent_runtime = AgentRuntime(workflow=self.workflow)

        # Get manager context
        self.manager = await self.agent_runtime.manager()

        # Extract agent configurations
        agent_config = self.config.get("agent_workflow", {})
        build_executors = self.config.get("build_executors", ["build-executor"])
        compute_executors = self.config.get("compute_executors", ["compute-executor"])

        # Extract paths
        paths = self.config.get("paths", {})
        install_dir = Path(paths.get("install_dir", "./installs"))

        # Launch MetisAgent
        metis = await self.manager.launch(
            MetisAgent,
            args=(str(install_dir / "metis"),),
            agent_workflow_config=agent_config,
            agent_workflow_include=build_executors,
        )
        self.agents["metis"] = metis

        # Launch WPSAgent
        wps = await self.manager.launch(
            WPSAgent,
            args=(str(install_dir / "wps"),),
            agent_workflow_config=agent_config,
            agent_workflow_include=build_executors,
        )
        self.agents["wps"] = wps

        # Launch MPASLimitedAreaAgent
        mpas_la = await self.manager.launch(
            MPASLimitedAreaAgent,
            args=(str(install_dir / "mpas_limited_area"),),
            agent_workflow_config=agent_config,
            agent_workflow_include=build_executors,
        )
        self.agents["mpas_limited_area"] = mpas_la

        # Launch MPASAgent
        mpas = await self.manager.launch(
            MPASAgent,
            args=(str(install_dir / "mpas"),),
            agent_workflow_config=agent_config,
            agent_workflow_include=compute_executors,
        )
        self.agents["mpas"] = mpas

    async def build_phase(self) -> None:
        """Build all software components in parallel.

        Downloads and compiles:
        - Metis (gpmetis utility)
        - WPS (ungrib utility)
        - MPAS-Limited-Area (create_region utility)
        - MPAS (init_atmosphere_model and atmosphere_model)

        This phase runs all builds concurrently for efficiency.
        """
        # Download all components in parallel
        download_tasks = [
            self.agents["metis"].download_metis(),
            self.agents["wps"].download_wps(),
            self.agents["mpas_limited_area"].download(),
            self.agents["mpas"].download_mpas(),
        ]
        await asyncio.gather(*download_tasks)

        # Build all components in parallel
        build_tasks = [
            self.agents["metis"].build(),
            self.agents["wps"].build(),
            self.agents["mpas_limited_area"].build(),
            self.agents["mpas"].build(),
        ]
        await asyncio.gather(*build_tasks)

    async def mesh_phase(self) -> Dict[str, str]:
        """Generate and partition regional mesh.

        Returns
        -------
        dict
            Dictionary with mesh file paths

        Notes
        -----
        Sequential steps:
        1. Download global mesh for specified resolution
        2. Create regional CONUS mesh
        3. Partition mesh for init and forecast MPI ranks
        """
        model_config = self.config.get("model", {})
        resolution = model_config.get("resolution", "120km")
        paths = self.config.get("paths", {})
        mesh_dir = Path(paths.get("mesh_dir", "./mesh"))

        # Download global mesh
        global_mesh = await self.agents["mpas_limited_area"].download_global_mesh(
            resolution
        )

        # Create regional mesh
        region_spec = model_config.get("region", "conus")
        regional_mesh = await self.agents["mpas_limited_area"].create_conus_mesh(
            global_static=global_mesh["static"],
            global_graph=global_mesh["graph"],
            region_spec=region_spec,
            output_dir=str(mesh_dir),
        )

        # Partition mesh for different MPI configurations
        init_ranks = model_config.get("init_ranks", 24)
        forecast_ranks = model_config.get("forecast_ranks", 48)

        partition_tasks = [
            self.agents["metis"].partition_mesh(
                regional_mesh["graph"], init_ranks
            ),
            self.agents["metis"].partition_mesh(
                regional_mesh["graph"], forecast_ranks
            ),
        ]
        await asyncio.gather(*partition_tasks)

        return regional_mesh

    async def preprocess_phase(self) -> Dict[str, str]:
        """Fetch and process input data.

        Returns
        -------
        dict
            Dictionary with preprocessed data paths

        Notes
        -----
        Parallel processing of:
        - Initial conditions (ICS)
        - Lateral boundary conditions (LBCS)
        """
        # TODO: Implement data fetching and preprocessing
        # Reference: Old MPAS app get_ics, get_lbcs, ungrib steps
        # - Fetch GRIB data for ICS and LBCS
        # - Run ungrib for ICS
        # - Run ungrib for LBCS (parallel)
        # - Return paths to processed data

        paths = self.config.get("paths", {})
        grib_dir = Path(paths.get("grib_data_dir", "./grib_data"))
        ungrib_dir = Path(paths.get("ungrib_dir", "./ungrib_output"))

        # Placeholder for implementation
        return {
            "ics": str(ungrib_dir / "ics"),
            "lbcs": str(ungrib_dir / "lbcs"),
        }

    async def initialization_phase(
        self, mesh: Dict[str, str], preprocessed_data: Dict[str, str]
    ) -> Dict[str, str]:
        """Initialize MPAS with processed data.

        Parameters
        ----------
        mesh : dict
            Mesh file paths from mesh_phase
        preprocessed_data : dict
            Preprocessed data paths from preprocess_phase

        Returns
        -------
        dict
            Dictionary with initialized file paths

        Notes
        -----
        Parallel initialization of:
        - Initial conditions
        - Lateral boundary conditions
        """
        # TODO: Implement MPAS initialization
        # Reference: Old MPAS app mpas_init_ics, mpas_init_lbcs steps
        # - Initialize ICS
        # - Initialize LBCS (parallel)
        # - Return paths to initialized files

        paths = self.config.get("paths", {})
        init_dir = Path(paths.get("init_dir", "./initialization"))

        # Placeholder for implementation
        return {
            "ics_file": str(init_dir / "init.nc"),
            "lbcs_dir": str(init_dir / "lbcs"),
        }

    async def forecast_phase(
        self, mesh: Dict[str, str], initialized: Dict[str, str]
    ) -> str:
        """Run MPAS forecast.

        Parameters
        ----------
        mesh : dict
            Mesh file paths
        initialized : dict
            Initialized data paths

        Returns
        -------
        str
            Path to forecast output directory
        """
        # TODO: Implement MPAS forecast
        # Reference: Old MPAS app mpas_forecast step
        # - Run atmosphere_model
        # - Return path to forecast output

        paths = self.config.get("paths", {})
        forecast_dir = Path(paths.get("forecast_dir", "./forecast"))

        # Placeholder for implementation
        return str(forecast_dir)

    async def run(self) -> str:
        """Execute complete MPAS forecast workflow.

        Returns
        -------
        str
            Path to forecast output directory

        Raises
        ------
        Exception
            If any workflow phase fails
        """
        try:
            # Setup phase
            await self.setup_agents()

            # Build phase - parallel compilation
            await self.build_phase()

            # Mesh phase - sequential mesh generation and partitioning
            mesh = await self.mesh_phase()

            # Preprocess phase - parallel data preparation
            preprocessed_data = await self.preprocess_phase()

            # Initialization phase - parallel MPAS initialization
            initialized = await self.initialization_phase(mesh, preprocessed_data)

            # Forecast phase - run model
            forecast_output = await self.forecast_phase(mesh, initialized)

            return forecast_output

        finally:
            # Always cleanup, even if errors occurred
            await self.cleanup()

    async def cleanup(self) -> None:
        """Shutdown all agents and clean up workflow.

        Ensures proper cleanup of:
        - Agent handles
        - Manager context
        - Workflow executors
        """
        if self.manager is not None:
            # Shutdown agents gracefully
            for agent_name, agent_handle in self.agents.items():
                try:
                    await agent_handle.shutdown()
                except Exception as e:
                    print(f"Warning: Error shutting down {agent_name}: {e}")

            # Exit manager context
            try:
                await self.manager.__aexit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error exiting manager context: {e}")

        if self.workflow is not None:
            try:
                self.workflow.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up workflow: {e}")
