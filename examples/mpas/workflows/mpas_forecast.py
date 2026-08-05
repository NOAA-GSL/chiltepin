# SPDX-License-Identifier: Apache-2.0

"""MPASForecastWorkflow - Orchestrates MPAS multi-agent forecast.

This module provides the main workflow orchestration class that coordinates
all MPAS component agents to build, configure, and run MPAS forecasts.
"""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from chiltepin import AgentRuntime, Workflow
from chiltepin.manager import Manager

from agents import MeshAgent, MPASAgent, WPSAgent

# Maps config "type" strings to agent classes
AGENT_TYPES: Dict[str, Type] = {
    "mesh": MeshAgent,
    "wps": WPSAgent,
    "mpas": MPASAgent,
}


class MPASForecastWorkflow:
    """Orchestrates MPAS multi-agent forecast workflow.

    This class manages the complete MPAS forecast workflow by coordinating
    multiple specialized agents:
    - MeshAgent: Mesh generation and partitioning
    - WPSAgent: Data preprocessing
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
        self._agents_by_type: Dict[str, List[str]] = defaultdict(list)

    async def setup_workflow(self) -> None:
        """Create Workflow and AgentRuntime.

        Raises
        ------
        RuntimeError
            If workflow setup fails
        """
        workflow_config = self.config.get("workflow_config", {})

        self.workflow = Workflow(
            workflow_config,
            run_dir=str(Path(self.config.get("experiment_dir")) / "parsl_logs")
        )
        self.workflow.start()

        self.agent_runtime = AgentRuntime(workflow=self.workflow)

    def agents_by_type(self, agent_type: str) -> List[Any]:
        """Return all launched agent handles of the given type."""
        return [self.agents[name] for name in self._agents_by_type.get(agent_type, [])]

    def _build_agent_args(self, agent_conf: Dict[str, Any]) -> tuple:
        """Build positional init args for an agent based on its type."""
        agent_type = agent_conf["type"]
        work_dir = str(Path(agent_conf.get("work_dir", "/tmp")))

        if agent_type == "mesh":
            # MeshAgent(install_dir, mesh_config)
            model = self.config.get("model", {})
            mesh_config = dict(model.get("mesh", {"resolution": "120km"}))
            mesh_config["init_ranks"] = model.get("init_ranks")
            mesh_config["forecast_ranks"] = model.get("forecast_ranks")
            return (work_dir, mesh_config)

        # Default: single work_dir positional arg (wps, mpas, etc.)
        return (work_dir,)

    async def setup_agents(self) -> None:
        """Launch all agents defined in the config list.

        Each entry in ``config["agents"]`` must have:
        - ``name``: unique identifier for this agent instance
        - ``type``: agent class key (mesh, wps, mpas)
        - ``executor``: which workflow executor runs the agent process
        - ``workflow_config``: executor config passed into the agent
        - ``work_dir``: working directory for the agent
        """
        agent_configs = self.config.get("agents", [])

        for agent_conf in agent_configs:
            name = agent_conf["name"]
            agent_type = agent_conf["type"]

            if name in self.agents:
                raise ValueError(f"Duplicate agent name: '{name}'")

            agent_cls = AGENT_TYPES.get(agent_type)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent type '{agent_type}' for agent '{name}'. "
                    f"Valid types: {list(AGENT_TYPES)}"
                )

            args = self._build_agent_args(agent_conf)
            handle = await self.manager.launch(
                agent_cls,
                args=args,
                agent_workflow_config=agent_conf["workflow_config"],
                executor=agent_conf.get("executor"),
            )

            self.agents[name] = handle
            self._agents_by_type[agent_type].append(name)

    # async def build_phase(self) -> None:
    #     """Build all software components in parallel.

    #     Downloads and compiles:
    #     - WPS (ungrib utility)
    #     - MPAS (init_atmosphere_model and atmosphere_model)

    #     Mesh tools are installed on demand by generate_mesh().
    #     """
    #     build_tasks = [
    #         # self.agents["wps"].build(),
    #         # self.agents["mpas"].build(),
    #     ]
    #     if build_tasks:
    #         await asyncio.gather(*build_tasks)

    # async def mesh_phase(self) -> Dict[str, Any]:
    #     """Generate and partition mesh on all mesh agents concurrently.
    #
    #     Returns
    #     -------
    #     dict
    #         Per-agent results keyed by agent name
    #     """
    #     mesh_agents = self.agents_by_type("mesh")
    #     results = await asyncio.gather(*[
    #         a.generate_mesh() for a in mesh_agents
    #     ])
    #     return dict(zip(self._agents_by_type["mesh"], results))

    # async def preprocess_phase(self) -> Dict[str, str]:
    #     """Fetch and process input data.

    #     Returns
    #     -------
    #     dict
    #         Dictionary with preprocessed data paths

    #     Notes
    #     -----
    #     Parallel processing of:
    #     - Initial conditions (ICS)
    #     - Lateral boundary conditions (LBCS)
    #     """
    #     # TODO: Implement data fetching and preprocessing
    #     # Reference: Old MPAS app get_ics, get_lbcs, ungrib steps
    #     # - Fetch GRIB data for ICS and LBCS
    #     # - Run ungrib for ICS
    #     # - Run ungrib for LBCS (parallel)
    #     # - Return paths to processed data

    #     paths = self.config.get("paths", {})
    #     grib_dir = Path(paths.get("grib_data_dir", "./grib_data"))
    #     ungrib_dir = Path(paths.get("ungrib_dir", "./ungrib_output"))

    #     # Placeholder for implementation
    #     return {
    #         "ics": str(ungrib_dir / "ics"),
    #         "lbcs": str(ungrib_dir / "lbcs"),
    #     }

    # async def initialization_phase(
    #     self, mesh: Dict[str, str], preprocessed_data: Dict[str, str]
    # ) -> Dict[str, str]:
    #     """Initialize MPAS with processed data.

    #     Parameters
    #     ----------
    #     mesh : dict
    #         Mesh file paths from mesh_phase
    #     preprocessed_data : dict
    #         Preprocessed data paths from preprocess_phase

    #     Returns
    #     -------
    #     dict
    #         Dictionary with initialized file paths

    #     Notes
    #     -----
    #     Parallel initialization of:
    #     - Initial conditions
    #     - Lateral boundary conditions
    #     """
    #     # TODO: Implement MPAS initialization
    #     # Reference: Old MPAS app mpas_init_ics, mpas_init_lbcs steps
    #     # - Initialize ICS
    #     # - Initialize LBCS (parallel)
    #     # - Return paths to initialized files

    #     paths = self.config.get("paths", {})
    #     init_dir = Path(paths.get("init_dir", "./initialization"))

    #     # Placeholder for implementation
    #     return {
    #         "ics_file": str(init_dir / "init.nc"),
    #         "lbcs_dir": str(init_dir / "lbcs"),
    #     }

    # async def forecast_phase(
    #     self, mesh: Dict[str, str], initialized: Dict[str, str]
    # ) -> str:
    #     """Run MPAS forecast.

    #     Parameters
    #     ----------
    #     mesh : dict
    #         Mesh file paths
    #     initialized : dict
    #         Initialized data paths

    #     Returns
    #     -------
    #     str
    #         Path to forecast output directory
    #     """
    #     # TODO: Implement MPAS forecast
    #     # Reference: Old MPAS app mpas_forecast step
    #     # - Run atmosphere_model
    #     # - Return path to forecast output

    #     paths = self.config.get("paths", {})
    #     forecast_dir = Path(paths.get("forecast_dir", "./forecast"))

    #     # Placeholder for implementation
    #     return str(forecast_dir)

    async def mpas_build_test_phase(self) -> None:
        """Run MPAS download/build actions on all MPAS agents.

        This is a lightweight test phase intended for validating MPAS
        toolchain and build environment using the current experiment config.
        """
        mpas_agents = self.agents_by_type("mpas")
        if not mpas_agents:
            raise RuntimeError(
                "MPAS build test requires at least one MPAS agent."
            )

        download_results = await asyncio.gather(
            *[agent.download_mpas() for agent in mpas_agents],
            return_exceptions=True,
        )
        for idx, result in enumerate(download_results):
            label = self._agents_by_type.get("mpas", [])[idx]
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"MPAS download failed for agent '{label}': {result}"
                ) from result
            print(f"MPAS source downloaded for '{label}': {result}")

        build_results = await asyncio.gather(
            *[agent.build() for agent in mpas_agents],
            return_exceptions=True,
        )
        for idx, result in enumerate(build_results):
            label = self._agents_by_type.get("mpas", [])[idx]
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"MPAS build failed for agent '{label}': {result}"
                ) from result
            print(f"MPAS build output for '{label}': {result}")

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
            # Setup workflow and agent runtime
            await self.setup_workflow()

            async with await self.agent_runtime.manager() as manager:
                self.manager = manager

                # Launch agents
                await self.setup_agents()

                print("Running MPAS build-only test phase...")
                await self.mpas_build_test_phase()
                await self.shutdown_agents()
                return ""

                # Generate mesh on all mesh agents concurrently
                mesh_agents = self.agents_by_type("mesh")
                mesh_results = await asyncio.gather(
                    *[a.generate_mesh() for a in mesh_agents],
                    return_exceptions=True,
                )
                for idx, mesh_result in enumerate(mesh_results):
                    mesh_label = self._agents_by_type.get("mesh", [])[idx]
                    if isinstance(mesh_result, Exception):
                        raise RuntimeError(
                            f"Mesh generation failed for agent '{mesh_label}': "
                            f"{mesh_result}"
                        ) from mesh_result

                    plot_path = mesh_result.get("plot")
                    plot_error = mesh_result.get("plot_error")
                    if plot_error:
                        raise RuntimeError(
                            f"Mesh plot generation failed for agent '{mesh_label}': "
                            f"{plot_error}"
                        )
                    if plot_path:
                        print(f"Mesh plot generated for '{mesh_label}': {plot_path}")

                # GEOG interpolation is temporarily disabled while focusing on
                # MPAS download/build testing.
                # model_cfg = self.config.get("model", {})
                # mesh_cfg = model_cfg.get("mesh", {})
                # regional_cfg = mesh_cfg.get("regional", {})
                # geog_interp_cfg = model_cfg.get("geog_interpolation", {})
                # geog_interp_enabled = bool(geog_interp_cfg.get("enabled", False))
                # if (
                #     isinstance(regional_cfg, dict)
                #     and "project_hexes" in regional_cfg
                #     and geog_interp_enabled
                # ):
                #     mpas_agents = self.agents_by_type("mpas")
                #     if not mpas_agents:
                #         raise RuntimeError(
                #             "project_hexes mesh requires MPAS geog interpolation, "
                #             "but no MPAS agent is configured."
                #         )
                #
                #     namelist_file = geog_interp_cfg.get("namelist_file")
                #     streams_file = geog_interp_cfg.get("streams_file")
                #     if not namelist_file or not streams_file:
                #         raise RuntimeError(
                #             "Missing model.geog_interpolation.namelist_file or "
                #             "model.geog_interpolation.streams_file in config."
                #         )
                #
                #     mpas_agent = mpas_agents[0]
                #     geog_info = await mpas_agent.download_geog_data(
                #         model_cfg.get("geog", {})
                #     )
                #     geog_dir = geog_info["geog_dir"]
                #
                #     geog_tasks = []
                #     geog_labels = []
                #     output_subdir = geog_interp_cfg.get("output_subdir", "mpas_init_geog")
                #     for idx, mesh_result in enumerate(mesh_results):
                #         mesh_label = self._agents_by_type.get("mesh", [])[idx]
                #         mesh_file = mesh_result.get("mesh")
                #         if not mesh_file:
                #             continue
                #         geog_output_dir = str(Path(mesh_file).parent / output_subdir)
                #         geog_tasks.append(
                #             mpas_agent.interpolate_geog_only(
                #                 mesh_file=mesh_file,
                #                 namelist_file=namelist_file,
                #                 streams_file=streams_file,
                #                 output_dir=geog_output_dir,
                #                 geog_dir=geog_dir,
                #             )
                #         )
                #         geog_labels.append(mesh_label)
                #
                #     geog_results = await asyncio.gather(*geog_tasks, return_exceptions=True)
                #     for mesh_label, geog_result in zip(geog_labels, geog_results):
                #         if isinstance(geog_result, Exception):
                #             raise RuntimeError(
                #                 f"GEOG interpolation failed for mesh agent "
                #                 f"'{mesh_label}': {geog_result}"
                #             ) from geog_result
                #         print(
                #             f"GEOG interpolation completed for '{mesh_label}': "
                #             f"{geog_result['output_dir']}"
                #         )

                # Create regional mesh
                # await self.agents["mesh"].create_region()

                # Build phase - parallel compilation
                # await self.build_phase()

                # # Mesh phase - sequential mesh generation and partitioning
                # mesh = await self.mesh_phase()

                # # Preprocess phase - parallel data preparation
                # preprocessed_data = await self.preprocess_phase()

                # # Initialization phase - parallel MPAS initialization
                # initialized = await self.initialization_phase(mesh, preprocessed_data)

                # # Forecast phase - run model
                # forecast_output = await self.forecast_phase(mesh, initialized)

                # Shutdown agents while manager/exchange client is still alive
                await self.shutdown_agents()

                # return forecast_output
                return ""

        finally:
            # Cleanup workflow (agents already shut down inside manager context)
            await self.cleanup()

    async def shutdown_agents(self) -> None:
        """Shutdown all agents gracefully.

        Must be called while the manager context (exchange client) is still active.
        """
        for agent_name, agent_handle in self.agents.items():
            try:
                await agent_handle.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down {agent_name}: {e}")

    async def cleanup(self) -> None:
        """Clean up workflow resources.

        Handles cleanup of:
        - Workflow executors

        Note: Agent shutdown is handled by shutdown_agents() which must be
        called inside the manager context before the exchange client is torn down.
        """
        if self.workflow is not None:
            try:
                self.workflow.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up workflow: {e}")
