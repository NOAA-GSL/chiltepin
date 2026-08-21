# SPDX-License-Identifier: Apache-2.0

"""MPASForecastWorkflow - Orchestrates MPAS multi-agent forecast.

This module provides the main workflow orchestration class that coordinates
all MPAS component agents to build, configure, and run MPAS forecasts.
"""

import asyncio
import os
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
        """Build (positional_args, keyword_args) for an agent based on its type."""
        agent_type = agent_conf["type"]
        work_dir = str(Path(agent_conf.get("work_dir", "/tmp")))
        extra_kwargs = dict(agent_conf.get("kwargs", {}))

        if agent_type == "mesh":
            # MeshAgent(work_dir, **kwargs)
            return (work_dir,), extra_kwargs
        if agent_type == "mpas":
            # MPASAgent(install_dir, mpas_config)
            model = self.config.get("model", {})
            mpas_init_config = dict(model.get("mpas_init", {}))
            mpas_fcst_config = dict(model.get("mpas_forecast", {}))
            return (work_dir, mpas_init_config, mpas_fcst_config), extra_kwargs

        # Default: single work_dir positional arg
        return (work_dir,), extra_kwargs

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

            args, agent_kwargs = self._build_agent_args(agent_conf)
            handle = await self.manager.launch(
                agent_cls,
                args=args,
                kwargs=agent_kwargs,
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

    async def mesh_phase(self) -> Dict[str, Any]:
        """Generate and partition mesh on all mesh agents concurrently.
    
        Returns
        -------
        dict
            Per-agent results keyed by agent name
        """
        mesh_agents = self.agents_by_type("mesh")
        model = self.config.get("model", {})
        mesh_config = dict(model.get("mesh", {"resolution": "120km"}))
        mesh_config["init_ranks"] = model.get("init_ranks")
        mesh_config["forecast_ranks"] = model.get("forecast_ranks")
        mesh_prompt = mesh_config.pop("prompt", None)
        mesh_name = mesh_config.get("name")
        llm_config = model.get("llm", {})
        llm_model = llm_config.get("model", "ollama_chat/qwen2.5:3b")
        api_key = llm_config.get("api_key") or (
            os.environ.get(llm_config["api_key_env"]) if llm_config.get("api_key_env") else None
        )
        api_base = llm_config.get("api_base")

        tasks = []
        for label, agent in zip(self._agents_by_type["mesh"], mesh_agents):
            agent_conf = self.config.get("agents", [])
            matched = next((c for c in agent_conf if c.get("name") == label), None)
            if matched is None:
                raise RuntimeError(f"Missing configuration for mesh agent '{label}'")

            mesh_data_dir = str(Path(matched.get("work_dir", "/tmp")) / "mesh_data")
            if mesh_prompt:
                tasks.append(
                    agent.create_mesh_from_prompt(
                        mesh_prompt,
                        mesh_data_dir,
                        model=llm_model,
                        api_key=api_key,
                        api_base=api_base,
                        mesh_name=mesh_name,
                    )
                )
            else:
                tasks.append(agent.generate_mesh(dict(mesh_config), mesh_data_dir))

        raw_results = await asyncio.gather(*tasks)

        if mesh_prompt:
            results = []
            for result in raw_results:
                mesh_result = dict(result["mesh_result"])
                mesh_result["mesh_config"] = result["mesh_config"]
                results.append(mesh_result)
        else:
            results = raw_results

        return dict(zip(self._agents_by_type["mesh"], results))

    async def mpas_phase(self) -> Dict[str, Any]:
        """Run MPAS forecast on all MPAS agents concurrently.
    
        Returns
        -------
        dict
            Per-agent results keyed by agent name
        """
        mpas_agents = self.agents_by_type("mpas")
        results = await asyncio.gather(*[
            a.install_mpas() for a in mpas_agents
        ])
        return dict(zip(self._agents_by_type["mpas"], results))

    async def download_geog_phase(self) -> Dict[str, Any]:
        """Download GEOG data on all MPAS agents concurrently.
    
        Returns
        -------
        dict
            Per-agent results keyed by agent name
        """
        mpas_agents = self.agents_by_type("mpas")
        results = await asyncio.gather(*[
            a.download_geog_data() for a in mpas_agents
        ])
        return dict(zip(self._agents_by_type["mpas"], results))

    async def interpolate_geog_phase(
        self, mesh_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run GEOG interpolation for meshes produced by project_hexes.

        Skips interpolation if the mesh config does not use project_hexes.

        Parameters
        ----------
        mesh_results : dict
            Per-agent mesh results from mesh_phase()

        Returns
        -------
        dict
            Per-mesh-agent results keyed by agent name, or empty if skipped
        """
        model_cfg = self.config.get("model", {})
        mesh_cfg = model_cfg.get("mesh", {})
        regional_cfg = mesh_cfg.get("regional")

        if not isinstance(regional_cfg, dict) or "project_hexes" not in regional_cfg:
            return {}

        mpas_agents = self.agents_by_type("mpas")
        if not mpas_agents:
            raise RuntimeError(
                "project_hexes mesh requires GEOG interpolation, "
                "but no MPAS agent is configured."
            )
        mpas_agent = mpas_agents[0]

        init_ranks = model_cfg.get("init_ranks", 1)

        tasks = []
        labels = []
        for mesh_label, mesh_result in mesh_results.items():
            mesh_file = mesh_result.get("mesh")
            if not mesh_file:
                continue
            tasks.append(mpas_agent.interpolate_geog_only(
                mesh_file=mesh_file,
                num_ranks=init_ranks,
            ))
            labels.append(mesh_label)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"GEOG interpolation failed for mesh '{label}': {result}"
                ) from result
            output[label] = result

        return output

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

    async def _prepare_mpas_agent(self, agent: Any, label: str) -> Dict[str, Any]:
        """Download and build MPAS for one agent with clear stage errors."""
        try:
            source_dir = await agent.download_mpas()
        except Exception as e:
            raise RuntimeError(
                f"MPAS download failed for agent '{label}': {e}"
            ) from e

        try:
            build_result = await agent.build()
        except Exception as e:
            raise RuntimeError(
                f"MPAS build failed for agent '{label}': {e}"
            ) from e

        return {
            "source": source_dir,
            "build": build_result,
        }

    async def prepare_core_assets_phase(self) -> Dict[str, Dict[str, Any]]:
        """Prepare MPAS, mesh, and optional GEOG data concurrently.

        Returns
        -------
        dict
            {
              "mpas": {agent_name: {"source": str, "build": dict}},
              "mesh": {agent_name: dict},
              "geog": {agent_name: dict},
            }
        """
        mpas_agents = self.agents_by_type("mpas")
        mesh_agents = self.agents_by_type("mesh")

        if not mpas_agents:
            raise RuntimeError("Workflow requires at least one MPAS agent.")
        if not mesh_agents:
            raise RuntimeError("Workflow requires at least one mesh agent.")

        model_cfg = self.config.get("model", {})
        geog_cfg = model_cfg.get("geog")
        mesh_config = dict(model_cfg.get("mesh", {"resolution": "120km"}))
        mesh_config["init_ranks"] = model_cfg.get("init_ranks")
        mesh_config["forecast_ranks"] = model_cfg.get("forecast_ranks")

        phase_tasks: List[asyncio.Task] = []
        phase_labels: List[str] = []

        for idx, agent in enumerate(mpas_agents):
            label = self._agents_by_type.get("mpas", [])[idx]
            phase_tasks.append(
                asyncio.create_task(self._prepare_mpas_agent(agent, label))
            )
            phase_labels.append(f"mpas:{label}")

        for idx, agent in enumerate(mesh_agents):
            label = self._agents_by_type.get("mesh", [])[idx]
            matched = next(
                (c for c in self.config.get("agents", []) if c.get("name") == label),
                None,
            )
            if matched is None:
                raise RuntimeError(f"Missing configuration for mesh agent '{label}'")

            mesh_data_dir = str(Path(matched.get("work_dir", "/tmp")) / "mesh_data")
            phase_tasks.append(
                asyncio.create_task(
                    agent.generate_mesh(dict(mesh_config), mesh_data_dir)
                )
            )
            phase_labels.append(f"mesh:{label}")

        if geog_cfg:
            geog_path = geog_cfg.get("path")
            for idx, agent in enumerate(mpas_agents):
                label = self._agents_by_type.get("mpas", [])[idx]
                phase_tasks.append(asyncio.create_task(agent.download_geog_data(path=geog_path)))
                phase_labels.append(f"geog:{label}")

        phase_results = await asyncio.gather(*phase_tasks, return_exceptions=True)

        prepared: Dict[str, Dict[str, Any]] = {
            "mpas": {},
            "mesh": {},
            "geog": {},
        }
        for phase_label, result in zip(phase_labels, phase_results):
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"Core asset preparation failed in '{phase_label}': {result}"
                ) from result

            phase_type, agent_label = phase_label.split(":", 1)
            prepared[phase_type][agent_label] = result

            if phase_type == "mpas":
                print(
                    f"MPAS build completed for '{agent_label}': "
                    f"{result['build']}"
                )
            elif phase_type == "mesh":
                plot_path = result.get("plot")
                plot_error = result.get("plot_error")
                if plot_error:
                    raise RuntimeError(
                        f"Mesh plot generation failed for agent "
                        f"'{agent_label}': {plot_error}"
                    )
                if plot_path:
                    print(f"Mesh plot generated for '{agent_label}': {plot_path}")
            elif phase_type == "geog":
                print(
                    f"GEOG data ready for '{agent_label}': "
                    f"{result['geog_dir']}"
                )

        return prepared

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

                mesh_results = await self.mesh_phase()
                print(f"Mesh generation completed for agents: {list(mesh_results)}")
                print(f"Mesh results: {mesh_results}")

                # mpas_results = await self.mpas_phase()
                # print(f"MPAS build completed for agents: {list(mpas_results)}")
                # print(f"MPAS results: {mpas_results}")

                # geog_results = await self.download_geog_phase()
                # print(f"GEOG download completed for agents: {list(geog_results)}")
                # print(f"GEOG results: {geog_results}")

                # interp_results = await self.interpolate_geog_phase(mesh_results)
                # if interp_results:
                #     print(f"GEOG interpolation completed: {interp_results}")
                # else:
                #     print("GEOG interpolation skipped (not a project_hexes mesh)")

                # print(
                #     "Preparing core assets (MPAS build, mesh generation, "
                #     "and optional GEOG download) concurrently..."
                # )
                # prepared_assets = await self.prepare_core_assets_phase()

                # GEOG interpolation is temporarily disabled until the
                # geog_interpolation configuration path is finalized.
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
                #     for idx, mesh_result in enumerate(mesh_results):
                #         mesh_label = self._agents_by_type.get("mesh", [])[idx]
                #         mesh_file = mesh_result.get("mesh")
                #         if not mesh_file:
                #             continue
                #         geog_tasks.append(
                #             mpas_agent.interpolate_geog_only(
                #                 mesh_file=mesh_file,
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
