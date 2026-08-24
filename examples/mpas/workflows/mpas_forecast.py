# SPDX-License-Identifier: Apache-2.0

"""MPASForecastWorkflow - Orchestrates MPAS multi-agent forecast.

This module provides the main workflow orchestration class that coordinates
all MPAS component agents to build, configure, and run MPAS forecasts.
"""

import asyncio
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from agents import MeshAgent, MPASAgent, WPSAgent

from chiltepin import AgentRuntime, Workflow
from chiltepin.manager import Manager

logger = logging.getLogger(__name__)

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

    @staticmethod
    def resolve_llm_api_key(config: Dict[str, Any]) -> Optional[str]:
        """Resolve LLM API key from config or environment variable."""
        llm_cfg = config.get("model", {}).get("llm", {})
        key = llm_cfg.get("api_key")
        if key:
            return key
        env_name = llm_cfg.get("api_key_env")
        if env_name:
            return os.environ.get(env_name)
        return None

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
            run_dir=str(Path(self.config.get("experiment_dir")) / "parsl_logs"),
        )
        self.workflow.start()

        self.agent_runtime = AgentRuntime(workflow=self.workflow)

    def agents_by_type(self, agent_type: str) -> List[Any]:
        """Return all launched agent handles of the given type."""
        return [self.agents[name] for name in self._agents_by_type.get(agent_type, [])]

    def _build_agent_args(self, agent_conf: Dict[str, Any]) -> tuple:
        """Build (positional_args, keyword_args) for an agent based on its type."""
        agent_type = agent_conf["type"]
        work_dir = str(
            Path(agent_conf.get("work_dir") or tempfile.mkdtemp(prefix="chiltepin_"))
        )
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
        api_key = self.resolve_llm_api_key(self.config)
        api_base = llm_config.get("api_base")

        tasks = []
        for label, agent in zip(self._agents_by_type["mesh"], mesh_agents):
            agent_conf = self.config.get("agents", [])
            matched = next((c for c in agent_conf if c.get("name") == label), None)
            if matched is None:
                raise RuntimeError(f"Missing configuration for mesh agent '{label}'")

            mesh_data_dir = str(
                Path(matched.get("work_dir") or tempfile.mkdtemp(prefix="chiltepin_"))
                / "mesh_data"
            )
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
        results = await asyncio.gather(*[a.install_mpas() for a in mpas_agents])
        return dict(zip(self._agents_by_type["mpas"], results))

    async def download_geog_phase(self) -> Dict[str, Any]:
        """Download GEOG data on all MPAS agents concurrently.

        Returns
        -------
        dict
            Per-agent results keyed by agent name
        """
        mpas_agents = self.agents_by_type("mpas")
        results = await asyncio.gather(*[a.download_geog_data() for a in mpas_agents])
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
            tasks.append(
                mpas_agent.interpolate_geog_only(
                    mesh_file=mesh_file,
                    num_ranks=init_ranks,
                )
            )
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

    async def _prepare_mpas_agent(self, agent: Any, label: str) -> Dict[str, Any]:
        """Download and build MPAS for one agent with clear stage errors."""
        try:
            source_dir = await agent.download_mpas()
        except Exception as e:
            raise RuntimeError(f"MPAS download failed for agent '{label}': {e}") from e

        try:
            build_result = await agent.build()
        except Exception as e:
            raise RuntimeError(f"MPAS build failed for agent '{label}': {e}") from e

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

            mesh_data_dir = str(
                Path(matched.get("work_dir") or tempfile.mkdtemp(prefix="chiltepin_"))
                / "mesh_data"
            )
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
                phase_tasks.append(
                    asyncio.create_task(agent.download_geog_data(path=geog_path))
                )
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
                logger.info(
                    "MPAS build completed for '%s': %s", agent_label, result["build"]
                )
            elif phase_type == "mesh":
                plot_path = result.get("plot")
                plot_error = result.get("plot_error")
                if plot_error:
                    logger.warning(
                        "Mesh plot generation failed for agent '%s': %s",
                        agent_label,
                        plot_error,
                    )
                if plot_path:
                    logger.info(
                        "Mesh plot generated for '%s': %s", agent_label, plot_path
                    )
            elif phase_type == "geog":
                logger.info(
                    "GEOG data ready for '%s': %s", agent_label, result["geog_dir"]
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
                logger.info(
                    "Mesh generation completed for agents: %s", list(mesh_results)
                )
                logger.info("Mesh results: %s", mesh_results)

                # Shutdown agents while manager/exchange client is still alive
                await self.shutdown_agents()

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
                logger.warning("Error shutting down %s: %s", agent_name, e)

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
                logger.warning("Error cleaning up workflow: %s", e)
