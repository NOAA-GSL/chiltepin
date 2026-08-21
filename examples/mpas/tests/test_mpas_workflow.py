# SPDX-License-Identifier: Apache-2.0

"""Tests for MPASForecastWorkflow."""

from copy import deepcopy

import pytest
from workflows import MPASForecastWorkflow


class TestMPASForecastWorkflow:
    """Test suite for MPASForecastWorkflow."""

    def test_init(self, test_config):
        """Test MPASForecastWorkflow initialization."""
        workflow = MPASForecastWorkflow(test_config)

        assert workflow.config == test_config
        assert workflow.workflow is None  # Not started yet
        assert workflow.agent_runtime is None
        assert workflow.manager is None
        assert workflow.agents == {}

    @pytest.mark.asyncio
    async def test_mesh_phase_uses_prompt_when_configured(self, test_config):
        """mesh_phase should use create_mesh_from_prompt when model.mesh_prompt is set."""
        workflow_config = deepcopy(test_config)
        workflow_config["model"] = {
            "mesh_prompt": "Generate a 15km MPAS mesh that covers Japan",
            "llm": {
                "model": "qwen2.5:3b",
                "url": "http://localhost:11434/api/chat",
            },
            "init_ranks": 2,
            "forecast_ranks": 4,
        }
        workflow_config["agents"] = [
            {
                "name": "mesh-foo",
                "type": "mesh",
                "work_dir": "/tmp/mesh-foo",
                "workflow_config": {},
            }
        ]

        class StubMeshAgent:
            def __init__(self):
                self.calls = []

            async def create_mesh_from_prompt(
                self, prompt, mesh_data_dir, model, llm_url
            ):
                self.calls.append((prompt, mesh_data_dir, model, llm_url))
                return {
                    "mesh_config": {"resolution": "15km", "name": "japan_15km"},
                    "mesh_result": {
                        "mesh": "mesh.nc",
                        "graph": "graph.info",
                        "partitions": {},
                    },
                }

        workflow = MPASForecastWorkflow(workflow_config)
        stub_agent = StubMeshAgent()
        workflow.agents = {"mesh-foo": stub_agent}
        workflow._agents_by_type["mesh"].append("mesh-foo")

        result = await workflow.mesh_phase()

        assert stub_agent.calls == [
            (
                "Generate a 15km MPAS mesh that covers Japan",
                "/tmp/mesh-foo/mesh_data",
                "qwen2.5:3b",
                "http://localhost:11434/api/chat",
            )
        ]
        assert result["mesh-foo"]["mesh"] == "mesh.nc"
        assert result["mesh-foo"]["mesh_config"]["resolution"] == "15km"

    # TODO: Add tests for workflow phases
    # These tests will require mocking agent actions and workflow components
    #
    # Example tests to add:
    # - test_setup_agents() - test agent launching
    # - test_build_phase() - test parallel builds
    # - test_mesh_phase() - test mesh generation and partitioning
    # - test_preprocess_phase() - test data preprocessing
    # - test_initialization_phase() - test MPAS initialization
    # - test_forecast_phase() - test forecast execution
    # - test_cleanup() - test proper cleanup
    # - test_run() - test complete workflow execution
    # - test_error_handling() - test cleanup on errors
