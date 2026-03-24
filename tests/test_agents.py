# SPDX-License-Identifier: Apache-2.0

"""Tests for chiltepin.agents module."""

import pytest


class TestAgentSystem:
    """Test the AgentSystem class."""

    def test_agent_system_requires_started_workflow(self):
        """Test that AgentSystem raises error if workflow not started."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        # Create workflow but don't start it
        config = {
            "test-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
            }
        }
        workflow = Workflow(config)

        # Create AgentSystem (should work)
        agent_system = AgentSystem(
            workflow=workflow,
            executor_names=["test-executor"],
            exchange_address="https://test.example.com",
        )

        # But trying to create executors should fail
        with pytest.raises(RuntimeError, match="Workflow must be started"):
            agent_system._create_executors()

    def test_agent_system_creates_executors(self, tmp_path):
        """Test that AgentSystem creates ParslPoolExecutors correctly."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem
        import pathlib

        # Get project root for PYTHONPATH
        project_root = pathlib.Path(__file__).parent.parent.resolve()

        config = {
            "test-executor-1": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            },
            "test-executor-2": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            },
        }

        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow,
                executor_names=["test-executor-1", "test-executor-2"],
                exchange_address="https://test.example.com",
            )

            # Executors should not be created yet
            assert agent_system.executors is None

            # Create executors explicitly
            agent_system._create_executors()

            # Now executors should exist
            assert agent_system.executors is not None
            assert len(agent_system.executors) == 2
            assert "test-executor-1" in agent_system.executors
            assert "test-executor-2" in agent_system.executors

            # Verify they're ParslPoolExecutors
            from parsl.concurrent import ParslPoolExecutor

            for executor in agent_system.executors.values():
                assert isinstance(executor, ParslPoolExecutor)

        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_system_full_integration(self, tmp_path):
        """Full integration test with real Manager and Agent.

        This test validates the complete AgentSystem workflow including:
        - Creating a real Manager with HttpExchangeFactory
        - Launching an actual Agent on a ParslPoolExecutor
        - Calling an action on the agent that executes a Parsl task
        - Testing @loop decorator for autonomous agent behavior
        - Verifying the agent action returns the expected result
        """
        import asyncio
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem
        from chiltepin.tasks import python_task
        from academy.agent import Agent, action, loop
        import pathlib

        project_root = pathlib.Path(__file__).parent.parent.resolve()

        @python_task
        def simple_computation(x: int) -> int:
            return x * 2

        class TestAgent(Agent):
            def __init__(self, config):
                self.config = config
                self.counter = 0

            async def agent_on_startup(self) -> None:
                from chiltepin import Workflow

                self.workflow = Workflow(self.config, include=["test-executor"])
                self.dfk = self.workflow.start()

            async def agent_on_shutdown(self) -> None:
                self.workflow.cleanup()

            @action
            async def compute(self, x: int) -> int:
                import asyncio

                return await asyncio.wrap_future(simple_computation(x))

            @action
            async def get_counter(self) -> int:
                """Return the current counter value."""
                return self.counter

            @loop
            async def increment_counter(self, shutdown: asyncio.Event) -> None:
                """Autonomous loop that increments counter."""
                while not shutdown.is_set():
                    await asyncio.sleep(0.5)
                    self.counter += 1

        config = {
            "test-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{project_root}"],
            }
        }

        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow,
                executor_names=["test-executor"],
            )

            async with await agent_system.manager() as manager:
                # Launch a test agent
                agent = await manager.launch(
                    TestAgent,
                    args=(config,),
                    executor="test-executor",
                )

                # Test @action decorator with task execution
                result = await agent.compute(x=21)
                assert result == 42

                # Test @loop decorator - verify autonomous behavior
                initial_count = await agent.get_counter()
                await asyncio.sleep(2)  # Let the loop run
                final_count = await agent.get_counter()

                # Counter should have incremented (at least 3 times in 2 seconds with 0.5s sleep)
                assert final_count > initial_count
                assert final_count >= initial_count + 3

        finally:
            workflow.cleanup()

