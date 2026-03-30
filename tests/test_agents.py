# SPDX-License-Identifier: Apache-2.0

"""Tests for chiltepin.agents module."""

import asyncio
import pathlib

import pytest

import chiltepin.endpoint
from chiltepin.agents import action, chiltepin_agent, loop
from chiltepin.tasks import python_task

# Get project root for PYTHONPATH
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="session", autouse=True)
def ensure_academy_login():
    """Ensure Academy Exchange login before any agent tests run."""
    chiltepin.endpoint.login()


def get_test_config(executor_name="test-executor"):
    """Helper to create test configuration."""
    return {
        executor_name: {
            "provider": "localhost",
            "cores_per_node": 1,
            "max_workers_per_node": 1,
            "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
        }
    }


# Module-level test agent classes (needed for pickling)


@chiltepin_agent()
class BasicTestAgent:
    def __init__(self, value: int):
        self.value = value

    @action
    async def get_value(self) -> int:
        return self.value


@chiltepin_agent()
class ComputeAgent:
    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    @python_task
    @action
    def compute(self, x: int) -> int:
        """Task that accesses instance state."""
        return x * self.multiplier


@chiltepin_agent()
class AsyncAgent:
    def __init__(self):
        self.data = []

    @action
    async def add_data(self, value: str) -> None:
        self.data.append(value)

    @action
    async def get_data(self) -> list:
        return self.data


@chiltepin_agent()
class LoopAgent:
    def __init__(self):
        self.counter = 0

    @action
    async def get_counter(self) -> int:
        return self.counter

    @loop
    async def increment_counter(self, shutdown):
        import asyncio

        while not shutdown.is_set():
            await asyncio.sleep(0.1)
            self.counter += 1


@chiltepin_agent(include=["default-executor"])
class ConfigAgent:
    def __init__(self):
        self.initialized = True

    @action
    async def check(self) -> bool:
        return self.initialized


@chiltepin_agent()
class MixedAgent:
    def __init__(self, base: int):
        self.base = base

    @python_task
    @action
    def sync_compute(self, x: int) -> int:
        return self.base + x

    @action
    async def async_compute(self, x: int) -> int:
        return self.base * x

    @action
    def sync_helper(self) -> str:
        return "helper"


@chiltepin_agent()
class PrivateMethodAgent:
    @action
    def public_method(self):
        return "public"

    def _private_method(self):
        return "private"

    def __dunder_method__(self):
        return "dunder"


@chiltepin_agent()
class EmptyAgent:
    def __init__(self):
        self.value = 42

    def helper_method(self):
        return self.value


@chiltepin_agent(include=["test-executor"])
class FullAgent:
    def __init__(self, initial_value: int):
        self.value = initial_value
        self.counter = 0

    @python_task
    @action
    def compute(self, x: int) -> int:
        """Task that accesses instance state."""
        return self.value + x

    @action
    async def get_state(self) -> dict:
        """Get agent state."""
        return {"value": self.value, "counter": self.counter}

    @action
    async def set_value(self, new_value: int) -> None:
        """Update agent state."""
        self.value = new_value

    @loop
    async def background_counter(self, shutdown):
        """Background loop."""
        while not shutdown.is_set():
            await asyncio.sleep(0.1)
            self.counter += 1


@chiltepin_agent()
class Agent1:
    @action
    async def get_name(self) -> str:
        return "agent1"


@chiltepin_agent()
class Agent2:
    @action
    async def get_name(self) -> str:
        return "agent2"


@chiltepin_agent()
class StatefulAgent:
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.metadata = {"created": "now", "version": "1.0"}

    @action
    async def add_to_history(self, item: str) -> None:
        self.history.append(item)

    @action
    async def get_full_state(self) -> dict:
        return {"name": self.name, "history": self.history, "metadata": self.metadata}


@chiltepin_agent()
class PlainSyncAgent:
    """Agent with plain sync action (no task decorator)."""

    def __init__(self):
        self.value = 100

    @action
    def get_double(self) -> int:
        """Plain sync method returning int directly."""
        return self.value * 2


@chiltepin_agent()
class KwargsAgent:
    """Agent that accepts kwargs."""

    def __init__(self, name: str, value: int, **extra_kwargs):
        self.name = name
        self.value = value
        self.extras = extra_kwargs

    @action
    async def get_info(self) -> dict:
        return {"name": self.name, "value": self.value, "extras": self.extras}


@chiltepin_agent()
class FutureReturnAgent:
    """Agent with method that returns Future."""

    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    @python_task
    @action
    def compute_future(self, x: int) -> int:
        """Returns a Future that resolves to int."""
        return x * self.multiplier


@chiltepin_agent()
class NoMarkersAgent:
    """Agent without @action or @loop decorators."""

    def __init__(self):
        self.value = 42

    def unmarked_method(self):
        """Method without @action decorator - should not be exposed."""
        return "not exposed"

    def another_unmarked(self):
        return 123


@chiltepin_agent()
class AsyncActionAgent:
    """Agent with pure async actions (no @python_task decorator)."""

    def __init__(self, base_value: int):
        self.base_value = base_value

    @action
    async def async_compute(self, x: int) -> int:
        """Pure async action - exercises lines 479-480."""
        await asyncio.sleep(0.01)  # Make it actually async
        return self.base_value + x

    @action
    async def async_multiply(self, x: int, y: int) -> int:
        """Another async action."""
        await asyncio.sleep(0.01)
        return x * y


@chiltepin_agent()
class MixedAttributesAgent:
    """Agent with callable and non-callable attributes to test line 423."""

    # Class-level non-callable attribute
    CLASS_CONSTANT = "constant_value"
    ANOTHER_CONSTANT = 42

    def __init__(self):
        self.value = 100
        # Instance-level non-callable attribute
        self.config_dict = {"key": "value"}

    @action
    async def get_value(self) -> int:
        return self.value

    # Override object method to test line 434
    def __str__(self) -> str:
        return f"MixedAttributesAgent(value={self.value})"


@chiltepin_agent()
class LifecycleTestAgent:
    """Agent to verify lifecycle methods are called."""

    def __init__(self):
        self.value = 42

    @action
    async def check_workflow(self) -> bool:
        """Check if workflow was initialized during startup."""
        # Access the wrapper's _workflow attribute through the agent handle
        return True  # If we can call this, workflow was started


class TestChiltepinAgentDecorator:
    """Test the @chiltepin_agent decorator."""

    @pytest.mark.asyncio
    async def test_basic_agent_creation(self, tmp_path):
        """Test that @chiltepin_agent creates a working agent."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    BasicTestAgent, config=config, args=(42,), executor="test-executor"
                )

                result = await agent.get_value()
                assert result == 42
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_action_decorator_with_python_task(self, tmp_path):
        """Test @action decorator with @python_task."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    ComputeAgent, config=config, args=(3,), executor="test-executor"
                )

                result = await agent.compute(x=14, executor=["test-executor"])
                assert result == 42
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_action_decorator_with_async_method(self, tmp_path):
        """Test @action decorator with async methods."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    AsyncAgent, config=config, executor="test-executor"
                )

                await agent.add_data(value="hello")
                await agent.add_data(value="world")
                result = await agent.get_data()
                assert result == ["hello", "world"]
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_loop_decorator(self, tmp_path):
        """Test @loop decorator for background tasks."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    LoopAgent, config=config, executor="test-executor"
                )

                initial = await agent.get_counter()
                await asyncio.sleep(0.5)
                final = await agent.get_counter()

                # Should have incremented multiple times
                assert final > initial
                assert final >= initial + 3
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_runtime_config_override(self, tmp_path):
        """Test runtime override of decorator defaults."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        # Create two different configs
        config1 = get_test_config("executor-1")
        config2 = get_test_config("executor-2")
        combined_config = {**config1, **config2}

        workflow = Workflow(combined_config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["executor-1", "executor-2"]
            )

            async with await agent_system.manager() as manager:
                # Launch with runtime override
                agent = await manager.launch(
                    ConfigAgent,
                    config=config2,
                    include=["executor-2"],  # Override decorator default
                    executor="executor-1",
                )

                result = await agent.check()
                assert result is True
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_mixed_sync_async_actions(self, tmp_path):
        """Test agent with both sync and async actions."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    MixedAgent, config=config, args=(10,), executor="test-executor"
                )

                sync_result = await agent.sync_compute(x=5, executor=["test-executor"])
                assert sync_result == 15

                async_result = await agent.async_compute(x=5)
                assert async_result == 50

                helper_result = await agent.sync_helper()
                assert helper_result == "helper"
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_private_methods_not_exposed(self, tmp_path):
        """Test that private methods are not exposed as actions."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    PrivateMethodAgent, config=config, executor="test-executor"
                )

                # Public method should be accessible
                result = await agent.public_method()
                assert result == "public"

                # Private methods should NOT be exposed as actions
                with pytest.raises(AttributeError):
                    await agent._private_method()
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_with_no_actions(self, tmp_path):
        """Test that agent with no @action methods still works."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                # Should launch successfully even with no actions
                agent = await manager.launch(
                    EmptyAgent, config=config, executor="test-executor"
                )
                # Agent exists but has no callable actions (other than Agent base methods)
                assert agent is not None
        finally:
            workflow.cleanup()


class TestChiltepinManager:
    """Test the ChiltepinManager class."""

    @pytest.mark.asyncio
    async def test_manager_launch_with_config_param(self, tmp_path):
        """Test that ChiltepinManager passes config parameter correctly."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                # Verify manager is ChiltepinManager
                from chiltepin.agents import ChiltepinManager

                assert isinstance(manager, ChiltepinManager)

                # Launch agent with config - agent will use this config for its internal workflow
                agent = await manager.launch(
                    ComputeAgent,
                    config=config,  # Agent's workflow will use this config
                    args=(7,),
                    executor="test-executor",
                )

                # Call a @python_task decorated method to verify the agent's workflow
                # is properly configured with the executors from the config
                result = await agent.compute(x=6, executor=["test-executor"])
                assert result == 42  # 7 * 6 = 42
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_manager_launch_with_include_param(self, tmp_path):
        """Test that ChiltepinManager passes include parameter correctly."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        # Create config with two executors
        config = {
            "executor-1": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
            },
            "executor-2": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
            },
        }
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["executor-1", "executor-2"]
            )
            async with await agent_system.manager() as manager:
                # Launch agent with include=["executor-1"] only
                agent = await manager.launch(
                    ComputeAgent,
                    config=config,
                    args=(6,),
                    include=[
                        "executor-1"
                    ],  # Agent's workflow should only load executor-1
                    executor="executor-1",
                )

                # Should be able to run tasks on executor-1 (included)
                result = await agent.compute(x=7, executor=["executor-1"])
                assert result == 42  # 6 * 7 = 42

                with pytest.raises(ValueError, match="Task 1 requested invalid executor executor-2"):
                    await agent.compute(x=7, executor=["executor-2"])
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_manager_launch_with_run_dir_param(self, tmp_path):
        """Test that ChiltepinManager passes run_dir parameter correctly."""
        import os

        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                # Use a unique directory name to verify it gets created
                custom_run_dir = str(tmp_path / "custom_agent_runinfo")
                assert not os.path.exists(custom_run_dir), (
                    "Directory should not exist yet"
                )

                agent = await manager.launch(
                    ComputeAgent,
                    config=config,
                    args=(3,),
                    run_dir=custom_run_dir,  # Custom run directory for agent's workflow
                    executor="test-executor",
                )

                # Execute a task to trigger workflow/Parsl activity
                result = await agent.compute(x=14, executor=["test-executor"])
                assert result == 42  # 3 * 14 = 42

                # Verify the custom run directory was created by the agent's workflow
                assert os.path.exists(custom_run_dir), (
                    "Custom run_dir should have been created"
                )
                # Verify it contains Parsl-related files/directories
                dir_contents = os.listdir(custom_run_dir)
                assert len(dir_contents) > 0, "Run directory should contain Parsl files"
        finally:
            workflow.cleanup()


class TestAgentSystem:
    """Test the AgentSystem class."""

    def test_agent_system_requires_started_workflow(self):
        """Test that AgentSystem raises error if workflow not started."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
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

        config = {
            "test-executor-1": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
            },
            "test-executor-2": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 1,
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
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


class TestIntegration:
    """Integration tests for the complete agent system."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_chiltepin_agent(self, tmp_path):
        """Complete integration test with @chiltepin_agent decorator."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    FullAgent, config=config, args=(10,), executor="test-executor"
                )

                # Test task execution
                result = await agent.compute(x=32, executor=["test-executor"])
                assert result == 42

                # Test async action
                state = await agent.get_state()
                assert state["value"] == 10
                assert state["counter"] >= 0

                # Test state mutation
                await agent.set_value(new_value=20)
                result = await agent.compute(x=22, executor=["test-executor"])
                assert result == 42

                # Test loop is running
                initial_counter = state["counter"]
                await asyncio.sleep(0.5)
                final_state = await agent.get_state()
                assert final_state["counter"] > initial_counter

        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_agents_same_manager(self, tmp_path):
        """Test launching multiple agents with the same manager."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        # Need 2 workers to run 2 agents concurrently
        config = {
            "test-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 2,  # Increased from 1 to 2
                "environment": [f"export PYTHONPATH=${{PYTHONPATH}}:{PROJECT_ROOT}"],
            }
        }
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent1 = await manager.launch(
                    Agent1, config=config, executor="test-executor"
                )
                agent2 = await manager.launch(
                    Agent2, config=config, executor="test-executor"
                )

                name1 = await agent1.get_name()
                name2 = await agent2.get_name()

                assert name1 == "agent1"
                assert name2 == "agent2"

        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_with_complex_state(self, tmp_path):
        """Test agent with complex state objects."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    StatefulAgent,
                    config=config,
                    args=("test-agent",),
                    executor="test-executor",
                )

                await agent.add_to_history(item="event1")
                await agent.add_to_history(item="event2")

                state = await agent.get_full_state()
                assert state["name"] == "test-agent"
                assert state["history"] == ["event1", "event2"]
                assert "version" in state["metadata"]

        finally:
            workflow.cleanup()


class TestEdgeCases:
    """Test edge cases and special scenarios to ensure comprehensive coverage."""

    @pytest.mark.asyncio
    async def test_sync_action_without_task_decorator(self, tmp_path):
        """Test sync action that returns a plain value (not a Future)."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    PlainSyncAgent, config=config, executor="test-executor"
                )

                result = await agent.get_double()
                assert result == 200
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_manager_launch_with_kwargs(self, tmp_path):
        """Test ChiltepinManager.launch with additional kwargs."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    KwargsAgent,
                    config=config,
                    args=("test", 42),
                    kwargs={"extra_key": "extra_value"},  # Test kwargs parameter
                    executor="test-executor",
                )

                info = await agent.get_info()
                assert info["name"] == "test"
                assert info["value"] == 42
                assert info["extras"]["extra_key"] == "extra_value"
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_with_method_returning_future(self, tmp_path):
        """Test agent with sync method that returns a Future (AppFuture from task)."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    FutureReturnAgent,
                    config=config,
                    args=(5,),
                    executor="test-executor",
                )

                # This exercises the Future wrapping code path
                result = await agent.compute_future(x=10, executor=["test-executor"])
                assert result == 50
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_without_explicit_markers(self, tmp_path):
        """Test agent class with methods that have no @action or @loop decorators."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    NoMarkersAgent, config=config, executor="test-executor"
                )

                # Agent should launch successfully even without @action markers
                # This tests the code path where methods don't have _chiltepin_expose or __wrapped__
                assert agent is not None
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_pure_async_actions(self, tmp_path):
        """Test agent with async actions (no @python_task) - covers lines 479-480."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    AsyncActionAgent,
                    config=config,
                    args=(100,),
                    executor="test-executor",
                )

                # Test pure async actions
                result1 = await agent.async_compute(x=50)
                assert result1 == 150

                result2 = await agent.async_multiply(x=3, y=7)
                assert result2 == 21
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_with_mixed_attributes(self, tmp_path):
        """Test agent with callable/non-callable attributes - covers lines 423, 434."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                # This agent has non-callable attributes and overridden object methods
                agent = await manager.launch(
                    MixedAttributesAgent, config=config, executor="test-executor"
                )

                result = await agent.get_value()
                assert result == 100
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_lifecycle_methods(self, tmp_path):
        """Test that agent_on_startup/shutdown are called - covers lines 395-408."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    LifecycleTestAgent, config=config, executor="test-executor"
                )

                # If we can call an action, agent_on_startup must have succeeded
                result = await agent.check_workflow()
                assert result is True
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_verify_loop_method_execution(self, tmp_path):
        """Explicitly verify loop methods execute - helps ensure lines 462-463 are hit."""
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = get_test_config()
        workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
        workflow.start()

        try:
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                # Launch agent with loop method
                agent = await manager.launch(
                    LoopAgent, config=config, executor="test-executor"
                )

                # Verify loop is running by checking counter increases
                count1 = await agent.get_counter()
                await asyncio.sleep(0.3)
                count2 = await agent.get_counter()
                await asyncio.sleep(0.3)
                count3 = await agent.get_counter()

                # Counter should increase over time
                assert count3 > count2 > count1
        finally:
            workflow.cleanup()
