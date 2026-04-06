# SPDX-License-Identifier: Apache-2.0

"""Tests for chiltepin.agents module."""

import asyncio
import pathlib

import pytest
from academy.agent import Agent

import chiltepin.endpoint as endpoint
from chiltepin.agents import agent_action, agent_loop, chiltepin_agent
from chiltepin.tasks import bash_task, join_task, python_task

# Get project root for PYTHONPATH
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture(scope="session", autouse=True)
def ensure_academy_login():
    """Make sure we are already logged in to the Academy Exchange before any agent tests run.

    In CI environments, the login will obtain credentials from environment variables when
    login_required() is called and will not prompt for input.
    """
    if endpoint.login_required():
        raise RuntimeError(
            "Chiltepin login is required to run agent tests. Please log in before running tests."
        )


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

    @agent_action
    async def get_value(self) -> int:
        return self.value


@chiltepin_agent()
class ComputeAgent:
    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    @python_task
    @agent_action
    def compute(self, x: int) -> int:
        """Task that accesses instance state."""
        return x * self.multiplier


@chiltepin_agent()
class AsyncAgent:
    def __init__(self):
        self.data = []

    @agent_action
    async def add_data(self, value: str) -> None:
        self.data.append(value)

    @agent_action
    async def get_data(self) -> list:
        return self.data


@chiltepin_agent()
class LoopAgent:
    def __init__(self):
        self.counter = 0

    @agent_action
    async def get_counter(self) -> int:
        return self.counter

    @agent_loop
    async def increment_counter(self, shutdown):
        import asyncio

        while not shutdown.is_set():
            await asyncio.sleep(0.1)
            self.counter += 1


@chiltepin_agent(agent_workflow_include=["default-executor"])
class ConfigAgent:
    def __init__(self):
        self.initialized = True

    @agent_action
    async def check(self) -> bool:
        return self.initialized


@chiltepin_agent()
class MixedAgent:
    def __init__(self, base: int):
        self.base = base

    @python_task
    @agent_action
    def sync_compute(self, x: int) -> int:
        return self.base + x

    @agent_action
    async def async_compute(self, x: int) -> int:
        return self.base * x

    @agent_action
    def sync_helper(self) -> str:
        return "helper"


@chiltepin_agent()
class PrivateMethodAgent:
    @agent_action
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


@chiltepin_agent()
class PositionalArgsAgent:
    """Agent to test positional arguments in agent actions."""

    def __init__(self, base: int):
        self.base = base

    @agent_action
    @python_task
    def add_numbers(self, a: int, b: int, c: int = 0) -> int:
        """Sync action with positional and keyword args."""
        return self.base + a + b + c

    @agent_action
    async def multiply_numbers(self, x: int, y: int) -> int:
        """Async action with positional args."""
        return self.base * x * y

    @agent_action
    async def concat_strings(self, s1: str, s2: str, sep: str = " ") -> str:
        """Async action with mixed positional and keyword args."""
        return f"{s1}{sep}{s2}"


@chiltepin_agent()
class BashTaskAgent:
    """Agent to test @bash_task decorated methods."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    @agent_action
    @bash_task
    def echo_message(self, message):
        """Return bash command to echo a message."""
        return f"echo '{self.prefix}: {message}'"

    @agent_action
    @bash_task
    def write_file(self, filepath, content):
        """Return bash command to write content to a file."""
        return f"echo '{content}' > {filepath}"


@chiltepin_agent()
class JoinTaskAgent:
    """Agent to test @join_task decorated methods."""

    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    @python_task
    def _compute_part(self, x: int) -> int:
        """Helper task for join demonstration."""
        return x * self.multiplier

    @python_task
    def _sum_results(self, a: int, b: int) -> int:
        """Sum two values - will receive resolved future values."""
        return a + b

    @agent_action
    @join_task
    def compute_sum(self, a: int, b: int):
        """Join task that combines multiple task futures.

        The join_task launches multiple tasks and returns the final future.
        Parsl will automatically resolve dependencies.
        """
        # Launch multiple tasks
        future_a = self._compute_part(a, executor=["test-executor"])
        future_b = self._compute_part(b, executor=["test-executor"])
        # Use another task to sum them - Parsl handles dependency resolution
        # The @python_task decorator will automatically wait for future_a and future_b
        return self._sum_results(future_a, future_b, executor=["test-executor"])


@chiltepin_agent(agent_workflow_include=["test-executor"])
class FullAgent:
    def __init__(self, initial_value: int):
        self.value = initial_value
        self.counter = 0

    @python_task
    @agent_action
    def compute(self, x: int) -> int:
        """Task that accesses instance state."""
        return self.value + x

    @agent_action
    async def get_state(self) -> dict:
        """Get agent state."""
        return {"value": self.value, "counter": self.counter}

    @agent_action
    async def set_value(self, new_value: int) -> None:
        """Update agent state."""
        self.value = new_value

    @agent_loop
    async def background_counter(self, shutdown):
        """Background agent_loop."""
        while not shutdown.is_set():
            await asyncio.sleep(0.1)
            self.counter += 1


@chiltepin_agent()
class Agent1:
    @agent_action
    async def get_name(self) -> str:
        return "agent1"


@chiltepin_agent()
class Agent2:
    @agent_action
    async def get_name(self) -> str:
        return "agent2"


@chiltepin_agent()
class StatefulAgent:
    def __init__(self, name: str):
        self.name = name
        self.history = []
        self.metadata = {"created": "now", "version": "1.0"}

    @agent_action
    async def add_to_history(self, item: str) -> None:
        self.history.append(item)

    @agent_action
    async def get_full_state(self) -> dict:
        return {"name": self.name, "history": self.history, "metadata": self.metadata}


@chiltepin_agent()
class PlainSyncAgent:
    """Agent with plain sync agent_action (no task decorator)."""

    def __init__(self):
        self.value = 100

    @agent_action
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

    @agent_action
    async def get_info(self) -> dict:
        return {"name": self.name, "value": self.value, "extras": self.extras}


@chiltepin_agent()
class FutureReturnAgent:
    """Agent with method that returns Future."""

    def __init__(self, multiplier: int):
        self.multiplier = multiplier

    @python_task
    @agent_action
    def compute_future(self, x: int) -> int:
        """Returns a Future that resolves to int."""
        return x * self.multiplier


@chiltepin_agent()
class NoMarkersAgent:
    """Agent without @agent_action or @agent_loop decorators."""

    def __init__(self):
        self.value = 42

    def unmarked_method(self):
        """Method without @agent_action decorator - should not be exposed."""
        return "not exposed"

    def another_unmarked(self):
        return 123


@chiltepin_agent()
class AsyncActionAgent:
    """Agent with pure async actions (no @python_task decorator)."""

    def __init__(self, base_value: int):
        self.base_value = base_value

    @agent_action
    async def async_compute(self, x: int) -> int:
        """Pure async agent_action."""
        await asyncio.sleep(0.01)
        return self.base_value + x

    @agent_action
    async def async_multiply(self, x: int, y: int) -> int:
        """Another async agent_action."""
        await asyncio.sleep(0.01)
        return x * y


@chiltepin_agent()
class MixedAttributesAgent:
    """Agent with callable and non-callable attributes."""

    # Class-level non-callable attribute
    CLASS_CONSTANT = "constant_value"
    ANOTHER_CONSTANT = 42

    def __init__(self):
        self.value = 100
        # Instance-level non-callable attribute
        self.config_dict = {"key": "value"}

    @agent_action
    async def get_value(self) -> int:
        return self.value

    # Override object method
    def __str__(self) -> str:
        return f"MixedAttributesAgent(value={self.value})"


@chiltepin_agent()
class LifecycleTestAgent:
    """Agent to verify lifecycle methods are called."""

    def __init__(self):
        self.value = 42

    @agent_action
    async def check_workflow(self) -> bool:
        """Check if workflow was initialized during startup."""
        # Access the wrapper's _workflow attribute through the agent handle
        return True  # If we can call this, workflow was started


@chiltepin_agent()
class ReverseOrderAgent:
    def __init__(self, value: int):
        self.value = value

    @agent_action
    @python_task
    def compute_action_outer(self, x: int) -> int:
        """@agent_action outer, @python_task inner"""
        return self.value + x

    @python_task
    @agent_action
    def compute_task_outer(self, x: int) -> int:
        """@python_task outer, @agent_action inner (recommended)"""
        return self.value * x


# Agent composition classes - demonstrating agents that use other agents
@chiltepin_agent()
class LowererAgent:
    """Agent that converts text to lowercase."""

    @agent_action
    async def lower(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()


@chiltepin_agent()
class ReverserAgent:
    """Agent that reverses text."""

    @agent_action
    async def reverse(self, text: str) -> str:
        """Reverse text."""
        return text[::-1]


@chiltepin_agent()
class CoordinatorAgent:
    """Agent that coordinates other agents to process text."""

    def __init__(self, lowerer, reverser):
        """Initialize with handles to other agents.

        Parameters
        ----------
        lowerer : Handle[LowererAgent]
            Handle to the lowerer agent
        reverser : Handle[ReverserAgent]
            Handle to the reverser agent
        """
        self.lowerer = lowerer
        self.reverser = reverser

    @agent_action
    async def process(self, text: str) -> str:
        """Process text by lowering then reversing it.

        This demonstrates agent composition - calling actions on other agents.
        """
        text = await self.lowerer.lower(text)
        text = await self.reverser.reverse(text)
        return text


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
            # It is ok to use public Academy exchange for tests
            agent_system = AgentSystem(
                workflow=workflow, executor_names=["test-executor"]
            )
            async with await agent_system.manager() as manager:
                agent = await manager.launch(
                    BasicTestAgent,
                    agent_workflow_config=config,
                    args=(42,),
                    executor="test-executor",
                )

                result = await agent.get_value()
                assert result == 42
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_action_decorator_with_python_task(self, tmp_path):
        """Test @agent_action decorator with @python_task."""
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
                    ComputeAgent,
                    agent_workflow_config=config,
                    args=(3,),
                    executor="test-executor",
                )

                result = await agent.compute(x=14, executor=["test-executor"])
                assert result == 42
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_action_decorator_with_async_method(self, tmp_path):
        """Test @agent_action decorator with async methods."""
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
                    AsyncAgent, agent_workflow_config=config, executor="test-executor"
                )

                await agent.add_data(value="hello")
                await agent.add_data(value="world")
                result = await agent.get_data()
                assert result == ["hello", "world"]
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_actions_with_positional_arguments(self, tmp_path):
        """Test that agent actions support positional arguments."""
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
                    PositionalArgsAgent,
                    agent_workflow_config=config,
                    args=(10,),  # base=10
                    executor="test-executor",
                )

                # Test sync action with positional args
                result = await agent.add_numbers(5, 3)  # 10 + 5 + 3 = 18
                assert result == 18

                # Test sync action with positional + keyword args
                result = await agent.add_numbers(5, 3, c=2)  # 10 + 5 + 3 + 2 = 20
                assert result == 20

                # Test sync action with all keyword args
                result = await agent.add_numbers(a=1, b=2, c=3)  # 10 + 1 + 2 + 3 = 16
                assert result == 16

                # Test async action with positional args
                result = await agent.multiply_numbers(2, 3)  # 10 * 2 * 3 = 60
                assert result == 60

                # Test async action with keyword args
                result = await agent.multiply_numbers(x=4, y=5)  # 10 * 4 * 5 = 200
                assert result == 200

                # Test async action with mixed positional and keyword args
                result = await agent.concat_strings("hello", "world")
                assert result == "hello world"

                result = await agent.concat_strings("foo", "bar", sep="-")
                assert result == "foo-bar"

                result = await agent.concat_strings(s1="a", s2="b", sep=":")
                assert result == "a:b"
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_actions_with_bash_task(self, tmp_path):
        """Test that agent actions work with @bash_task decorator."""
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
                    BashTaskAgent,
                    agent_workflow_config=config,
                    args=("TEST",),  # prefix="TEST"
                    executor="test-executor",
                )

                # Test bash_task action - should return exit code 0 on success
                result = await agent.echo_message("Hello World")
                assert result == 0, "Bash task should return exit code 0 on success"

                # Test bash_task action that writes a file
                test_file = tmp_path / "test_output.txt"
                result = await agent.write_file(str(test_file), "test content")
                assert result == 0, "Bash task should return exit code 0 on success"

                # Verify the file was created
                assert test_file.exists(), "Bash task should have created the file"
                assert test_file.read_text().strip() == "test content"
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_actions_with_join_task(self, tmp_path):
        """Test that agent actions work with @join_task decorator."""
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
                    JoinTaskAgent,
                    agent_workflow_config=config,
                    args=(5,),  # multiplier=5
                    executor="test-executor",
                )

                # Test join_task action - combines results from multiple tasks
                # compute_sum(2, 3) -> (2*5) + (3*5) = 10 + 15 = 25
                result = await agent.compute_sum(2, 3)
                assert result == 25, "Join task should combine results correctly"

                # Test with different values
                result = await agent.compute_sum(4, 6)
                assert result == 50, (
                    "Join task should work with different inputs"
                )  # (4*5) + (6*5) = 50
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_loop_decorator(self, tmp_path):
        """Test @agent_loop decorator for background tasks."""
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
                    LoopAgent, agent_workflow_config=config, executor="test-executor"
                )

                # Get the initial counter value (could be 0 or already incremented due to race)
                # We can't assert it's 0 because the loop starts immediately and runs async
                initial = await agent.get_counter()

                # Poll until we see an increment (with timeout)
                for _ in range(20):  # 2 second timeout (20 * 0.1s)
                    await asyncio.sleep(0.1)
                    current = await agent.get_counter()
                    if current > initial:
                        break

                final = await agent.get_counter()
                # Verify the background loop is running and incrementing
                assert final > initial, (
                    "Background loop should have incremented counter"
                )
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
                    agent_workflow_config=config2,
                    agent_workflow_include=["executor-2"],  # Override decorator default
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
                    MixedAgent,
                    agent_workflow_config=config,
                    args=(10,),
                    executor="test-executor",
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
                    PrivateMethodAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
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
        """Test that agent with no @agent_action methods still works."""
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
                    EmptyAgent, agent_workflow_config=config, executor="test-executor"
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
                    agent_workflow_config=config,  # Agent's workflow will use this config
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
                # Launch agent with agent_workflow_include=["executor-1"] only
                agent = await manager.launch(
                    ComputeAgent,
                    agent_workflow_config=config,
                    args=(6,),
                    agent_workflow_include=[
                        "executor-1"
                    ],  # Agent's workflow should only load executor-1
                    executor="executor-1",
                )

                # Should be able to run tasks on executor-1 (included)
                result = await agent.compute(x=7, executor=["executor-1"])
                assert result == 42  # 6 * 7 = 42

                with pytest.raises(
                    ValueError, match="Task 1 requested invalid executor executor-2"
                ):
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
                    agent_workflow_config=config,
                    args=(3,),
                    agent_workflow_run_dir=custom_run_dir,  # Custom run directory for agent's workflow
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
                    FullAgent,
                    agent_workflow_config=config,
                    args=(10,),
                    executor="test-executor",
                )

                # Test task execution
                result = await agent.compute(x=32, executor=["test-executor"])
                assert result == 42

                # Test async agent_action
                state = await agent.get_state()
                assert state["value"] == 10
                assert state["counter"] >= 0

                # Test state mutation
                await agent.set_value(new_value=20)
                result = await agent.compute(x=22, executor=["test-executor"])
                assert result == 42

                # Test agent_loop is running
                initial_counter = state["counter"]
                await asyncio.sleep(0.5)
                final_state = await agent.get_state()
                assert final_state["counter"] > initial_counter

        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_agents_same_manager(self, tmp_path):
        """Test launching multiple agents with the same manager.

        This also implicitly tests that auto-generated agent_workflow_run_dir values
        prevent Parsl directory collisions when multiple agents are launched without
        explicit run_dir specification. Each agent gets a unique UUID-based run_dir.
        """
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
                # Launch agents without agent_workflow_run_dir - each gets auto-generated unique path
                agent1 = await manager.launch(
                    Agent1, agent_workflow_config=config, executor="test-executor"
                )
                agent2 = await manager.launch(
                    Agent2, agent_workflow_config=config, executor="test-executor"
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
                    agent_workflow_config=config,
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
        """Test sync agent_action that returns a plain value (not a Future)."""
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
                    PlainSyncAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
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
                    agent_workflow_config=config,
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
                    agent_workflow_config=config,
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
        """Test agent class with methods that have no @agent_action or @agent_loop decorators."""
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
                    NoMarkersAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
                )

                # Agent should launch successfully even without @agent_action markers
                # This tests the code path where methods don't have _chiltepin_expose or __wrapped__
                assert agent is not None
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_pure_async_actions(self, tmp_path):
        """Test agent with async actions (no @python_task)."""
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
                    agent_workflow_config=config,
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
        """Test agent with callable/non-callable attributes."""
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
                    MixedAttributesAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
                )

                result = await agent.get_value()
                assert result == 100
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_lifecycle_methods(self, tmp_path):
        """Test that agent_on_startup/shutdown are called."""
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
                    LifecycleTestAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
                )

                # If we can call an agent_action, agent_on_startup must have succeeded
                result = await agent.check_workflow()
                assert result is True
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_verify_loop_method_execution(self, tmp_path):
        """Explicitly verify agent_loop methods execute."""
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
                # Launch agent with agent_loop method
                agent = await manager.launch(
                    LoopAgent, agent_workflow_config=config, executor="test-executor"
                )

                # Verify agent_loop is running by checking counter increases
                count1 = await agent.get_counter()
                await asyncio.sleep(0.3)
                count2 = await agent.get_counter()
                await asyncio.sleep(0.3)
                count3 = await agent.get_counter()

                # Counter should increase over time
                assert count3 > count2 > count1
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_decorator_order_equivalence(self, tmp_path):
        """Test that @agent_action/@python_task order does not affect behavior."""
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
                    ReverseOrderAgent,
                    agent_workflow_config=config,
                    args=(10,),
                    executor="test-executor",
                )

                # Both should work and return correct results
                result_outer_action = await agent.compute_action_outer(
                    x=5, executor=["test-executor"]
                )
                result_outer_task = await agent.compute_task_outer(
                    x=5, executor=["test-executor"]
                )
                assert result_outer_action == 15  # 10 + 5
                assert result_outer_task == 50  # 10 * 5
        finally:
            workflow.cleanup()

    @pytest.mark.asyncio
    async def test_agent_composition(self, tmp_path):
        """Test that agents can use other agents (agent composition pattern).

        This demonstrates the pattern from agent_example1.py where a coordinator
        agent receives handles to other agents and orchestrates them.
        """
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        # Need 3 workers to run 3 agents concurrently
        config = {
            "test-executor": {
                "provider": "localhost",
                "cores_per_node": 1,
                "max_workers_per_node": 3,  # Need at least 3 for composition
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
                # Launch the lowerer and reverser agents first
                lowerer = await manager.launch(
                    LowererAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
                )
                reverser = await manager.launch(
                    ReverserAgent,
                    agent_workflow_config=config,
                    executor="test-executor",
                )

                # Launch the coordinator agent, passing in handles to the other agents
                coordinator = await manager.launch(
                    CoordinatorAgent,
                    agent_workflow_config=config,
                    args=(lowerer, reverser),
                    executor="test-executor",
                )

                # Test that the coordinator can orchestrate the other agents
                text = "DEADBEEF"
                expected = "feebdaed"  # lowercase then reversed
                result = await coordinator.process(text)
                assert result == expected, f"Expected {expected}, got {result}"

                # Test with different input
                text2 = "Hello World"
                expected2 = "dlrow olleh"  # lowercase then reversed
                result2 = await coordinator.process(text2)
                assert result2 == expected2, f"Expected {expected2}, got {result2}"
        finally:
            workflow.cleanup()


class DummyAcademyAgent(Agent):
    """A native Academy agent (not decorated with @chiltepin_agent)."""

    def __init__(self, value):
        super().__init__()
        self.value = value


def test_chiltepin_manager_rejects_native_academy_agent(tmp_path):
    """Test that ChiltepinManager rejects non-chiltepin agents with a clear error."""
    import pytest

    from chiltepin import Workflow
    from chiltepin.agents import AgentSystem

    config = {"test-executor": {"provider": "localhost"}}
    workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
    workflow.start()

    try:
        agent_system = AgentSystem(workflow=workflow, executor_names=["test-executor"])

        async def try_launch():
            async with await agent_system.manager() as manager:
                with pytest.raises(
                    TypeError,
                    match="only supports agents decorated with @chiltepin_agent",
                ):
                    await manager.launch(
                        DummyAcademyAgent, args=(123,), executor="test-executor"
                    )

        import asyncio

        asyncio.run(try_launch())
    finally:
        workflow.cleanup()


def test_academy_action_decorator_rejected():
    """Test that using Academy's @action decorator is detected and rejected."""

    import pytest
    from academy.agent import action as academy_action

    with pytest.raises(TypeError, match="uses Academy's @action decorator"):

        @chiltepin_agent()
        class BadAgent:
            @academy_action
            async def bad_method(self):
                pass


def test_academy_loop_decorator_rejected():
    """Test that using Academy's @loop decorator is detected and rejected."""
    import asyncio

    import pytest
    from academy.agent import loop as academy_loop

    with pytest.raises(TypeError, match="uses Academy's @loop decorator"):

        @chiltepin_agent()
        class BadAgent:
            @academy_loop
            async def bad_loop(self, shutdown: asyncio.Event) -> None:
                pass


def test_mixed_academy_chiltepin_action_decorators_rejected():
    """Test that stacking Academy's @action with @agent_action is rejected."""
    import pytest
    from academy.agent import action as academy_action

    with pytest.raises(TypeError, match="has both Academy and Chiltepin decorators"):

        @chiltepin_agent()
        class BadAgent:
            @agent_action
            @academy_action
            async def mixed_method(self):
                pass


def test_mixed_academy_chiltepin_loop_decorators_rejected():
    """Test that stacking Academy's @loop with @agent_loop is rejected."""
    import asyncio

    import pytest
    from academy.agent import loop as academy_loop

    with pytest.raises(TypeError, match="has both Academy and Chiltepin decorators"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            @academy_loop
            async def mixed_loop(self, shutdown: asyncio.Event) -> None:
                pass


def test_agent_loop_requires_async():
    """Test that @agent_loop raises TypeError when applied to non-async methods."""
    import pytest

    with pytest.raises(
        TypeError, match="@agent_loop can only be applied to async methods"
    ):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            def sync_loop(self, shutdown):  # Missing 'async'
                pass


def test_agent_loop_requires_shutdown_parameter():
    """Test that @agent_loop validates the method accepts a shutdown parameter."""
    import pytest

    with pytest.raises(TypeError, match="must accept a 'shutdown' parameter"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            async def bad_loop(self):  # Missing shutdown parameter
                pass


def test_agent_loop_rejects_args():
    """Test that @agent_loop rejects methods with *args."""
    import pytest

    with pytest.raises(TypeError, match="should not use \\*args"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            async def bad_loop(self, *args):
                pass


def test_agent_loop_rejects_kwargs():
    """Test that @agent_loop rejects methods with **kwargs."""
    import pytest

    with pytest.raises(TypeError, match="should not use \\*\\*kwargs"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            async def bad_loop(self, **kwargs):
                pass


def test_agent_loop_rejects_required_params_after_shutdown():
    """Test that @agent_loop rejects additional required parameters."""
    import pytest

    with pytest.raises(TypeError, match="must accept exactly one parameter"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            async def bad_loop(
                self, shutdown, extra_param
            ):  # extra_param has no default
                pass


def test_agent_loop_rejects_optional_params():
    """Test that @agent_loop rejects additional parameters even with defaults."""
    import pytest

    with pytest.raises(TypeError, match="must accept exactly one parameter"):

        @chiltepin_agent()
        class BadAgent:
            @agent_loop
            async def bad_loop(self, shutdown, interval=1.0):  # Even with default
                pass


def test_bash_task_decorator_compatibility():
    """Test that @agent_action works with @bash_task decorator.

    This test verifies that the @bash_task decorator (which uses MethodWrapper)
    properly preserves the _chiltepin_expose attribute so that @chiltepin_agent
    can detect and wrap it correctly as an Academy action.
    """

    # Define agent class with bash_task decorated method
    @chiltepin_agent()
    class BashTaskTestAgent:
        def __init__(self):
            self.prefix = "TEST"

        @agent_action
        @bash_task
        def run_command(self, cmd):
            """Return bash command to execute."""
            return f"echo '{self.prefix}: {cmd}'"

    # If decorators are compatible, class should be created successfully
    assert BashTaskTestAgent._is_chiltepin_agent
    assert hasattr(BashTaskTestAgent, "run_command")

    # The method should have been wrapped as an Academy action
    # (which means @chiltepin_agent successfully detected the @agent_action marker)
    method = getattr(BashTaskTestAgent, "run_command")
    assert hasattr(method, "_agent_method_type")
    assert method._agent_method_type == "action"

    # Verify it's a callable method
    assert callable(method)


def test_join_task_decorator_compatibility():
    """Test that @agent_action works with @join_task decorator.

    This test verifies that the @join_task decorator (which uses MethodWrapper)
    properly preserves the _chiltepin_expose attribute so that @chiltepin_agent
    can detect and wrap it correctly as an Academy action.
    """

    # Define agent class with join_task decorated method
    @chiltepin_agent()
    class JoinTaskTestAgent:
        def __init__(self):
            self.value = 10

        @agent_action
        @join_task
        def combine_results(self, x, y):
            """Join task that combines results."""
            return x + y + self.value

    # If decorators are compatible, class should be created successfully
    assert JoinTaskTestAgent._is_chiltepin_agent
    assert hasattr(JoinTaskTestAgent, "combine_results")

    # The method should have been wrapped as an Academy action
    # (which means @chiltepin_agent successfully detected the @agent_action marker)
    method = getattr(JoinTaskTestAgent, "combine_results")
    assert hasattr(method, "_agent_method_type")
    assert method._agent_method_type == "action"

    # Verify it's a callable method
    assert callable(method)


def test_agent_inheritance_pattern():
    """Test that inheritance works when decorating the child class.

    The supported pattern is to decorate only the final class in the hierarchy,
    not intermediate parent classes. When you decorate a child class, both
    parent and child methods marked with @agent_action or @agent_loop are
    discovered and wrapped.
    """

    # Define base behavior class WITHOUT @chiltepin_agent decorator
    class BaseAgent:
        def __init__(self, base_value: int):
            self.base_value = base_value

        @agent_action
        async def base_method(self) -> int:
            """Method from base class."""
            return self.base_value

        def helper_method(self) -> str:
            """Helper method not exposed as action."""
            return "helper"

    # Define child class and ONLY decorate the child
    @chiltepin_agent()
    class ChildAgent(BaseAgent):
        def __init__(self, base_value: int, child_value: int):
            super().__init__(base_value)
            self.child_value = child_value

        @agent_action
        async def child_method(self) -> int:
            """Method from child class."""
            return self.child_value

        @agent_action
        async def combined_method(self) -> int:
            """Method that uses both base and child state."""
            return self.base_value + self.child_value

    # Verify the child class is marked as a chiltepin agent
    assert ChildAgent._is_chiltepin_agent

    # Verify both base and child methods are wrapped as actions
    assert hasattr(ChildAgent, "base_method")
    assert hasattr(ChildAgent, "child_method")
    assert hasattr(ChildAgent, "combined_method")

    # Verify they have the _agent_method_type marker from Academy's @action
    base_method = getattr(ChildAgent, "base_method")
    assert hasattr(base_method, "_agent_method_type")
    assert base_method._agent_method_type == "action"

    child_method = getattr(ChildAgent, "child_method")
    assert hasattr(child_method, "_agent_method_type")
    assert child_method._agent_method_type == "action"

    combined_method = getattr(ChildAgent, "combined_method")
    assert hasattr(combined_method, "_agent_method_type")
    assert combined_method._agent_method_type == "action"

    # Verify undecorated method is not exposed on the decorated agent wrapper
    # It should exist in the base class but not be exposed on the agent
    assert hasattr(BaseAgent, "helper_method")
    # The decorated agent only exposes @agent_action/@agent_loop methods
    assert not hasattr(ChildAgent, "helper_method")


def test_agent_actions_can_call_unexposed_helpers():
    """Test that exposed actions can call unexposed helper methods internally.

    While helper methods are not exposed on the agent wrapper, they remain
    accessible to action methods because actions execute on the behavior
    instance which has normal inheritance.
    """

    # Define base class with helper method
    class BehaviorWithHelper:
        def __init__(self, value: int):
            self.value = value

        def _private_helper(self) -> str:
            """Private helper - not exposed."""
            return f"helper_result_{self.value}"

        def public_helper(self) -> int:
            """Public helper - not exposed (no decorator)."""
            return self.value * 2

        @agent_action
        async def action_using_helpers(self) -> dict:
            """Action that calls unexposed helper methods."""
            # Both helpers should be accessible via self
            private_result = self._private_helper()
            public_result = self.public_helper()
            return {
                "private": private_result,
                "public": public_result,
                "value": self.value,
            }

    # Decorate the class
    @chiltepin_agent()
    class AgentWithHelpers(BehaviorWithHelper):
        pass

    # Verify helpers are not exposed on the agent wrapper
    assert not hasattr(AgentWithHelpers, "_private_helper")
    assert not hasattr(AgentWithHelpers, "public_helper")

    # Verify the action IS exposed
    assert hasattr(AgentWithHelpers, "action_using_helpers")
    action = getattr(AgentWithHelpers, "action_using_helpers")
    assert hasattr(action, "_agent_method_type")
    assert action._agent_method_type == "action"

    # Test that the behavior instance (which actions execute on) can call helpers
    # Create a wrapper instance (passing None for agent infrastructure params)
    wrapper = AgentWithHelpers(42, agent_workflow_config=None)

    # The wrapper should have created a behavior instance internally
    assert hasattr(wrapper, "_behavior")
    behavior = wrapper._behavior

    # Verify the behavior instance is of the correct type
    assert isinstance(behavior, BehaviorWithHelper)

    # The behavior instance should have access to helpers
    assert behavior._private_helper() == "helper_result_42"
    assert behavior.public_helper() == 84

    # Most importantly: verify the action method can execute and call helpers
    import asyncio

    result = asyncio.run(behavior.action_using_helpers())
    assert result == {
        "private": "helper_result_42",
        "public": 84,
        "value": 42,
    }


def test_agent_multiple_inheritance_mixin_pattern():
    """Test that multiple inheritance (mixins) works with @chiltepin_agent.

    When a class inherits from multiple parent classes (mixins), all methods
    marked with @agent_action or @agent_loop from all parents should be
    discovered and wrapped when the final child class is decorated.
    """

    # Define mixin classes WITHOUT @chiltepin_agent decorator
    class StorageMixin:
        """Mixin providing storage operations."""

        @agent_action
        async def save(self, data: str) -> str:
            """Save data."""
            return f"saved: {data}"

    class ProcessingMixin:
        """Mixin providing processing operations."""

        @agent_action
        async def process(self, data: str) -> str:
            """Process data."""
            return data.upper()

    class MonitoringMixin:
        """Mixin providing monitoring operations."""

        @agent_action
        async def get_status(self) -> str:
            """Get status."""
            return "running"

        def _internal_check(self) -> bool:
            """Not decorated - won't be exposed."""
            return True

    # Combine mixins in a decorated class
    @chiltepin_agent()
    class CombinedAgent(StorageMixin, ProcessingMixin, MonitoringMixin):
        """Agent combining multiple mixins."""

        def __init__(self, name: str):
            self.name = name

        @agent_action
        async def get_name(self) -> str:
            """Method from the main class."""
            return self.name

    # Verify the class is marked as a chiltepin agent
    assert CombinedAgent._is_chiltepin_agent

    # Verify all methods from all mixins plus the child are wrapped
    assert hasattr(CombinedAgent, "save")  # From StorageMixin
    assert hasattr(CombinedAgent, "process")  # From ProcessingMixin
    assert hasattr(CombinedAgent, "get_status")  # From MonitoringMixin
    assert hasattr(CombinedAgent, "get_name")  # From CombinedAgent

    # Verify they all have the _agent_method_type marker
    save_method = getattr(CombinedAgent, "save")
    assert hasattr(save_method, "_agent_method_type")
    assert save_method._agent_method_type == "action"

    process_method = getattr(CombinedAgent, "process")
    assert hasattr(process_method, "_agent_method_type")
    assert process_method._agent_method_type == "action"

    status_method = getattr(CombinedAgent, "get_status")
    assert hasattr(status_method, "_agent_method_type")
    assert status_method._agent_method_type == "action"

    name_method = getattr(CombinedAgent, "get_name")
    assert hasattr(name_method, "_agent_method_type")
    assert name_method._agent_method_type == "action"

    # Verify non-decorated private method remains on the mixin itself
    assert hasattr(MonitoringMixin, "_internal_check")
    # Private methods (starting with _) are not exposed on the agent wrapper
    assert not hasattr(CombinedAgent, "_internal_check")


def test_extending_decorated_agent_raises_error():
    """Test that trying to extend a decorated agent raises a helpful error.

    Once a class is decorated with @chiltepin_agent, it becomes an Agent wrapper
    and cannot be extended. This test verifies that attempting to do so raises
    a clear TypeError with guidance on the correct pattern.
    """

    # Create a decorated agent
    @chiltepin_agent()
    class ParentAgent:
        @agent_action
        async def parent_action(self) -> str:
            return "parent"

    # Attempting to extend the decorated agent should raise TypeError
    with pytest.raises(TypeError) as exc_info:

        @chiltepin_agent()
        class ChildAgent(ParentAgent):  # ← This should fail
            @agent_action
            async def child_action(self) -> str:
                return "child"

    # Verify the error message is helpful
    error_message = str(exc_info.value)
    assert "Cannot extend decorated agent class" in error_message
    assert "ParentAgent" in error_message
    assert "@chiltepin_agent decorator wraps classes in an Agent" in error_message
    assert "undecorated base class" in error_message
    assert "Agent Inheritance" in error_message


def test_extending_decorated_agent_grandparent_raises_error():
    """Test that extending a decorated agent is caught even when not immediate parent.

    The check should use mro() to catch decorated agents anywhere in the inheritance
    chain, not just immediate parents. This test verifies that a decorated grandparent
    is detected.
    """

    # Create a decorated grandparent
    @chiltepin_agent()
    class GrandParentAgent:
        @agent_action
        async def grandparent_action(self) -> str:
            return "grandparent"

    # Create undecorated middle parent - inherits from decorated agent
    class MiddleParent(GrandParentAgent):
        @agent_action
        async def middle_action(self) -> str:
            return "middle"

    # Attempting to decorate a class that inherits from MiddleParent
    # should detect GrandParentAgent in the mro() and fail
    with pytest.raises(TypeError) as exc_info:

        @chiltepin_agent()
        class ChildAgent(MiddleParent):  # ← Should fail - GrandParentAgent is in mro()
            @agent_action
            async def child_action(self) -> str:
                return "child"

    # Verify the error message mentions GrandParentAgent
    error_message = str(exc_info.value)
    assert "Cannot extend decorated agent class" in error_message
    assert "GrandParentAgent" in error_message  # Should catch the grandparent


def test_double_decorating_agent_raises_error():
    """Test that applying @chiltepin_agent twice to the same class raises an error.

    Double-decoration should be caught and raise a clear error message.
    This prevents accidental misuse of the decorator.
    """

    # First decoration is fine
    @chiltepin_agent()
    class MyAgent:
        @agent_action
        async def my_action(self) -> str:
            return "action"

    # Attempting to decorate the already-decorated class should fail
    with pytest.raises(TypeError) as exc_info:
        MyAgent = chiltepin_agent()(MyAgent)  # Double decoration!

    # Verify the error message is helpful
    error_message = str(exc_info.value)
    assert "already decorated" in error_message
    assert "MyAgent" in error_message
    assert "Double-decoration is not supported" in error_message


def test_launching_undecorated_subclass_of_decorated_agent_raises_error(tmp_path):
    """Test that launching a subclass of a decorated agent without decorating it fails.

    If a user creates an undecorated subclass of a decorated agent and tries to launch it,
    ChiltepinManager should detect this and provide a helpful error message. This prevents
    confusing runtime failures where the subclass's methods aren't discovered.
    """
    from chiltepin import Workflow
    from chiltepin.agents import AgentSystem

    # Create a decorated parent agent
    @chiltepin_agent()
    class ParentAgent:
        @agent_action
        async def parent_action(self) -> str:
            return "parent"

    # Create an undecorated subclass (problematic pattern)
    class ChildAgent(ParentAgent):
        @agent_action
        async def child_action(self) -> str:
            return "child"

    # Set up a minimal workflow and manager
    config = {
        "local": {
            "provider": "local",
            "max_workers_per_node": 1,
        }
    }

    workflow = Workflow(config, run_dir=str(tmp_path / "runinfo"))
    workflow.start()

    try:
        agent_system = AgentSystem(workflow=workflow, executor_names=["local"])

        # Attempting to launch the undecorated subclass should fail
        async def try_launch():
            async with await agent_system.manager() as manager:
                with pytest.raises(TypeError) as exc_info:
                    await manager.launch(
                        ChildAgent, agent_workflow_config=config, executor="local"
                    )

                # Verify the error message is helpful
                error_message = str(exc_info.value)
                assert "subclass of decorated agent" in error_message
                assert "ParentAgent" in error_message
                assert "not itself decorated" in error_message
                assert "undecorated base behavior class" in error_message

        import asyncio

        asyncio.run(try_launch())
    finally:
        workflow.cleanup()
