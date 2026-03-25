# SPDX-License-Identifier: Apache-2.0

"""Agent system integration for Chiltepin workflows.

This module provides simplified interfaces for integrating Academy agents
with Chiltepin workflows and Parsl executors.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from academy.agent import Agent
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.manager import Manager
from parsl.concurrent import ParslPoolExecutor


class AgentSystem:
    """Simplified agent management system for Chiltepin workflows.

    This class wraps the complexity of setting up Academy Manager with
    ParslPoolExecutors and HttpExchangeFactory, providing a clean interface
    for users who always use this pattern.

    Args:
        workflow: The Workflow instance to use for executing tasks and agents
        executor_names: List of executor names for running agents on Parsl executors
        exchange_address: The exchange server address
        auth_method: Authentication method (default: "globus")

    Example:
        ```python
        from chiltepin import Workflow
        from chiltepin.agents import AgentSystem

        config = {"my-executor": {...}}
        workflow = Workflow(config)
        workflow.start()

        agent_system = AgentSystem(
            workflow=workflow,
            executor_names=["my-executor"],
            exchange_address="https://exchange.academy-agents.org"
        )

        async with await agent_system.manager() as manager:
            # Launch and interact with agents
            agent = await manager.launch(MyAgent, executor="my-executor")
            result = await agent.some_action()

        workflow.cleanup()
        ```
    """

    def __init__(
        self,
        workflow,
        executor_names: list[str],
        exchange_address: str = "https://exchange.academy-agents.org",
        auth_method: str = "globus",
    ):
        """Initialize the AgentSystem.

        Args:
            workflow: The Workflow instance with started dfk
            executor_names: List of executor names for running agents on Parsl executors
            exchange_address: The exchange server address
            auth_method: Authentication method for accessing the exchange (default: "globus")
        """
        self.workflow = workflow
        self.executor_names = executor_names
        self.exchange_address = exchange_address
        self.auth_method = auth_method
        self._executors: Optional[dict] = None

    def _create_executors(self):
        """Create ParslPoolExecutors for all configured executor names."""
        if self.workflow.dfk is None:
            raise RuntimeError(
                "Workflow must be started before creating AgentSystem executors. "
                "Call workflow.start() first."
            )

        self._executors = {
            name: ParslPoolExecutor(dfk=self.workflow.dfk, executors=[name])
            for name in self.executor_names
        }

    async def manager(self):
        """Create and return an Academy Manager context manager.

        This method returns an Academy Manager configured with HttpExchangeFactory
        using Globus authentication. The Manager is created with ParslPoolExecutors
        for all configured executors.

        Returns:
            An async context manager for the Academy Manager

        Example:
            ```python
            async with await agent_system.manager() as manager:
                agent = await manager.launch(MyAgent, executor="my-executor")
                result = await agent.some_action()
            ```
        """
        # Create executors if not already created
        if self._executors is None:
            self._create_executors()

        # Return the Manager context manager
        return await Manager.from_exchange_factory(
            factory=HttpExchangeFactory(
                self.exchange_address,
                auth_method=self.auth_method,
            ),
            executors=self._executors,
        )

    @property
    def executors(self):
        """Access the created ParslPoolExecutors.

        Returns None if executors haven't been created yet.
        """
        return self._executors


class ChiltepinAgent(Agent):
    """Base class for Chiltepin agents with automatic workflow management.

    This class automatically manages workflow lifecycle, eliminating boilerplate
    code that users would otherwise need to write in agent_on_startup and
    agent_on_shutdown methods.

    The workflow is started when the agent starts and cleaned up when the agent
    shuts down. Users simply define their actions with @action decorator and the
    workflow automatically handles task execution.

    Args:
        config: Either a path to a YAML configuration file or a configuration dictionary
        include: List of resource labels to load. If None, all resources are loaded.
        run_dir: Directory for Parsl runtime files. If None, uses Parsl's default.
        **kwargs: Additional keyword arguments (passed to subclass __init__)

    Example:
        ```python
        from chiltepin.agents import ChiltepinAgent
        from chiltepin.tasks import python_task
        from academy.agent import action

        @python_task
        def run_model(temperature: float) -> str:
            return f"Predicted: {temperature:.2f} degrees"

        class MyModel(ChiltepinAgent):
            def __init__(self, config, temperature: float):
                super().__init__(config, include=["ursa-compute"])
                self.temperature = temperature

            @action
            async def predict(self) -> str:
                import asyncio
                # Tasks automatically use the managed workflow
                return await asyncio.wrap_future(run_model(self.temperature))
        ```
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        *,
        include: Optional[List[str]] = None,
        run_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the ChiltepinAgent with workflow configuration.

        Args:
            config: Either a path to a YAML configuration file or a configuration dictionary
            include: List of resource labels to load. If None, all resources are loaded.
            run_dir: Directory for Parsl runtime files. If None, uses Parsl's default.
            **kwargs: Additional keyword arguments (available for subclass use)
        """
        super().__init__()
        self._config = config
        self._include = include
        self._run_dir = run_dir
        self._workflow = None
        self._dfk = None

    async def agent_on_startup(self) -> None:
        """Start the workflow when the agent starts.

        This method is called automatically by the Academy Agent framework.
        Users can override this method but should call super().agent_on_startup()
        to ensure the workflow is started.
        """
        from chiltepin import Workflow

        self._workflow = Workflow(
            self._config,
            include=self._include,
            run_dir=self._run_dir,
        )
        self._dfk = self._workflow.start()

    async def agent_on_shutdown(self) -> None:
        """Clean up the workflow when the agent shuts down.

        This method is called automatically by the Academy Agent framework.
        Users can override this method but should call super().agent_on_shutdown()
        to ensure the workflow is cleaned up.
        """
        if self._workflow is not None:
            self._workflow.cleanup()


def expose(func):
    """Marker decorator to indicate a method should be exposed as an agent action.

    Use this decorator on methods in classes decorated with @chiltepin_agent to
    mark them as actions that should be exposed through the agent interface.

    This is an alternative to using Academy's @action decorator, which requires
    async methods. Use @expose for sync methods (especially @python_task methods)
    and either @expose or @action for async methods.

    Example:
        ```python
        @chiltepin_agent(include=["compute"])
        class MyModel:
            @expose
            @python_task
            def compute(self):
                return "result"

            @expose  # or @action
            async def get_status(self):
                return "ready"
        ```
    """
    func._chiltepin_expose = True
    return func


def chiltepin_agent(*, include: Optional[List[str]] = None, run_dir: Optional[str] = None):
    """Decorator that wraps a regular Python class (behavior) in an Academy Agent.

    This decorator allows you to write agent behavior as a regular, serializable
    Python class where task-decorated methods can access instance state directly.
    The decorator automatically creates an Agent wrapper that manages the workflow
    lifecycle and exposes the behavior's methods as actions.

    Only methods marked with @expose, @action, or @loop decorators are exposed as
    agent actions. Use @expose for sync methods (especially @python_task methods)
    and @expose or @action for async methods. @loop methods are automatically exposed.

    The behavior class must accept `config` as its first __init__ parameter.

    Args:
        include: List of resource labels to load. If None, all resources are loaded.
        run_dir: Directory for Parsl runtime files. If None, uses Parsl's default.

    Returns:
        A decorator function that wraps the behavior class in an Agent.

    Example:
        ```python
        from chiltepin.agents import chiltepin_agent, expose
        from chiltepin.tasks import python_task
        from academy.agent import action, loop

        @chiltepin_agent(include=["ursa-compute"])
        class MyModel:
            '''Regular Python class - fully serializable!'''

            def __init__(self, config, temperature: float):
                self.config = config
                self.temperature = temperature

            @expose  # ← Use @expose for sync/task-decorated methods
            @python_task
            def run_model(self) -> str:
                # Import modules inside methods for serialization
                import random
                # Can directly access self.temperature!
                return f"Predicted: {self.temperature + random.uniform(0, 5):.2f} degrees"

            @expose  # ← Use @expose or @action for async methods
            async def get_status(self) -> str:
                return f"Temperature: {self.temperature:.2f}"

            @loop  # ← Loop methods are automatically exposed
            async def update_temperature(self, shutdown) -> None:
                # Import modules inside methods for serialization
                import asyncio
                import random

                while not shutdown.is_set():
                    await asyncio.sleep(1)
                    self.temperature += random.uniform(-3, 3)

            def _private_helper(self):
                # Not decorated with @expose, won't be exposed
                pass

        # Usage is the same as regular agents
        model = await manager.launch(MyModel, args=(config, 25))
        result = await model.run_model()
        status = await model.get_status()
        ```
    """
    from academy.agent import action, loop as loop_decorator
    import asyncio
    import inspect

    def decorator(behavior_class):
        """Inner decorator that receives the behavior class."""

        # Create a wrapper Agent class dynamically
        class ChiltepinAgentWrapper(Agent):
            def __init__(self, config, *args, **kwargs):
                """Initialize with config as first param, rest go to behavior."""
                super().__init__()
                # Store config for workflow setup
                self._config = config
                self._include = include
                self._run_dir = run_dir
                self._workflow = None
                self._dfk = None
                # Create the behavior instance
                self._behavior = behavior_class(config, *args, **kwargs)

            async def agent_on_startup(self) -> None:
                """Start the workflow when the agent starts."""
                from chiltepin import Workflow

                self._workflow = Workflow(
                    self._config,
                    include=self._include,
                    run_dir=self._run_dir,
                )
                self._dfk = self._workflow.start()

            async def agent_on_shutdown(self) -> None:
                """Clean up the workflow when the agent shuts down."""
                if self._workflow is not None:
                    self._workflow.cleanup()

        # Scan the behavior class for methods to wrap as actions
        # Only wrap methods that are marked with @action or @loop decorators
        for name in dir(behavior_class):
            if name.startswith("_"):
                continue

            attr = getattr(behavior_class, name)
            if not callable(attr):
                continue

            # Skip if it's inherited from base object class
            if name in dir(object):
                continue

            # Check if this method was marked with @expose, @action, or @loop
            is_loop_method = False
            is_exposed = False

            # Check for @loop - has 'shutdown' parameter
            if inspect.iscoroutinefunction(attr):
                sig = inspect.signature(attr)
                params = list(sig.parameters.keys())
                if 'shutdown' in params:
                    is_loop_method = True

            # Check for @expose marker (our custom decorator)
            if hasattr(attr, '_chiltepin_expose'):
                is_exposed = True

            # Check for @action marker - these decorators typically set __wrapped__
            if hasattr(attr, '__wrapped__'):
                is_exposed = True

            # Only wrap methods that are explicitly marked
            if is_loop_method:
                # This is a @loop method - wrap it appropriately
                def make_loop_method(method_name):
                    @loop_decorator
                    async def loop_method(self, shutdown: asyncio.Event) -> None:
                        method = getattr(self._behavior, method_name)
                        await method(shutdown)

                    loop_method.__name__ = method_name
                    loop_method.__doc__ = getattr(behavior_class, method_name).__doc__
                    return loop_method

                setattr(ChiltepinAgentWrapper, name, make_loop_method(name))

            elif is_exposed:
                # Method was decorated with @expose or @action
                # Wrap it as an action
                if inspect.iscoroutinefunction(attr):
                    # Async action
                    def make_async_action(method_name):
                        @action
                        async def action_method(self, **kwargs):
                            method = getattr(self._behavior, method_name)
                            return await method(**kwargs)

                        action_method.__name__ = method_name
                        action_method.__doc__ = getattr(behavior_class, method_name).__doc__
                        return action_method

                    setattr(ChiltepinAgentWrapper, name, make_async_action(name))
                else:
                    # Sync action (might be task-decorated)
                    def make_action(method_name):
                        @action
                        async def action_method(self, **kwargs):
                            method = getattr(self._behavior, method_name)
                            result = method(**kwargs)
                            # Check if it's a Future (from task decorator)
                            if hasattr(result, 'result') and callable(result.result):
                                # It's a Parsl AppFuture - wrap it
                                return await asyncio.wrap_future(result)
                            return result

                        action_method.__name__ = method_name
                        action_method.__doc__ = getattr(behavior_class, method_name).__doc__
                        return action_method

                    setattr(ChiltepinAgentWrapper, name, make_action(name))

        # Set better names for debugging
        ChiltepinAgentWrapper.__name__ = behavior_class.__name__
        ChiltepinAgentWrapper.__qualname__ = behavior_class.__qualname__
        ChiltepinAgentWrapper.__module__ = behavior_class.__module__
        ChiltepinAgentWrapper.__doc__ = behavior_class.__doc__

        return ChiltepinAgentWrapper

    return decorator
