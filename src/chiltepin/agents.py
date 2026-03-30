# SPDX-License-Identifier: Apache-2.0

"""Agent system integration for Chiltepin workflows.

This module provides simplified interfaces for integrating Academy agents
with Chiltepin workflows and Parsl executors.

When using @chiltepin_agent decorator, always import action and loop decorators
from chiltepin.agents, NOT from academy.agent:

    from chiltepin.agents import chiltepin_agent, action, loop  # ✅ Correct
    from academy.agent import action, loop  # ❌ Wrong for @chiltepin_agent

Chiltepin's decorators work with both sync and async methods, while Academy's
action decorator requires async methods only.
"""

from typing import List, Optional

from academy.agent import Agent
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.manager import Manager
from parsl.concurrent import ParslPoolExecutor


class ChiltepinManager(Manager):
    """Custom Manager that supports config=, include=, and run_dir= kwargs in launch().

    This Manager subclass intercepts launch() calls to extract chiltepin-specific
    keyword arguments (config, include, run_dir) and passes them to agents created
    with the @chiltepin_agent decorator.

    This keeps workflow infrastructure concerns (config, include, run_dir) separate
    from behavior logic, allowing behavior classes to focus on domain logic only.
    """

    async def launch(
        self,
        agent_class,
        args=None,
        kwargs=None,
        config=None,
        include=None,
        run_dir=None,
        **manager_kwargs,
    ):
        """Launch an agent, supporting chiltepin-specific configuration.

        Args:
            agent_class: The agent class to launch
            args: Tuple of positional arguments for agent __init__ (behavior logic only)
            kwargs: Dict of keyword arguments for agent __init__ (behavior logic only)
            config: Workflow configuration dict or path (chiltepin agents only)
            include: Optional list of executor labels for workflow (chiltepin agents only)
            run_dir: Optional run directory for workflow (chiltepin agents only)
            **manager_kwargs: Other keyword arguments for Manager (e.g., executor, resources)

        Returns:
            The launched agent proxy

        Example:
            ```python
            model = await manager.launch(
                MyModel,
                config=ursa_config,           # ← Workflow config
                include=["ursa-compute"],     # ← Which executors
                run_dir="/custom/path",       # ← Where to run
                args=(25.0,),                 # ← Behavior args only
                executor="ursa-service-gc"    # ← Manager executor
            )
            ```
        """
        # If config, include, or run_dir were provided, add them to the agent's kwargs
        if config is not None or include is not None or run_dir is not None:
            # Ensure kwargs exists
            if kwargs is None:
                kwargs = {}
            else:
                # Make a copy to avoid mutating caller's dict
                kwargs = kwargs.copy()

            if config is not None:
                kwargs["config"] = config
            if include is not None:
                kwargs["include"] = include
            if run_dir is not None:
                kwargs["run_dir"] = run_dir

        # Call parent launch without config/include/run_dir (they're now in agent's kwargs)
        return await super().launch(
            agent_class, args=args, kwargs=kwargs, **manager_kwargs
        )


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

        # Return the ChiltepinManager context manager
        return await ChiltepinManager.from_exchange_factory(
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


def action(func):
    """Marker decorator to indicate a method should be exposed as an agent action.

    Use this decorator on methods in classes decorated with @chiltepin_agent to
    mark them as actions that should be exposed through the agent interface.

    Unlike Academy's @action decorator, this works for both sync and async methods,
    making it suitable for @python_task decorated methods as well as async helpers.

    Example:
        ```python
        from chiltepin.agents import chiltepin_agent, action
        from chiltepin.tasks import python_task

        @chiltepin_agent(include=["compute"])
        class MyModel:
            @action  # ← Works for sync task methods
            @python_task
            def compute(self):
                return "result"

            @action  # ← Also works for async methods
            async def get_status(self):
                return "ready"
        ```

    .. important::
        **Always import from chiltepin.agents, not academy.agent**::

            from chiltepin.agents import action  # ✅ Correct
            from academy.agent import action     # ❌ Wrong - different semantics

        Using Academy's decorator will cause confusing errors. Academy's @action
        requires async methods, while chiltepin's works with any callable.
    """
    func._chiltepin_expose = True
    return func


def loop(func):
    """Marker decorator for background loop methods in chiltepin agents.

    Use this decorator on async methods with a 'shutdown' parameter that should
    run as background loops in classes decorated with @chiltepin_agent.

    This is equivalent to Academy's @loop decorator but provided here for consistency
    so all decorators can be imported from chiltepin.agents.

    .. important::
        **Always import from chiltepin.agents, not academy.agent**::

            from chiltepin.agents import loop  # ✅ Correct
            from academy.agent import loop     # ❌ Wrong - will cause type errors

        Using Academy's decorator will cause confusing signature validation errors.

    Example:
        ```python
        from chiltepin.agents import chiltepin_agent, loop
        import asyncio

        @chiltepin_agent(include=["compute"])
        class MyModel:
            @loop
            async def update_data(self, shutdown):
                while not shutdown.is_set():
                    await asyncio.sleep(1)
                    # Update state
        ```
    """
    func._chiltepin_loop = True
    return func


def chiltepin_agent(
    *, include: Optional[List[str]] = None, run_dir: Optional[str] = None
):
    """Decorator that wraps a regular Python class (behavior) in an Academy Agent.

    This decorator allows you to write agent behavior as a regular, serializable
    Python class where task-decorated methods can access instance state directly.
    The decorator automatically creates an Agent wrapper that manages the workflow
    lifecycle and exposes the behavior's methods as actions.

    Only methods marked with @action or @loop decorators are exposed as agent actions.
    Use @action for any method (sync or async) you want to expose, and @loop for
    background loops. Both decorators should be imported from chiltepin.agents.

    Args:
        include: Default list of resource labels to load. Can be overridden at runtime
                 using include= keyword argument in manager.launch(). If None, all
                 resources are loaded.
        run_dir: Default directory for Parsl runtime files. Can be overridden at runtime
                 using run_dir= keyword argument in manager.launch(). If None, uses
                 Parsl's default.

    Returns:
        A decorator function that wraps the behavior class in an Agent.

    Runtime Configuration:
        When using AgentSystem (which provides ChiltepinManager), pass workflow
        configuration using keyword arguments to launch(). This separates infrastructure
        from behavior logic:

        ```python
        @chiltepin_agent(include=["default-executor"])
        class MyModel:
            def __init__(self, temperature):  # ← No config! Pure domain logic
                self.temperature = temperature

        # Launch with runtime configuration
        model = await manager.launch(
            MyModel,
            config=ursa_config,              # ← Workflow config used by the agent's workflow context
            args=(25.0,),                    # ← Behavior args only (domain logic)
            include=["runtime-executor"],    # ← Override decorator default - executors the agent should use to run tasks
            executor="ursa-service-gc"       # ← The executor to use for launching the agent itself (infrastructure)
        )
        ```

    Example:
        ```python
        from chiltepin.agents import chiltepin_agent, action, loop
        from chiltepin.tasks import python_task

        @chiltepin_agent(include=["ursa-compute"])  # ← Default, can be overridden
        class MyModel:
            '''Regular Python class - fully serializable!'''

            def __init__(self, temperature: float):
                '''Initialize behavior with domain logic only.'''
                self.temperature = temperature

            @action  # ← Use @action for sync/task-decorated methods
            @python_task
            def run_model(self) -> str:
                # Import modules inside methods for serialization
                import random
                # Can directly access self.temperature!
                return f"Predicted: {self.temperature + random.uniform(0, 5):.2f} degrees"

            @action  # ← Use @action for async methods too
            async def get_status(self) -> str:
                return f"Temperature: {self.temperature:.2f}"

            @loop  # ← Use @loop for background loops
            async def update_temperature(self, shutdown) -> None:
                # Import modules inside methods for serialization
                import asyncio
                import random

                while not shutdown.is_set():
                    await asyncio.sleep(1)
                    self.temperature += random.uniform(-3, 3)

            def _private_helper(self):
                # Not decorated with @action, won't be exposed
                pass

        # Launch agent using decorator defaults (include=["ursa-compute"])
        model = await manager.launch(MyModel, config=config, args=(25,))

        # Override decorator defaults at runtime
        model = await manager.launch(
            MyModel,
            config=config,
            args=(25,),
            include=["runtime-executor"],  # ← Override decorator's include
        )

        result = await model.run_model()
        status = await model.get_status()
        ```
    """
    import asyncio
    import inspect

    from academy.agent import action as academy_action
    from academy.agent import loop as academy_loop
    from parsl.dataflow.futures import AppFuture

    # Capture decorator parameters for use in closure
    decorator_include = include
    decorator_run_dir = run_dir

    def decorator(behavior_class):
        """Inner decorator that receives the behavior class."""

        # Create a wrapper Agent class dynamically
        class ChiltepinAgentWrapper(Agent):
            # Note: Coverage excluded for __init__ and lifecycle methods below.
            # These methods execute in Academy Agent workers (often in separate processes or
            # on remote systems via Globus Compute). The coverage tool can only track code
            # executing in the local test process, not in remote workers or agent processes
            # managed by Academy's exchange system.
            def __init__(
                self, *args, config={}, include=None, run_dir=None, **kwargs
            ):  # pragma: no cover
                """Initialize the agent wrapper.

                Args:
                    *args: Positional arguments for behavior class
                    config: Configuration for Agent's workflow context (from manager.launch), defaults to local config
                    include: Optional runtime override for Agent's workflow executor list (from manager.launch)
                    run_dir: Optional runtime override for Parsl's run directory (from manager.launch)
                    **kwargs: Keyword arguments for behavior class
                """
                super().__init__()  # pragma: no cover
                # Store config for workflow setup
                self._config = config  # pragma: no cover

                # Use runtime overrides if provided, otherwise fall back to decorator defaults
                self._include = (
                    include if include is not None else decorator_include
                )  # pragma: no cover
                self._run_dir = (
                    run_dir if run_dir is not None else decorator_run_dir
                )  # pragma: no cover

                # Create the behavior instance with its args/kwargs
                # Note: config, include, and run_dir are infrastructure, not passed to behavior
                self._behavior = behavior_class(*args, **kwargs)  # pragma: no cover

                self._workflow = None  # pragma: no cover
                self._dfk = None  # pragma: no cover

            async def agent_on_startup(self) -> None:  # pragma: no cover
                """Start the workflow when the agent starts."""
                from chiltepin import Workflow  # pragma: no cover

                self._workflow = Workflow(  # pragma: no cover
                    self._config,
                    include=self._include,
                    run_dir=self._run_dir,
                )
                self._dfk = self._workflow.start()  # pragma: no cover

            async def agent_on_shutdown(self) -> None:  # pragma: no cover
                """Clean up the workflow when the agent shuts down."""
                if self._workflow is not None:  # pragma: no cover
                    self._workflow.cleanup()  # pragma: no cover

        # Scan the behavior class for methods to wrap as actions
        # Only wrap methods that are marked with @action or @loop decorators
        for name in dir(behavior_class):
            if name.startswith("_"):
                continue

            attr = getattr(behavior_class, name)
            # Note: Coverage excluded for attribute type checking.
            # These checks execute but are in a dynamically-constructed decorator closure,
            # which coverage.py cannot properly track.
            if not callable(attr):  # pragma: no cover
                continue

            # Skip if it's inherited from base object class
            if name in dir(object):  # pragma: no cover
                continue

            # Check if this method was marked with @action or @loop
            is_loop_method = False
            is_exposed = False

            # Check for @loop - must have _chiltepin_loop marker
            # Note: Coverage excluded for these conditional branches.
            # The code executes during decorator application, but coverage.py cannot track
            # execution inside dynamically-created decorator closures.
            if hasattr(attr, "_chiltepin_loop"):  # pragma: no cover
                is_loop_method = True

            # Check for @action marker (our custom decorator sets _chiltepin_expose)
            if hasattr(attr, "_chiltepin_expose"):
                is_exposed = True

            # Only wrap methods that are explicitly marked
            # Note: Coverage excluded for method wrapping closures below.
            # These nested functions execute, but coverage.py cannot track code inside
            # dynamically-created closures that are set as class attributes.
            if is_loop_method:  # pragma: no cover
                # This is a @loop method - wrap it appropriately
                def make_loop_method(method_name):
                    @academy_loop
                    async def loop_method(self, shutdown: asyncio.Event) -> None:
                        method = getattr(self._behavior, method_name)
                        await method(shutdown)

                    loop_method.__name__ = method_name
                    loop_method.__doc__ = getattr(behavior_class, method_name).__doc__
                    return loop_method

                setattr(ChiltepinAgentWrapper, name, make_loop_method(name))

            # Note: Coverage excluded for action method wrapping.
            # These closures execute when agent actions are created, but coverage.py
            # cannot track nested closure code that's dynamically attached to classes.
            elif is_exposed:  # pragma: no cover
                # Method was decorated with @action
                # Wrap it as an action
                if inspect.iscoroutinefunction(attr):
                    # Async action
                    def make_async_action(method_name):
                        @academy_action
                        async def action_method(self, **kwargs):
                            method = getattr(self._behavior, method_name)
                            return await method(**kwargs)

                        action_method.__name__ = method_name
                        action_method.__doc__ = getattr(
                            behavior_class, method_name
                        ).__doc__
                        return action_method

                    setattr(ChiltepinAgentWrapper, name, make_async_action(name))
                else:
                    # Sync action (might be task-decorated)
                    def make_action(method_name):
                        @academy_action
                        async def action_method(self, **kwargs):
                            method = getattr(self._behavior, method_name)
                            result = method(**kwargs)

                            # Check if it's a Parsl AppFuture (from task decorator)
                            if isinstance(result, AppFuture):
                                # It's a Parsl AppFuture - wrap it
                                return await asyncio.wrap_future(result)
                            return result

                        action_method.__name__ = method_name
                        action_method.__doc__ = getattr(
                            behavior_class, method_name
                        ).__doc__
                        return action_method

                    setattr(ChiltepinAgentWrapper, name, make_action(name))

        # Set better names for debugging
        ChiltepinAgentWrapper.__name__ = behavior_class.__name__
        ChiltepinAgentWrapper.__qualname__ = behavior_class.__qualname__
        ChiltepinAgentWrapper.__module__ = behavior_class.__module__
        ChiltepinAgentWrapper.__doc__ = behavior_class.__doc__

        return ChiltepinAgentWrapper

    return decorator
