# SPDX-License-Identifier: Apache-2.0

"""Agent system integration for Chiltepin workflows.

This module provides simplified interfaces for integrating Academy agents
with Chiltepin workflows and Parsl executors.

When using @chiltepin_agent decorator, always import agent_action and agent_loop decorators
from chiltepin.agents, NOT from academy.agent:

.. code-block:: python

    from chiltepin.agents import chiltepin_agent, agent_action, agent_loop  # ✅ Correct
    from academy.agent import action, loop  # ❌ Wrong for @chiltepin_agent

Chiltepin's decorators work with both sync and async methods, while Academy's
action decorator requires async methods only.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from academy.agent import Agent
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.handle import Handle
from academy.manager import Manager
from parsl.concurrent import ParslPoolExecutor

if TYPE_CHECKING:
    from academy.agent import AgentT

    from chiltepin.workflow import Workflow
else:
    AgentT = TypeVar("AgentT")


class ChiltepinManager(Manager):
    """Custom Manager that supports agent_workflow_config=, agent_workflow_include=, and agent_workflow_run_dir= kwargs in launch().

    This Manager subclass intercepts launch() calls to extract chiltepin-specific
    keyword arguments (agent_workflow_config, agent_workflow_include, agent_workflow_run_dir) and passes them to agents
    created with the @chiltepin_agent decorator.

    This keeps workflow infrastructure concerns (Parsl configuration) separate
    from behavior logic, allowing behavior classes to focus on domain logic only.
    """

    async def launch(
        self,
        agent_class: Type[AgentT],
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        agent_workflow_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        agent_workflow_include: Optional[List[str]] = None,
        agent_workflow_run_dir: Optional[str] = None,
        **manager_kwargs: Any,
    ) -> Handle[AgentT]:
        """Launch an agent, supporting chiltepin-specific configuration.

        Parameters
        ----------
        agent_class : Type[AgentT]
            The agent class to launch
        args : Optional[Tuple[Any, ...]]
            Tuple of positional arguments for agent __init__ (behavior logic only)
        kwargs : Optional[Dict[str, Any]]
            Dict of keyword arguments for agent __init__ (behavior logic only)
        agent_workflow_config : Optional[Union[str, Path, Dict[str, Any]]]
            Workflow configuration dict or path (chiltepin agents only)
        agent_workflow_include : Optional[List[str]]
            Optional list of executor labels for workflow (chiltepin agents only)
        agent_workflow_run_dir : Optional[str]
            Optional run directory for workflow (chiltepin agents only).
            **Important**: When launching multiple agents on shared filesystems,
            provide unique run_dir values to avoid Parsl directory collisions.
            If omitted, a unique directory is auto-generated.
        **manager_kwargs : Any
            Other keyword arguments for Manager (e.g., executor, resources)

        Returns
        -------
        Handle[AgentT]
            The launched agent proxy

        Examples
        --------

        .. code-block:: python

            model = await manager.launch(
                MyModel,
                agent_workflow_config=ursa_config,           # ← Workflow config
                agent_workflow_include=["ursa-compute"],     # ← Which executors
                agent_workflow_run_dir="/custom/path",       # ← Where to run
                args=(25.0,),                       # ← Behavior args only
                executor="ursa-service-gc"          # ← Manager executor
            )
        """
        # Validate that the agent class is properly decorated with @chiltepin_agent
        # Use __dict__ to ensure the class itself is decorated, not just inheriting the flag
        is_directly_decorated = agent_class.__dict__.get("_is_chiltepin_agent", False)
        is_inherited_decorated = getattr(agent_class, "_is_chiltepin_agent", False)

        if not is_directly_decorated:
            # Check if this is a subclass of a decorated agent (problematic pattern)
            if is_inherited_decorated:
                # Find the decorated parent to provide a helpful error message
                decorated_parent = None
                for base in agent_class.mro()[1:]:
                    if isinstance(base, type) and base.__dict__.get(
                        "_is_chiltepin_agent", False
                    ):
                        decorated_parent = base
                        break

                parent_name = (
                    decorated_parent.__name__
                    if decorated_parent
                    else "decorated parent"
                )
                original_behavior_name = (
                    getattr(decorated_parent, "_behavior_class_name", parent_name)
                    if decorated_parent
                    else "Behavior"
                )

                raise TypeError(
                    f"Cannot launch '{agent_class.__name__}' - it is a subclass of decorated agent '{parent_name}' but is not itself decorated.\n\n"
                    f"Subclassing decorated agents is not supported. To use inheritance:\n\n"
                    f"1. Create an undecorated base behavior class:\n"
                    f"   class {original_behavior_name}Base:\n"
                    f"       @agent_action\n"
                    f"       async def shared_method(self): ...\n\n"
                    f"2. Decorate each implementation separately:\n"
                    f"   @chiltepin_agent()\n"
                    f"   class {parent_name}({original_behavior_name}Base):\n"
                    f"       pass\n\n"
                    f"   @chiltepin_agent()\n"
                    f"   class {agent_class.__name__}({original_behavior_name}Base):\n"
                    f"       @agent_action\n"
                    f"       async def new_method(self): ...\n\n"
                    f"See the 'Agent Inheritance' section in the documentation for details."
                )
            else:
                # Not decorated at all
                raise TypeError(
                    f"ChiltepinManager only supports agents decorated with @chiltepin_agent. "
                    f"Got: {agent_class.__module__}.{agent_class.__name__}. "
                    "Use the base Academy Manager for native agents."
                )

        # If agent_workflow_config, agent_workflow_include, or agent_workflow_run_dir were provided, add them to the agent's kwargs
        if (
            agent_workflow_config is not None
            or agent_workflow_include is not None
            or agent_workflow_run_dir is not None
        ):
            # Ensure kwargs exists
            if kwargs is None:
                kwargs = {}
            else:
                # Make a copy to avoid mutating caller's dict
                kwargs = kwargs.copy()

            if agent_workflow_config is not None:
                kwargs["agent_workflow_config"] = agent_workflow_config
            if agent_workflow_include is not None:
                kwargs["agent_workflow_include"] = agent_workflow_include
            if agent_workflow_run_dir is not None:
                kwargs["agent_workflow_run_dir"] = agent_workflow_run_dir

        # Call parent launch without agent_workflow_* params (they're now in agent's kwargs)
        return await super().launch(
            agent_class, args=args, kwargs=kwargs, **manager_kwargs
        )


class AgentSystem:
    """Simplified agent management system for Chiltepin workflows.

    This class wraps the complexity of setting up Academy Manager with
    ParslPoolExecutors and HttpExchangeFactory, providing a clean interface
    for users who always use this pattern.

    Parameters
    ----------
    workflow : Workflow
        The Workflow instance to use for executing tasks and agents
    executor_names : List[str]
        List of executor names for running agents on Parsl executors
    exchange_address : str
        The exchange server address
    auth_method : str
        Authentication method (default: "globus")

    Examples
    --------

    .. code-block:: python

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
    """

    def __init__(
        self,
        workflow: Workflow,
        executor_names: List[str],
        exchange_address: str = "https://exchange.academy-agents.org",
        auth_method: str = "globus",
    ) -> None:
        """Initialize the AgentSystem.

        Parameters
        ----------
        workflow : Workflow
            The Workflow instance with started dfk
        executor_names : List[str]
            List of executor names for running agents on Parsl executors
        exchange_address : str
            The exchange server address
        auth_method : str
            Authentication method for accessing the exchange (default: "globus")
        """
        self.workflow = workflow
        self.executor_names = executor_names
        self.exchange_address = exchange_address
        self.auth_method = auth_method
        self._executors: Optional[Dict[str, ParslPoolExecutor]] = None

    def _create_executors(self) -> None:
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

    async def manager(self) -> ChiltepinManager:
        """Create and return a Chiltepin Manager context manager.

        This method returns a Chiltepin Manager configured with HttpExchangeFactory
        using Globus authentication. The Manager is created with ParslPoolExecutors
        for all configured executors.

        Returns
        -------
        ChiltepinManager
            An async context manager for the Chiltepin Manager

        Examples
        --------

        .. code-block:: python

            async with await agent_system.manager() as manager:
                agent = await manager.launch(MyAgent, executor="my-executor")
                result = await agent.some_action()
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
    def executors(self) -> Optional[Dict[str, ParslPoolExecutor]]:
        """Access the created ParslPoolExecutors.

        Returns None if executors haven't been created yet.
        """
        return self._executors


def agent_action(func: Callable) -> Callable:
    """Marker decorator to indicate a method should be exposed as an agent action.

    Use this decorator on methods in classes decorated with @chiltepin_agent to
    mark them as actions that should be exposed through the agent interface.

    Unlike Academy's @action decorator, this works for both sync and async methods,
    making it suitable for @python_task decorated methods as well as async helpers.

    Examples
    --------

    .. code-block:: python

        from chiltepin.agents import chiltepin_agent, agent_action
        from chiltepin.tasks import python_task

        @chiltepin_agent(agent_workflow_include=["compute"])
        class MyModel:
            @python_task
            @agent_action  # ← Works for sync task methods
            def compute(self):
                return "result"

            @agent_action  # ← Also works for async methods
            async def get_status(self):
                return "ready"

    .. important::
        **Always import from chiltepin.agents, not academy.agent**::

            from chiltepin.agents import agent_action  # ✅ Correct
            from academy.agent import action     # ❌ Wrong - different semantics

        Using Academy's decorator will cause confusing errors. Academy's @action
        requires async methods, while chiltepin's works with any callable.
    """
    func._chiltepin_expose = True
    return func


def agent_loop(func: Callable) -> Callable:
    """Marker decorator for background agent_loop methods in chiltepin agents.

    Use this decorator on async methods with a 'shutdown' parameter that should
    run as background loops in classes decorated with @chiltepin_agent.

    This is equivalent to Academy's @loop decorator but provided here for consistency
    so all decorators can be imported from chiltepin.agents.

    .. important::
        **The decorated method MUST be async.** The decorator validates this at
        decoration time and will raise a TypeError if applied to a non-async method.
        Background loops need to be async to properly cooperate with the agent's
        event loop.

    .. important::
        **Always import from chiltepin.agents, not academy.agent**::

            from chiltepin.agents import agent_loop  # ✅ Correct
            from academy.agent import loop     # ❌ Wrong - will cause type errors

        Using Academy's decorator will cause confusing signature validation errors.

    Examples
    --------

    .. code-block:: python

        from chiltepin.agents import chiltepin_agent, agent_loop
        import asyncio

        @chiltepin_agent(agent_workflow_include=["compute"])
        class MyModel:
            @agent_loop
            async def update_data(self, shutdown):
                while not shutdown.is_set():
                    await asyncio.sleep(1)
                    # Update state
    """
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"@agent_loop can only be applied to async methods. "
            f"Method '{func.__name__}' is not async. "
            f"Did you forget the 'async' keyword?"
        )

    # Validate that the method has the correct signature for Academy's loop protocol
    # Academy will call the method with exactly one argument: shutdown (asyncio.Event)
    # Expected signature: async def loop_method(self, shutdown): ...
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Filter out 'self' parameter (should be first for instance methods)
    non_self_params = [p for p in params if p.name != "self"]

    if not non_self_params:
        raise TypeError(
            f"@agent_loop method '{func.__name__}' must accept a 'shutdown' parameter. "
            f"Expected signature: async def {func.__name__}(self, shutdown: asyncio.Event): "
            f"The shutdown parameter is an asyncio.Event used to signal loop termination."
        )

    # Should have exactly one non-self parameter
    if len(non_self_params) != 1:
        raise TypeError(
            f"@agent_loop method '{func.__name__}' must accept exactly one parameter (shutdown). "
            f"Found {len(non_self_params)} parameters: {', '.join(p.name for p in non_self_params)}. "
            f"Expected signature: async def {func.__name__}(self, shutdown: asyncio.Event):"
        )

    # That parameter should not be *args or **kwargs
    first_param = non_self_params[0]

    if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
        raise TypeError(
            f"@agent_loop method '{func.__name__}' should not use *args. "
            f"Expected signature: async def {func.__name__}(self, shutdown: asyncio.Event): "
            f"Academy will call the loop with exactly one argument (shutdown Event)."
        )

    if first_param.kind == inspect.Parameter.VAR_KEYWORD:
        raise TypeError(
            f"@agent_loop method '{func.__name__}' should not use **kwargs. "
            f"Expected signature: async def {func.__name__}(self, shutdown: asyncio.Event): "
            f"Academy will call the loop with exactly one argument (shutdown Event)."
        )

    func._chiltepin_loop = True
    return func


def chiltepin_agent(
    *,
    agent_workflow_include: Optional[List[str]] = None,
    agent_workflow_run_dir: Optional[str] = None,
) -> Callable[[Type], Type[Agent]]:
    """Decorator that wraps a regular Python class (behavior) in an Academy Agent.

    This decorator allows you to write agent behavior as a regular, serializable
    Python class where task-decorated methods can access instance state directly.
    The decorator automatically creates an Agent wrapper that manages the workflow
    lifecycle and exposes the behavior's methods as actions.

    Only methods marked with @agent_action or @agent_loop decorators are exposed as agent actions.
    Use @agent_action for any method (sync or async) you want to expose, and @agent_loop for
    background loops. Both decorators should be imported from chiltepin.agents.

    Parameters
    ----------
    agent_workflow_include : Optional[List[str]]
        Default list of resource labels to load. Can be overridden at runtime
        using agent_workflow_include= keyword argument in manager.launch(). If None, all
        resources are loaded.
    agent_workflow_run_dir : Optional[str]
        Default directory for Parsl runtime files. Can be overridden at runtime
        using agent_workflow_run_dir= keyword argument in manager.launch(). If None,
        a unique directory is auto-generated to prevent collisions when multiple agents
        run on shared filesystems.

    Returns
    -------
    Callable[[Type], Type[Agent]]
        A decorator function that wraps the behavior class in an Agent.

    Notes
    -----
    Runtime Configuration:
        When using AgentSystem (which provides ChiltepinManager), pass workflow
        configuration using keyword arguments to launch(). This separates infrastructure
        from behavior logic.

    Examples
    --------
    Runtime configuration:

    .. code-block:: python

            @chiltepin_agent(agent_workflow_include=["default-executor"])
            class MyModel:
                def __init__(self, temperature):  # ← No parsl config! Pure domain logic
                    self.temperature = temperature

            # Launch with runtime configuration
            model = await manager.launch(
                MyModel,
                agent_workflow_config=ursa_config,              # ← Workflow config used by the agent's workflow context
                args=(25.0,),                          # ← Behavior args only (domain logic)
                agent_workflow_include=["runtime-executor"],    # ← Override decorator default - executors the agent should use to run tasks
                executor="ursa-service-gc"             # ← The executor to use for launching the agent itself (infrastructure)
            )

    Basic agent creation:

    .. code-block:: python

        from chiltepin.agents import chiltepin_agent, agent_action, agent_loop
        from chiltepin.tasks import python_task

        @chiltepin_agent(agent_workflow_include=["ursa-compute"])  # ← Default, can be overridden
        class MyModel:
            '''Regular Python class - fully serializable!'''

            def __init__(self, temperature: float):
                '''Initialize behavior with domain logic only.'''
                self.temperature = temperature

            @agent_action  # ← Use @agent_action for sync/task-decorated methods
            @python_task
            def run_model(self) -> str:
                # Import modules inside methods for serialization
                import random
                # Can directly access self.temperature!
                return f"Predicted: {self.temperature + random.uniform(0, 5):.2f} degrees"

            @agent_action  # ← Use @agent_action for async methods too
            async def get_status(self) -> str:
                return f"Temperature: {self.temperature:.2f}"

            @agent_loop  # ← Use @agent_loop for background loops
            async def update_temperature(self, shutdown) -> None:
                # Import modules inside methods for serialization
                import asyncio
                import random

                while not shutdown.is_set():
                    await asyncio.sleep(1)
                    self.temperature += random.uniform(-3, 3)

            def _private_helper(self):
                # Not decorated with @agent_action, won't be exposed
                pass

        # Launch agent using decorator defaults (agent_workflow_include=["ursa-compute"])
        model = await manager.launch(MyModel, agent_workflow_config=config, args=(25,))

        # Override decorator defaults at runtime
        model = await manager.launch(
            MyModel,
            agent_workflow_config=config,
            args=(25,),
            agent_workflow_include=["runtime-executor"],  # ← Override decorator's agent_workflow_include
        )

        result = await model.run_model()
        status = await model.get_status()
    """
    import asyncio
    import inspect

    from academy.agent import action as academy_action
    from academy.agent import loop as academy_loop
    from parsl.dataflow.futures import AppFuture

    # Capture decorator parameters for use in closure
    decorator_include = agent_workflow_include
    decorator_run_dir = agent_workflow_run_dir

    def decorator(behavior_class: Type) -> Type[Agent]:
        """Inner decorator that receives the behavior class."""

        # Check if the class itself is already decorated (double-decoration)
        # Use __dict__ to distinguish direct decoration from inherited flag
        if behavior_class.__dict__.get("_is_chiltepin_agent", False):
            raise TypeError(
                f"Cannot apply @chiltepin_agent to '{behavior_class.__name__}' - it is already decorated.\n\n"
                f"Double-decoration is not supported. Remove one of the @chiltepin_agent() decorators.\n\n"
                f"If you meant to create a subclass, create an undecorated behavior class first:\n"
                f"   class {behavior_class.__name__}Behavior:\n"
                f"       @agent_action\n"
                f"       async def method(self): ...\n\n"
                f"   @chiltepin_agent()\n"
                f"   class {behavior_class.__name__}({behavior_class.__name__}Behavior):\n"
                f"       pass"
            )

        # Check if user is trying to extend a decorated agent (unsupported pattern)
        # Use mro() to check entire inheritance chain, not just immediate parents
        # Use __dict__ to identify the actually-decorated class (not intermediate classes
        # that merely inherited the flag), so error messages are accurate
        for base in behavior_class.mro()[1:]:  # Skip first element (class itself)
            if isinstance(base, type) and base.__dict__.get(
                "_is_chiltepin_agent", False
            ):
                # Found a decorated agent in the inheritance chain (directly decorated, not inherited)
                original_name = getattr(base, "_behavior_class_name", base.__name__)
                raise TypeError(
                    f"Cannot extend decorated agent class '{base.__name__}'.\n\n"
                    f"The @chiltepin_agent decorator wraps classes in an Agent, making them "
                    f"unsuitable as base classes. To use inheritance:\n\n"
                    f"1. Create an undecorated base class with shared behavior:\n"
                    f"   class {original_name}Base:\n"
                    f"       @agent_action\n"
                    f"       async def shared_method(self): ...\n\n"
                    f"2. Extend and decorate your specific implementation:\n"
                    f"   @chiltepin_agent()\n"
                    f"   class {behavior_class.__name__}({original_name}Base):\n"
                    f"       @agent_action\n"
                    f"       async def custom_method(self): ...\n\n"
                    f"See the 'Agent Inheritance' section in the documentation for details."
                )

        # Create a wrapper Agent class dynamically
        class ChiltepinAgentWrapper(Agent):
            # Note: Coverage excluded for __init__ and lifecycle methods below.
            # These methods execute in Academy Agent workers (often in separate processes or
            # on remote systems via Globus Compute). The coverage tool can only track code
            # executing in the local test process, not in remote workers or agent processes
            # managed by Academy's exchange system.
            def __init__(
                self,
                *args: Any,
                agent_workflow_config: Optional[
                    Union[str, Path, Dict[str, Any]]
                ] = None,
                agent_workflow_include: Optional[List[str]] = None,
                agent_workflow_run_dir: Optional[str] = None,
                **kwargs: Any,
            ) -> None:  # pragma: no cover
                """Initialize the agent wrapper.

                Parameters
                ----------
                *args : Any
                    Positional arguments for behavior class
                agent_workflow_config : Optional[Union[str, Path, Dict[str, Any]]]
                    Configuration for Agent's workflow context (from manager.launch), defaults to local config
                agent_workflow_include : Optional[List[str]]
                    Optional runtime override for Agent's workflow executor list (from manager.launch)
                agent_workflow_run_dir : Optional[str]
                    Optional runtime override for Parsl's run directory (from manager.launch).
                    If None, a unique directory is auto-generated to prevent collisions.
                **kwargs : Any
                    Keyword arguments for behavior class
                """
                super().__init__()  # pragma: no cover
                # Store config for workflow setup
                self._config = agent_workflow_config  # pragma: no cover

                # Use runtime overrides if provided, otherwise fall back to decorator defaults
                self._include = (
                    agent_workflow_include
                    if agent_workflow_include is not None
                    else decorator_include
                )  # pragma: no cover

                # Auto-generate unique run_dir if not specified to prevent collisions
                # when multiple agents run on shared filesystems
                if agent_workflow_run_dir is not None:
                    self._run_dir = agent_workflow_run_dir
                elif decorator_run_dir is not None:
                    self._run_dir = decorator_run_dir
                else:
                    # Generate unique run_dir using UUID to avoid Parsl directory collisions
                    self._run_dir = (
                        f"parsl_runinfo_{uuid.uuid4().hex[:8]}"  # pragma: no cover
                    )

                # Create the behavior instance with its args/kwargs
                # Note: agent_workflow_config, agent_workflow_include, and agent_workflow_run_dir are infrastructure, not passed to behavior
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
        # Only wrap methods that are marked with @agent_action or @agent_loop decorators
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

            # Check if this method was marked with @agent_action or @agent_loop
            is_loop_method = False
            is_exposed = False

            # First, detect if user accidentally used Academy's @action or @loop decorators
            # Academy's decorators set _agent_method_type attribute
            if hasattr(attr, "_agent_method_type"):
                method_type = getattr(attr, "_agent_method_type", "unknown")
                has_chiltepin_marker = hasattr(attr, "_chiltepin_expose") or hasattr(
                    attr, "_chiltepin_loop"
                )

                if has_chiltepin_marker:
                    # Mixed usage - both Academy and Chiltepin decorators
                    raise TypeError(
                        f"Method '{name}' in class '{behavior_class.__name__}' has both Academy and Chiltepin decorators. "
                        f"This is not supported and will cause double-wrapping issues. "
                        f"Remove the Academy decorator (@action or @loop) and use only Chiltepin decorators:\n"
                        f"  from chiltepin.agents import agent_action, agent_loop  # Use these only"
                    )
                elif method_type == "action":
                    raise TypeError(
                        f"Method '{name}' in class '{behavior_class.__name__}' uses Academy's @action decorator. "
                        f"Use @agent_action from chiltepin.agents instead:\n"
                        f"  from chiltepin.agents import agent_action  # Not 'from academy.agent import action'"
                    )
                elif method_type == "loop":
                    raise TypeError(
                        f"Method '{name}' in class '{behavior_class.__name__}' uses Academy's @loop decorator. "
                        f"Use @agent_loop from chiltepin.agents instead:\n"
                        f"  from chiltepin.agents import agent_loop  # Not 'from academy.agent import loop'"
                    )

            # Check for @agent_loop - must have _chiltepin_loop marker
            # Note: Coverage excluded for these conditional branches.
            # The code executes during decorator application, but coverage.py cannot track
            # execution inside dynamically-created decorator closures.
            if hasattr(attr, "_chiltepin_loop"):  # pragma: no cover
                is_loop_method = True

            # Check for @agent_action marker (our custom decorator sets _chiltepin_expose)
            if hasattr(attr, "_chiltepin_expose"):
                is_exposed = True

            # Only wrap methods that are explicitly marked
            # Note: Coverage excluded for method wrapping closures below.
            # These nested functions execute, but coverage.py cannot track code inside
            # dynamically-created closures that are set as class attributes.
            if is_loop_method:  # pragma: no cover
                # This is a @agent_loop method - wrap it appropriately
                def make_loop_method(method_name):
                    @academy_loop
                    async def loop_method(self, shutdown: asyncio.Event) -> None:
                        method = getattr(self._behavior, method_name)
                        await method(shutdown)

                    loop_method.__name__ = method_name
                    loop_method.__doc__ = getattr(behavior_class, method_name).__doc__
                    return loop_method

                setattr(ChiltepinAgentWrapper, name, make_loop_method(name))

            # Note: Coverage excluded for agent_action method wrapping.
            # These closures execute when agent actions are created, but coverage.py
            # cannot track nested closure code that's dynamically attached to classes.
            elif is_exposed:  # pragma: no cover
                # Method was decorated with @agent_action
                # Wrap it as an agent_action
                if inspect.iscoroutinefunction(attr):
                    # Async agent_action
                    def make_async_action(method_name):
                        @academy_action
                        async def action_method(self, *args, **kwargs):
                            method = getattr(self._behavior, method_name)
                            return await method(*args, **kwargs)

                        action_method.__name__ = method_name
                        action_method.__doc__ = getattr(
                            behavior_class, method_name
                        ).__doc__
                        return action_method

                    setattr(ChiltepinAgentWrapper, name, make_async_action(name))
                else:
                    # Sync agent_action (might be task-decorated)
                    def make_action(method_name):
                        @academy_action
                        async def action_method(self, *args, **kwargs):
                            method = getattr(self._behavior, method_name)
                            result = method(*args, **kwargs)

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

        # Mark this as a chiltepin agent for ChiltepinManager validation
        ChiltepinAgentWrapper._is_chiltepin_agent = True

        # Store original behavior class name for better error messages
        ChiltepinAgentWrapper._behavior_class_name = behavior_class.__name__

        return ChiltepinAgentWrapper

    return decorator
