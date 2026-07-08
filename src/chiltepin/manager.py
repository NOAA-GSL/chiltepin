# SPDX-License-Identifier: Apache-2.0

"""Chiltepin Manager for launching agents with workflow configuration.

This module provides the Manager class that extends Academy's Manager to support
Chiltepin-specific workflow configuration parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from academy.handle import Handle
from academy.manager import Manager as AcademyManager

if TYPE_CHECKING:
    from academy.agent import AgentT
else:
    AgentT = TypeVar("AgentT")


class Manager(AcademyManager):
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
                    f"Manager only supports agents decorated with @chiltepin_agent. "
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
