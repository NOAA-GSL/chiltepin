# SPDX-License-Identifier: Apache-2.0

"""Agent runtime infrastructure for Chiltepin workflows.

This module provides the AgentRuntime class that encapsulates the complexity
of setting up Academy Manager with ParslPoolExecutors and HttpExchangeFactory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from academy.exchange.cloud.client import HttpExchangeFactory
from parsl.concurrent import ParslPoolExecutor

from .manager import Manager

if TYPE_CHECKING:
    from chiltepin.workflow import Workflow


class AgentRuntime:
    """Simplified agent runtime infrastructure for Chiltepin workflows.

    This class wraps the complexity of setting up Academy Manager with
    ParslPoolExecutors and HttpExchangeFactory, providing a clean interface
    for users who always use this pattern.

    Parameters
    ----------
    workflow : Workflow
        The Workflow instance to use for executing tasks and agents
    executor_names : List[str]
        List of executor names for running agents on Parsl executors
    exchange_address : Optional[str]
        The exchange server address. If None (default), academy's built-in
        default exchange URL is used, which matches the API version of the
        installed academy release.
    auth_method : str
        Authentication method (default: "globus")

    Examples
    --------

    .. code-block:: python

        from chiltepin import Workflow, AgentRuntime

        config = {"my-executor": {...}}
        workflow = Workflow(config)
        workflow.start()

        agent_runtime = AgentRuntime(
            workflow=workflow,
            executor_names=["my-executor"],
        )

        async with await agent_runtime.manager() as manager:
            # Launch and interact with agents
            agent = await manager.launch(MyAgent, executor="my-executor")
            result = await agent.some_action()

        workflow.cleanup()
    """

    def __init__(
        self,
        workflow: Workflow,
        executor_names: List[str],
        exchange_address: Optional[str] = None,
        auth_method: str = "globus",
    ) -> None:
        """Initialize the AgentRuntime.

        Parameters
        ----------
        workflow : Workflow
            The Workflow instance with started dfk
        executor_names : List[str]
            List of executor names for running agents on Parsl executors
        exchange_address : Optional[str]
            The exchange server address. If None (default), academy's built-in
            default exchange URL is used, which matches the API version of the
            installed academy release.
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
                "Workflow must be started before creating AgentRuntime executors. "
                "Call workflow.start() first."
            )

        self._executors = {
            name: ParslPoolExecutor(dfk=self.workflow.dfk, executors=[name])
            for name in self.executor_names
        }

    async def manager(self) -> Manager:
        """Create and return a Manager context manager.

        This method returns a Manager configured with HttpExchangeFactory
        using Globus authentication. The Manager is created with ParslPoolExecutors
        for all configured executors.

        Returns
        -------
        Manager
            An async context manager for the Manager

        Examples
        --------

        .. code-block:: python

            async with await agent_runtime.manager() as manager:
                agent = await manager.launch(MyAgent, executor="my-executor")
                result = await agent.some_action()
        """
        # Create executors if not already created
        if self._executors is None:
            self._create_executors()

        # Only pass url= when the caller supplied an explicit address. When
        # None, academy uses its own DEFAULT_EXCHANGE_URL, which tracks the
        # exchange server's API version for the installed academy release.
        # (Academy 0.5.0 moved the public exchange to a versioned "/v1" URL;
        # hardcoding the old unversioned address makes the server reject
        # messages with 400 Bad Request.)
        factory_kwargs: Dict[str, Any] = {"auth_method": self.auth_method}
        if self.exchange_address is not None:
            factory_kwargs["url"] = self.exchange_address

        # Return the Manager context manager
        return await Manager.from_exchange_factory(
            factory=HttpExchangeFactory(**factory_kwargs),
            executors=self._executors,
        )

    @property
    def executors(self) -> Optional[Dict[str, ParslPoolExecutor]]:
        """Access the created ParslPoolExecutors.

        Returns None if executors haven't been created yet.
        """
        return self._executors
