# SPDX-License-Identifier: Apache-2.0

"""Agent system integration for Chiltepin workflows.

This module provides simplified interfaces for integrating Academy agents
with Chiltepin workflows and Parsl executors.
"""

from typing import Optional

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
