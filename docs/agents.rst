Agents
======

Chiltepin integrates with `Academy Agents <https://docs.academy-agents.org/latest/get-started/>`_ to support
distributed agent-based workflows. Agents enable long-running, stateful computations that can
be launched on remote resources and interacted with asynchronously through an action-based API.

.. note::
   Chiltepin's agent system builds on Academy Agents to provide:

   1. **Automatic workflow lifecycle management**: Agents manage their own Parsl workflow context
   2. **Simplified agent creation**: Use ``@chiltepin_agent`` decorator on regular Python classes
   3. **Runtime configuration**: Pass workflow config, executors, and paths via ``manager.launch()``
   4. **Serializable behavior**: Decorated classes remain fully serializable for remote execution

   For more information about Academy Agents, see the
   `Academy documentation <https://docs.academy-agents.org/latest/get-started/>`_.

Overview
--------

Chiltepin provides five main components for agent-based workflows:

- **@chiltepin_agent**: Decorator to wrap a regular Python class as an agent
- **@action**: Decorator to mark methods that should be exposed as agent actions
- **@loop**: Decorator to mark methods that should run as background loops
- **AgentSystem**: Helper class to simplify Academy Manager setup with Parsl executors
- **ChiltepinManager**: Custom Manager that supports workflow configuration parameters

When to Use Agents
------------------

Use agents when you need:

- **Long-running services**: Agents that persist beyond a single task execution
- **Stateful computations**: Maintaining state across multiple action invocations
- **Background processing**: Loops that update state while handling requests
- **Autonomous behavior**: Agents that can make decisions and act without external prompts
- **Remote interaction**: Asynchronous communication with computations on remote resources

For one-off tasks without shared state, use :doc:`tasks` instead.

Basic Usage
-----------

Creating an Agent
^^^^^^^^^^^^^^^^^

Use the ``@chiltepin_agent`` decorator to wrap a regular Python class:

.. code-block:: python

   from chiltepin.agents import chiltepin_agent, action, loop
   from chiltepin.tasks import python_task

   @chiltepin_agent(include=["compute"])
   class WeatherModel:
       """A simple weather model agent."""
       
       def __init__(self, temperature: float):
           self.temperature = temperature
       
       @action
       @python_task
       def forecast(self) -> str:
           """Generate a forecast based on current temperature."""
           import random
           conditions = ["sunny", "cloudy", "rainy"]
           return f"{random.choice(conditions)} at {self.temperature}°C"
       
       @action
       async def get_temperature(self) -> float:
           """Get the current temperature."""
           return self.temperature
       
       @loop
       async def update_temperature(self, shutdown):
           """Background loop that updates temperature."""
           import asyncio
           import random
           while not shutdown.is_set():
               await asyncio.sleep(1)
               self.temperature += random.uniform(-2, 2)

Key Features
^^^^^^^^^^^^

1. **Regular Python class**: No inheritance required, fully serializable
2. **Access instance state**: Task-decorated methods can access ``self.temperature``
3. **Mixed sync/async**: Use ``@action`` on both sync and async methods
4. **Background loops**: Use ``@loop`` for continuous background processing or autonomous behavior
5. **Infrastructure separation**: Workflow config passed via ``manager.launch()``, not ``__init__``

Launching Agents
^^^^^^^^^^^^^^^^

Use ``AgentSystem`` to create a manager and launch agents:

.. code-block:: python

   from chiltepin import Workflow, AgentSystem
   
   # Configuration for the manager's workflow (where agents run)
   manager_config = {
       "manager-executor": {
           "endpoint": ENDPOINT_UUID,
           "provider": "localhost",
       }
   }
   
   # Configuration for the agent's internal workflow (where tasks run)
   agent_config = {
       "compute": {
           "provider": "slurm",
           "partition": "compute",
           # ... other config
       }
   }
   
   # Start workflow for hosting agents
   workflow = Workflow(manager_config, include=["manager-executor"])
   workflow.start()
   
   # Create agent system
   agent_system = AgentSystem(
       workflow=workflow,
       executor_names=["manager-executor"],
   )
   
   # Launch and interact with agent
   async with await agent_system.manager() as manager:
       model = await manager.launch(
           WeatherModel,
           config=agent_config,         # Agent's workflow config
           include=["compute"],         # Which executors to use
           args=(25.0,),                # Arguments for __init__
           executor="manager-executor"  # Where to run the agent
       )
       
       # Call agent actions
       temp = await model.get_temperature()
       forecast = await model.forecast(executor=["compute"])
   
   workflow.cleanup()

Runtime Configuration
---------------------

Infrastructure concerns (workflow config, executors, directories) are passed to
``manager.launch()`` rather than the behavior class:

.. code-block:: python

   model = await manager.launch(
       WeatherModel,
       config=agent_config,         # Workflow configuration dict or YAML path
       include=["compute"],         # List of executors to include (None = all)
       run_dir="/custom/path",      # Directory for Parsl runtime files
       args=(25.0,),                # Behavior arguments (domain logic)
       kwargs={"units": "C"},       # Behavior keyword arguments
       executor="manager-executor"  # Agent executor (where agent runs)
   )

This separation keeps behavior classes focused on domain logic:

.. code-block:: python

   @chiltepin_agent()
   class WeatherModel:
       def __init__(self, temperature: float, units: str = "C"):
           # Only domain logic, no infrastructure concerns
           self.temperature = temperature
           self.units = units

Decorator Parameters
^^^^^^^^^^^^^^^^^^^^

The ``@chiltepin_agent`` decorator accepts default values that can be overridden at runtime:

.. code-block:: python

   @chiltepin_agent(include=["default-compute"], run_dir="./runs")
   class MyAgent:
       pass
   
   # Use decorator defaults
   agent1 = await manager.launch(MyAgent, config=cfg)
   
   # Override at runtime
   agent2 = await manager.launch(
       MyAgent,
       config=cfg,
       include=["special-compute"],  # Overrides decorator default
       run_dir="/tmp/runs"           # Overrides decorator default
   )

Action Decorators
-----------------

Use ``@action`` to expose methods as agent actions. The decorator works with both
synchronous and asynchronous methods:

Synchronous Actions
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from chiltepin.agents import chiltepin_agent, action
   from chiltepin.tasks import python_task
   
   @chiltepin_agent()
   class DataProcessor:
       @action
       @python_task
       def process_data(self, data: str) -> str:
           """Synchronous task-decorated method."""
           return data.upper()
       
       @action
       def get_config(self) -> dict:
           """Synchronous helper method."""
           return {"version": "1.0"}

Asynchronous Actions
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @chiltepin_agent()
   class AsyncService:
       @action
       async def fetch_data(self, url: str) -> str:
           """Async method using httpx, aiohttp, etc."""
           import httpx  # ✅ Import inside method for serializability
           # ✅ Create client temporarily, don't store in self.__init__
           async with httpx.AsyncClient() as client:
               response = await client.get(url)
               return response.text

Task-Decorated Actions
^^^^^^^^^^^^^^^^^^^^^^

When using ``@python_task`` with ``@action``, the order matters:

.. code-block:: python

   @chiltepin_agent()
   class Computer:
       @python_task  # ← Second
       @action       # ← First
       def compute(self, x: int) -> int:
           return x ** 2

This allows the task to access instance state (``self``) while still executing remotely.

Loop Decorators
---------------

Use ``@loop`` to create background tasks that run continuously:

.. code-block:: python

   from chiltepin.agents import chiltepin_agent, loop
   import asyncio
   
   @chiltepin_agent()
   class Monitor:
       def __init__(self):
           self.status = "initializing"
           self.count = 0
       
       @loop
       async def heartbeat(self, shutdown: asyncio.Event):
           """Background loop that runs until agent shuts down."""
           self.status = "running"
           while not shutdown.is_set():
               await asyncio.sleep(1)
               self.count += 1
               if self.count % 10 == 0:
                   print(f"Heartbeat: {self.count}")
           self.status = "stopped"

The ``shutdown`` event is provided automatically and signals when the agent is shutting down.

AgentSystem Helper
-------------------

The ``AgentSystem`` class simplifies setup by wrapping the complexity of creating
an Academy Manager with ParslPoolExecutors:

.. code-block:: python

   from chiltepin import Workflow, AgentSystem
   
   # Without AgentSystem (manual setup)
   from academy.manager import Manager
   from academy.exchange.cloud.client import HttpExchangeFactory
   from parsl.concurrent import ParslPoolExecutor
   
   executors = {
       "my-exec": ParslPoolExecutor(dfk=workflow.dfk, executors=["my-exec"])
   }
   
   async with await Manager.from_exchange_factory(
       factory=HttpExchangeFactory(
           "https://exchange.academy-agents.org",
           auth_method="globus"
       ),
       executors=executors
   ) as manager:
       # Use manager
       pass
   
   # With AgentSystem (simplified)
   agent_system = AgentSystem(
       workflow=workflow,
       executor_names=["my-exec"],
   )
   
   async with await agent_system.manager() as manager:
       # Use manager - ChiltepinManager with config/include/run_dir support
       pass

ChiltepinManager
----------------

``ChiltepinManager`` is a custom ``Manager`` subclass that intercepts ``launch()``
to support Chiltepin-specific parameters (``config``, ``include``, ``run_dir``).
It's created automatically by ``AgentSystem.manager()``.

You can also create it directly:

.. code-block:: python

   from chiltepin.agents import ChiltepinManager
   from academy.exchange.cloud.client import HttpExchangeFactory
   
   async with await ChiltepinManager.from_exchange_factory(
       factory=HttpExchangeFactory(
           "https://exchange.academy-agents.org",
           auth_method="globus"
       ),
       executors=my_executors
   ) as manager:
       agent = await manager.launch(
           MyAgent,
           config=agent_config,
           include=["compute"]
       )

Best Practices
--------------

Import Decorators Correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Always import ``action`` and ``loop`` from ``chiltepin.agents``, not ``academy.agent``:

.. code-block:: python

   # ✅ Correct
   from chiltepin.agents import chiltepin_agent, action, loop
   
   # ❌ Wrong - Academy's decorators have different semantics
   from academy.agent import action, loop

Academy's ``@action`` requires async methods, while Chiltepin's works with both sync and async.

Keep Behavior Classes Serializable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since agents can run remotely, behavior classes must be serializable:

.. code-block:: python

   @chiltepin_agent()
   class GoodAgent:
       def __init__(self, value: int):
           self.value = value  # ✅ Serializable types
       
       @action
       @python_task
       def compute(self):
           # ✅ Import modules inside methods for remote execution
           import numpy as np
           return np.array([self.value])
   
   @chiltepin_agent()
   class BadAgent:
       def __init__(self, value: int):
           import numpy as np  # ❌ Don't import at class level
           self.np = np        # ❌ Modules may not serialize
           self.value = value

Separate Infrastructure from Logic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass infrastructure concerns via ``manager.launch()``:

.. code-block:: python

   @chiltepin_agent()
   class MyAgent:
       def __init__(self, threshold: float):  # ✅ Domain parameters only
           self.threshold = threshold
   
   # ❌ Bad - mixing infrastructure with domain logic
   # def __init__(self, threshold: float, workflow_config: dict):
   #     self.threshold = threshold
   #     self.config = workflow_config

Use Type Hints
^^^^^^^^^^^^^^

Type hints improve code clarity and enable better IDE support:

.. code-block:: python

   from typing import List
   
   @chiltepin_agent()
   class TypedAgent:
       def __init__(self, values: List[float]):
           self.values = values
       
       @action
       @python_task
       def mean(self) -> float:
           return sum(self.values) / len(self.values)

Complete Example
----------------

Here's a complete example combining all features:

.. code-block:: python

   import asyncio
   import logging
   from chiltepin import Workflow, AgentSystem
   from chiltepin.agents import chiltepin_agent, action, loop
   from chiltepin.tasks import python_task
   
   logger = logging.getLogger(__name__)
   
   @chiltepin_agent(include=["compute"])
   class TemperatureModel:
       """Agent that forecasts temperature with background updates."""
       
       def __init__(self, initial_temp: float, location: str):
           self.temperature = initial_temp
           self.location = location
           self.forecast_count = 0
       
       @action
       @python_task
       def forecast(self) -> str:
           """Generate forecast using current temperature."""
           import random
           conditions = ["sunny", "cloudy", "rainy", "snowy"]
           condition = random.choice(conditions)
           return f"{self.location}: {condition}, {self.temperature:.1f}°C"
       
       @action
       async def get_stats(self) -> dict:
           """Get current statistics."""
           return {
               "temperature": self.temperature,
               "location": self.location,
               "forecasts_generated": self.forecast_count
           }
       
       @action
       async def set_temperature(self, temp: float) -> None:
           """Manually update temperature."""
           self.temperature = temp
       
       @loop
       async def update_temperature(self, shutdown: asyncio.Event):
           """Simulate temperature changes."""
           import asyncio
           import random
           
           while not shutdown.is_set():
               await asyncio.sleep(2)
               # Random walk
               self.temperature += random.uniform(-1, 1)
               # Keep reasonable bounds
               self.temperature = max(-50, min(50, self.temperature))
   
   async def main():
       # Manager workflow configuration
       manager_config = {
           "manager-executor": {
               "endpoint": "your-endpoint-uuid",
               "provider": "localhost",
           }
       }
       
       # Agent workflow configuration
       agent_config = {
           "compute": {
               "provider": "slurm",
               "partition": "compute",
               "cores_per_node": 48,
               "walltime": "01:00:00",
           }
       }
       
       # Start manager workflow
       workflow = Workflow(manager_config, include=["manager-executor"])
       workflow.start()
       
       # Create agent system
       agent_system = AgentSystem(
           workflow=workflow,
           executor_names=["manager-executor"],
       )
       
       async with await agent_system.manager() as manager:
           # Launch agent with runtime configuration
           model = await manager.launch(
               TemperatureModel,
               config=agent_config,
               include=["compute"],
               args=(20.0, "Boulder, CO"),
               executor="manager-executor"
           )
           
           # Interact with agent
           logger.info("Getting initial stats...")
           stats = await model.get_stats()
           logger.info(f"Stats: {stats}")
           
           logger.info("Generating forecast...")
           forecast = await model.forecast(executor=["compute"])
           logger.info(f"Forecast: {forecast}")
           
           logger.info("Waiting for temperature updates...")
           await asyncio.sleep(5)
           
           stats = await model.get_stats()
           logger.info(f"Updated stats: {stats}")
           
           logger.info("Setting temperature manually...")
           await model.set_temperature(25.0)
           
           forecast = await model.forecast(executor=["compute"])
           logger.info(f"New forecast: {forecast}")
       
       workflow.cleanup()
       logger.info("Done!")
   
   if __name__ == "__main__":
       logging.basicConfig(level=logging.INFO)
       asyncio.run(main())

Troubleshooting
---------------

Serialization Errors
^^^^^^^^^^^^^^^^^^^^

If you get serialization errors when launching agents:

1. Check that behavior class doesn't inherit from non-serializable classes
2. Move imports inside methods rather than at class level
3. Avoid storing non-serializable objects (file handles, connections) in ``self``

Action Not Found
^^^^^^^^^^^^^^^^

If an action isn't available on the agent proxy:

1. Check that the method is decorated with ``@action``
2. Verify you're importing ``action`` from ``chiltepin.agents``, not ``academy.agent``
3. Ensure the method name doesn't start with underscore (private methods aren't exposed)

Workflow Not Starting
^^^^^^^^^^^^^^^^^^^^^

If the agent's internal workflow doesn't start:

1. Check that ``config`` is passed to ``manager.launch()``
2. Verify the configuration dict is valid (see :doc:`configuration`)
3. Check that ``include`` parameter matches actual executor names in config
4. Check that requested executors are available and can start (e.g. Slurm partition is correct)
5. Check that resources are available (e.g. Slurm queue isn't full)

See Also
--------

- :doc:`tasks` - For information about task decorators
- :doc:`configuration` - For workflow configuration details
- :doc:`quickstart` - For getting started with Chiltepin
- `Academy Agents Documentation <https://docs.academy-agents.org/latest/get-started/>`_ - For more on Academy
