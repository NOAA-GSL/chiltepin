MPAS Global Mesh Agent
======================

``chiltepin.agents.mpas.MeshAgent`` Generates MPAS meshes for use in NWP
workflows. The current implementation downloads precomputed global MPAS
meshes from UCAR. Regional meshes and partitioning are not yet supported.

For the API reference, see :doc:`../api/agents/mpas_mesh`.

Requirements
------------

* Chiltepin installed in the agent and task execution environments; no extra
  Python dependencies are required.
* Authentication via ``chiltepin login`` before launching the agent.
* Network access to the Academy Exchange from the agent and to UCAR from
  the download resource.

Launch Arguments
----------------

Pass these arguments to ``manager.launch(MeshAgent, ...)``:

``kwargs={"work_dir": "/path/to/work"}``
   Required agent argument. Supplies a writable working-data directory on the
   system hosting the agent, where the agent writes its logs and installs tools.

``executor``
   Resource in the launching workflow on which to deploy the agent.

``agent_workflow_config``
   Configuration dictionary defining the resources the agent will use to run
   its tasks. Must define a ``service`` resource having internet access, as
   described below.

``agent_workflow_include``
   Optional resource-label list restricting which configured resources are
   loaded. If supplied, it must include ``"service"``.

``agent_workflow_run_dir``
   Optional directory for the agent's workflow runtime files. Use a unique
   directory for each agent; if omitted, a unique directory is generated.

Agent Configuration
-------------------

The agent workflow configuration must contain this resource:

``service``
   Runs agent tasks that require internet access. It must have write access
   to the mesh output directory. The agent and this resource must share that
   directory at the same absolute path.

For local execution:

.. code-block:: python

   agent_config = {"service": {"provider": "localhost"}}

Actions
-------

``download_global_mesh(resolution, mesh_data_dir)``
   Downloads the requested global mesh. Returns a dictionary with ``"static"``
   (the static NetCDF file) and ``"graph"`` (the graph file).

   * ``resolution``: The horizontal resolution of the global mesh to download
   * ``mesh_data_dir``: Where the agent will write the mesh files

``generate_mesh(mesh_config, mesh_data_dir)``
   Generates the requested mesh and creates named symlinks in a subdirectory of
   ``mesh_data_dir``. Returns ``"mesh"`` and ``"graph"`` paths to those links.

   * ``mesh_config``: Dictionary with optional ``resolution`` (default
     ``"120km"``) and mesh ``name`` (default ``"mesh_<resolution>"``).
   * ``mesh_data_dir``: Output directory shared by the agent and its tasks.

Both actions return absolute paths and reuse mesh files already downloaded to
``mesh_data_dir``.

Supply absolute paths appropriate to the system hosting the agent, not the
client launching it. Avoid concurrent writes to the same mesh files.

Supported global resolutions are ``480km``, ``384km``, ``240km``, ``120km``, ``60km``,
``48km``, ``30km``, ``24km``, ``15km``, ``12km``, ``10km``, ``7.5km``, ``5km``,
``4km``, ``3.75km``, and ``3km``.


Deployment Example
------------------

This example launches the agent and its tasks locally:

.. code-block:: python

   import asyncio
   from pathlib import Path

   from chiltepin import AgentRuntime, Workflow
   from chiltepin.agents.mpas import MeshAgent

   async def main():
       work_dir = str(Path.home() / "mpas-work")
       mesh_data_dir = str(Path.home() / "mpas-meshes")
       controller_config = {"agent-executor": {"provider": "localhost"}}
       agent_config = {"service": {"provider": "localhost"}}

       workflow = Workflow(controller_config, run_dir="mesh-controller-runinfo")
       with workflow:
           runtime = AgentRuntime(workflow)
           async with await runtime.manager() as manager:
               mesh = await manager.launch(
                   MeshAgent,
                   executor="agent-executor",
                   kwargs={"work_dir": work_dir},
                   agent_workflow_config=agent_config,
                   agent_workflow_run_dir="mesh-agent-runinfo",
               )
               files = await mesh.download_global_mesh("480km", mesh_data_dir)
               print(files["static"])
               print(files["graph"])
               named = await mesh.generate_mesh(
                   {"resolution": "480km", "name": "forecast"}, mesh_data_dir
               )
               print(named["mesh"])
               print(named["graph"])

   asyncio.run(main())
