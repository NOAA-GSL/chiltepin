MPAS Global Mesh Agent
======================

``chiltepin.agents.mpas.MeshAgent`` Generates MPAS meshes for use in NWP
workflows. The current implementation downloads precomputed global MPAS
meshes from UCAR and can partition their graphs for MPI with Metis. Regional
meshes are not yet supported.

For the API reference, see :doc:`../api/agents/mpas_mesh`.

Requirements
------------

* Chiltepin installed in the agent and task execution environments; no extra
  Python dependencies are required.
* Authentication via ``chiltepin login`` before launching the agent.
* Network access to the Academy Exchange from the agent and to UCAR and GitHub
  from the ``service`` resource.
* For partitioning, ``git``, ``wget``, and ``tar`` on the ``service``
  resource, and ``cmake``, ``make``, and a C compiler on the ``compute``
  resource, to download and build Metis.

Launch Arguments
----------------

Pass these arguments to ``manager.launch(MeshAgent, ...)``:

``kwargs={"work_dir": "/path/to/work"}``
   Required agent argument. Supplies a writable working-data directory on the
   system hosting the agent, where the agent writes its logs and installs tools.
   A relative path or one starting with ``~`` is made absolute on the agent.
   The ``service`` and ``compute`` resources must see this directory at the same
   absolute path. Give each agent its own ``work_dir``, since agents sharing one
   could install tools over each other.

``kwargs={"metis_version": "5.2.1"}``
   Optional agent argument. The Metis version to install for partitioning, as
   it appears in the release tag without the leading ``v``, such as ``5.2.1``.
   Defaults to ``5.2.1``. Anything else raises ``ValueError``.

``executor``
   Resource in the launching workflow on which to deploy the agent.

``agent_workflow_config``
   Configuration dictionary defining the resources the agent will use to run
   its tasks. Must define a ``service`` resource having internet access, and a
   ``compute`` resource if meshes will be partitioned, as described below.

``agent_workflow_include``
   Optional resource-label list restricting which configured resources are
   loaded. If supplied, it must include ``"service"``, and ``"compute"`` if
   meshes will be partitioned.

``agent_workflow_run_dir``
   Optional directory for the agent's workflow runtime files. Use a unique
   directory for each agent; if omitted, a unique directory is generated.

Agent Configuration
-------------------

The agent workflow configuration must contain these resources:

``service``
   Runs agent tasks that require internet access, such as downloading meshes
   and Metis. It must have write access to the mesh output directory and to
   ``work_dir``. The agent and this resource must share these directories at
   the same absolute paths.

``compute``
   Runs tasks that do not need the internet, such as building Metis and
   partitioning meshes. It needs ``cmake``, ``make`` and a C compiler, and it must have
   write access to ``work_dir`` and the mesh output directory at the same
   absolute paths as the agent. Requests that only download global meshes
   without partitioning do not use it.

For local execution:

.. code-block:: python

   agent_config = {
       "service": {"provider": "localhost"},
       "compute": {"provider": "localhost"},
   }

Actions
-------

``download_global_mesh(resolution, mesh_data_dir)``
   Downloads the requested global mesh. Returns a dictionary with ``"static"``
   (the static NetCDF file) and ``"graph"`` (the graph file).

   * ``resolution``: The horizontal resolution of the global mesh to download
   * ``mesh_data_dir``: Where the agent will write the mesh files

``generate_mesh(mesh_config, mesh_data_dir)``
   Generates the requested mesh and creates named symlinks in a subdirectory of
   ``mesh_data_dir``. Returns ``"static"`` and ``"graph"`` paths to those links.

   * ``mesh_config``: Dictionary with optional ``resolution`` (default
     ``"120km"``), mesh ``name`` (default ``"mesh_<resolution>"``), and
     ``init_ranks`` and ``forecast_ranks``. The rank entries are positive
     integers giving MPI rank counts to partition the mesh graph for. A count
     of 1 needs no partition. A name must not be that of a downloaded global
     mesh file, such as ``x1.2562.graph.info``. Any other key raises
     ``ValueError``, so a misspelled key is not silently ignored.
   * ``mesh_data_dir``: Output directory shared by the agent and its tasks.

   If partitions are requested, the result also has ``"partitions"``, a
   dictionary mapping each rank count to its partition file in the mesh
   subdirectory. Metis is installed first if it is not already installed.
   Before it creates anything or downloads anything, the action checks that the
   workflow defines the resources the request needs, ``service`` and, if
   partitions are requested, ``compute``. If one is missing it raises
   ``RuntimeError`` naming it.

   A mesh with the ``mesh_config``'s name must not already exist in the specified
   ``mesh_data_dir``. An existing mesh raises ``FileExistsError`` without
   overwriting it. Choose a new name for another mesh. If a request fails, its
   mesh subdirectory and everything in it are removed so the name can be used
   again. Global mesh files already downloaded to
   ``mesh_data_dir`` are kept for reuse. If cleanup itself fails, for example
   because the request was interrupted while a partition was still running, the
   agent logs a warning naming the leftover directory, which must be removed
   before the name can be used again. A ``mesh_config`` that is not a
   dictionary, or a ``resolution`` that is not a supported string, raises
   ``ValueError``.

``install_metis()``
   Downloads and builds Metis in ``work_dir``, unless a finished install is
   already there. A finished build leaves a marker file,
   ``work_dir/metis/<metis_version>/.installed``, which is also how a restarted
   agent finds an earlier installation. An install counts only if the marker
   exists and ``bin/gpmetis`` is an executable file, and this is checked on every
   call, so a damaged or removed install is rebuilt, even in a running agent.
   A build always starts from an empty ``work_dir/metis/<metis_version>``, so an
   interrupted or failed install is replaced by the next one. After a successful
   build the downloaded sources in ``work_dir/metis/build`` are removed. After a
   failed build they stay for debugging. This runs
   automatically when a request needs Metis, so calling it is optional, but it
   can be called early to have Metis ready. Requests that need Metis while it
   is being installed wait for that install. If it fails, each waiting request
   tries again in turn and reports its own error. The install logs are
   ``metis_download.*`` and ``metis_build.*`` in ``work_dir/logs``. Installing
   needs both the ``service`` and ``compute`` resources. If one is missing, the
   action raises ``RuntimeError`` naming it, before it starts any download.
``partition_mesh(mesh_path, num_ranks)``
   Partitions a graph file with Metis, installing Metis first if needed.
   ``mesh_path`` must be a non-empty string. A relative path or one starting
   with ``~`` is made absolute on the agent, and the returned paths are
   absolute. Partition files are written next to the graph file.
   ``num_ranks`` is one integer greater than 1 or a non-empty list of at most 16
   distinct ones; anything else raises ``ValueError``. Returns the partition file path for an
   integer, or a dictionary of ``{rank: path}`` for a list. Each call writes its
   own logs, ``partition_mesh.<rank>.<id>.stdout`` and ``.stderr`` in
   ``work_dir/logs``, whose first lines show the graph file, and a failure
   names its ``.stderr`` file.

``ensure_tools_installed(mesh_config)``
   Installs the tools that a ``generate_mesh`` request with this configuration
   needs, so installation can overlap with other work. It checks the
   configuration first, exactly as ``generate_mesh`` does, and raises
   ``ValueError`` if it is invalid.

The download and generate actions return absolute paths and reuse mesh files
already downloaded to ``mesh_data_dir``.

Supply absolute paths appropriate to the system hosting the agent, not the
client launching it. Avoid concurrent writes to the same mesh files.

The ``partitions`` dictionary has integer keys. The agent exchange uses pickle by
default, which preserves them. If you configure the exchange to use JSON,
integer keys arrive as strings, a tuple sent as ``num_ranks`` arrives as a
list, and a set cannot be sent. Use a list of integers to be safe.

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
       agent_config = {
           "service": {"provider": "localhost"},
           "compute": {"provider": "localhost"},
       }

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
                   {"resolution": "480km", "name": "forecast", "forecast_ranks": 4},
                   mesh_data_dir,
               )
               print(named["static"])
               print(named["graph"])
               print(named["partitions"][4])

   asyncio.run(main())
