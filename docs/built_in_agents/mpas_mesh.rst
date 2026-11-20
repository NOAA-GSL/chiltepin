MPAS Mesh Agent
===============

``chiltepin.agents.mpas.MeshAgent`` Generates MPAS meshes for use in NWP
workflows. It downloads precomputed global MPAS meshes from UCAR, generates
regional meshes with the MPAS-Tools ``project_hexes`` program or cuts them from
a global mesh with the MPAS-Limited-Area ``create_region`` program, can
partition mesh graphs for MPI with Metis, and can draw images of meshes.

For the API reference, see :doc:`../api/agents/mpas_mesh`.

Requirements
------------

* Chiltepin installed in the agent and task execution environments.
  ``create_region`` regional meshes and plotting also need the ``mpas`` extra
  where the agent's tasks run on the ``compute`` resource, which installs
  numpy, netCDF4, matplotlib and cartopy:

  .. code-block:: console

     $ pip install "chiltepin[mpas]"

* Authentication via ``chiltepin login`` before launching the agent.
* Network access to the Academy Exchange from the agent and to UCAR and GitHub
  from the ``service`` resource.
* For partitioning, ``git``, ``wget``, and ``tar`` on the ``service``
  resource, and ``cmake``, ``make``, and a C compiler on the ``compute``
  resource, to download and build Metis.
* For ``project_hexes`` regional meshes, ``git`` on the ``service`` resource,
  and ``make``, a Fortran compiler, and the netCDF-C and netCDF-Fortran
  libraries on the ``compute`` resource, to download and build the
  ``project_hexes`` and ``grid_rotate`` programs from MPAS-Tools. The build
  finds netCDF through ``nf-config`` and ``nc-config``, so these must be on the
  ``PATH`` of the ``compute`` resource, for example by loading the
  netCDF-Fortran module in its ``environment`` setting.
* For ``create_region`` regional meshes, ``git`` on the ``service`` resource,
  to clone MPAS-Limited-Area, and the ``mpas`` extra on the ``compute``
  resource, whose Python runs the ``create_region`` program.
* For plotting, network access to Natural Earth on the ``service`` resource,
  to download the map layers the plots draw. cartopy publishes Linux wheels for
  Python 3.10 only for x86_64, so on an ARM machine with Python 3.10 the
  ``mpas`` extra tries to build cartopy, which needs the GEOS and PROJ
  libraries. Use Python 3.11 or newer there.

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
``kwargs={"mpas_tools_version": "2.0.0"}``
   Optional agent argument. The MPAS-Tools tag to install for regional meshes,
   such as ``2.0.0``, written without a ``v`` prefix, since the MPAS-Tools tags
   have none. Defaults to ``2.0.0``. Anything else raises ``ValueError``.
``kwargs={"limited_area_version": "2.2"}``
   Optional agent argument. The MPAS-Limited-Area version to install for
   ``create_region`` meshes, as it appears in the release tag without the
   leading ``v``, such as ``2.2``. Defaults to ``2.2``. Anything else raises
   ``ValueError``.

``executor``
   Resource in the launching workflow on which to deploy the agent.

``agent_workflow_config``
   Configuration dictionary defining the resources the agent will use to run
   its tasks. Must define a ``service`` resource having internet access, and a
   ``compute`` resource if meshes will be partitioned or regional, as described
   below.

``agent_workflow_include``
   Optional resource-label list restricting which configured resources are
   loaded. If supplied, it must include ``"service"``, and ``"compute"`` if
   meshes will be partitioned or regional.

``agent_workflow_run_dir``
   Optional directory for the agent's workflow runtime files. Use a unique
   directory for each agent; if omitted, a unique directory is generated.

Agent Configuration
-------------------

The agent workflow configuration must contain these resources:

``service``
   Runs agent tasks that require internet access, such as downloading meshes,
   Metis, MPAS-Tools, MPAS-Limited-Area, and the map
   layers plots draw. It must
   have write access to the mesh output directory and to
   ``work_dir``. The agent and this resource must share these directories at
   the same absolute paths.

``compute``
   Runs tasks that do not need the internet, such as building Metis and
   MPAS-Tools, running ``project_hexes``, ``grid_rotate`` and ``create_region``,
   partitioning meshes, and plotting them. ``create_region`` and plotting need
   the ``mpas`` extra. It needs ``cmake``, ``make`` and a C compiler to build Metis, and
   ``make``, a Fortran compiler and netCDF-C and netCDF-Fortran to build and run
   the MPAS-Tools programs. It must have
   write access to ``work_dir`` and the mesh output directory at the same
   absolute paths as the agent. Requests that only download global meshes
   without partitioning do not use it.

For local execution:

.. code-block:: python

   agent_config = {
       "service": {"provider": "localhost"},
       "compute": {"provider": "localhost"},
   }

The ``environment`` setting of a resource lists the commands that run before its
tasks, which is where to load modules. For regional meshes, the ``compute``
resource must provide ``nf-config`` and ``nc-config``. For example, with
spack-stack modules:

.. code-block:: python

   "compute": {
       "provider": "localhost",
       "environment": [
           "source /usr/lmod/lmod/init/bash",
           "module use /opt/spack-stack/envs/unified-env/modules/Core",
           "module load stack-gcc",
           "module load stack-openmpi",
           "module load netcdf-fortran",
           "module load cmake",
       ],
   },

Actions
-------

``download_global_mesh(resolution, mesh_data_dir)``
   Downloads the requested global mesh. Returns a dictionary with ``"static"``
   (the static NetCDF file) and ``"graph"`` (the graph file).

   * ``resolution``: The horizontal resolution of the global mesh to download
   * ``mesh_data_dir``: Where the agent will write the mesh files

``generate_mesh(mesh_config, mesh_data_dir)``
   Generates the requested mesh in a subdirectory of ``mesh_data_dir`` named
   after the mesh. Returns ``"static"`` and ``"graph"`` paths to the mesh files,
   named ``<name>.static.nc`` and ``<name>.graph.info``. For a global mesh these
   are symlinks to the downloaded files, and for a regional mesh they are the
   files the programs wrote.

   * ``mesh_config``: Dictionary with optional ``resolution`` (default
     ``"120km"``), mesh ``name`` (default ``"mesh_<resolution>"``),
     ``init_ranks`` and ``forecast_ranks``, ``regional``, and ``plot``. The
     rank entries
     are positive integers giving MPI rank counts to partition the mesh graph
     for. A count of 1 needs no partition. A name can be at most 200
     bytes long, which is 200 characters unless it has non-ASCII characters,
     must not contain control characters such as a newline, and must not be
     that of a downloaded global mesh file, such as ``x1.2562.graph.info``. Any other key
     raises ``ValueError``, so a misspelled key is not silently ignored.
     ``plot`` is ``True`` or ``False``, by default ``False``.
   * ``mesh_data_dir``: Output directory shared by the agent and its tasks.

   Without ``regional``, the request is for a global mesh, which is downloaded.
   Its ``resolution`` must be one of the supported global resolutions listed
   below.

   With ``regional``, a regional mesh is generated. ``regional`` must be a
   dictionary with exactly one key, ``project_hexes`` or ``create_region``,
   which chooses how. All the numbers in either block must be real numbers, not
   strings, and finite.

   ``project_hexes`` generates a new rectangular mesh. Its ``resolution`` can
   be any positive number of kilometers written as digits followed by ``km``,
   such as ``"2km"`` or ``"7.5km"``. No limit is placed on the size of the
   mesh, so a fine resolution over a large extent can ask for a very large
   mesh. Its value is a dictionary with these numbers:

   * ``center_lat``: Latitude of the center, from -90 to 90 degrees.
   * ``center_lon``: Longitude of the center, from -180 to 360 degrees.
   * ``extent_x_km`` and ``extent_y_km``: Width and height of the region in
     kilometers, each greater than 0.
   * ``rotation_degrees``: Optional. Counter-clockwise rotation of the mesh
     about its center. If it is not 0, ``grid_rotate`` is run.

   For example:

   .. code-block:: python

      {
          "resolution": "15km",
          "name": "conus",
          "regional": {
              "project_hexes": {
                  "center_lat": 39.0,
                  "center_lon": -98.0,
                  "extent_x_km": 5000,
                  "extent_y_km": 3000,
              }
          },
      }

   The mesh holds the cells of the region and also 7 layers of cells around
   it, which limited-area MPAS uses for its boundary, so it extends about 8
   cells beyond ``extent_x_km`` and ``extent_y_km`` on each side.

   ``project_hexes`` draws the rectangle on a Lambert conformal cone touching
   the sphere at ``center_lat``. The cone's apex is at the pole, and cells at or
   past it are misplaced, so a mesh, with its boundary layers, must not reach
   the pole: ``extent_y_km / 2`` plus 8 cells must be less than ``6371.229 /
   tan(|center_lat|)`` km. A request that reaches it raises ``ValueError``
   giving the largest ``extent_y_km`` for its center and cell size. Use
   ``create_region`` for a region around a pole. A center on the equator is
   allowed: the cone then touches 0.0001 degrees north of it, since
   ``project_hexes`` divides by the sine of the latitude it touches at. This
   moves cells by at most about 40 m on a mesh 10000 km across.

   A region that covers only a few cells can make ``project_hexes`` crash or
   write no mesh. The request then raises ``RuntimeError`` naming the log file,
   and nothing is left behind, so give a region several cells across in each
   direction.

   The mesh directory of a ``project_hexes`` mesh also keeps the inputs the
   programs read, ``namelist.projections`` and, if rotated, ``namelist.input``.
   The MPAS-Tools programs are installed first if they are not already
   installed. Each run writes its own logs, ``project_hexes.<id>.*`` and
   ``grid_rotate.<id>.*`` in ``work_dir/logs``, and a failure names both of
   its log files.

   ``create_region`` cuts a region out of a global mesh, which is downloaded to
   ``mesh_data_dir`` unless it is already there and is kept for reuse.
   Requests that need the same global mesh in the same ``mesh_data_dir`` at the
   same time wait for one download, so none replaces a file another is reading.
   The
   region keeps the global mesh's cells, so its ``resolution`` must be one of
   the supported global resolutions listed below. Its value is a dictionary
   with exactly one shape, whose value holds that shape's numbers. Sizes are in
   kilometers, latitudes from -90 to 90 degrees, and longitudes from -180 to 360
   degrees:

   * ``circle``: ``center_lat``, ``center_lon``, and ``radius_km``.
   * ``ellipse``: ``center_lat``, ``center_lon``, ``semi_major_axis_km`` and
     ``semi_minor_axis_km``, with the minor axis not longer than the major one,
     and optionally ``orientation_degrees``, the clockwise rotation of the major
     axis from due north, by default 0. Any angle can be given, and it is
     written as the same direction within one turn.

   * ``polygon``: ``vertices``, a list of at least 3 ``[lat, lon]`` pairs in
     order around the boundary, and ``inside_lat`` and ``inside_lon``, a point
     inside the region. MPAS-Limited-Area warns that a concave polygon can give
     wrong results when MPAS interpolates static fields, so keep it convex.
   * ``channel``: ``upper_lat`` and ``lower_lat``, the latitudes of a band
     around the globe, with ``upper_lat`` greater than ``lower_lat``.

   A radius or semi-axis must be greater than 0 and less than half the
   circumference of the Earth, 20015.8 km, since a larger one reaches past the
   opposite side of the globe.

   For example:

   .. code-block:: python

      {
          "resolution": "60km",
          "name": "colorado",
          "regional": {
              "create_region": {
                  "circle": {
                      "center_lat": 39.0,
                      "center_lon": -105.5,
                      "radius_km": 400,
                  }
              }
          },
      }

   The mesh holds the cells of the region and also 8 layers of cells around
   it, which limited-area MPAS uses for its boundary, so it extends about 8
   cells beyond the shape. ``create_region`` fails if it cannot find a region
   smaller than the whole globe, for example for a shape smaller than a cell, a
   polygon whose inside point is outside it or whose edges cross, a channel too
   narrow to hold a cell, or a shape that covers nearly the whole globe. The
   request then raises ``RuntimeError`` naming both log files. The program
   prints its errors to the ``.stdout`` log, not the ``.stderr`` one. The
   mesh directory also keeps ``region.pts``, the points file the program read.
   MPAS-Limited-Area is installed first if it is not already installed. Each
   run writes its own logs, ``create_region.<id>.*`` in ``work_dir/logs``.

   If partitions are requested, the result also has ``"partitions"``, a
   dictionary mapping each rank count to its partition file in the mesh
   subdirectory. Metis is installed first if it is not already installed.

   With ``"plot": True``, an image of the mesh is drawn as ``plot_mesh`` draws
   it with its defaults, once the mesh and its partitions are complete. The
   result then also has ``"plot"``, the path of ``<name>.png`` in the mesh
   subdirectory. A plot that fails does not fail the request, which returns the
   mesh with ``"plot_error"`` saying why instead of ``"plot"``. That includes a
   missing ``compute`` resource, a missing ``mpas`` extra, or a failed plotting
   install. A plot drawn without some map layers has both ``"plot"`` and
   ``"plot_error"``.
   Before it creates anything or downloads anything, the action checks that the
   workflow defines the resources the request needs: ``service`` for a global
   or ``create_region`` mesh, and ``compute`` for a regional mesh or if
   partitions are requested. If
   one is missing it raises ``RuntimeError`` naming it. Installing a tool that
   is missing also needs both ``service`` and ``compute``, and checks for them
   when it runs.

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
``install_mpas_tools()``
   Downloads and builds the ``project_hexes`` and ``grid_rotate`` programs from
   MPAS-Tools in ``work_dir``, unless a finished install is already there. The
   tag named by ``mpas_tools_version`` is cloned on the ``service`` resource and
   built on the ``compute`` resource, which needs a Fortran compiler and netCDF
   as described above. The programs are installed in
   ``work_dir/mpas-tools/<mpas_tools_version>/bin``. Like ``install_metis()``,
   a finished build leaves a ``.installed`` marker, an install counts only if the
   marker and both executable programs exist, and this is checked on every
   call, so a damaged install is rebuilt. A build starts from an empty install
   directory, the sources in ``work_dir/mpas-tools/build`` are removed after a
   successful build, and requests that need the programs while they are being
   installed wait for that install. The logs are ``mpas_tools_download.*`` and
   ``mpas_tools_build.*`` in ``work_dir/logs``. If ``nf-config`` or
   ``nc-config`` is missing on the ``compute`` resource, the build stops with a
   message saying so and leaves an existing install alone. Installing needs both
   the ``service`` and ``compute`` resources, and raises ``RuntimeError`` naming
   a missing one before it starts any download. This runs automatically when a
   ``project_hexes`` request needs it, so calling it is optional.
``install_limited_area()``
   Installs MPAS-Limited-Area, whose ``create_region`` program is a Python
   script, in ``work_dir``, unless a finished install is already there. On the
   ``service`` resource, the tag named by ``limited_area_version`` is cloned
   into ``work_dir/limited-area/<limited_area_version>/src``. A check on the
   ``compute`` resource then imports numpy and netCDF4 with the Python of the
   agent's tasks and runs the program with it, and only then writes the
   ``.installed`` marker. If the ``mpas`` extra is missing, the check fails with
   a message saying to install it. Like the other installs, an install counts
   only if the marker and the program exist, and this is checked on every call,
   so a damaged install is replaced. An install made by an earlier chiltepin,
   which ran ``create_region`` in its own virtual environment in ``venv``, is
   replaced too, so stop agents of that chiltepin sharing ``work_dir`` before
   upgrading. An install starts from an empty
   install directory, and requests that need it while it is being installed
   wait for that install. The logs are ``limited_area_install.*`` and
   ``limited_area_check.*`` in ``work_dir/logs``. Installing needs both the
   ``service`` and ``compute`` resources, and raises ``RuntimeError`` naming a
   missing one before it starts. This runs automatically when a
   ``create_region`` request needs it, so calling it is optional.
``install_plotting()``
   Prepares plotting, unless it is already prepared. The Natural Earth map
   layers the plots draw, coastlines, country borders, and state and province
   lines, are downloaded to ``work_dir/plot/natural_earth`` on the ``service``
   resource, since the ``compute`` resource may have no internet access. A check
   on the ``compute`` resource then confirms that the Python of its tasks has the
   ``mpas`` extra and that every layer has its ``.shp``, ``.shx`` and ``.dbf``
   files, and only then writes the
   ``work_dir/plot/.installed`` marker, so a restarted agent reuses it. The logs
   are ``plot_download.*`` and ``plot_check.*`` in ``work_dir/logs``. If the
   extra is missing, the check fails with a message saying to install it.
   Requests that need plotting while it is being prepared wait for that
   install. Installing needs both the ``service`` and ``compute`` resources, and
   raises ``RuntimeError`` naming a missing one before it starts. This runs
   automatically when a plot is requested, so calling it is optional.
``plot_mesh(mesh_path, title=None, zoom=None, center_lat=None, center_lon=None)``
   Draws a PNG image of a mesh on the ``compute`` resource, and returns its
   absolute path. The image is written next to the mesh as ``<name>.png``, for
   example ``conus.png`` for ``conus.static.nc``, and replaces an earlier one. It
   is written under a temporary name and renamed when complete, so a failed plot
   keeps an earlier image. Plotting is prepared first if it is not already.
   Meshes whose names differ only by ``.static`` or ``.nc``, such as ``a.nc``
   and ``a.static.nc`` in one directory, are drawn to the same ``a.png``.

   * ``mesh_path``: Path to a mesh's static or grid file. A relative path or one
     starting with ``~`` is made absolute on the agent.
   * ``title``: Title of the image, by default ``"MPAS mesh: <name>"``.
   * ``zoom``, ``center_lat`` and ``center_lon``: The view of a global mesh, as
     described below. Giving any of them for a regional mesh raises
     ``ValueError``, without drawing.

   A mesh is regional if any of its cells is in a boundary layer, as every
   ``project_hexes`` and ``create_region`` mesh is, or if its cells cover less
   than 99.9% of the sphere, and global otherwise. A region without boundary
   layers is drawn with every cell as interior.

   * A regional mesh is drawn on a flat map shaped like the region, with its
     cells colored by boundary layer, from white for the interior to purple for
     layer 7, on a fixed scale with a color bar. Coastlines and country borders
     are drawn over it, and state and province lines too on a map less than 60
     degrees across over North America, or 30 degrees elsewhere. A region
     across the 180th meridian is drawn in one piece, and one around a pole as
     a band up to the pole. The whole region is drawn at once, which takes
     about 2.5 GB of memory on the ``compute`` resource for each million
     cells, so the largest regional meshes may not fit.
   * A global mesh is drawn on a sphere with coastlines, seen from above
     ``center_lat`` and ``center_lon``, by default 0N 0E. ``zoom`` is greater
     than 0. The default makes the globe fill the image, and a larger zoom shows
     less of the globe, with larger cells. Above a zoom of 2, the coastlines
     come from the detailed layer.

   Cell edges are drawn only when cells come out at least 10 pixels across in
   the image, since smaller ones blur together. At the default zoom that means
   global meshes of 120km and coarser. A finer global mesh is drawn as a plain
   sphere unless zoomed in, and a 60km mesh shows its cells from a zoom of
   about 2.4.

   If a map layer is missing, the image is drawn without it, and a warning
   naming the log is logged. Each plot writes its own logs, ``plot_mesh.<id>.*``
   in ``work_dir/logs``, and a failure raises ``RuntimeError`` naming both.
   ``FileNotFoundError`` is raised for a mesh file that does not exist.
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
   needs, so installation can overlap with other work. That includes preparing
   plotting for a request with ``"plot": True``, and a failure to prepare it is
   raised here, though ``generate_mesh`` would only report it in
   ``"plot_error"``. It checks the
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

Supported global resolutions, for meshes without ``regional`` and for
``create_region`` meshes, are ``480km``, ``384km``, ``240km``, ``120km``, ``60km``,
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
