Testing
=======

The Chiltepin test suite uses pytest and requires an editable installation of the 
package (achieved using the ``pip install -e ".[test]"`` installation step).

.. note::
   The full test suite requires HPC resources (Slurm or PBSPro) and is designed to run on Linux
   HPC systems or within the Docker container. macOS and Windows users should use the Docker
   container (see :doc:`container`) to run the complete test suite.

Prerequisites
-------------

Before running tests, ensure you have:

1. Installed Chiltepin with test dependencies: ``pip install -e ".[test]"``
2. Access to an HPC system with Slurm or PBSPro, or use the Docker container
3. Authenticated with Globus (for Globus Compute and Globus Transfer tests)

Globus Authentication
---------------------

The Globus tests require authentication before running. Use the following 
command to authenticate:

.. code-block:: console

   $ chiltepin login

This will open a web browser, or provide a URL, for you to complete the authentication flow.
Compute, Transfer, and Academy scopes are requested together in one login flow.
Sign in with an identity that has access to the integration-test Globus data collections.
Login does not probe collection permissions.

Platform-Specific Testing
-------------------------

**Test Suite Requirements**
  The full test suite requires Slurm or PBSPro schedulers and Globus Compute endpoint configurations.
  You can run the full test suite by:

  1. Using an existing test config on the corresponding HPC system (Hera, Hercules, Ursa)
  2. Creating a new test config for your Slurm/PBSPro system using existing configs as examples
  3. Using the Docker container which provides a Slurm environment (see :doc:`container`)

  **macOS and Windows users**: Use the Docker container to run the complete test suite, as it
  provides the required Slurm scheduler.

**Endpoint Management Tests** (Linux only)
  Tests in ``test_endpoint.py`` (except platform-check tests) and ``test_globus_compute_*.py``
  require the ``globus-compute-endpoint`` package, which is only available on Linux.
  These tests will automatically skip on macOS and Windows with a clear reason message.

**Platform Check Tests**
  Tests that verify ``NotImplementedError`` is raised on non-Linux platforms. These run on
  Linux and macOS to verify proper error handling.

**What This Means:**

* **Linux developers**: Can run full test suite with access to HPC/Slurm or PBSPro
* **macOS developers**: Use Docker container for complete testing (includes endpoint management and Slurm)
* **Windows developers**: Use Docker container (native Windows not supported)
* **CI/CD**: Runs on Linux to ensure all functionality is tested

Running Tests
-------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~

To run the full test suite:

.. code-block:: console

   $ pytest --config=tests/configs/<platform>.yaml

Where ``<platform>`` is one of:

* ``docker`` - For the Docker container environment
* ``hera`` - For NOAA Hera HPC system
* ``hercules`` - For NOAA Hercules HPC system
* ``ursa`` - For NOAA Ursa HPC system

You can also provide a custom configuration file if you have specific settings or want
to test against a different environment.

Verbose Output
~~~~~~~~~~~~~~

For more detailed information during testing:

.. code-block:: console

   $ pytest -s -vvv --config=tests/configs/<platform>.yaml

Running Specific Tests
~~~~~~~~~~~~~~~~~~~~~~

To run a specific test file:

.. code-block:: console

   $ pytest -vvv --config=tests/configs/docker.yaml tests/test_endpoint.py

To run a specific test function:

.. code-block:: console

   $ pytest -vvv --config=tests/configs/docker.yaml tests/test_endpoint.py::TestEndpointIntegration::test_configure_default_config_dir

Coverage Reports
----------------

To run tests and generate a coverage report:

.. code-block:: console

   $ pytest --cov=src/chiltepin --cov-report=term --config=tests/configs/<platform>.yaml

This will display a coverage report showing the percentage of lines of code that were
executed during tests. The line numbers of uncovered code will also be displayed for
each file.

Test Organization
-----------------

The test suite is organized into several files:

* ``test_configure.py`` - Tests for configuration parsing and executor creation
* ``test_cli.py`` - Tests for command-line interface functionality
* ``test_auth.py`` - Offline tests for shared Chiltepin authentication, CI, and service consumers
* ``test_tasks.py`` - Tests for task decorators
* ``test_init.py`` - Tests for the chiltepin package initialization
* ``test_workflow.py`` - Tests for the ``Workflow`` context manager
* ``test_agents.py`` - Tests for the agent decorators, runtime and manager
* ``test_agents_package.py`` - Import and serialization compatibility for the agents package
* ``test_endpoint.py`` - Tests for Globus Compute endpoint management (Linux only, except platform-check tests)
* ``test_data.py`` - Tests for data handling utilities
* ``test_parsl_hello.py`` - Basic Parsl integration tests
* ``test_parsl_mpi.py`` - MPI-enabled Parsl integration tests
* ``test_globus_compute_hello.py`` - Basic Globus Compute integration tests (Linux only)
* ``test_globus_compute_mpi.py`` - MPI-enabled Globus Compute integration tests (Linux only)
* ``test_mpas_*.py`` - Tests for the MPAS mesh agent, as described below

MPAS Mesh Agent Tests
---------------------

Most of the MPAS mesh agent's tests run without ``--config``, with its programs
and downloads stood in, and need the ``mpas`` and ``llm`` extras, which the
``test`` extra includes. ``test_mpas_mesh.py`` also launches a local agent,
which needs Globus authentication. ``test_mpas_geo_lookup.py`` downloads
Natural Earth's countries and states, about 45 MB, and
``test_mpas_plot_script.py`` the 480km global mesh from UCAR:

* ``test_mpas_mesh.py`` - Request checks, installs and actions, and a local agent
  launched through the agent runtime
* ``test_mpas_plot_script.py`` - The plotting script
* ``test_mpas_geo_lookup.py`` - Looking up places in Natural Earth and Nominatim
* ``test_mpas_region_shapes.py`` - Fitting regions to the shapes the mesh tools make
* ``test_mpas_mesh_prompt.py`` - Reading a mesh request from a prompt with an LLM
* ``test_mpas_mesh_prompt_actions.py`` - The prompt actions

The others make real meshes through a launched agent, and run only with
``--config``. They download from UCAR, GitHub and Natural Earth, and build
Metis and the MPAS-Tools programs, so the platform's ``compute`` resource needs
the compilers and netCDF described in :doc:`built_in_agents/mpas_mesh`, as the
Docker cluster has:

* ``test_mpas_mesh_metis.py`` - Installing Metis and partitioning meshes
* ``test_mpas_mesh_regional.py`` - ``project_hexes`` meshes
* ``test_mpas_mesh_create_region.py`` - ``create_region`` meshes
* ``test_mpas_mesh_plot.py`` - Plots of meshes
* ``test_mpas_mesh_from_prompt.py`` - Meshes from prompts read by a real LLM

``test_mpas_mesh_from_prompt.py`` also runs only when ``CHILTEPIN_TEST_LLM_ENV``
names a file that the ``service`` resource sources before its tasks, holding
what the model needs, such as credentials or a server's address.
``CHILTEPIN_TEST_LLM_MODEL`` names the model, by default Claude Sonnet 5.5 on
Bedrock, and ``CHILTEPIN_TEST_LLM_REASONING`` its reasoning effort, if any. CI
uses a local Ollama model, which needs no credentials, started on the cluster's
frontend with ``.github/scripts/start-ollama.sh``:

.. code-block:: console

   $ .github/scripts/start-ollama.sh ~/ollama qwen3:8b <version> <sha256>
   $ CHILTEPIN_TEST_LLM_ENV=~/ollama/ollama.env \
     CHILTEPIN_TEST_LLM_MODEL=ollama_chat/chiltepin-test \
     CHILTEPIN_TEST_LLM_REASONING=none \
     pytest --config=tests/configs/docker.yaml tests/test_mpas_mesh_from_prompt.py

The Ollama version and its checksum are in ``.github/workflows/test-suite.yaml``.

Docker Container Testing
------------------------

When running tests in the Docker container, you may need to adjust the 
``cores_per_node`` setting in the configuration file to match the number of cores 
allocated to Docker on your system.

See :doc:`container` for more information on using the Docker environment.
