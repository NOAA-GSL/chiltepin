Testing
=======

The Chiltepin test suite uses pytest and requires an editable installation of the 
package (achieved using the ``pip install -e ".[test]"`` installation step).

.. note::
   The full test suite requires HPC resources (Slurm) and is designed to run on Linux
   or within the Docker container. macOS and Windows users should use the Docker
   container (see :doc:`container`) to run the complete test suite.

Prerequisites
-------------

Before running tests, ensure you have:

1. Installed Chiltepin with test dependencies: ``pip install -e ".[test]"``
2. Access to an HPC system with Slurm, or use the Docker container
3. Authenticated with Globus (for Globus Compute and Globus Transfer tests)

Globus Authentication
---------------------

The Globus tests require authentication before running. Use the following 
command to authenticate:

.. code-block:: console

   $ chiltepin login

This will open a web browser, or provide a URL, for you to complete the authentication flow.

Platform-Specific Testing
-------------------------

**Test Suite Requirements**
  The full test suite requires Slurm/HPC resources. On macOS and Windows, use the
  Docker container which provides a Slurm environment. See :doc:`container` for details.

**Endpoint Management Tests** (Linux only)
  Tests in ``test_endpoint.py`` (except platform-check tests) and ``test_globus_compute_*.py``
  require the ``globus-compute-endpoint`` package, which is only available on Linux.
  These tests will automatically skip on macOS and Windows with a clear reason message.

**Platform Check Tests** (All platforms)
  Tests that verify ``NotImplementedError`` is raised on non-Linux platforms run on all systems.

**What This Means:**

* **Linux developers**: Can run full test suite with access to HPC/Slurm
* **macOS and Windows developers**: Use Docker container for complete testing
* **All developers**: Platform check tests run on all platforms
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
* ``test_tasks.py`` - Tests for task decorators
* ``test_endpoint.py`` - Tests for Globus Compute endpoint management (Linux only, except platform-check tests)
* ``test_data.py`` - Tests for data handling utilities
* ``test_parsl_hello.py`` - Basic Parsl integration tests
* ``test_parsl_mpi.py`` - MPI-enabled Parsl integration tests
* ``test_globus_compute_hello.py`` - Basic Globus Compute integration tests (Linux only)
* ``test_globus_compute_mpi.py`` - MPI-enabled Globus Compute integration tests (Linux only)

Docker Container Testing
------------------------

When running tests in the Docker container, you may need to adjust the 
``cores_per_node`` setting in the configuration file to match the number of cores 
allocated to Docker on your system.

See :doc:`container` for more information on using the Docker environment.
