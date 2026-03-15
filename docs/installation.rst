Installation
============

Chiltepin is developed and tested on **Linux**. Task submission and data transfer use
platform-agnostic libraries and should work on macOS and Windows, though testing primarily
occurs on Linux. Endpoint management (creating, starting, stopping, and deleting endpoints)
requires Linux.

.. tip::
   **macOS/Windows users:** You can install and use Chiltepin for task submission and data
   transfer. For endpoint management or to run the full test suite, use the Docker container
   (see :doc:`container`).

Prerequisites
-------------

* Python 3.10 or higher
* Linux recommended (macOS/Windows supported for task submission and data transfer)
* Linux required for endpoint management

Installing from PyPI
--------------------

The recommended method for users is to install Chiltepin from PyPI:

.. code-block:: console

   $ pip install chiltepin

You can also install in a virtual environment (recommended):

**Using venv:**

.. code-block:: console

   $ python -m venv .chiltepin
   $ source .chiltepin/bin/activate
   $ pip install chiltepin

**Using conda:**

.. code-block:: console

   $ conda create -n "chiltepin" python=3.10
   $ conda activate chiltepin
   $ pip install chiltepin

Development Installation
------------------------

For development or testing, install from a git clone in editable mode:

.. code-block:: console

   $ git clone https://github.com/NOAA-GSL/chiltepin.git
   $ cd chiltepin
   $ python -m venv .chiltepin
   $ source .chiltepin/bin/activate
   $ pip install -e ".[test]"

.. note::

   The ``[test]`` option installs additional dependencies required for running the test suite.

Activating the Environment
---------------------------

Once installed, Chiltepin can be used simply by activating the environment using
the command appropriate for your environment type:

**For venv:**

.. code-block:: console

   $ source .chiltepin/bin/activate

**For conda:**

.. code-block:: console

   $ conda activate chiltepin

Dependencies
------------

Chiltepin has the following core dependencies:

* ``globus-compute-sdk`` (>=4.3.0,<4.7.0)
* ``globus-compute-endpoint`` (>=4.3.0,<4.7.0) - **Linux only**
* ``parsl`` (>=2025.12.1)

These will be automatically installed when you install Chiltepin.

.. note::
   ``globus-compute-endpoint`` is only available on Linux. On macOS and Windows, Chiltepin
   will skip this dependency. Task submission and data transfer work on all platforms;
   only endpoint management requires Linux.
