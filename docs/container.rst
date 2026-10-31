Docker Container
================

Chiltepin provides a Docker Compose environment that runs a Slurm cluster with
`spack-stack <https://github.com/NOAA-GSL/DockerSpackStackSlurmCluster>`_
installed.

This is particularly useful for:

* Full functionality on macOS (endpoint management requires Linux)
* Running Chiltepin on Windows (native Windows is not supported)
* Creating a consistent testing environment
* Demonstrating Chiltepin capabilities without access to HPC systems

Prerequisites
-------------

You need to have Docker and Docker Compose installed on your system:

* `Docker <https://docs.docker.com/get-docker/>`_
* `Docker Compose <https://docs.docker.com/compose/install/>`_

Container Architecture
----------------------

The Docker environment consists of:

* **Frontend node**: Where you interact with the system and submit jobs
* **Compute nodes**: Multiple Slurm compute nodes for running jobs
* **Master node**: The Slurm controller that manages job scheduling and resource allocation
* **Shared volume**: A shared directory for data and code accessible by all nodes

This simulates a real HPC cluster environment with job scheduling and multi-node execution.

Starting a Cluster
------------------

From the repository root:

.. code-block:: console

   $ cd docker
   $ docker compose -f docker-compose.yml up -d --wait

This downloads prebuilt images and starts a frontend, a Slurm controller, and
five compute nodes. The first launch may take several minutes while spack-stack
is copied into the cluster's shared volume.

No host ports are published. Each cluster has its own network and named volumes.

.. important::

   The Chiltepin repository on the host (``..`` relative to ``docker/``) is
   bind-mounted at ``/home/admin/chiltepin`` in the cluster. This means changes
   to Chiltepin in the cluster container are visible on the host and vice versa.
   Changes made in this mounted directory remain on the host after the cluster
   is removed.

Accessing the Container
-----------------------

Once the container has started, you can access it with a bash shell:

.. code-block:: console

   $ docker exec -it chiltepin-frontend bash -l

This will log you into the frontend node of the Slurm cluster.

Installing Chiltepin in a Cluster
---------------------------------

Inside the cluster container, the repository is mounted at ``~/chiltepin``. To install
Chiltepin in editable mode:

.. code-block:: console

   (container) $ python3 -m venv ~/.chiltepin
   (container) $ source ~/.chiltepin/bin/activate
   (container) $ cd ~/chiltepin
   (container) $ pip install -e '.[test]'

.. note::

   The ``[test]`` option ensures that dependencies required for running the tests
   are installed.

.. important::

   Keep the virtual environment outside ``~/chiltepin`` so it does not mix with
   environments on the host. The environment can be recreated using the commands
   above if the cluster's volumes are removed.

Running Tests in a Cluster
--------------------------

After installation, you can run the test suite using the Docker-specific configuration:

.. code-block:: console

   (container) $ pytest --config=tests/configs/docker.yaml

For more verbose output:

.. code-block:: console

   (container) $ pytest -s -vvv --config=tests/configs/docker.yaml

.. note::
   
   Depending on how many cores your machine has and how many you've allocated to Docker, 
   you may need to modify the ``cores_per_node`` setting in ``tests/configs/docker.yaml`` 
   to match your machine's specifications for all tests to pass.

Stopping and Removing a Cluster
-------------------------------

To stop the cluster environment:

.. code-block:: console

   $ cd docker
   $ docker compose -f docker-compose.yml down

To stop and remove all named-volume data (/opt):

.. code-block:: console

   $ cd docker
   $ docker compose -f docker-compose.yml down -v

.. note::
   Changes made in /home/admin/chiltepin in the cluster container will persist on
   the host even after the cluster is shut down and its volumes are removed.

Changing the Number of Compute Nodes in the Cluster
---------------------------------------------------

A tool is provided for generating compose files for clusters of different sizes.

From ``docker/``:

.. code-block:: console

   $ python3 generate-compose.py --nodes 3
   $ docker compose -f docker-compose.yml up -d --remove-orphans --wait

The supported range is **1-10 compute nodes**, matching the Slurm configuration
in the images. The ``--nodes`` flag is optional and defaults to 5.

The generator overwrites ``docker-compose.yml`` by default.  Use the ``--output``
option to write to a different file instead:

.. code-block:: console

   $ python3 generate-compose.py --nodes 3 --output docker-compose-3.yml
   $ docker compose -f docker-compose-3.yml up -d --remove-orphans --wait

.. important::

   Before reducing the node count, finish or cancel jobs on the nodes being
   removed. ``--remove-orphans`` removes their containers.

.. note::

   The test configuration requests three nodes for MPI blocks plus nodes for
   other work. Smaller clusters may require reducing ``nodes_per_block`` and
   concurrency in ``tests/configs/docker.yaml``. Unused nodes in the images'
   ten-node Slurm configuration may appear as down.

Running Concurrent Clusters
-------------------------------

.. warning::

   **Do not start multiple clusters from the same repository checkout.**
   Each cluster mounts its host checkout at ``/home/admin/chiltepin``. Sharing
   one checkout can cause conflicting source changes, build artifacts, and test
   output. Use a **separate clone of the repository for each cluster**.

Multiple clusters may coexist by starting them with a unique name that sets
both the container names and the Compose project name.  The default name is
``chiltepin``.  To start a cluster with a custom name both ``CLUSTER_NAME`` and
``-p`` must be set to that name at startup.

To start a new cluster named "mycluster":

.. code-block:: console

   $ CLUSTER_NAME=mycluster docker compose -p mycluster -f docker-compose.yml up -d --wait

The cluster can then be accessed with:

.. code-block:: console

   $ docker exec -it mycluster-frontend bash -l

The cluster can be stopped with:

.. code-block:: console

   $ docker compose -p mycluster -f docker-compose.yml down

The following example illustrates how one might use two concurrent clusters

First, create two separate clones on the host:

.. code-block:: console

   $ git clone https://github.com/NOAA-GSL/chiltepin.git chiltepin-a
   $ git clone https://github.com/NOAA-GSL/chiltepin.git chiltepin-b

In one terminal, start and access the first cluster:

.. code-block:: console

   $ cd chiltepin-a/docker
   $ CLUSTER_NAME=cluster-a docker compose -p cluster-a -f docker-compose.yml up -d --wait
   $ docker exec -it cluster-a-frontend bash -l

In a second terminal, starting from the directory containing both clones:

.. code-block:: console

   $ cd chiltepin-b/docker
   $ CLUSTER_NAME=cluster-b docker compose -p cluster-b -f docker-compose.yml up -d --wait
   $ docker exec -it cluster-b-frontend bash -l

Follow the installation and testing instructions above inside each frontend.
Each cluster now has its own mounted checkout, network, and named volumes.
Allocate sufficient Docker CPU, memory, and disk space for both clusters.

After exiting the container shells, stop each cluster from its own clone's
``docker/`` directory, using the same project name:

.. code-block:: console

   (chiltepin-a/docker) $ docker compose -p cluster-a -f docker-compose.yml down
   (chiltepin-b/docker) $ docker compose -p cluster-b -f docker-compose.yml down

Add ``-v`` to remove that cluster's named volumes. Neither command removes the
host checkout or changes made there, and stopping one cluster does not stop
the other.

Troubleshooting
---------------

**Container won't start:**
   Check that Docker has sufficient resources allocated (CPU, memory, disk space).

**Tests fail with resource errors:**
   Adjust core counts, node counts, and concurrency in the test configuration
   to match the available cluster resources.

**Repository is missing or not writable:**
   Ensure Docker has permission to access the repository directory on your host system. If not,
   copy ``/home/admin/chiltepin`` to another directory such as ``~/work`` in the container or
   re-clone the repository directly inside the container and install Chiltepin there.  Note that
   any work outside of ``/home/admin/chiltepin`` in the container will not survive a
   ``down -v``, so be sure to save work before destroying the cluster.
