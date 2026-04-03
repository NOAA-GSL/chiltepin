Chiltepin Documentation
=======================

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/pypi/v/chiltepin
   :target: https://pypi.org/project/chiltepin/
   :alt: PyPI - Version

.. image:: https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml/badge.svg
   :target: https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml
   :alt: ChiltepinTests

.. image:: https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml/badge.svg
   :target: https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml
   :alt: Documentation

.. image:: https://zenodo.org/badge/712000160.svg
   :target: https://doi.org/10.5281/zenodo.19195670
   :alt: DOI

**Chiltepin** is a Python library for exploring federated agentic workflow capabilities
using `Parsl <https://parsl-project.org/>`_, `Globus Compute <https://globus-compute.readthedocs.io/>`_,
and `Academy Agents <https://docs.academy-agents.org/>`_. It provides tools and demonstrations for
implementing distributed, agentic, exascale scientific workflows on HPC systems.

.. warning::

   This project is for research and exploration purposes only. It is not
   intended for use in operational production environments.

Platform Support
----------------

Chiltepin is **developed and tested on Linux**:

* ✅ **Linux**: Full support for all features
* 🍎 **macOS**: Task submission and data transfer supported (endpoint management not available)
* ❌ **Windows**: Not supported natively due to reliance on POSIX ``fork`` semantics (use Docker or WSL2)
* 🐳 **Docker**: Full support available via container on all platforms

.. note::
   Windows users can use Chiltepin via Docker container or WSL2 with a Linux distribution.
   Chiltepin and its use of Parsl both require fork-based multiprocessing which is not available on
   native Windows.

Overview
--------

This repository is a collection of tools and demonstrations used for
implementing distributed, agentic, exascale scientific workflows. The
project focuses on:

* **Workflow management** using Parsl
* **Federated distributed computing** with Globus Compute
* **Autonomous workflow agents** with Academy Agents integration
* **HPC integration** of multiple on-prem and/or cloud-based systems
* **Container-based testing** with Docker and Slurm

Key Features
------------

* Configuration-based resource management for both HPC platforms and laptops
* Support for both MPI (HPC) and non-MPI (HTC) applications
* Agentic workflows with Academy Agents integration
* Task decorators for seamless integration of Parsl and Globus Compute
* Globus Compute endpoint management utilities
* Dynamic distributed task execution across heterogeneous resources
* Docker container environment for development and testing
* Comprehensive test suite with high coverage for core modules

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tasks
   agents
   data
   configuration
   endpoints
   testing
   container

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Disclaimer
----------

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric
Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as
is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or
Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable
Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark,
manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the
Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not
be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States
Government.
