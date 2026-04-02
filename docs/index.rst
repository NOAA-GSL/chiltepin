Chiltepin Documentation
=======================

.. image:: https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml/badge.svg
   :target: https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml
   :alt: ChiltepinTests

**Chiltepin** is a Python library for exploring federated agentic workflow capabilities
using Parsl, Globus Compute, and Academy Agents. It provides tools and demonstrations for
implementing distributed scientific workflows on HPC systems.

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
implementing distributed exascale scientific workflows. The
project focuses on:

* **Workflow management** using Parsl
* **Federated distributed computing** with Globus Compute
* **HPC integration** of multiple on-prem and/or cloud-based systems
* **Container-based testing** with Docker and Slurm

Key Features
------------

* Configuration-based resource management for both HPC platforms and laptops
* Support for both MPI (HPC) and non-MPI (HTC) applications
* Agent-based workflows with Academy Agents integration
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

Legal Notice
------------

This repository is a scientific product and is not official communication of
the National Oceanic and Atmospheric Administration, or the United States
Department of Commerce. All NOAA GitHub project code is provided on an 'as is'
basis and the user assumes responsibility for its use.
