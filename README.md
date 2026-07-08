[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Version](https://img.shields.io/pypi/v/chiltepin)](https://pypi.org/project/chiltepin/)
[![TestSuite](https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml/badge.svg)](https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml)
[![Documentation](https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml/badge.svg)](https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml)
[![DOI](https://zenodo.org/badge/712000160.svg)](https://doi.org/10.5281/zenodo.19195670)

# Chiltepin

## Overview

This repository is a collection of tools and demonstrations used to explore
and test various technologies for implementing exascale scientific workflows.
This collection of resources is not intended for production use, and is for
research purposes only.

Chiltepin provides Python decorators and utilities for building scientific workflows
that can execute on distributed computing resources using [Parsl](https://parsl-project.org/),
[Globus](https://www.globus.org/) services, and [Academy Agents](https://docs.academy-agents.org/).
It supports both traditional task-based workflows and stateful agent-based workflows for
long-running, autonomous computations.

### Platform Support

Chiltepin is **developed and tested on Linux**:
- ✅ **Linux**: Full support for all features
- 🍎 **macOS**: Supported for task submission and data transfer (endpoint management not available)
- ❌ **Windows**: Native execution not supported; Chiltepin's use of Parsl relies on fork-based
  multiprocessing
- 🐳 **Docker**: Full Linux-based test suite and feature support available via container on all platforms

## Documentation

**📚 Full documentation is available at [Read the Docs](https://chiltepin.readthedocs.io/)**

Key documentation sections:
- [Installation Guide](https://chiltepin.readthedocs.io/en/latest/installation.html) - Installing Chiltepin
- [Quick Start](https://chiltepin.readthedocs.io/en/latest/quickstart.html) - Your first Chiltepin workflow
- [Tasks](https://chiltepin.readthedocs.io/en/latest/tasks.html) - Python, Bash, and Join task decorators
- [Agents](https://chiltepin.readthedocs.io/en/latest/agents.html) - Building stateful agent-based workflows
- [Configuration](https://chiltepin.readthedocs.io/en/latest/configuration.html) - Configuring compute resources
- [Endpoints](https://chiltepin.readthedocs.io/en/latest/endpoints.html) - Managing Globus Compute endpoints
- [Data Transfer](https://chiltepin.readthedocs.io/en/latest/data.html) - Using Globus for data movement
- [Testing Guide](https://chiltepin.readthedocs.io/en/latest/testing.html) - Running the test suite

## Quick Start

Install Chiltepin using pip:

```bash
pip install chiltepin
```

For detailed installation instructions including conda, Docker, and platform-specific guidance,
see the [Installation Guide](https://chiltepin.readthedocs.io/en/latest/installation.html).

## Examples

Chiltepin includes real-world example applications that demonstrate agent-based workflows for numerical weather prediction (NWP) models:

### MPAS Multi-Agent Forecast

A complete MPAS (Model for Prediction Across Scales) forecast application using a **component-based multi-agent architecture**. Each independent software component (Metis, WPS, MPAS-Limited-Area, MPAS) has its own agent managing its lifecycle.

**Key Features:**
- Multi-agent design showcasing loose coupling and component reusability
- Automated setup with conda environment for uwtools integration
- Complete workflow from mesh generation through forecast
- Platform-specific configurations for Hera, Hercules, and Jet
- Comprehensive documentation and tests

**Get Started:**
```bash
cd examples/mpas
./setup.sh
conda activate mpas-example
cp config/user_config.yaml.template config/user_config.yaml
# Edit config/user_config.yaml with your settings
python run_mpas_forecast.py config/user_config.yaml
```

See [examples/mpas/README.md](examples/mpas/README.md) for complete documentation.

## Contributing

Contributions are welcome! For development installation and running tests, clone the repository
and install in editable mode:

```bash
git clone https://github.com/NOAA-GSL/chiltepin.git
cd chiltepin
python -m venv .chiltepin
source .chiltepin/bin/activate
pip install -e ".[test]"
```

See the [Testing Guide](https://chiltepin.readthedocs.io/en/latest/testing.html) for more information.

## License

See [LICENSE](LICENSE) for details.

## NOAA Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
