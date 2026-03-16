[![ExascaleSandboxTests](https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml/badge.svg)](https://github.com/NOAA-GSL/chiltepin/actions/workflows/test-suite.yaml)
[![Documentation](https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml/badge.svg)](https://github.com/NOAA-GSL/chiltepin/actions/workflows/docs.yaml)

# Chiltepin

## Overview

This repository is a collection of tools and demonstrations used to explore
and test various technologies for implementing exascale scientific workflows.
This collection of resources is not intended for production use, and is for
research purposes only.

Chiltepin provides Python decorators and utilities for building scientific workflows
that can execute on distributed computing resources using [Parsl](https://parsl-project.org/)
and [Globus](https://www.globus.org/) services.

### Platform Support

Chiltepin is **developed and tested on Linux**:
- ✅ **Linux**: Full support for all features
- 🍎 **macOS**: Supported for task submission and data transfer (endpoint management not available)
- ❌ **Windows**: Native execution not supported; chiltepin and its use of Parsl relies on fork-based
multiprocessing
- 🐳 **Docker**: Full LInux-based test suite and feature support available via container on all platforms

## Documentation

**📚 Full documentation is available at [Read the Docs](https://chiltepin.readthedocs.io/)**

Key documentation sections:
- [Installation Guide](https://chiltepin.readthedocs.io/en/latest/installation.html) - Installing Chiltepin
- [Quick Start](https://chiltepin.readthedocs.io/en/latest/quickstart.html) - Your first Chiltepin workflow
- [Tasks](https://chiltepin.readthedocs.io/en/latest/tasks.html) - Python, Bash, and Join task decorators
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

