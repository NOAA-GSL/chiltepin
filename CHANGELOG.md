# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-12

### Changed
- Relaxed dependency version constraints for broader compatibility:
  - `globus-compute-sdk>=4.3.0,<4.7.0` (was >=4.5.0,<4.7.0)
  - `globus-compute-endpoint>=4.3.0,<4.7.0` (was >=4.5.0,<4.7.0)
  - `parsl>=2025.12.1` (was >=2026.1.5)
- These versions are tested and compatible with conda-forge distributions

## [0.1.0] - 2026-03-10

### Added
- Initial release of Chiltepin
- Python task decorator (`@python_task`) for workflow tasks
- Bash task decorator (`@bash_task`) for shell command tasks
- Join task decorator (`@join_task`) for task coordination
- Support for MPI tasks
- Configuration management for compute resources
- Support for Parsl executors (HTEX, MPI)
- Support for Globus Compute executors
- Globus endpoint management utilities
- CLI tool for endpoint configuration and management
- Comprehensive documentation at Read the Docs
- Full test suite with 100% coverage of core modules

### Features
- **Task Decorators**: Simple decorators for creating Python and Bash workflow tasks
- **MPI Support**: First-class support for MPI applications with task geometry specification
- **Multi-Executor**: Support for local, HPC (via Slurm/PBS), and Globus Compute execution
- **Resource Configuration**: YAML-based configuration for compute resources
- **Data Transfer**: Integration with Globus for efficient data movement
- **Workflow Context Managers**: Easy workflow initialization with `run_workflow()`

[0.1.0]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.0
