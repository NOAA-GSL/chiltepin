# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-04-02

### Added
- **Agent System Integration**: Added support for stateful agent-based workflows with Academy Agents
  - `@chiltepin_agent` decorator to wrap Python classes as Academy agents
  - `@agent_action` decorator for exposing methods as agent actions (supports sync and async)
  - `@agent_loop` decorator for background loops (requires async methods)
  - `AgentSystem` helper class for simplified agent management
  - `ChiltepinManager` custom manager supporting workflow configuration at launch time
  - Comprehensive agent documentation and examples
  - Full test suite with 100% coverage of agent module
- **Enhanced Workflow API**: `Workflow` now accepts `None` for default configuration
  - `Workflow()` or `Workflow(None)` creates default local executor automatically
  - More intuitive API for simple use cases
  - Fully backward compatible with existing code

### Changed
- **Validation Improvements**: Added comprehensive decorator validation
  - Detects and rejects mixed usage of Academy and Chiltepin decorators
  - Validates `@agent_loop` methods are async with correct signature
  - Clear error messages guide users to correct decorator usage
- **Documentation Updates**: Enhanced documentation for agent workflows
  - Added agents module to API reference
  - Updated README to mention Academy Agents and agent-based workflows
  - Added documentation for multi-agent deployments on shared filesystems
  - Clarified decorator naming (Chiltepin vs Academy)

### Fixed
- **Collision Prevention**: Auto-generated unique run directories for agents
  - UUID-based naming prevents Parsl directory collisions
  - Important for multi-agent deployments on shared filesystems
- **Runtime Robustness**: Improved action wrapper argument handling
  - Agent actions now accept both positional and keyword arguments
  - Proper forwarding of `*args` and `**kwargs` to underlying methods
- **Test Reliability**: Fixed race condition in loop decorator test
  - Removed racy assertion on initial counter value
  - Test now validates loop functionality without timing assumptions

[0.1.5]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.5

## [0.1.4] - 2026-03-23

### Changed
- **API Simplification**: Removed `run_workflow()`, `run_workflow_from_file()`, and
  `run_workflow_from_dict()` functions in favor of unified `Workflow` class
  - `Workflow` class supports both context manager pattern (`with Workflow(...)`) and
    explicit lifecycle management (`workflow.start()` / `workflow.cleanup()`)
  - Simplified package exports to only include `Workflow` class
  - Updated all documentation and examples to reflect new API
- **Improved Exception Handling**: Enhanced cleanup robustness with proper exception chaining
  - All cleanup operations are attempted even if some fail
  - Cleanup exceptions properly chained using `__cause__` for full error context
  - User exceptions always take precedence over cleanup exceptions
- **Enhanced State Management**: Workflow state preserved when cleanup operations fail
  - DataFlowKernel and logger handler references preserved on cleanup failure for debugging
  - State only reset after successful cleanup operations

### Added
- Comprehensive test coverage for workflow lifecycle and exception handling scenarios
  - Added 5 lifecycle tests covering state preservation and startup failures
  - Added 8 exception handling tests for cleanup robustness
  - Added 5 tests for user exception precedence
  - Maintained 100% code coverage of workflow module

[0.1.4]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.4

## [0.1.3] - 2026-03-17

### Changed
- **Clarified Windows is not supported**: Discovered during conda-forge builds that Parsl's
  fork-based multiprocessing is incompatible with Windows
  - Linux: Full support for all features
  - macOS: Task submission and data transfer (endpoint management requires Linux)
  - Windows: Not supported natively; use Docker or WSL2
- Updated all documentation to accurately reflect platform limitations
- Improved error messages in endpoint management to guide users by platform

### Added
- Explicitly declared previously implicit dependencies:
  - `psutil` (used in endpoint management)
  - `pyyaml` (used in configuration)
  - `globus-sdk` (imported directly)

[0.1.3]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.3

## [0.1.2] - 2026-03-16

### Changed
- Gracefully isolate Linux-only endpoint management features
  - Added platform detection with informative error messages for non-Linux systems
  - Endpoint management (start/stop/configure) now limited to Linux platforms
  - Task submission and data transfer continue to work on all platforms (Linux, macOS, Windows)
  - Enhanced test suite to skip endpoint tests on unsupported platforms
  - Updated documentation to clarify platform support

### Fixed
- Proper handling of `globus-compute-endpoint` import errors on non-Linux platforms
- Improved error messages to guide users on platform-specific limitations

[0.1.2]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.2

## [0.1.1] - 2026-03-12

### Changed
- Relaxed dependency version constraints for broader compatibility:
  - `globus-compute-sdk>=4.3.0,<4.7.0` (was >=4.5.0,<4.7.0)
  - `globus-compute-endpoint>=4.3.0,<4.7.0` (was >=4.5.0,<4.7.0)
  - `parsl>=2025.12.1` (was >=2026.1.5)
- These versions are tested and compatible with conda-forge distributions

[0.1.1]: https://github.com/NOAA-GSL/chiltepin/releases/tag/v0.1.1

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
