# MPAS Multi-Agent Forecast Example

This example demonstrates chiltepin's agent-based workflow capabilities by implementing a complete MPAS (Model for Prediction Across Scales) forecast system using a **component-based multi-agent architecture**.

## Architecture Overview

This example uses **multiple specialized agents** rather than a monolithic agent. Each independent component has its own agent that manages its complete lifecycle (download → build → execute):

- **MeshAgent** - Downloads and builds Metis and MPAS-Limited-Area tools, generates regional mesh from global mesh data, partitions mesh for MPI parallel execution
- **WPSAgent** - Downloads and builds WPS (WRF Preprocessing System), runs ungrib to prepare initial and boundary conditions
- **MPASAgent** - Downloads and builds MPAS model, initializes conditions, runs forecast

### Why Multi-Agent Architecture?

- **Loose Coupling**: Components don't depend on each other's implementation details
- **Independent Testing**: Each agent can be tested and developed in isolation
- **Component Reusability**: Agents can be reused in other workflows (e.g., WPSAgent for WRF)
- **Clear Separation of Concerns**: Each agent owns its component's complete lifecycle
- **Better Scalability**: Different agents can run on different executors/platforms

### Workflow Coordination

The `MPASForecastWorkflow` orchestrates all agents, managing dependencies and executing phases:

1. **Build Phase**: Parallel builds of all components (Metis, MPAS-Limited-Area, WPS, MPAS)
2. **Mesh Phase**: Generate regional mesh, partition for different MPI configurations
3. **Preprocess Phase**: Fetch and process input data (ICS and LBCS)
4. **Initialization Phase**: Initialize MPAS with processed data
5. **Forecast Phase**: Run MPAS forecast

## Prerequisites

- Linux system (tested on NOAA RDHPCS: Hera, Hercules, Jet, and Docker container)
- Access to Slurm-based HPC system (or localhost for Docker)
- Internet access for downloading model source code and data
- Globus account for Academy Agent exchange (see main chiltepin documentation)

## Quick Start

### 1. Setup Environment

The setup script will install Miniforge (if needed), create a conda environment with uwtools, and pip install chiltepin:

```bash
cd examples/mpas
./setup.sh
```

This creates a conda environment named `mpas-example` with:
- **Miniforge**: conda-forge channel by default + mamba for faster operations
- uwtools (conda-only, for configuration management)
- chiltepin (pip installed from PyPI)
- All necessary dependencies

Activate the environment:

```bash
conda activate mpas-example
```

### 2. Configure Your Experiment

Copy the user configuration template:

```bash
cp config/user_config.yaml.template config/user_config.yaml
```

Edit `config/user_config.yaml` to customize:
- Platform selection (docker, hera, hercules, jet)
- Experiment directory path
- Model resolution
- Forecast length and settings
- Data source locations

Platform-specific executor configurations are in `config/platforms/`.

### 3. Run MPAS Forecast

Execute the forecast workflow:

```bash
python run_mpas_forecast.py config/user_config.yaml
```

The workflow will:
1. Launch all component agents
2. Build all software components in parallel
3. Generate and partition the MPAS mesh
4. Download and process input data
5. Initialize MPAS
6. Run the forecast

Logs and output files will be written to your configured experiment directory.

## Directory Structure

```
examples/mpas/
├── README.md                          # This file
├── setup.sh                           # Environment setup script
├── environment.yml                    # Conda environment specification
├── run_mpas_forecast.py              # Main entry point
├── agents/                            # Agent implementations
│   ├── __init__.py
│   ├── mesh_agent.py                 # MeshAgent (Metis + MPAS-Limited-Area)
│   ├── wps_agent.py                  # WPSAgent
│   └── mpas_agent.py                 # MPASAgent
├── workflows/                         # Workflow orchestration
│   ├── __init__.py
│   └── mpas_forecast.py              # MPASForecastWorkflow
├── config/                            # Configuration files
│   ├── default_config.yaml           # Default settings
│   ├── user_config.yaml.template     # User customization template
│   └── platforms/                     # Platform-specific configs
│       ├── hera.yaml
│       ├── hercules.yaml
│       └── jet.yaml
├── tests/                             # Test suite
│   ├── conftest.py
│   ├── test_mesh_agent.py
│   ├── test_wps_agent.py
│   ├── test_mpas_agent.py
│   └── test_mpas_workflow.py
└── docs/                              # Additional documentation
    ├── architecture.md                # Detailed architecture design
    ├── configuration.md               # Configuration guide
    └── platforms.md                   # Platform-specific notes
```

## Configuration

Configuration uses a three-level hierarchy:

1. **default_config.yaml** - Base configuration with sensible defaults
2. **platforms/{platform}.yaml** - Platform-specific executor settings (Slurm partitions, modules, etc.)
3. **user_config.yaml** - User customizations (paths, domain, resolution, etc.)

See [docs/configuration.md](docs/configuration.md) for detailed configuration guide.

## Development and Testing

### Running Tests

This example has its own test suite independent of the main chiltepin tests:

```bash
conda activate mpas-example
pytest tests/ -v
```

Tests include:
- Unit tests for each agent
- Integration test for workflow orchestration
- Setup script validation

### Agent Development

Each agent is a standard chiltepin agent using `@chiltepin_agent` decorator:

```python
from chiltepin.agents import chiltepin_agent, agent_action
from chiltepin.tasks import bash_task

@chiltepin_agent(agent_workflow_include=["build-executor"])
class MeshAgent:
    def __init__(self, install_dir: str, metis_tag: str = "5.2.1",
                 limited_area_version: str = "master"):
        self.install_dir = install_dir
        self.metis_tag = metis_tag
        self.limited_area_version = limited_area_version
        self.gpmetis_path = None
    
    @agent_action
    async def install(self):
        """Download and build Metis and MPAS-Limited-Area."""
        # Implementation here
        pass
    
    @agent_action
    async def download_global_mesh(self, resolution: str):
        """Download global mesh files."""
        # Implementation here
        pass
    
    @agent_action
    async def create_regional_mesh(self, global_static, global_graph,
                                   region_spec, output_dir):
        """Create regional mesh from global mesh."""
        # Implementation here
        pass
    
    @agent_action
    async def partition_mesh(self, mesh_path: str, num_ranks: int):
        """Partition mesh for MPI execution."""
        # Implementation here
        pass
```

See [docs/architecture.md](docs/architecture.md) for detailed agent design patterns.

## Platform-Specific Notes

### Hera
- Modules: Uses Intel compiler and Intel MPI
- Partitions: `hera` partition for compute, `service` for builds
- Storage: `/scratch1/{user}` for experiment directories

### Hercules
- Modules: Uses Intel compiler and Intel MPI
- Partitions: `hercules` partition for compute
- Storage: `/work/{user}` for experiment directories

### Jet
- Modules: Uses Intel compiler and Intel MPI
- Partitions: `xjet` partition for compute, `sjet` for builds
- Storage: `/lfs4/{user}` or `/mnt/lfs4/{user}` for experiment directories

See [docs/platforms.md](docs/platforms.md) for detailed platform instructions.

## Troubleshooting

### Environment Issues

If conda environment creation fails:
```bash
# Remove existing environment and retry
conda env remove -n mpas-example
./setup.sh
```

### Build Failures

Check module availability and compiler versions:
```bash
module list
which mpicc
mpicc --version
```

Build logs are written to `{experiment_dir}/logs/build_*.log`

### Workflow Failures

Check Parsl logs and task output:
```bash
# Parsl logs
ls {experiment_dir}/runinfo/

# Task output
ls {experiment_dir}/logs/
```

## Additional Resources

- Main chiltepin documentation: https://chiltepin.readthedocs.io/
- MPAS documentation: https://mpas-dev.github.io/
- uwtools documentation: (conda package documentation)
- Old MPAS app reference: https://github.com/NOAA-GSL/ExascaleWorkflowSandbox/tree/747241f/apps/mpas

## Contributing

To contribute to this example:
1. Follow the main chiltepin development guidelines
2. Ensure all tests pass: `pytest tests/`
3. Follow the multi-agent architecture pattern
4. Update documentation for any new features

## License

This example is part of chiltepin and uses the same Apache 2.0 license.
See [LICENSE](../../LICENSE) in the main repository.
