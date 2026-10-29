# MPAS Configuration Guide

This document provides detailed information about configuring the MPAS multi-agent forecast example.

## Configuration Hierarchy

Configuration is loaded in this order (later overrides earlier):

1. **default_config.yaml** - Base configuration with sensible defaults
2. **platforms/{platform}.yaml** - Platform-specific settings (executors, modules, paths)
3. **user_config.yaml** - Your experiment customizations

## Configuration Structure

### Top-Level Sections

```yaml
platform: hera                # Platform selection
workflow: {...}               # Workflow executor configurations
agent_workflow: {...}         # Agent-internal executor configurations
manager_executors: [...]      # Executors for agent runtime
build_executors: [...]        # Executors for build agents
compute_executors: [...]      # Executors for compute agents
model: {...}                  # Model settings
paths: {...}                  # Directory paths
data_sources: {...}           # Data source URLs
versions: {...}               # Software versions
logging: {...}                # Logging configuration
```

## Detailed Configuration

### Platform Selection

```yaml
platform: hera  # Options: docker, hera, hercules, jet
```

Automatically loads platform-specific configuration from `config/platforms/{platform}.yaml`.

### Workflow Executors

Executors used by the workflow to run agents:

```yaml
workflow:
  manager-executor:
    provider: "localhost"
    cores_per_node: 1
    max_workers_per_node: 1

  build-executor:
    provider: "slurm"
    partition: "service"
    nodes_per_block: 1
    cores_per_node: 4
    max_blocks: 2
    walltime: "01:00:00"
    environment:
      - "module purge"
      - "module load intel/2022.1.2"
      - "module load impi/2022.1.2"

  compute-executor:
    provider: "slurm"
    partition: "compute"
    nodes_per_block: 2
    cores_per_node: 40
    max_blocks: 1
    walltime: "02:00:00"
    environment:
      - "module purge"
      - "module load intel/2022.1.2"
      - "module load impi/2022.1.2"
      - "module load netcdf/4.7.4"
```

### Agent Workflow Configuration

Executors passed to agents for their internal tasks:

```yaml
agent_workflow:
  build-executor:
    provider: "slurm"
    partition: "service"
    nodes_per_block: 1
    cores_per_node: 4
    walltime: "01:00:00"

  compute-executor:
    provider: "slurm"
    partition: "compute"
    nodes_per_block: 2
    cores_per_node: 40
    walltime: "02:00:00"
```

### Model Configuration

MPAS model settings:

```yaml
model:
  # Mesh resolution
  resolution: "120km"  # Options: 120km, 60km, 30km, 15km

  # Regional domain
  region: "conus"      # CONUS domain specification

  # MPI configuration
  init_ranks: 24       # Ranks for initialization
  forecast_ranks: 48   # Ranks for forecast

  # Forecast settings
  forecast_length_hours: 24
  output_interval_hours: 1

  # Physics options
  physics:
    microphysics: "wsm6"
    radiation: "rrtmg"
    pbl: "ysu"
    cumulus: "kain_fritsch"
```

### Paths Configuration

Directory structure for experiment:

```yaml
paths:
  experiment_dir: /scratch1/${USER}/mpas_experiments/run_20260708_120000
  install_dir: ${experiment_dir}/installs
  mesh_dir: ${experiment_dir}/mesh
  grib_data_dir: ${experiment_dir}/grib_data
  ungrib_dir: ${experiment_dir}/ungrib_output
  init_dir: ${experiment_dir}/initialization
  forecast_dir: ${experiment_dir}/forecast
  run_dir: ${experiment_dir}/runinfo
```

**Environment variable expansion**: `${USER}` expands to your username.

**Path resolution**: Relative paths resolved from experiment_dir.

### Data Sources

URLs for input data:

```yaml
data_sources:
  ics_source: "GFS"
  ics_url_template: "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{yyyymmdd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000"
  
  lbcs_source: "GFS"
  lbcs_url_template: "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{yyyymmdd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{hhh}"
  lbcs_interval_hours: 3
```

### Software Versions

Versions to download and build:

```yaml
versions:
  metis: "5.1.0"
  wps: "4.5"
  mpas_limited_area: "master"
  mpas: "v8.2.0"
```

## Platform-Specific Configuration

### Hera

```yaml
# config/platforms/hera.yaml
workflow:
  build-executor:
    partition: "service"  # Service partition for builds
  compute-executor:
    partition: "hera"     # Compute partition
    cores_per_node: 40    # 40 cores per node

paths:
  experiment_dir: /scratch1/${USER}/mpas_experiments/run_$(date +%Y%m%d_%H%M%S)
```

**Modules**:
- Intel compiler and MPI
- NetCDF, HDF5, PNetCDF, PIO

### Hercules

```yaml
# config/platforms/hercules.yaml
workflow:
  compute-executor:
    partition: "hercules"
    cores_per_node: 64    # 64 cores per node (AMD EPYC)

paths:
  experiment_dir: /work/noaa/${USER}/mpas_experiments/run_$(date +%Y%m%d_%H%M%S)
```

### Jet

```yaml
# config/platforms/jet.yaml
workflow:
  build-executor:
    partition: "sjet"     # Service partition
  compute-executor:
    partition: "xjet"     # Compute partition
    cores_per_node: 24    # 24 cores per node

paths:
  experiment_dir: /lfs4/HFIP/${USER}/mpas_experiments/run_$(date +%Y%m%d_%H%M%S)
```

## User Configuration

Create your user configuration:

```bash
cp config/user_config.yaml.template config/user_config.yaml
```

### Minimal User Configuration

Only specify what you want to override:

```yaml
# config/user_config.yaml
platform: hera

paths:
  experiment_dir: /scratch1/john.doe/my_mpas_test
```

### Typical User Configuration

```yaml
# config/user_config.yaml
platform: hera

paths:
  experiment_dir: /scratch1/john.doe/mpas_120km_test

model:
  resolution: 120km
  forecast_length_hours: 24
```

### Advanced User Configuration

Override specific executor settings:

```yaml
# config/user_config.yaml
platform: hera

paths:
  experiment_dir: /scratch1/john.doe/mpas_60km_large

model:
  resolution: 60km
  forecast_length_hours: 48
  forecast_ranks: 96  # More ranks for finer mesh

workflow:
  compute-executor:
    nodes_per_block: 4   # More nodes
    walltime: "04:00:00" # Longer walltime
```

## Configuration Tips

### 1. Start Simple

Begin with minimal user configuration. Add overrides only as needed.

### 2. Check Platform Defaults

Review `config/platforms/{platform}.yaml` to see what's already configured for your platform.

### 3. Test with Small Domain

Start with coarse resolution (120km) and short forecast (6-12 hours) for testing.

### 4. Adjust Resources

Monitor job resource usage and adjust:
- `nodes_per_block`: Number of nodes
- `walltime`: Maximum run time
- `max_blocks`: Maximum concurrent jobs

### 5. Module Environment

Platform configs include appropriate modules. Verify modules are available:

```bash
module avail intel
module avail netcdf
```

## Troubleshooting Configuration

### Path Issues

**Problem**: Paths not resolved correctly

**Solution**: Use absolute paths in user_config.yaml:
```yaml
paths:
  experiment_dir: /full/path/to/experiment
```

### Module Issues

**Problem**: Modules not found or conflicts

**Solution**: Check platform configuration modules match your system:
```bash
module avail  # List available modules
```

Override modules in user_config.yaml if needed:
```yaml
workflow:
  compute-executor:
    environment:
      - "module purge"
      - "module load your-intel-version"
      - "module load your-netcdf-version"
```

### Executor Issues

**Problem**: Jobs not starting or failing

**Solution**: Verify partition names and resource limits:
```bash
sinfo  # List partitions
scontrol show partition <partition_name>  # Partition details
```

### Configuration Validation

Validate configuration before running:

```python
import yaml

with open('config/user_config.yaml') as f:
    config = yaml.safe_load(f)
    print(yaml.dump(config, default_flow_style=False))
```

## Example Configurations

See `config/user_config.yaml.template` for additional examples and all available options.

## Further Reading

- [Parsl Configuration Documentation](https://parsl.readthedocs.io/en/stable/userguide/configuring.html)
- [Platform-Specific Notes](platforms.md)
- [Architecture Overview](architecture.md)
