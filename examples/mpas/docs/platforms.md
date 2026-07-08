# Platform-Specific Notes

This document provides detailed information for running the MPAS multi-agent example on supported HPC platforms.

## Supported Platforms

- [Docker Container](#docker-container)
- [NOAA RDHPCS Hera](#hera)
- [MSU-HPC Hercules](#hercules)
- [NOAA RDHPCS Jet](#jet)

---

## Docker Container

### Overview

- **Environment**: Docker container (development/testing)
- **Provider**: localhost (no Slurm)
- **Use Case**: Testing, development, small-scale experiments

### Configuration

Docker configuration uses localhost provider for all executors:
```yaml
platform: docker

paths:
  experiment_dir: /home/admin/mpas_experiments/test_run

# Reduce MPI ranks for container resources
model:
  init_ranks: 2
  forecast_ranks: 4
```

### Best Practices

1. **Resource Limits**: Container has limited resources. Use small domain:
   ```yaml
   model:
     resolution: 120km
     forecast_length_hours: 6
   ```

2. **No Module System**: Container has dependencies pre-installed, no module loading needed

3. **Testing Focus**: Docker is ideal for:
   - Testing workflow structure
   - Developing agent logic
   - Validating configuration
   - Small-scale integration tests

4. **Not for Production**: For production forecasts, use HPC platforms (Hera, Hercules, Jet)

### Known Limitations

- Limited CPU/memory resources
- No MPI across multiple nodes
- Slower I/O compared to HPC scratch filesystems
- Single-node execution only

---

## Hera

### Overview

- **Location**: NOAA RDHPCS, Boulder, CO
- **Login**: `hera-login-[1-4].boulder.rdhpcs.noaa.gov`
- **Compute Nodes**: 40 cores per node (2x Intel Xeon Gold 6148)
- **Partitions**: `hera` (compute), `service` (builds, transfers)

### Storage

- **Home**: `/home/{user}` (50 GB quota)
- **Scratch**: `/scratch1/{user}` and `/scratch2/{user}` (5 TB quota)
- **Recommended for experiments**: `/scratch1/{user}`

### Module Environment

Default configuration uses:
```bash
module purge
module load intel/2022.1.2
module load impi/2022.1.2
module load netcdf/4.7.4
module load hdf5/1.10.6
module load pnetcdf/1.12.1
module load pio/2.5.7
```

Verify modules are available:
```bash
module avail intel
module avail netcdf
```

### Configuration

Example `config/user_config.yaml` for Hera:
```yaml
platform: hera

paths:
  experiment_dir: /scratch1/${USER}/mpas_experiments/test_run
```

### Job Submission

Check queue status:
```bash
squeue -u $USER
```

Check partition info:
```bash
sinfo -p hera
sinfo -p service
```

### Known Issues

1. **Scratch space**: Regularly clean up old experiments. Files inactive > 30 days may be purged.

2. **Module conflicts**: If you get module conflicts, start with `module purge` in executor environment.

---

## Hercules

### Overview

- **Location**: MSU-HPC, Mississippi State University
- **Login**: `hercules-login-[1-2].hpc.msstate.edu`
- **Compute Nodes**: 64 cores per node (2x AMD EPYC 7713)
- **Partitions**: `hercules` (compute and builds)

### Storage

- **Home**: `/home/{user}` (50 GB quota)
- **Work**: `/work/noaa/{user}` (quota varies)
- **Recommended for experiments**: `/work/noaa/{user}`

### Module Environment

Default configuration uses:
```bash
module purge
module load intel/2022.1.2
module load impi/2022.1.2
module load netcdf/4.9.2
module load hdf5/1.14.0
module load pnetcdf/1.12.3
```

Check available modules:
```bash
module spider intel
module spider netcdf
```

### Configuration

Example `config/user_config.yaml` for Hercules:
```yaml
platform: hercules

paths:
  experiment_dir: /work/noaa/${USER}/mpas_experiments/test_run

model:
  forecast_ranks: 64  # Utilize full node
```

### Job Submission

Check queue:
```bash
squeue -u $USER
```

Partition information:
```bash
sinfo -p hercules
```

### Known Issues

1. **AMD vs Intel**: Hercules uses AMD EPYC processors. Intel-compiled code works but may not be optimal. Consider AMD-optimized compilers if available.

2. **64 cores per node**: Adjust MPI rank counts to multiples of 64 for full node utilization.

---

## Jet

### Overview

- **Location**: NOAA RDHPCS, Boulder, CO
- **Login**: `jet-login-[1-8].boulder.rdhpcs.noaa.gov`
- **Compute Nodes**: 24 cores per node (2x Intel Xeon E5-2695v4)
- **Partitions**: `xjet` (compute), `sjet` (service/builds)

### Storage

- **Home**: `/home/{user}` (50 GB quota)
- **LFS4**: `/lfs4/{project}/{user}` (quota varies by project)
- **Alternative**: `/mnt/lfs4/{project}/{user}`
- **Recommended for experiments**: `/lfs4/HFIP/{user}` (if HFIP project member)

### Module Environment

Default configuration uses:
```bash
module purge
module load intel/2022.1.2
module load impi/2022.1.2
module load netcdf/4.7.4
module load hdf5/1.10.6
module load pnetcdf/1.12.1
```

Check modules:
```bash
module avail intel
module avail netcdf
```

### Configuration

Example `config/user_config.yaml` for Jet:
```yaml
platform: jet

paths:
  experiment_dir: /lfs4/HFIP/${USER}/mpas_experiments/test_run

model:
  forecast_ranks: 24  # One full node
```

### Job Submission

Check queue:
```bash
squeue -u $USER
```

Partition details:
```bash
sinfo -p xjet
sinfo -p sjet
```

### Known Issues

1. **LFS4 paths**: Use `/lfs4` not `/mnt/lfs4` in batch jobs. Both work from login nodes, but `/lfs4` is preferred.

2. **24 cores per node**: Smaller nodes than Hera/Hercules. Adjust resource requests accordingly.

3. **Project allocations**: Ensure you have appropriate project allocation for LFS4 access.

---

## General Platform Tips

### 1. Test on Small Domain

Start with quick test on coarse resolution:
```yaml
model:
  resolution: 120km
  forecast_length_hours: 6
```

### 2. Monitor Resource Usage

After test run, check resource utilization:
```bash
sacct -j <jobid> --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS
```

Adjust executor configuration based on actual usage.

### 3. Walltime Selection

- Builds: 30-60 minutes usually sufficient
- Mesh generation: 15-30 minutes
- 120km forecast (24h): 30-60 minutes
- 60km forecast (24h): 1-2 hours
- 30km forecast (24h): 2-4 hours

Add buffer (1.5-2x) to walltime estimates.

### 4. Node vs Core Allocation

For MPI jobs, consider:
- **Full node allocation**: More efficient, less overhead
- **Partial node**: Useful for small tests, but may have contention

Example for full nodes:
```yaml
model:
  forecast_ranks: 40  # Hera (40 cores/node)
  # or
  forecast_ranks: 64  # Hercules (64 cores/node)
  # or
  forecast_ranks: 24  # Jet (24 cores/node)
```

### 5. File I/O Optimization

MPAS is I/O intensive:
- Use scratch filesystems (not home directories)
- Consider I/O aggregation settings in MPAS namelist
- Clean up old runs regularly

### 6. Environment Troubleshooting

If jobs fail with library errors:
```bash
# Check loaded modules in job
module list

# Verify library paths
ldd /path/to/executable

# Test executable on login node
./atmosphere_model --help
```

### 7. Getting Help

Platform-specific support:
- **Hera**: rdhpcs.help@noaa.gov
- **Hercules**: MSU HPC support (check documentation)
- **Jet**: rdhpcs.help@noaa.gov

Include in support requests:
- Job ID
- Error messages from logs
- Configuration files used

---

## Platform Comparison

| Feature | Docker | Hera | Hercules | Jet |
|---------|--------|------|----------|-----|
| Cores/Node | 4 | 40 | 64 | 24 |
| Processor | Container | Intel Xeon Gold | AMD EPYC | Intel Xeon E5 |
| Compute Partition | localhost | `hera` | `hercules` | `xjet` |
| Build Partition | localhost | `service` | `hercules` | `sjet` |
| Scratch Space | `/home/admin` | `/scratch1`, `/scratch2` | `/work/noaa` | `/lfs4` |
| Typical Home Quota | Container limit | 50 GB | 50 GB | 50 GB |
| Use Case | Testing/Dev | Production | Production | Production |

---

## Adding New Platforms

To add support for a new platform:

1. Create `config/platforms/{platform_name}.yaml`
2. Define workflow and agent_workflow executors
3. Specify correct modules and environment
4. Set default paths
5. Test with small domain first
6. Document platform-specific notes here

Example template:
```yaml
# config/platforms/my_platform.yaml
workflow:
  build-executor:
    provider: "slurm"
    partition: "build_partition"
    # ... other settings

  compute-executor:
    provider: "slurm"
    partition: "compute_partition"
    # ... other settings
    environment:
      - "module load ..."

agent_workflow:
  # Similar structure

paths:
  experiment_dir: /path/on/platform/${USER}/mpas_experiments
```
