# MPAS Multi-Agent Architecture

This document explains the multi-agent architecture design for the MPAS forecast example.

## Overview

The MPAS example uses a **component-based multi-agent architecture** where each independent software component has its own agent that manages its complete lifecycle. This stands in contrast to a monolithic agent that would handle all operations.

## Why Multi-Agent Architecture?

### 1. Loose Coupling

Each component agent is independent and doesn't depend on implementation details of other components. Agents communicate only through the orchestration workflow, which passes data between them.

**Example**: MeshAgent doesn't know or care how WPSAgent works. It handles the full mesh lifecycle internally (generation and partitioning) and passes the result to the workflow.

### 2. Independent Testing

Each agent can be tested in isolation without requiring other components to be available.

**Example**: You can test MeshAgent's partition functionality with sample mesh files, without needing to build MPAS or WPS.

### 3. Component Reusability

Agents can be reused in other workflows that need the same functionality.

**Example**: WPSAgent could be used in a WRF forecast workflow with minimal changes. MeshAgent could generate and partition meshes for any model that uses Metis.

### 4. Clear Separation of Concerns

Each agent owns one component's complete lifecycle:
- Download source code
- Build executables
- Execute component-specific operations

**Example**: MPASAgent is solely responsible for MPAS model operations. It doesn't know how meshes are created or partitioned—that's the job of other agents.

### 5. Better Scalability

Different agents can run on different executors or platforms based on their needs.

**Example**: 
- MeshAgent and WPSAgent run on `build-executor` (service nodes for compilation)
- MPASAgent runs on `compute-executor` (compute nodes for MPI jobs)

## Agent Responsibilities

### MeshAgent

**Purpose**: Manage mesh generation and partitioning (Metis + MPAS-Limited-Area)

**Lifecycle**:
1. Download and build Metis (`gpmetis` utility)
2. Download and build MPAS-Limited-Area (`create_region` utility)
3. Download global mesh files
4. Create regional mesh from global mesh
5. Partition mesh files for specified MPI rank counts

**Why separate?**: Mesh generation and partitioning are tightly coupled preprocessing steps independent of the model itself. They share a strict dependency (partition always follows mesh creation) and represent a single conceptual workflow phase.

### WPSAgent

**Purpose**: Manage WRF Preprocessing System (ungrib utility)

**Lifecycle**:
1. Download WPS source from GitHub
2. Build `ungrib` executable (minimal WPS build)
3. Run ungrib to process GRIB data into intermediate format

**Why separate?**: WPS is shared between WRF and MPAS. This agent could be reused in WRF workflows.

### MPASAgent

**Purpose**: Manage MPAS model initialization and forecasting

**Lifecycle**:
1. Download MPAS-Model source
2. Build `init_atmosphere_model` and `atmosphere_model` executables
3. Initialize MPAS with processed initial conditions
4. Initialize MPAS with processed lateral boundary conditions
5. Run MPAS forecast

**Why separate?**: MPAS model operations are distinct from preprocessing. This agent focuses purely on the model, receiving preprocessed inputs from other agents.

## Workflow Orchestration

The `MPASForecastWorkflow` class orchestrates all agents without being tightly coupled to their implementations. It:

1. Launches all agents via `AgentRuntime` and `Manager`
2. Calls agent actions in the correct dependency order
3. Passes data between agents
4. Handles errors and cleanup

### Workflow Phases

```
Phase 1: Setup
├─ Launch MeshAgent
├─ Launch WPSAgent
└─ Launch MPASAgent

Phase 2: Build (PARALLEL)
├─ mesh.install() (downloads + builds Metis and MPAS-Limited-Area)
├─ wps.download_wps() + wps.build()
└─ mpas.download_mpas() + mpas.build()

Phase 3: Mesh (SEQUENTIAL)
├─ mesh.download_global_mesh()
├─ mesh.create_regional_mesh()
├─ mesh.partition_mesh(init_ranks)
└─ mesh.partition_mesh(forecast_ranks)

Phase 4: Preprocess (PARALLEL)
├─ Fetch ICS GRIB data
├─ Fetch LBCS GRIB data
├─ wps.run_ungrib(ICS)
└─ wps.run_ungrib(LBCS)

Phase 5: Initialize (PARALLEL)
├─ mpas.initialize_ics()
└─ mpas.initialize_lbcs()

Phase 6: Forecast
└─ mpas.run_forecast()

Phase 7: Cleanup
├─ Shutdown all agents
└─ Cleanup workflow
```

## Data Flow

```
MeshAgent
  ↓ (generates and partitions mesh)
WPSAgent
  ↓ (processes GRIB data)
MPASAgent
  ↓ (runs forecast)
Output Files
```

## Communication Pattern

Agents communicate through the orchestration workflow:

```python
# Workflow coordinates agents
mesh = await mesh_agent.create_regional_mesh(...)
partitioned = await mesh_agent.partition_mesh(mesh["graph"], ...)
ungrib_output = await wps_agent.run_ungrib(...)
initialized = await mpas_agent.initialize_ics(ungrib_output, mesh, ...)
forecast = await mpas_agent.run_forecast(initialized, ...)
```

Agents **never** directly call methods on other agents. The workflow orchestrates all interactions.

## Comparison: Multi-Agent vs Monolithic

### Monolithic Agent (Not Used)

```python
@chiltepin_agent()
class MPASForecastAgent:
    # One agent does everything
    def download_all_software(self):
        """Download Metis, WPS, MPAS-LA, MPAS"""
        pass
    
    def build_all_software(self):
        """Build all components"""
        pass
    
    def generate_mesh(self):
        """Handle mesh generation"""
        pass
    
    def run_forecast(self):
        """Run complete forecast"""
        pass
```

**Problems**:
- **Tight coupling**: All components bundled together
- **Testing difficulty**: Can't test components independently
- **No reusability**: Can't use individual pieces in other workflows
- **Single failure point**: Error in one component affects all
- **Scaling issues**: All operations use same executor config

### Multi-Agent (Used)

```python
@chiltepin_agent()
class MeshAgent:
    """Focused on mesh generation and partitioning"""
    def install(self): pass
    def download_global_mesh(self): pass
    def create_regional_mesh(self): pass
    def partition_mesh(self): pass

@chiltepin_agent()
class WPSAgent:
    """Focused on WPS only"""
    def download_wps(self): pass
    def build(self): pass
    def run_ungrib(self): pass

# ... similar for other agents
```

**Benefits**:
- **Loose coupling**: Components independent
- **Easy testing**: Test each agent separately
- **Reusable**: Use agents in different workflows
- **Isolated failures**: Error in one agent doesn't break others
- **Flexible scaling**: Each agent can use different executors

## Best Practices

### When to Create a New Agent

Create a separate agent when:
1. **Component has independent lifecycle** (download → build → execute)
2. **Component could be reused** in other workflows
3. **Component needs different executor** configuration
4. **Component can be tested independently**

### When to Use One Agent

Use a single agent when:
1. **Operations are tightly coupled** and always used together
2. **Component is workflow-specific** and won't be reused
3. **Operations are simple** and don't need complex lifecycle management

### Agent Design Guidelines

1. **Single Responsibility**: Each agent manages one software component
2. **Stateful but Serializable**: Store component state (paths, versions) as simple types
3. **Action-Based Interface**: Expose operations as `@agent_action` decorated methods
4. **Defensive Programming**: Check prerequisites (e.g., build before run)
5. **Clear Documentation**: Document each action's inputs, outputs, and dependencies

## Future Extensions

The multi-agent architecture makes it easy to add new capabilities:

### Add Post-Processing

```python
@chiltepin_agent()
class UPPAgent:
    """Unified Post Processor for MPAS output"""
    def download_upp(self): pass
    def build(self): pass
    def process_output(self, mpas_output): pass
```

Add to workflow:
```python
# In workflow orchestration
upp_output = await upp_agent.process_output(forecast_output)
```

### Add Data Assimilation

```python
@chiltepin_agent()
class DAAgent:
    """Data assimilation with JEDI"""
    def download_jedi(self): pass
    def build(self): pass
    def run_analysis(self, background, observations): pass
```

### Support Multiple Models

Reuse preprocessing agents for WRF:
```python
# WRF workflow reuses WPSAgent and MeshAgent
wrf_workflow = WRFWorkflow(
    agents=[wps_agent, mesh_agent, wrf_agent, upp_agent]
)
```

## Conclusion

The multi-agent architecture provides:
- **Maintainability**: Clear component boundaries
- **Flexibility**: Easy to add/remove/modify components
- **Robustness**: Isolated failures
- **Scalability**: Optimize each component independently

This design pattern is recommended for complex workflows with multiple independent software components.
