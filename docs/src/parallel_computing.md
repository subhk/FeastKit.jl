# Parallel Computing

FeastKit.jl provides multiple parallelization strategies to accelerate eigenvalue computations. The FEAST algorithm is naturally parallelizable because each contour integration point can be computed independently.

## Table of Contents

- [Overview](#overview)
- [Checking Capabilities](#checking-capabilities)
- [Threading (Shared Memory)](#threading-shared-memory)
- [Distributed Computing](#distributed-computing)
- [MPI Parallelization](#mpi-parallelization)
- [Hybrid Parallelization](#hybrid-parallelization)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

FeastKit supports three parallel backends:

| Backend | Best For | Setup | Scalability |
|---------|----------|-------|-------------|
| **Threading** | Single node, shared memory | `julia --threads=N` | Up to ~16 cores |
| **Distributed** | Multi-process Julia | `addprocs(N)` | Multiple nodes |
| **MPI** | HPC clusters | MPI installation | 1000s of cores |

### How FEAST Parallelizes

The FEAST algorithm computes eigenvalues using contour integration:

```
                    Im(z)
                      ↑
      z₄ ●───────────●───────────● z₁
         │           │           │
      z₃ ●───────────●───────────● z₂    Each zₙ is computed
         │           │           │       independently!
      z₅ ●───────────●───────────● z₈
         │           │           │
      z₆ ●───────────●───────────● z₇
         └───────────┴───────────→ Re(z)
               Emin        Emax
```

Each integration point requires solving a linear system `(z*B - A)*Y = X`. These solves are independent and can be distributed across workers.

---

## Checking Capabilities

Before using parallel features, check available backends:

```julia
using FeastKit

# Check all available backends
capabilities = feast_parallel_capabilities()
println(capabilities)
# Dict(:threads => true, :distributed => false, :mpi => false)

# Detailed information
feast_parallel_info()
# FeastKit Parallel Computing Capabilities
# ========================================
# Threading:
#   Available threads: 8
#   Status: Enabled
#
# Distributed Computing:
#   Available workers: 1
#   Status: Disabled
#
# MPI:
#   MPI initialized: No
#   Status: Disabled
```

---

## Threading (Shared Memory)

The simplest parallelization - uses Julia's built-in threading.

### Setup

Start Julia with multiple threads:

```bash
# Command line
julia --threads=8

# Or use auto-detection
julia --threads=auto

# Environment variable
export JULIA_NUM_THREADS=8
julia
```

### Usage

```julia
using FeastKit, LinearAlgebra

# Create test problem
n = 5000
A = SymTridiagonal(2.0*ones(n), -ones(n-1))
B = Matrix(1.0I, n, n)

# Threaded computation
result = feast(A, B, (0.5, 1.5), M0=20, parallel=:threads)

# Or explicitly using feast_parallel
result = feast_parallel(A, B, (0.5, 1.5), M0=20, use_threads=true)

println("Found $(result.M) eigenvalues using $(Threads.nthreads()) threads")
```

### Direct RCI Interface

For more control, use the parallel RCI (Reverse Communication Interface):

```julia
using FeastKit

# Create parallel state
state = ParallelFeastState{Float64}(ne, M0, use_parallel=true, use_threads=true)

# RCI loop
while true
    pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)

    if state.ijob == Int(Feast_RCI_PARALLEL_SOLVE)
        # Solve all contour points in parallel
        pfeast_compute_all_contour_points!(state, A, B, work, M0)
    elseif state.ijob == Int(Feast_RCI_DONE)
        break
    end
end
```

---

## Distributed Computing

For multi-process parallelization using Julia's `Distributed` module.

### Setup

```julia
using Distributed

# Add local workers
addprocs(4)  # Add 4 worker processes

# Or add remote workers
addprocs([("node1", 2), ("node2", 2)])  # 2 workers each on node1 and node2

# Verify workers
println("Workers: $(workers())")
println("Number of workers: $(nworkers())")
```

### Usage

```julia
using Distributed
@everywhere using FeastKit
using LinearAlgebra

# Create problem on main process
n = 10000
A = sprandn(n, n, 0.001)
A = A + A' + 10I
B = sparse(1.0I, n, n)

# Distributed computation
result = feast(A, B, (9.0, 11.0), M0=30, parallel=:distributed)

println("Found $(result.M) eigenvalues using $(nworkers()) workers")
```

### How It Works

FeastKit distributes contour points across workers:

```julia
# Show distribution
using FeastKit
pfeast_show_distribution(16, nworkers())
# Worker 1: points 1-4
# Worker 2: points 5-8
# Worker 3: points 9-12
# Worker 4: points 13-16
```

---

## MPI Parallelization

For high-performance computing clusters with thousands of cores.

### Prerequisites

1. Install MPI on your system (OpenMPI, MPICH, or Intel MPI)
2. Install MPI.jl: `Pkg.add("MPI")`
3. Enable MPI in FeastKit: `ENV["FEASTKIT_ENABLE_MPI"] = "true"`

### Setup

```bash
# Install MPI.jl and configure
julia -e 'using Pkg; Pkg.add("MPI"); using MPI; MPI.install_mpiexecjl()'

# Set environment variable before running
export FEASTKIT_ENABLE_MPI=true
```

### Usage

Create a script `feast_mpi.jl`:

```julia
using MPI
MPI.Init()

using FeastKit, LinearAlgebra, SparseArrays

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Create problem (same on all ranks)
n = 10000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = sparse(1.0I, n, n)

# MPI FEAST
result = mpi_feast(A, B, (0.0, 0.1), M0=20, comm=comm)

if rank == 0
    println("Found $(result.M) eigenvalues")
    println("Eigenvalues: $(result.lambda[1:result.M])")
end

MPI.Finalize()
```

Run with MPI:

```bash
mpiexec -n 8 julia --project feast_mpi.jl
```

### MPI-Specific Functions

```julia
# Direct MPI interface
result = mpi_feast(A, B, interval, M0=M0, comm=comm, fpm=fpm)

# Check MPI availability
if mpi_available()
    println("MPI is ready!")
end
```

---

## Hybrid Parallelization

Combine MPI (across nodes) with threading (within nodes) for maximum performance.

### Setup

```bash
# 4 MPI ranks, each with 8 threads
export JULIA_NUM_THREADS=8
mpiexec -n 4 julia --threads=8 feast_hybrid.jl
```

### Usage

```julia
using MPI
MPI.Init()

using FeastKit

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Hybrid FEAST: MPI + threads
result = feast_hybrid(A, B, interval,
                      M0=20,
                      comm=comm,
                      use_threads=true)

if rank == 0
    println("Hybrid computation complete")
    println("MPI ranks: $(MPI.Comm_size(comm))")
    println("Threads per rank: $(Threads.nthreads())")
    println("Total parallelism: $(MPI.Comm_size(comm) * Threads.nthreads())")
end

MPI.Finalize()
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MPI Communicator                          │
├───────────────┬───────────────┬───────────────┬─────────────┤
│    Rank 0     │    Rank 1     │    Rank 2     │   Rank 3    │
│  ┌─────────┐  │  ┌─────────┐  │  ┌─────────┐  │ ┌─────────┐ │
│  │Thread 1 │  │  │Thread 1 │  │  │Thread 1 │  │ │Thread 1 │ │
│  │Thread 2 │  │  │Thread 2 │  │  │Thread 2 │  │ │Thread 2 │ │
│  │Thread 3 │  │  │Thread 3 │  │  │Thread 3 │  │ │Thread 3 │ │
│  │Thread 4 │  │  │Thread 4 │  │  │Thread 4 │  │ │Thread 4 │ │
│  └─────────┘  │  └─────────┘  │  └─────────┘  │ └─────────┘ │
│ Points: 1-4   │ Points: 5-8   │ Points: 9-12  │Points: 13-16│
└───────────────┴───────────────┴───────────────┴─────────────┘
```

---

## Performance Tuning

### Choosing the Right Backend

| Scenario | Recommended Backend |
|----------|---------------------|
| Laptop/workstation (1-16 cores) | `:threads` |
| Single node server (16-64 cores) | `:threads` or `:distributed` |
| Multi-node cluster | `:mpi` |
| HPC with many cores per node | `:hybrid` (MPI + threads) |

### Integration Points vs Workers

The number of integration points should match or exceed your worker count:

```julia
# Rule of thumb: points = 2 × workers
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[2] = 2 * nworkers()  # Set integration points

result = feast(A, B, interval, M0=20, fpm=fpm, parallel=:distributed)
```

### Benchmarking

Compare parallel performance:

```julia
using FeastKit

# Compare backends
feast_parallel_comparison(A, B, interval, M0=20)

# Detailed benchmarks
pfeast_rci_benchmark(A, B, interval, M0, compare_serial=true)
# Parallel RCI Performance Comparison
# =====================================
# Matrix size: 5000
# Integration points: 16
# Threads available: 8
# Workers available: 4
#
# Parallel FeastKit (threaded):
# Time: 2.345 seconds
# Eigenvalues found: 15
# Convergence loops: 3
#
# Serial FeastKit:
# Time: 12.567 seconds
# Thread speedup: 5.36x
```

### Memory Considerations

Parallel computation increases memory usage:

| Backend | Memory per Worker | Total Overhead |
|---------|------------------|----------------|
| Threading | Shared | 1× base |
| Distributed | Full copy | N× base |
| MPI | Full copy | N× base |

For memory-constrained systems, use threading or reduce `M0`.

---

## Troubleshooting

### Threading Not Working

```julia
# Check thread count
println(Threads.nthreads())  # Should be > 1

# Solution: Restart Julia with threads
# julia --threads=8
```

### Distributed Workers Not Found

```julia
# Check workers
println(nworkers())  # Should be > 1

# Add workers if needed
using Distributed
addprocs(4)

# Make sure FeastKit is loaded on all workers
@everywhere using FeastKit
```

### MPI Initialization Fails

```julia
# Check MPI availability
println(mpi_available())  # Should be true

# Common fixes:
# 1. Set environment variable
ENV["FEASTKIT_ENABLE_MPI"] = "true"

# 2. Ensure MPI.jl is properly installed
using Pkg
Pkg.add("MPI")
using MPI
MPI.install_mpiexecjl()

# 3. Run under mpiexec
# mpiexec -n 4 julia your_script.jl
```

### Performance Not Scaling

1. **Check linear solver time**: If solving `(z*B - A)*Y = X` dominates, parallel overhead may be significant
2. **Increase problem size**: Small problems have too much communication overhead
3. **Use matrix-free**: For very large problems, matrix-free with parallel linear solvers scales better

```julia
# Monitor parallel efficiency
@time result_serial = feast(A, B, interval, M0=20, parallel=:serial)
@time result_parallel = feast(A, B, interval, M0=20, parallel=:threads)

speedup = result_serial.time / result_parallel.time
efficiency = speedup / Threads.nthreads()
println("Speedup: $(speedup)x, Efficiency: $(efficiency * 100)%")
```

---

## API Reference

### High-Level Functions

```julia
# Automatic backend selection
feast(A, B, interval; parallel=:auto)
feast(A, B, interval; parallel=:threads)
feast(A, B, interval; parallel=:distributed)
feast(A, B, interval; parallel=:mpi, comm=MPI.COMM_WORLD)

# Direct parallel interface
feast_parallel(A, B, interval; use_threads=true)
mpi_feast(A, B, interval; comm=comm)
feast_hybrid(A, B, interval; comm=comm, use_threads=true)
```

### State Management

```julia
# Parallel RCI state
state = ParallelFeastState{Float64}(ne, M0, use_parallel, use_threads)

# RCI functions
pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)
pfeast_compute_all_contour_points!(state, A, B, work, M0)
```

### Utilities

```julia
# Check capabilities
feast_parallel_capabilities()
feast_parallel_info()
mpi_available()

# Distribution helpers
pfeast_show_distribution(ne, nworkers)
determine_parallel_backend(parallel, comm)
```

---

<div align="center">
  <p><strong>Scale your eigenvalue computations!</strong></p>
  <a href="performance.md">Performance Tips</a> · <a href="examples.md">Examples</a> · <a href="api_reference.md">API Reference</a>
</div>
