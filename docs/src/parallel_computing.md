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
| **MPI** | HPC clusters | MPI installation + `using MPI` | 1000s of cores |

MPI is a weak dependency provided by the `FeastKitMPIExt` package extension:
`using MPI` alongside `using FeastKit` is what gives `mpi_feast` and the other
`mpi_feast_*` drivers their methods. Threading and `Distributed` need no extra
packages. Iterative (IFEAST) solves additionally need `using Krylov`.

High-level production support is intentionally narrower than the lower-level
interfaces:

| Backend | Supported high-level problems | Fallback behavior |
|---------|--------------------------------|-------------------|
| `:serial` | Real symmetric, complex Hermitian, and general problems | None |
| `:threads` | Dense **and** sparse real symmetric standard/generalized problems | Explicit requests throw on unsupported inputs |
| `:distributed` | Sparse real symmetric standard/generalized problems with workers | Explicit requests throw if workers are missing |
| `:mpi` | Real symmetric standard/generalized plus dense/sparse complex Hermitian/general problems with an MPI communicator | Explicit requests throw if MPI is unavailable or storage is unsupported |
| `:auto` | Best available supported backend | Falls back to serial when needed |

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

Run this in a session started with more than one thread (see above);
`backend=:threads` throws in a single-threaded session rather than pretending to
parallelise.

```julia
using FeastKit, LinearAlgebra, SparseArrays

# Create test problem
n = 5000
A = SymTridiagonal(2.0*ones(n), -ones(n-1))
B = Matrix(1.0I, n, n)

# The eigenvalues are 2 - 2cos(kπ/(n+1)): (0.5, 1.5) would hold 948 of them,
# far more than M0. Bracket the ten smallest.
interval = (0.0, 4.15e-5)

A_sparse = sparse(A)
B_sparse = sparse(B)
result = feast(A_sparse, B_sparse, interval, M0=20, backend=:threads)

# Use automatic selection when serial fallback is acceptable.
result = feast(A_sparse, B_sparse, interval, M0=20, backend=:auto)

println("Found $(result.M) eigenvalues using $(Threads.nthreads()) threads")
```

### Direct RCI Interface

For more control, use the parallel RCI (Reverse Communication Interface):

```julia
using FeastKit

# ne = contour points, M0 = subspace size, N = matrix size; work/workc/Aq/Sq,
# lambda/q/res are the caller-owned RCI buffers (see `FeastWorkspaceReal`).
state = ParallelFeastState{Float64}(ne, M0, true, true)

# RCI loop
while true
    pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)

    if state.ijob == Int(FeastKit.Feast_RCI_PARALLEL_SOLVE)
        # Solve all contour points in parallel
        pfeast_compute_all_contour_points!(state, A, B, work, M0)
    elseif state.ijob == Int(Feast_RCI_MULT_A)
        work[:, 1:state.mode] .= A * q[:, 1:state.mode]
    elseif state.ijob == Int(Feast_RCI_MULT_B)
        work[:, 1:state.mode] .= B * q[:, 1:state.mode]
    elseif state.ijob == Int(Feast_RCI_DONE)
        break
    end
end
```

---

```@docs
feast_parallel
pfeast_srci!
```

## Distributed Computing

For multi-process parallelization using Julia's `Distributed` module.

The high-level distributed backend currently supports sparse real symmetric
standard/generalized problems. Requesting `backend=:distributed` requires active
Julia workers; use `backend=:auto` if serial fallback is acceptable.

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

Add workers first (see above); `backend=:distributed` throws when there are
none rather than silently running serial.

```julia
using Distributed
@everywhere using FeastKit
using LinearAlgebra, SparseArrays

# Create problem on main process
n = 10000
A = sprandn(n, n, 0.001)
A = A + A' + 10I
B = sparse(1.0I, n, n)

# Distributed computation. Throws if no workers are available.
result = feast(A, B, (9.0, 11.0), M0=30, backend=:distributed)

println("Found $(result.M) eigenvalues using $(nworkers()) workers")
```

### How It Works

FeastKit distributes contour points across workers:

```julia
# Show distribution. The worker/thread count comes from the session, so the
# only argument is the number of contour points; `use_threads` picks which
# layout to report.
using FeastKit
pfeast_show_distribution(16; use_threads=false)
# Worker 1: points 1-4
# Worker 2: points 5-8
# Worker 3: points 9-12
# Worker 4: points 13-16
```

---

## MPI Parallelization

For high-performance computing clusters with thousands of cores.

The high-level MPI backend supports real symmetric standard/generalized
problems and dense/sparse complex Hermitian/general problems. Pass the communicator
explicitly when using the public `feast` and `feast_general` APIs so MPI
remains an explicit opt-in.

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
result = feast(A, B, (0.0, 0.1), M0=20, backend=:mpi, comm=comm)

# Dense or sparse complex general MPI FEAST
Az = sparse(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 + 0.2im, 2.0 - 0.1im]))
Bz = spdiagm(0 => ones(ComplexF64, 3))
general = feast_general(Az, Bz, 1.0 + 0.1im, 1.5;
                        M0=3, backend=:mpi, comm=comm)

if rank == 0
    println("Found $(result.M) eigenvalues")
    println("Eigenvalues: $(result.lambda[1:result.M])")
end

MPI.Finalize()
```

Run with MPI:

```bash
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 8 julia --project feast_mpi.jl`)'
```

### MPI-Specific Functions

MPI honors custom contours registered in `fpm` on the communicator's root.
The contour nodes and weights are broadcast to every rank, and general-problem
eigenvalue selection uses that same geometry. A rank-local factorization or
direct shifted-solve failure returns `Feast_ERROR_LAPACK` collectively;
iterative projection failures return `Feast_ERROR_NO_CONVERGENCE`.
Projected eigenproblem and distributed residual failures are also synchronized
before ranks enter the next collective. Ranks use the root's Ritz vectors and
stopping decision. Supply the same matrices and solver settings on all ranks.
This synchronization also covers failures while forming the shifted-system
right-hand side `B * Q`, before the contour solves begin.

For iterative MPI solves, the default inner GMRES tolerance is one percent of
the precision-aware outer tolerance, with a machine-epsilon floor. This leaves
accuracy headroom for outer refinement without requesting unattainable precision
from `ComplexF32` solves. An explicit `solver_tol` overrides this default.

`fpm[10]=0` disables retained LU factors: each rank factors and solves one local
shift at a time on each refinement sweep. This mode also uses serial contour
solves within a hybrid rank to bound factor storage; `fpm[10]=1` retains factors
and permits threaded reuse. Real and complex `mpi_feast` entry points accept
either the `FeastParameters` returned by `feastinit()` or its raw `.fpm` vector.

```julia
# Direct MPI interface
result = mpi_feast(A, B, interval, M0=M0, comm=comm, fpm=fpm)

# FEAST-compatible PFEAST alias with explicit MPI communicator
result = pdfeast_scsrgv!(A, B, interval[1], interval[2], M0, fpm; comm=comm)

# Dense/sparse complex Hermitian/general PFEAST aliases with explicit MPI communicator
Ahd = Matrix(Diagonal(ComplexF64[0.5, 1.0, 2.0]))
Bhd = Matrix{ComplexF64}(I, 3, 3)
dense_hz = pzfeast_hegv!(Ahd, Bhd, 0.0, 2.5, M0, fpm; comm=comm)
Ahz = sparse(Diagonal(ComplexF64[0.5, 1.0, 2.0]))
hz = pzfeast_hcsrgv!(Ahz, Bz, 0.0, 2.5, M0, fpm; comm=comm)
gz = pzifeast_gcsrgv!(Az, Bz, 1.0 + 0.1im, 1.5, M0, fpm;
                      comm=comm, solver_tol=1e-10)

# Check MPI availability
if mpi_available()
    println("MPI is ready!")
end
```

The real symmetric PFEAST-compatible aliases are `psfeast_syev!`,
`pdfeast_syev!`, `psfeast_sygv!`, `pdfeast_sygv!`, `psfeast_scsrev!`,
`pdfeast_scsrev!`, `psfeast_scsrgv!`, `pdfeast_scsrgv!`, `psfeast_srci!`,
and `pdfeast_srci!`. Without `comm=`, these call the threaded/distributed
parallel kernels. With `comm=`, the dense and sparse generalized/standard
aliases use the MPI kernels.

Complex MPI aliases include dense `pcfeast_heev!`, `pzfeast_heev!`,
`pcfeast_hegv!`, `pzfeast_hegv!`, `pcfeast_geev!`, `pzfeast_geev!`,
`pcfeast_gegv!`, `pzfeast_gegv!`, sparse `pcfeast_hcsrev!`,
`pzfeast_hcsrev!`, `pcfeast_hcsrgv!`, `pzfeast_hcsrgv!`,
`pcfeast_gcsrev!`, `pzfeast_gcsrev!`, `pcfeast_gcsrgv!`,
`pzfeast_gcsrgv!`, and their GMRES-backed `pcifeast_*`/`pzifeast_*`
counterparts.

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
                      use_threads_per_rank=true)

if rank == 0
    println("Hybrid computation complete")
    println("MPI ranks: $(MPI.Comm_size(comm))")
    println("Threads per rank: $(Threads.nthreads())")
    println("Total parallelism: $(MPI.Comm_size(comm) * Threads.nthreads())")
end

MPI.Finalize()
```

### Architecture

Hybrid execution uses the same filtered-subspace solver as the real MPI backend,
with independent contour solves threaded within each rank. MPI collectives run
on the calling thread. Rank-local factorization or solve failures are propagated
collectively as `Feast_ERROR_LAPACK`.

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

Needs workers, as above.

```julia
using FeastKit, Distributed, SparseArrays, LinearAlgebra

n = 2000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = sparse(1.0I, n, n)
interval = (0.0, 2.5e-5)   # the ten smallest eigenvalues

# Rule of thumb: points = 2 × workers
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[2] = max(2 * nworkers(), 8)  # Set integration points

result = feast(A, B, interval, M0=20, fpm=fpm, backend=:distributed)
```

### Benchmarking

Compare parallel performance:

```julia
using FeastKit, SparseArrays, LinearAlgebra

n = 2000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = sparse(1.0I, n, n)
interval = (0.0, 2.5e-5)

# Compare backends (M0 is positional here)
feast_parallel_comparison(A, B, interval, 20)

# Detailed benchmarks
FeastKit.pfeast_rci_benchmark(A, B, interval, 20; compare_serial=true)
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

Needs a multi-threaded session.

```julia
# Monitor parallel efficiency. FeastResult has no timing field, so measure the
# calls themselves rather than reading `result.time`.
t_serial   = @elapsed result_serial   = feast(A, B, interval, M0=20, backend=:serial)
t_parallel = @elapsed result_parallel = feast(A, B, interval, M0=20, backend=:threads)

speedup = t_serial / t_parallel
efficiency = speedup / Threads.nthreads()
println("Speedup: $(speedup)x, Efficiency: $(efficiency * 100)%")
```

---

## API Reference

### High-Level Functions

Signature catalogue -- `A`, `B`, `interval`, `M0` and the RCI buffers are yours:

```julia
# Automatic backend selection
feast(A, B, interval; backend=:auto)
feast(A, B, interval; backend=:threads)
feast(A, B, interval; backend=:distributed)
feast(A, B, interval; backend=:mpi, comm=MPI.COMM_WORLD)

# Direct parallel interface
feast_parallel(A, B, interval; use_threads=true)
mpi_feast(A, B, interval; comm=comm)
feast_hybrid(A, B, interval; comm=comm, use_threads_per_rank=true)
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
pfeast_show_distribution(ne; use_threads=true)
determine_parallel_backend(parallel, comm)
```

---

<div align="center">
  <p><strong>Scale your eigenvalue computations!</strong></p>
  <a href="performance.md">Performance Tips</a> · <a href="examples.md">Examples</a> · <a href="api_reference.md">API Reference</a>
</div>
