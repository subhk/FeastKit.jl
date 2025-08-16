# FEAST.jl

[![Build Status](https://github.com/subhk/FEAST.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/FEAST.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia implementation of the FEAST eigenvalue solver for finding eigenvalues and eigenvectors of large-scale eigenvalue problems within a specified region.

## Overview

FEAST.jl is a pure Julia translation of the original FEAST (Fast Eigenvalue Algorithm using Spectral Transformations) library. FEAST is a numerical algorithm for solving both standard and generalized eigenvalue problems by computing eigenvalues located inside a given region in the complex plane.

### Key Features

- **Interval-based eigenvalue computation**: Find eigenvalues in specified intervals or regions
- **Multiple matrix formats**: Support for dense, sparse, and banded matrices  
- **Generalized eigenvalue problems**: Solve both Ax = λx and Ax = λBx problems
- **Complex arithmetic**: Handle both real symmetric/Hermitian and general complex matrices
- **Parallel computation**: Multi-threaded and distributed computing for contour integration
- **Reverse Communication Interface (RCI)**: Advanced interface for custom linear solvers
- **Polynomial eigenvalue problems**: Support for polynomial eigenvalue problems

## Installation

```julia
using Pkg
Pkg.add("FEAST")
```

## Quick Start

### Basic Usage

```julia
using FEAST
using LinearAlgebra

# Create a test matrix
n = 100
A = diagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Find eigenvalues in the interval [0.5, 1.5]
result = feast(A, (0.5, 1.5), M0=10)

println("Found $(result.M) eigenvalues")
println("Eigenvalues: ", result.lambda)
```

### Generalized Eigenvalue Problems

```julia
# For generalized problem Ax = λBx
B = diagm(0 => ones(n))
result = feast(A, B, (0.5, 1.5), M0=10)
```

### Sparse Matrices

```julia
using SparseArrays

# Create sparse matrix
A_sparse = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
result = feast(A_sparse, (0.5, 1.5), M0=10)
```

### Complex Eigenvalue Problems

```julia
# For general complex matrices, use circular search region
A_complex = randn(ComplexF64, 50, 50)
B_complex = Matrix{ComplexF64}(I, 50, 50)

# Search in circle centered at origin with radius 2
center = 0.0 + 0.0im
radius = 2.0
result = feast_general(A_complex, B_complex, center, radius, M0=15)
```

## Advanced Usage

### Custom FEAST Parameters

```julia
# Initialize FEAST parameters
fpm = feastinit()

# Customize parameters
feast_set_defaults!(fpm.fpm, 
                   print_level=1,
                   integration_points=16,  # More integration points
                   tolerance_exp=14,       # Higher precision
                   max_refinement=30)      # More refinement loops

# Use custom parameters
result = feast(A, (0.5, 1.5), M0=10, fpm=fpm.fpm)
```

### Banded Matrices

```julia
# For banded matrices stored in LAPACK format
n = 100
k = 2  # Number of super-diagonals
A_banded = zeros(k+1, n)
# ... fill A_banded with appropriate values ...

result = feast_banded(A_banded, k, (0.5, 1.5), M0=10)
```

### Matrix-Free Operations

```julia
# Define matrix-vector operations
function A_mul!(y, x)
    # Implement y = A*x
    y .= A * x
end

function B_mul!(y, x)
    # Implement y = B*x  
    y .= B * x
end

# Use matrix-free interface
result = feast_matvec(A_mul!, B_mul!, n, (0.5, 1.5), M0=10)
```

## Parallel Computing

FEAST.jl supports parallel computation where each contour integration point is solved independently, leading to significant speedups for large problems.

### Multi-threaded Execution

```julia
# Enable parallel computation using threads
result = feast(A, (0.5, 1.5), M0=10, parallel=:threads)

# Or use the explicit parallel interface
result = feast_parallel(A, B, (0.5, 1.5), M0=10, use_threads=true)
```

### Distributed Computing

```julia
using Distributed

# Add worker processes
addprocs(4)

# Use distributed computing for contour integration
result = feast(A, (0.5, 1.5), M0=10, parallel=:distributed)
```

### MPI Support for HPC Clusters

FEAST.jl provides full MPI support for high-performance computing clusters:

```julia
using MPI
using FEAST

# Initialize MPI (if not already done)
MPI.Init()

# Basic MPI FEAST
result = feast(A, B, (0.5, 1.5), M0=10, parallel=:mpi)

# Explicit MPI interface with communicator
comm = MPI.COMM_WORLD
result = mpi_feast(A, B, (0.5, 1.5), M0=10, comm=comm)

# Hybrid MPI + threading (best for modern HPC)
result = feast_hybrid(A, B, (0.5, 1.5), M0=10, 
                     comm=comm, use_threads_per_rank=true)
```

#### Running on HPC Systems

Create a job script for SLURM/PBS:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load julia mpi

# Run with MPI + threading
mpirun -np 32 julia --threads=4 feast_mpi_example.jl
```

Julia script (`feast_mpi_example.jl`):
```julia
using MPI
using FEAST
using LinearAlgebra

MPI.Init()

# Create distributed problem
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

if rank == 0
    println("Running FEAST on $size MPI processes with $(Threads.nthreads()) threads each")
end

# Large eigenvalue problem
n = 10000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = sparse(I, n, n)

# Hybrid MPI + threading execution
result = feast_hybrid(A, B, (0.5, 1.5), M0=20, 
                     comm=comm, use_threads_per_rank=true)

if rank == 0
    println("Found $(result.M) eigenvalues")
    println("Computation time measured on each rank")
end

MPI.Finalize()
```

### Parallel RCI Interface

For advanced users, a parallel RCI interface is available:

```julia
using FEAST

# Create parallel state
state = ParallelFeastState{Float64}(ne=8, M0=10, use_parallel=true, use_threads=true)

# Initialize workspace
N = size(A, 1)
work = randn(N, M0)
workc = zeros(ComplexF64, N, M0)
Aq = zeros(M0, M0)
Sq = zeros(M0, M0)
lambda = zeros(M0)
q = zeros(N, M0)
res = zeros(M0)

# RCI loop
while true
    pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, 
                Emin, Emax, M0, lambda, q, res)
    
    if state.ijob == FEAST_RCI_PARALLEL_SOLVE.value
        # Solve all contour points in parallel
        pfeast_compute_all_contour_points!(state, A, B, work, M0)
        
    elseif state.ijob == FEAST_RCI_MULT_A.value
        # Compute A*q for residual calculation
        M = state.mode
        work[:, 1:M] .= A * q[:, 1:M]
        
    elseif state.ijob == FEAST_RCI_DONE.value
        break
    end
end
```

### Performance Benchmarking

```julia
# Compare serial vs parallel performance
pfeast_benchmark(A, B, (0.5, 1.5), 10)

# Compare all available parallel backends
feast_parallel_comparison(A, B, (0.5, 1.5), 10)

# Check parallel computing capabilities
feast_parallel_info()

# Output example:
# FEAST Parallel Backend Comparison
# ==================================================
# Matrix size: 1000
# Search interval: (0.5, 1.5)
# Available backends:
#   Threads: 8
#   Workers: 4  
#   MPI: Yes
# 
# 1. Serial execution:
#    Time: 2.345 seconds
#    Eigenvalues: 7
# 
# 2. Threading execution:
#    Time: 0.412 seconds
#    Eigenvalues: 7
#    Speedup: 5.69x
# 
# 3. Distributed execution:
#    Time: 0.523 seconds
#    Eigenvalues: 7
#    Speedup: 4.48x
# 
# 4. MPI execution:
#    Time: 0.298 seconds
#    Eigenvalues: 7
#    Speedup: 7.87x
# 
# 5. Hybrid MPI+Threading:
#    Time: 0.156 seconds
#    Eigenvalues: 7
#    Speedup: 15.03x
```

### Automatic Backend Selection

```julia
# FEAST automatically selects the best available backend
result = feast(A, B, (0.5, 1.5), M0=10, parallel=:auto)

# Manual backend selection
result = feast(A, B, (0.5, 1.5), M0=10, parallel=:mpi)      # Force MPI
result = feast(A, B, (0.5, 1.5), M0=10, parallel=:threads)  # Force threading
result = feast(A, B, (0.5, 1.5), M0=10, parallel=:serial)   # Force serial
```

## Algorithm Overview

FEAST uses contour integration in the complex plane to compute eigenvalues. The key steps are:

1. **Contour Definition**: Define an integration contour enclosing the desired eigenvalues
2. **Moment Computation**: Compute spectral projector moments using numerical integration  
3. **Subspace Extraction**: Extract eigenspace using computed moments
4. **Refinement**: Iteratively refine the solution until convergence

The algorithm is particularly effective for:
- Large sparse matrices
- Finding eigenvalues in specific intervals
- Parallel computation (via the original PFEAST routines)

## Result Structure

FEAST returns a `FeastResult` object containing:

```julia
struct FeastResult{T<:Real, VT}
    lambda::Vector{T}      # Computed eigenvalues
    q::Matrix{VT}          # Computed eigenvectors  
    M::Int                 # Number of eigenvalues found
    res::Vector{T}         # Residual norms
    info::Int              # Exit status (0 = success)
    epsout::T              # Final residual
    loop::Int              # Number of refinement loops
end
```

## Error Codes

- `info = 0`: Successful convergence
- `info = 1`: Invalid matrix size
- `info = 2`: Invalid search subspace size  
- `info = 3`: Invalid search interval
- `info = 5`: No convergence achieved
- `info = 6`: Memory allocation error
- `info = 8`: LAPACK error

## Performance Tips

1. **Choose appropriate M0**: Set M0 slightly larger than expected number of eigenvalues
2. **Integration points**: More points (fpm[2]) improve accuracy but increase cost
3. **Sparse matrices**: Use sparse format for large problems with few non-zeros
4. **Initial guess**: Provide good initial guess when available (fpm[5] = 1)
5. **Parallel execution**: Use `parallel=:auto` to automatically select the best backend
6. **HPC clusters**: Use `parallel=:mpi` or `feast_hybrid()` for optimal cluster performance  
7. **Hybrid parallelism**: Combine MPI processes with threading for maximum performance
8. **Load balancing**: FEAST automatically distributes contour points for optimal load balancing

## Limitations

This is a Julia translation of the original FEAST library. Some features from the original may not be fully implemented:

- Custom contour integration is partially implemented
- Some advanced PFEAST (parallel) features are not included
- Matrix-free interface is simplified compared to the original

## References

1. E. Polizzi, "Density-matrix-based algorithm for solving eigenvalue problems", Physical Review B 79, 115112 (2009)
2. FEAST official website: http://www.feast-solver.org

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the same terms as the original FEAST library.
