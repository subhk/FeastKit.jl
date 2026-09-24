# FeastKit.jl

[![CI](https://github.com/subhk/FeastKit.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/subhk/FeastKit.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Stable documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/FeastKit.jl/stable/)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/FeastKit.jl/dev/)

FeastKit.jl is a pure Julia translation of the original FEAST library. FEAST is a numerical algorithm for solving both standard and generalized eigenvalue problems by computing eigenvalues located inside a given region in the complex plane.

### Key Features

- **Interval-based eigenvalue computation**: Find eigenvalues in specified intervals or regions
- **Multiple matrix formats**: Support for dense, sparse, and banded matrices  
- **Generalized eigenvalue problems**: Solve both Ax = λx and Ax = λBx problems
- **Complex arithmetic**: Handle both real symmetric/Hermitian and general complex matrices
- **Precision flexibility**: Works seamlessly with `Float64`/`ComplexF64` and `Float32`/`ComplexF32`
- **Parallel computation**: Multi-threaded, distributed, and optional MPI-based contour integration
- **Reverse Communication Interface (RCI)**: Advanced interface for custom linear solvers
- **Polynomial eigenvalue problems**: Support for polynomial eigenvalue problems

## Installation

```julia
using Pkg
Pkg.add("FeastKit")
```

### Optional dependencies

Krylov and MPI are weak dependencies, loaded through package extensions, so the
base install stays small. Load them alongside FeastKit when you need them:

```julia
using FeastKit
using Krylov   # enables the iterative IFEAST paths (solver=:gmres)
using MPI      # enables mpi_feast and the other distributed drivers
```

Direct (factorization-based) FEAST needs neither. Calling an iterative variant
without Krylov, or an MPI driver without MPI, reports what to load.

New performance controls in this checkout include `initial_subspace` for
serial/matrix-free warm starts, `subspace_size=:auto` with a width cap, and
opt-in dense serial `mixed_precision=true`. See the
[performance guide](docs/src/performance.md) for supported paths and
[measured tradeoffs](benchmark/README.md).

## Quick Start

### Basic Usage

```julia
using FeastKit
using LinearAlgebra

# Create a test matrix
n = 100
A = diagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Find eigenvalues in the interval [0.5, 1.5]
result = feast(A, (0.5, 1.5); subspace_size=30, tol=1e-10)

@assert result.converged result.message
println("Found $(result.M) eigenvalues")
println("Eigenvalues: ", result.values)
```

Use `subspace_size` to leave room for all eigenvalues in the search region.
`M0` remains an alias. Common controls are `tol`, `maxiter`, and
`quadrature_points`; `fpm` remains available for advanced settings. Conflicting
named and `fpm` settings raise an error. Tolerances round down to the next
decimal power (for example, `tol=3e-9` uses `1e-9`); Float32 retains its precision
floor of `sqrt(eps(Float32))`.

Full contours can be passed directly, including for non-Hermitian problems:

```julia
C = Matrix(Diagonal(ComplexF64[-0.3+0.2im, 0.4-0.1im, 2.5]))
region = feast_rectangle(-1, 1, -1, 1)
result = feast(C, region; subspace_size=3)
@assert result.converged result.message
```

For iterative shifted solves, load `Krylov` and pass `solver=:gmres` with
`solver_opts=(rtol=1e-13, maxiter=500, restart=30)`. Assembled matrices default
to `solver=:direct`; matrix-free operators default to `:gmres` and also accept
a solver callback. These keyword names work across both interfaces.

### Generalized Eigenvalue Problems

```julia
# For generalized problem Ax = λBx
B = diagm(0 => ones(n))
result = feast(A, B, (0.5, 1.5); subspace_size=30)
```

### Sparse Matrices

```julia
using SparseArrays

# Create sparse matrix
A_sparse = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
result = feast(A_sparse, (0.5, 1.5); subspace_size=30)
```

### Complex Eigenvalue Problems

```julia
using Random

# For general complex matrices, use circular search region
A_complex = randn(MersenneTwister(1), ComplexF64, 50, 50)
B_complex = Matrix{ComplexF64}(I, 50, 50)

# Search in circle centered at origin with radius 2
center = 0.0 + 0.0im
radius = 2.0
result = feast_general(A_complex, B_complex, center, radius, M0=15)
```

## Advanced Usage

### Custom FeastKit Parameters

```julia
# Initialize Feast parameters
fpm = feastinit()

# Customize parameters
feast_set_defaults!(fpm.fpm, 
                   print_level=1,
                   integration_points=16,  # More integration points
                   tolerance_exp=14,       # Higher precision
                   max_refinement=30)      # More refinement loops

# Use custom parameters. The interval holds 19 eigenvalues, so leave room.
result = feast(A, (0.5, 1.5); subspace_size=30, fpm=fpm.fpm)
@assert result.converged result.message
```

### Banded Matrices

```julia
# For banded matrices stored in LAPACK upper-band format
n = 100
k = 1  # Number of super-diagonals
A_banded = full_to_banded(A, k)  # (k+1) x n band storage of the tridiagonal A

result = feast_banded(A_banded, k, (0.5, 1.5); subspace_size=30)
@assert result.converged result.message
```

### FEAST-Compatible Routine Names

The native Julia APIs are type-generic, but FEAST-style precision-prefixed names
are available for porting existing code:

```julia
# Double precision real symmetric dense FEAST
result = dfeast_syev!(A64, 0.5, 1.5, M0, fpm)

# Double precision complex Hermitian sparse FEAST
result = zfeast_hcsrev!(A_sparse_z, 0.5, 1.5, M0, fpm)

# Single precision real banded FEAST
result = sfeast_sbev!(A_band_f32, ka, 0.5f0, 1.5f0, M0, fpm)
```

Supported prefixes are `sfeast_*` (`Float32`), `dfeast_*` (`Float64`),
`cfeast_*` (`ComplexF32`), and `zfeast_*` (`ComplexF64`) for the dense,
sparse, banded, custom-contour, and polynomial FEAST families implemented by
FeastKit.

PFEAST-compatible aliases are available for the implemented parallel backends:

```julia
# Double precision sparse generalized PFEAST using Julia Distributed workers
result = pdfeast_scsrgv!(A_sparse, B_sparse, 0.5, 1.5, M0, fpm;
                         use_threads=false)

# The same alias can route to MPI when a communicator is supplied
result = pdfeast_scsrgv!(A_sparse, B_sparse, 0.5, 1.5, M0, fpm;
                         comm=MPI.COMM_WORLD)

# Complex Hermitian/general dense and sparse aliases also route to MPI with comm=
result = pzfeast_hegv!(A_dense_z, B_dense_z, 0.5, 1.5, M0, fpm;
                       comm=MPI.COMM_WORLD)
result = pzfeast_hcsrgv!(A_sparse_z, B_sparse_z, 0.5, 1.5, M0, fpm;
                         comm=MPI.COMM_WORLD)
result = pzifeast_gcsrgv!(A_general_z, B_general_z, center, radius, M0, fpm;
                          comm=MPI.COMM_WORLD, solver_tol=1e-10)
```

Supported PFEAST prefixes are `psfeast_*` (`Float32`) and `pdfeast_*`
(`Float64`) for real symmetric dense/sparse standard and generalized problems,
plus `pcfeast_*` (`ComplexF32`) and `pzfeast_*` (`ComplexF64`) for dense/sparse
complex Hermitian/general MPI paths. Iterative complex MPI aliases use
`pcifeast_*` and `pzifeast_*`. `psfeast_srci!` and `pdfeast_srci!` expose the
parallel RCI state machine.

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
# Matrix-free solves default to GMRES, which needs Krylov.jl loaded.
using Krylov
result = feast_matvec(A_mul!, B_mul!, n, (0.5, 1.5); M0=30)
@assert result.converged result.message
```

## Parallel Computing

FeastKit.jl supports parallel computation where each contour integration point is solved independently, leading to significant speedups for large problems.

Production backend support is intentionally explicit:

| Backend | Supported high-level problems |
| --- | --- |
| `:serial` | Real symmetric, complex Hermitian, and general problems through the serial solvers |
| `:threads` | Dense and sparse real symmetric standard/generalized problems |
| `:distributed` | Sparse real symmetric standard/generalized problems with Julia workers |
| `:mpi` | Real symmetric standard/generalized plus dense/sparse complex Hermitian/general problems with an initialized MPI communicator |
| `:auto` | Best available backend; unsupported selections fall back to serial |

Explicit backend requests fail fast when the backend is unavailable or does not
support the requested problem. Use `backend=:auto` when serial fallback is
acceptable.

### Multi-threaded Execution

```julia
# Prefer the explicit backend keyword for new code.
# Threaded backend supports dense and sparse real symmetric problems.
result = feast(A_sparse, (0.5, 1.5), subspace_size=30, backend=:threads)

# Let FeastKit choose a backend and fall back if needed.
result = feast(A_sparse, (0.5, 1.5), subspace_size=30, backend=:auto)
```

The older `parallel=:threads` keyword remains supported as an alias. Both dense
and sparse threaded solvers complete the conjugate contour before extracting
the eigenspace.

### Distributed Computing

```julia
using Distributed

# Add worker processes
addprocs(4)

# Use distributed computing for contour integration
result = feast(A_sparse, (0.5, 1.5), subspace_size=30, backend=:distributed)
```

### MPI Support for HPC Clusters

FeastKit.jl provides full MPI support for high-performance computing clusters:

```julia
using MPI
using FeastKit

# Initialize MPI (if not already done)
MPI.Init()

# Basic MPI FeastKit
comm = MPI.COMM_WORLD
result = feast(A, B, (0.5, 1.5), subspace_size=30, backend=:mpi, comm=comm)

# Explicit MPI interface with communicator
result = mpi_feast(A, B, (0.5, 1.5), M0=30, comm=comm)

# Hybrid MPI + threading (best for modern HPC)
result = feast_hybrid(A, B, (0.5, 1.5), M0=30,
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
using FeastKit
using LinearAlgebra

MPI.Init()

# Create distributed problem
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

if rank == 0
    println("Running FeastKit on $size MPI processes with $(Threads.nthreads()) threads each")
end

# Large eigenvalue problem. FEAST targets a slice of the spectrum: (1.0, 1.01)
# holds 19 of the 10,000 eigenvalues. Keep M0 about 1.5x that count.
n = 10000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = sparse(1.0I, n, n)

# Hybrid MPI + threading execution
result = feast_hybrid(A, B, (1.0, 1.01), M0=30,
                     comm=comm, use_threads_per_rank=true)

if rank == 0
    println("Found $(result.M) eigenvalues")
    println("Computation time measured on each rank")
end

MPI.Finalize()
```

### Parallel RCI Interface

For advanced users, a parallel RCI interface is available. The kernel asks
for contour solves (`PARALLEL_SOLVE`) and for products with `A` and `B`:

```julia
using FeastKit, LinearAlgebra

n = 100
A = diagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
B = Matrix{Float64}(I, n, n)
Emin, Emax, M0 = 0.5, 1.5, 30
fpm = feastinit().fpm

# Arguments: contour points, subspace size, use_parallel, use_threads
state = ParallelFeastState{Float64}(fpm[2], M0, true, true)

work = randn(n, M0)                 # initial trial subspace
workc = zeros(ComplexF64, n, M0)
Aq, Sq = zeros(M0, M0), zeros(M0, M0)
lambda, q, res = zeros(M0), zeros(n, M0), zeros(M0)

while true
    pfeast_srci!(state, n, work, workc, Aq, Sq, fpm,
                 Emin, Emax, M0, lambda, q, res)

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

@assert state.info == 0
println("Found $(state.mode) eigenvalues: ", lambda[1:state.mode])
```

### Automatic Backend Selection

```julia
# FeastKit automatically selects the best available backend and can fall back
result = feast(A, B, (0.5, 1.5), subspace_size=30, backend=:auto)

# Manual backend selection fails if that backend cannot run the problem
result = feast(A, B, (0.5, 1.5), subspace_size=30, backend=:mpi, comm=comm)
result = feast(A_sparse, B_sparse, (0.5, 1.5), subspace_size=30, backend=:threads)
result = feast(A, B, (0.5, 1.5), subspace_size=30, backend=:serial)
```

## Algorithm Overview

FeastKit uses contour integration in the complex plane to compute eigenvalues. The key steps are:

1. **Contour Definition**: Define an integration contour enclosing the desired eigenvalues
2. **Moment Computation**: Compute spectral projector moments using numerical integration  
3. **Subspace Extraction**: Extract eigenspace using computed moments
4. **Refinement**: Iteratively refine the solution until convergence

The algorithm is particularly effective for:
- Large sparse matrices
- Finding eigenvalues in specific intervals
- Parallel computation (via the original PFEAST routines)

## Result Structure

Symmetric/Hermitian solves return `FeastResult`; full-contour solves return
`FeastGeneralResult`. Both provide `values`, `vectors`, `converged`, and
`message`, plus a compact REPL display. `converged` is true only for `info == 0`;
small residuals with a saturated subspace do not establish completeness.
For the convenience wrappers, use
`eigvals_feast(A, interval; check=true, ...)` or
`eigen_feast(A, B, interval; check=true, ...)` to throw an error with recovery
guidance for any nonzero FEAST status. Both default to `check=false` for
compatibility, and then log a warning when the solve did not converge. Use
`feast` directly to inspect status and partial results. A search region that
contains no eigenvalues is a complete answer: `info == 0` with `M == 0`.
The original fields remain available:

```julia
struct FeastResult{T<:Real, VT}
    lambda::Vector{T}      # Computed eigenvalues
    q::Matrix{VT}          # Computed eigenvectors  
    M::Int                 # Number of eigenvalues found
    res::Vector{T}         # Relative residuals (see below)
    info::Int              # Exit status (0 = success)
    epsout::T              # Final residual
    loop::Int              # Number of refinement loops
end
```

Residuals are `‖A x - λ B x‖ / (‖B x‖ max(|λ|, σ))`, where `σ` is the spectral
scale of the pencil, measured on random probes before the first contour sweep.
Pairs with `|λ| ≥ σ` are judged relative to their eigenvalue and smaller ones by
backward error, and a change of units (rescaling `A`) leaves every residual and
convergence decision unchanged. Eigenvectors are normalized to unit 2-norm; for
generalized problems they are `B`-orthogonal but not `B`-normalized.

## Error Codes

- `info = 0`: Successful convergence (including an empty search region, `M == 0`)
- `info = 1`: Invalid matrix size N
- `info = 2`: Invalid or saturated search subspace; increase `subspace_size` or narrow the region
- `info = 3`: Invalid search interval (Emin >= Emax)
- `info = 4`: Invalid center/radius for complex problems
- `info = 5`: No convergence achieved
- `info = 6`: Memory allocation error
- `info = 7`: Internal error
- `info = 8`: LAPACK error
- `info = 9`: Invalid FeastKit parameters

### Porting from Fortran FEAST

The precision-prefixed aliases (`dfeast_syev!` and friends) take FEAST's
arguments, but results follow FeastKit's conventions. Check code that branches
on status codes or relies on the eigenvector normalization:

| Situation | Fortran FEAST `info` | FeastKit `info` |
| --- | --- | --- |
| Success | 0 | 0 |
| No eigenvalue in the search region | 1 (warning) | 0, with `M == 0` |
| No convergence within the loop budget | 2 | 5 |
| Subspace `M0` too small | 3 | 2 |
| Problem with `Emin`/`Emax` or `Emid`/`r` | 200 | 3 or 4 |
| Problem with `M0` | 201 | 2 |
| Problem with `N` | 202 | 1 |

Fortran FEAST returns `B`-orthonormal eigenvectors (`XᴴBX = I`); FeastKit
normalizes each eigenvector to unit 2-norm. To recover FEAST's scaling, divide
column `j` by `sqrt(real(x_j' * B * x_j))`.

## Performance Tips

1. **Choose appropriate M0**: Use about 1.5x the number of eigenvalues in the region; `feast_estimate_count` estimates that number, and `subspace_size=:auto` sizes the subspace for you
2. **Integration points**: More points (fpm[2]) improve accuracy but increase cost
3. **Sparse matrices**: Use sparse format for large problems with few non-zeros
4. **Initial guess**: Provide good initial guess when available (fpm[5] = 1)
5. **Parallel execution**: Use `backend=:auto` to automatically select the best backend
6. **HPC clusters**: Use `backend=:mpi` or `feast_hybrid()` for optimal cluster performance  
7. **Hybrid parallelism**: Combine MPI processes with threading for maximum performance
8. **Load balancing**: FeastKit automatically distributes contour points for optimal load balancing

## Scope

FeastKit focuses on the public APIs covered by the test suite rather than
mirroring every optional routine from the original FEAST library:

- Custom contour integration is supported for the FEAST interfaces documented here
- Advanced PFEAST routines outside the documented backend matrix are out of scope
- Matrix-free APIs use FeastKit's Julia-native operator interface

## References

1. E. Polizzi, "Density-matrix-based algorithm for solving eigenvalue problems", Physical Review B 79, 115112 (2009)
2. Feast official website: http://www.feast-solver.org

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the same terms as the original Feast library.
