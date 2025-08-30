# API Reference

Complete reference for all FeastKit.jl functions, types, and interfaces.

## Table of Contents

- [Main Interfaces](#main-interfaces)
- [Matrix-Free Interface](#matrix-free-interface) 
- [Contour Integration](#contour-integration)
- [Parallel Computing](#parallel-computing)
- [Types and Structures](#types-and-structures)
- [Utility Functions](#utility-functions)
- [Error Codes](#error-codes)

---

## Main Interfaces

### feast

Main Feast interface for symmetric/Hermitian eigenvalue problems.

```julia
feast(A, interval; M0=10, fpm=nothing, kwargs...)
feast(A, B, interval; M0=10, fpm=nothing, kwargs...)
```

**Arguments:**
- `A::AbstractMatrix`: System matrix (symmetric/Hermitian)
- `B::AbstractMatrix`: Mass matrix (optional, defaults to identity)
- `interval::Tuple{Real,Real}`: Search interval `(Emin, Emax)`

**Keyword Arguments:**
- `M0::Int=10`: Maximum number of eigenvalues to find
- `fpm::Vector{Int}`: Feast parameter array (auto-initialized if `nothing`)
- `parallel::Union{Bool,Symbol}=false`: Parallelization mode
- `use_threads::Bool=true`: Enable threading
- `comm`: MPI communicator (if using MPI)

**Returns:**
- `FeastResult`: Results structure with eigenvalues and eigenvectors

**Examples:**
```julia
# Standard eigenvalue problem
result = feast(A, (0.5, 1.5), M0=10)

# Generalized eigenvalue problem  
result = feast(A, B, (0.1, 0.8), M0=15)

# With custom parameters
fpm = zeros(Int, 64)
fpm[2] = 16  # 16 integration points
result = feast(A, (0, 1), M0=20, fpm=fpm)
```

### feast_general

Feast interface for general (non-Hermitian) eigenvalue problems using circular contours.

```julia
feast_general(A, B, center, radius; M0=10, fpm=nothing)
```

**Arguments:**
- `A::AbstractMatrix{Complex}`: System matrix
- `B::AbstractMatrix{Complex}`: Mass matrix  
- `center::Complex`: Center of circular search region
- `radius::Real`: Radius of circular search region

**Examples:**
```julia
# Complex eigenvalue problem
A = randn(ComplexF64, 100, 100)
B = Matrix{ComplexF64}(I, 100, 100)
result = feast_general(A, B, 1.0+0.5im, 2.0, M0=10)
```

### feast_banded

Feast interface for banded matrices.

```julia  
feast_banded(A, kla, interval; B=nothing, klb=0, M0=10, fpm=nothing)
```

**Arguments:**
- `A::Matrix`: Banded matrix in LAPACK banded format
- `kla::Int`: Number of super-diagonals of A
- `interval::Tuple{Real,Real}`: Search interval
- `B::Matrix`: Banded mass matrix (optional)
- `klb::Int`: Number of super-diagonals of B

---

## Matrix-Free Interface

### LinearOperator

Matrix-free operator type for large-scale problems.

```julia
LinearOperator{T}(A_mul!, size; kwargs...)
```

**Arguments:**
- `A_mul!::Function`: Function `(y, x) -> y = A*x`
- `size::Tuple{Int,Int}`: Matrix dimensions

**Keyword Arguments:**
- `issymmetric::Bool=false`: Matrix is symmetric
- `ishermitian::Bool=false`: Matrix is Hermitian
- `isposdef::Bool=false`: Matrix is positive definite
- `At_mul!::Function`: Function for `A'*x` (optional)
- `Ac_mul!::Function`: Function for `A†*x` (optional)

**Examples:**
```julia
# Define matrix-vector multiplication
function A_mul!(y, x)
    # Tridiagonal: [-1 2 -1] stencil
    n = length(x)
    y[1] = 2*x[1] - x[2]
    for i in 2:n-1
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    end
    y[n] = -x[n-1] + 2*x[n]
end

# Create operator
A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)

# Use with Feast
result = feast(A_op, (0.5, 1.5), M0=10, solver=:cg)
```

```@docs
FeastKit.LinearOperator
FeastKit.MatrixVecFunction
FeastKit.MatrixFreeOperator
FeastKit.feast
FeastKit.feast_general
FeastKit.feast_polynomial
FeastKit.feast_matfree_srci!
FeastKit.feast_matfree_grci!
FeastKit.allocate_matfree_workspace
FeastKit.create_iterative_solver
FeastKit.create_direct_solver
```

### MatrixVecFunction

Alternative matrix-free operator with data storage.

```julia
MatrixVecFunction{T}(mul!, size; kwargs...)
```

**Arguments:**
- `mul!::Function`: Function `(y, op, x) -> y = op*x`  
- `size::Tuple{Int,Int}`: Matrix dimensions

### feast (Matrix-Free)

Matrix-free Feast interfaces.

```julia
# Symmetric/Hermitian problems
feast(A_op::MatrixFreeOperator, interval; kwargs...)
feast(A_op::MatrixFreeOperator, B_op::MatrixFreeOperator, interval; kwargs...)

# General problems  
feast_general(A_op::MatrixFreeOperator{Complex}, B_op, center, radius; kwargs...)
```

**Additional Keyword Arguments:**
- `solver::Union{Symbol,Function}=:gmres`: Linear solver type
- `solver_opts::NamedTuple`: Solver-specific options
- `tol::Real=1e-12`: Convergence tolerance
- `maxiter::Int=20`: Maximum refinement iterations

**Solvers:**
- `:cg`: Conjugate Gradient (symmetric positive definite)
- `:gmres`: Generalized Minimal Residual (general)
- `:bicgstab`: BiConjugate Gradient Stabilized
- Custom function: `(Y, z, X) -> solve (z*B - A)*Y = X`

### create_iterative_solver

Create iterative linear solver for matrix-free Feast.

```julia
create_iterative_solver(A_op, B_op, solver_type=:gmres; kwargs...)
```

**Keyword Arguments:**
- `rtol::Float64=1e-6`: Relative tolerance
- `maxiter::Int=1000`: Maximum iterations  
- `restart::Int=30`: GMRES restart parameter
- `preconditioner`: Preconditioner (optional)

---

## Contour Integration

### feast_contour

Generate elliptical integration contour for real intervals.

```julia
feast_contour(Emin, Emax, fpm)
```

**Returns:**
- `FeastContour`: Contour with nodes `Zne` and weights `Wne`

### feast_contour_expert  

Advanced contour generation with full control.

```julia
feast_contour_expert(Emin, Emax, ne, integration_type=0, ellipse_ratio=100)
```

**Arguments:**
- `Emin, Emax::Real`: Interval bounds
- `ne::Int`: Number of integration points
- `integration_type::Int`: 0=Gauss-Legendre, 1=Trapezoidal, 2=Zolotarev
- `ellipse_ratio::Int`: Aspect ratio a/b × 100 (100 = circle)

**Examples:**
```julia
# High-accuracy Gauss-Legendre with 16 points
contour = feast_contour_expert(-1.0, 1.0, 16, 0, 100)

# Zolotarev integration (optimal for ellipses)  
contour = feast_contour_expert(0.0, 2.0, 12, 2, 100)

# Flat ellipse (aspect ratio 0.5)
contour = feast_contour_expert(-1.0, 1.0, 10, 0, 50)
```

```@docs
FeastKit.feast_contour_expert
FeastKit.feast_contour_custom_weights!
FeastKit.feast_rational_expert
```

### feast_gcontour

Generate circular contour for general problems.

```julia
feast_gcontour(center, radius, fpm)
```

### feast_contour_custom_weights!

Custom contour with user-provided nodes and weights.

```julia
feast_contour_custom_weights!(Zne, Wne)
```

**Arguments:**
- `Zne::Vector{Complex}`: Integration nodes
- `Wne::Vector{Complex}`: Integration weights (modified in-place)

### feast_rational_expert

Evaluate rational function using custom contour.

```julia
feast_rational_expert(Zne, Wne, lambda)
```

**Arguments:**
- `Zne::Vector{Complex}`: Integration nodes
- `Wne::Vector{Complex}`: Integration weights  
- `lambda::Vector`: Eigenvalues to evaluate

**Returns:**
- `Vector`: Rational function values (≈1 inside contour, ≈0 outside)

---

## Parallel Computing

### feast (Parallel)

Parallel Feast interfaces.

```julia
feast(A, interval; parallel=:mpi, comm=MPI.COMM_WORLD, kwargs...)
```

**Parallel Options:**
- `parallel=false`: Serial execution
- `parallel=:threads`: Shared-memory threading  
- `parallel=:mpi`: MPI parallelization
- `parallel=:hybrid`: MPI + threads

### mpi_feast

Direct MPI interface.

```julia
mpi_feast(A, B, interval, comm; kwargs...)
```

### ParallelFeastState

State structure for parallel Feast calculations.

```julia
state = ParallelFeastState(comm, A, B, interval, M0)
result = feast_parallel!(state)
```

---

## Types and Structures

### FeastResult

Result structure returned by Feast calculations.

```julia
struct FeastResult{T<:Real, VT}
    lambda::Vector{T}    # Eigenvalues found
    q::Matrix{VT}        # Eigenvectors (columns)
    M::Int               # Number of eigenvalues found
    res::Vector{T}       # Individual residuals
    info::Int            # Status code (0 = success)
    epsout::T           # Final residual
    loop::Int           # Refinement iterations used
end
```

**Access patterns:**
```julia
result = feast(A, (0, 1), M0=10)

eigenvalues = result.lambda[1:result.M]
eigenvectors = result.q[:, 1:result.M]  
success = (result.info == 0)
```

### FeastContour

Integration contour structure.

```julia
struct FeastContour{T<:Real}
    Zne::Vector{Complex{T}}  # Integration nodes
    Wne::Vector{Complex{T}}  # Integration weights
end
```

### FeastParameters

Feast parameter structure.

```julia
struct FeastParameters
    fpm::Vector{Int}  # 64-element parameter array
end
```

### MatrixFreeOperator

Abstract base type for matrix-free operators.

```julia
abstract type MatrixFreeOperator{T} end
```

**Concrete types:**
- `LinearOperator{T}`
- `MatrixVecFunction{T}`

---

## Utility Functions

### feastinit!

Initialize Feast parameter array.

```julia
feastinit!(fpm::Vector{Int})
```

Sets default values for all Feast parameters.

### feastdefault!

Reset Feast parameters to defaults.

```julia
feastdefault!(fpm::Vector{Int})
```

### feast_set_defaults!

Set common Feast parameters with user-friendly names.

```julia
feast_set_defaults!(fpm; print_level=1, integration_points=8, 
                   tolerance_exp=12, max_refinement=20)
```

### feast_validate_interval

Validate search interval and estimate eigenvalue bounds.

```julia
feast_validate_interval(A, interval)
```

**Returns:**
- `Tuple{Real,Real}`: Estimated eigenvalue bounds using Gershgorin circles

### feast_summary

Print summary of Feast results.

```julia
feast_summary(result::FeastResult)
```

### eigvals_feast

Extract only eigenvalues from Feast calculation.

```julia
eigvals_feast(A, interval; kwargs...)
```

**Returns:**
- `Vector`: Eigenvalues found

### eigen_feast  

Return Eigen object from Feast calculation.

```julia
eigen_feast(A, interval; kwargs...)
```

**Returns:**
- `Eigen`: LinearAlgebra.Eigen object with `values` and `vectors`

### allocate_matfree_workspace

Allocate workspace for matrix-free operations.

```julia
allocate_matfree_workspace(T, N, M0)
```

**Arguments:**
- `T::Type`: Element type (Float64, ComplexF64, etc.)
- `N::Int`: Matrix size
- `M0::Int`: Maximum eigenvalues

---

## Error Codes

Feast functions return status codes in `result.info`:

| Code | Name | Description |
|------|------|-------------|
| 0 | `Feast_SUCCESS` | Success |
| 1 | `Feast_ERROR_N` | Invalid matrix size N |
| 2 | `Feast_ERROR_M0` | Invalid M0 parameter |
| 3 | `Feast_ERROR_EMIN_EMAX` | Invalid search interval |
| 4 | `Feast_ERROR_EMID_R` | Invalid center/radius for complex problems |
| 5 | `Feast_ERROR_NO_CONVERGENCE` | No convergence achieved |
| 6 | `Feast_ERROR_MEMORY` | Memory allocation failed |
| 7 | `Feast_ERROR_INTERNAL` | Internal error |
| 8 | `Feast_ERROR_LAPACK` | Linear algebra error |
| 9 | `Feast_ERROR_FPM` | Invalid Feast parameters |

**Error handling:**
```julia
result = feast(A, interval)

if result.info != 0
    error_name = ["Feast_SUCCESS", "Feast_ERROR_N", "Feast_ERROR_M0", 
                  "Feast_ERROR_EMIN_EMAX", "Feast_ERROR_EMID_R",
                  "Feast_ERROR_NO_CONVERGENCE", "Feast_ERROR_MEMORY",
                  "Feast_ERROR_INTERNAL", "Feast_ERROR_LAPACK", 
                  "Feast_ERROR_FPM"][result.info + 1]
    @warn "Feast failed with $error_name"
end
```

---

## Parameter Reference

The `fpm` parameter array controls Feast behavior:

| Index | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| `fpm[1]` | Print level | 1 | 0=silent, 1=summary, 2=detailed |
| `fpm[2]` | Integration points | 8 | Number of contour points |
| `fpm[3]` | Tolerance exponent | 12 | Convergence: 10^(-fpm[3]) |
| `fpm[4]` | Max iterations | 20 | Maximum refinement loops |
| `fpm[5]` | Initial subspace | 0 | 0=random, 1=user-provided |
| `fpm[16]` | Integration type | 0 | 0=Gauss, 1=Trapezoidal, 2=Zolotarev |
| `fpm[18]` | Ellipse ratio | 100 | Aspect ratio × 100 |

**Setting parameters:**
```julia
fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[1] = 2      # Detailed output
fpm[2] = 16     # 16 integration points  
fpm[3] = 14     # High precision (10^-14)
fpm[16] = 2     # Zolotarev integration

result = feast(A, interval, M0=10, fpm=fpm)
```

---

## Performance Guidelines

### Memory Usage

| Problem Type | Memory per Eigenvalue | Total Memory |
|--------------|----------------------|--------------|
| Dense N×N | ~16N bytes | ~16NM bytes |
| Sparse N×N | ~16N bytes | ~16NM bytes |  
| Matrix-free N×N | ~16N bytes | ~16NM bytes |

### Recommended Parameters

| Problem Size | M0 | Integration Points | Tolerance |
|--------------|----|--------------------|-----------|
| N < 1,000 | 10-20 | 8-12 | 1e-12 |
| 1,000 < N < 10,000 | 10-30 | 8-16 | 1e-10 |  
| 10,000 < N < 100,000 | 10-50 | 12-20 | 1e-8 |
| N > 100,000 | 10-100 | 16-32 | 1e-6 |

### Solver Selection

| Problem Type | Recommended Solver | Options |
|--------------|-------------------|---------|
| Symmetric positive definite | `:cg` | `rtol=1e-8` |
| Symmetric indefinite | `:gmres` | `restart=30` |
| General non-symmetric | `:gmres` | `restart=50, rtol=1e-6` |
| Well-conditioned | `:bicgstab` | `l=2` |

---

<div align="center">
  <p><strong>Complete API documentation for FeastKit.jl</strong></p>
  <p><a href="getting_started.html">← Getting Started</a> | <a href="examples.html">Examples →</a></p>
</div>
