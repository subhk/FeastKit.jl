# API Reference

Reference for the public interfaces, supported controls, and result conventions.

Blocks that show a bare call with no surrounding setup are **signatures**, not
runnable snippets: `A`, `B`, `interval` and `fpm` are supplied by you. Blocks labeled `@example` in the source execute during the build. Other
blocks are signatures or templates unless they include their own setup.

```@contents
Pages = ["api_reference.md"]
Depth = 2
```

---

## Main Interfaces

### feast

Main FeastKit interface for symmetric/Hermitian eigenvalue problems.

For assembled matrices, real/complex dispatch follows the element type. The
interval overload checks symmetry/Hermitian structure and rejects incompatible
input; it does not switch to a general solve automatically. See
[Matrix types: detected or declared?](@ref matrix-properties) for all four
matrix cases, Julia wrappers, and matrix-free declarations.

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
- `subspace_size`: Positive width (alias for `M0`) or `:auto` for count estimation and bounded growth
- `max_subspace_size`: Maximum width for `subspace_size=:auto` (default: matrix dimension)
- `initial_subspace`: Optional `N × k` seed; supported by serial assembled and matrix-free solves
- `mixed_precision`: Opt-in Float32 correction solves with Float64 residuals; dense Float64/ComplexF64, serial, direct solver only
- `tol`: Outer residual tolerance (default `1e-12`), rounded down to a decimal power in `[1e-16, 1]`; Float32 retains a `sqrt(eps(Float32))` floor
- `maxiter`: Maximum outer refinement iterations (default 20)
- `quadrature_points`: Half-contour node count (default 8); subject to the integration rule's valid counts
- `solver::Symbol=:direct`: Shifted linear solver, `:direct` or `:gmres` (`using Krylov` required for GMRES)
- `solver_opts::NamedTuple`: Inner controls `(rtol=..., maxiter=..., restart=...)`; only for iterative solves
- `fpm`: `Vector{Int}`, `FeastParameters`, or `nothing` (initialized automatically)
- `backend::Symbol=:serial`: Execution backend (`:serial`, `:auto`, `:threads`, `:distributed`, `:mpi`)
- `parallel::Union{Bool,Symbol}`: Legacy alias for `backend`
- `strict_backend::Bool=false`: Compatibility switch for legacy `parallel` requests; explicit `backend` requests already fail if unavailable or unsupported
- `use_threads::Bool=true`: Enable threading
- `comm`: MPI communicator (if using MPI)

**Returns:**
- `FeastResult`: Results structure with eigenvalues and eigenvectors

`result.values` and `result.vectors` alias `lambda` and `q`. Check
`result.converged` (equivalent to `info == 0`); `result.message` explains the
status and gives recovery guidance. The REPL display includes convergence,
residual, refinement count, and a short eigenvalue preview.

Named options that conflict with explicit `fpm` entries raise `ArgumentError`.
Named `fpm` overrides operate on a copy. Supplying both `M0` and `subspace_size`
is allowed only when they agree; `:auto` cannot be combined with `M0`.
`mixed_precision=true` corresponds to `fpm[42]=1`; `fpm[42]` now defaults to
`0`, and unsupported drivers reject `1` instead of silently ignoring it.
See [Performance Guide](performance.md) for sizing costs, warm-start checks,
adaptive inner tolerances, and mixed-precision fallback behavior.

Serial assembled drivers support both solvers. Complex MPI drivers also support
GMRES; real MPI, threaded, and distributed drivers support direct solves only.
An explicit unsupported backend raises an error; `backend=:auto` permits fallback.

Full contour overloads use the general solver and return `FeastGeneralResult`:

```julia
feast(A, contour::FeastKit.FeastContour; subspace_size=10, kwargs...)
feast(A, B, contour::FeastKit.FeastContour; subspace_size=10, kwargs...)
```

Pass a full contour from `feast_circle`, `feast_ellipse`, or `feast_rectangle`.
These overloads manage registration automatically and leave `fpm` unchanged.
Specify the node count in the contour constructor; `quadrature_points`, if
provided, must match. For matrix-free full-contour solves, use complex operators.

**Examples:**
```@example apifeast
using FeastKit, LinearAlgebra

n = 100
A = Matrix(SymTridiagonal(2.0 * ones(n), -1.0 * ones(n - 1)))
B = Matrix{Float64}(I, n, n)

# Standard eigenvalue problem. Size M0 to the number of eigenvalues the
# interval holds -- here 2 - 2cos(kπ/(n+1)) puts 7 of them below 0.05.
result = feast(A, (0.0, 0.05), M0=10)
@assert result.converged && result.M == 7

# Generalized eigenvalue problem
result = feast(A, B, (0.0, 0.05), M0=10)
@assert result.converged && result.M == 7

# With custom parameters. Always feastinit! before setting entries: on a bare
# zeros(Int, 64) the zeros read as user-supplied values, not as "unset".
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[2] = 16  # 16 integration points
result = feast(A, (0, 0.05), M0=10, fpm=fpm)
@assert result.converged && result.M == 7

result.info, result.M
```

### feast_general

FeastKit interface for general (non-Hermitian) eigenvalue problems using circular contours.

Accepts the same named controls as `feast`. Here `quadrature_points` sets the
full-contour count (`fpm[8]`, default 16), and the result is `FeastGeneralResult`.

```julia
feast_general(A, B, center, radius; M0=10, fpm=nothing)
```

**Arguments:**
- `A::AbstractMatrix`: Real or complex system matrix
- `B::AbstractMatrix`: Real or complex mass matrix; optional for a standard problem
- `center::Complex`: Center of circular search region
- `radius::Real`: Radius of circular search region

This high-level interface promotes the matrices to a common complex
floating-point type, including when the supplied matrices are real.

**Examples:**
```@example api_general
using FeastKit, LinearAlgebra
A = Matrix(Diagonal(ComplexF64[0.5+0.1im, 1.0+0.2im, 4.0]))
result = feast_general(A, 1.0+0.5im, 2.0; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 1.0+0.2im]; atol=1e-9)
result.values
```

### feast_banded

FeastKit interface for banded matrices.

Accepts `subspace_size` (including `:auto`), `max_subspace_size`,
`initial_subspace`, `mixed_precision`, `tol`, `maxiter`, `quadrature_points`, `solver`, and
`solver_opts` with the same meaning as the assembled interval interface.

```julia  
feast_banded(A, kla, interval; B=nothing, klb=0, M0=10, fpm=nothing)
```

**Arguments:**
- `A::Matrix`: Banded matrix in LAPACK banded format
- `kla::Int`: Number of super-diagonals of A
- `interval::Tuple{Real,Real}`: Search interval
- `B::Matrix`: Banded mass matrix (optional)
- `klb::Int`: Number of super-diagonals of B

Use `full_to_banded` for symmetric/Hermitian or upper-stored banded data.
Use `full_to_general_banded` for fully general non-Hermitian banded matrices
where both lower and upper bands must be preserved.
Direct banded solvers use LAPACK banded factorizations; `solver=:gmres` keeps
the same compact storage and applies shifted systems through banded matvecs.

### FEAST-compatible precision aliases

FeastKit's native solver names are type-generic. For users porting FEAST code,
precision-prefixed aliases are also available and forward to the same tested
implementations.

| FEAST prefix | Julia element type | Examples |
| --- | --- | --- |
| `sfeast_*` | `Float32` | `sfeast_syev!`, `sfeast_scsrgv!`, `sfeast_sbev!` |
| `dfeast_*` | `Float64` | `dfeast_syev!`, `dfeast_scsrgv!`, `dfeast_sbev!` |
| `sifeast_*` | `Float32` iterative polynomial | `sifeast_srcipev!`, `sifeast_scsrpev!` |
| `difeast_*` | `Float64` iterative | `difeast_sygv!`, `difeast_scsrgv!`, `difeast_srcipev!` |
| `cfeast_*` | `ComplexF32` | `cfeast_heev!`, `cfeast_gcsrgv!`, `cfeast_hbev!` |
| `zfeast_*` | `ComplexF64` | `zfeast_heev!`, `zfeast_gcsrgv!`, `zfeast_hbev!` |
| `cifeast_*` | `ComplexF32` iterative polynomial | `cifeast_grcipev!`, `cifeast_gcsrpev!` |
| `zifeast_*` | `ComplexF64` iterative | `zifeast_gegv!`, `zifeast_hcsrgv!`, `zifeast_grcipev!` |
| `psfeast_*` | `Float32` parallel real symmetric | `psfeast_syev!`, `psfeast_scsrgv!`, `psfeast_srci!` |
| `pdfeast_*` | `Float64` parallel real symmetric | `pdfeast_syev!`, `pdfeast_scsrgv!`, `pdfeast_srci!` |
| `pcfeast_*` | `ComplexF32` parallel Hermitian/general | `pcfeast_hegv!`, `pcfeast_hcsrgv!`, `pcfeast_gegv!` |
| `pzfeast_*` | `ComplexF64` parallel Hermitian/general | `pzfeast_hegv!`, `pzfeast_hcsrgv!`, `pzfeast_gegv!` |
| `pcifeast_*` | `ComplexF32` iterative parallel Hermitian/general | `pcifeast_hegv!`, `pcifeast_gcsrgv!` |
| `pzifeast_*` | `ComplexF64` iterative parallel Hermitian/general | `pzifeast_hegv!`, `pzifeast_gcsrgv!` |

The aliases cover dense, sparse CSC/CSR-style, banded, custom-contour `x`
variants, and polynomial FEAST entry points where the corresponding generic
FeastKit function exists.

The `psfeast_*` and `pdfeast_*` aliases cover the implemented real symmetric
PFEAST paths. Dense and sparse standard aliases construct the identity mass
matrix and call the corresponding generalized parallel solver. Passing
`comm=MPI.COMM_WORLD` routes supported sparse/dense real symmetric aliases to
the MPI kernels. Complex `pc/pz` and iterative `pci/pzi` aliases route dense
and sparse Hermitian/general problems to MPI when a communicator is supplied.

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
- `At_mul!`: Function `(y, x)` for `transpose(A)*x` (default `nothing`)
- `Ac_mul!`: Function `(y, x)` for `adjoint(A)*x` (default `nothing`)
- `solve!`: Stored solve callback (default `nothing`); pass a shifted solve
  explicitly as `solver=callback` when calling `feast`.

**Examples:**
```@example apiop
using FeastKit, LinearAlgebra

n = 200

# Define matrix-vector multiplication
function A_mul!(y, x)
    # Tridiagonal: [-1 2 -1] stencil
    y[1] = 2*x[1] - x[2]
    for i in 2:n-1
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    end
    y[n] = -x[n-1] + 2*x[n]
end

# Create operator
A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)

# Supply the shifted solve: z*I - A is tridiagonal, so a Thomas sweep is exact
# and O(n). Without `solver=` this falls back to unpreconditioned GMRES.
function tridiagonal_solve!(Y, z, X)
    d0 = ComplexF64(z) - 2
    c = Vector{ComplexF64}(undef, n - 1)
    d = Vector{ComplexF64}(undef, n)
    for j in axes(X, 2)
        c[1] = 1 / d0
        d[1] = X[1, j] / d0
        for i in 2:n
            m = d0 - c[i - 1]
            i < n && (c[i] = 1 / m)
            d[i] = (X[i, j] - d[i - 1]) / m
        end
        Y[n, j] = d[n]
        for i in (n - 1):-1:1
            Y[i, j] = d[i] - c[i] * Y[i + 1, j]
        end
    end
    return Y
end

result = feast(A_op, (0.0, 0.0025), M0=12, solver=tridiagonal_solve!)
@assert result.converged && result.M == 3
@assert isapprox(result.values, [2-2cos(k*π/(n+1)) for k in 1:3]; atol=1e-9)
result.info, result.M
```

```@docs
FeastKit.LinearOperator
FeastKit.FeastLinearOperator
FeastKit.feast_matvec
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
FeastKit.validate_companion_matrices
```

### MatrixVecFunction

Alternative callback wrapper. Capture payload data in the callback; the operator has no `data` field.

```julia
MatrixVecFunction{T}(mul!, size; kwargs...)
```

**Arguments:**
- `mul!::Function`: Function `(y, op, x) -> y = op*x`  
- `size::Tuple{Int,Int}`: Matrix dimensions

### feast (Matrix-Free)

Matrix-free FeastKit interfaces.

```julia
# Real symmetric operator problems
feast(A_op::MatrixFreeOperator, interval; kwargs...)
feast(A_op::MatrixFreeOperator, B_op::MatrixFreeOperator, interval; kwargs...)

# General problems  
feast_general(A_op::MatrixFreeOperator{<:Complex}, B_op, center, radius; kwargs...)
```

**Additional Keyword Arguments:**
- `solver::Union{Symbol,Function}=:gmres`: Linear solver type
- `solver_opts::NamedTuple`: Solver-specific options
- `tol::Real=1e-12`: Convergence tolerance
- `maxiter::Int=20`: Maximum refinement iterations

**Solvers:**
- `:gmres`: Generalized Minimal Residual (recommended)
- `:bicgstab`: BiConjugate Gradient Stabilized
- Custom function: `(Y, z, X) -> solve (z*B - A)*Y = X`

FEAST shifted systems use complex contour points, so `:cg` is rejected by
`create_iterative_solver`; use `:gmres` or `:bicgstab`.

### create_iterative_solver

Create iterative linear solver for matrix-free FeastKit.

```julia
create_iterative_solver(A_op, B_op, solver_type=:gmres; kwargs...)
```

**Keyword Arguments:**
- `rtol::Float64=1e-6`: Relative tolerance
- `maxiter::Int=1000`: Maximum iterations  
- `restart::Int=30`: GMRES restart parameter
- `preconditioner`: Optional left inverse-action operator. It must support
  `mul!(y, P, x)` on complex vectors, applying an approximate inverse rather
  than the matrix to be inverted. `nothing` disables preconditioning.

Both GMRES and BiCGSTAB use this operator. Their inner absolute tolerance is
zero; `rtol` is relative to the initial (preconditioned, when applicable)
residual, preserving the tolerance under uniform pencil scaling.
Each shifted system and its right-hand side are scaled together by the RHS
norm to avoid absolute breakdown thresholds on tiny pencils; the inverse-action
preconditioner is scaled consistently. Zero right-hand sides return zero.
If BiCGSTAB breaks down and produces a nonfinite solution, that right-hand
side is retried with GMRES using the same preconditioner and tolerances.

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
- `ellipse_ratio::Int`: Vertical/horizontal semiaxis ratio × 100
  (100 = circle, 50 = half as tall as it is wide)

**Examples:**
```@example apicontour
using FeastKit

# High-accuracy Gauss-Legendre with 16 points
contour = feast_contour_expert(-1.0, 1.0, 16, 0, 100)

# Zolotarev integration (optimal for ellipses)
contour = feast_contour_expert(0.0, 2.0, 12, 2, 100)

# Flat ellipse (aspect ratio 0.5). Integer bounds work too.
contour = feast_contour_expert(-1, 1, 10, 0, 50)

length(contour.Zne)
```

```@docs
FeastKit.feast_contour_expert
FeastKit.feast_contour_custom_weights!
FeastKit.feast_rational_expert
FeastKit.feast_rational
FeastKit.feast_rationalx
FeastKit.feast_grational
FeastKit.feast_grationalx
FeastKit.feast_clear_all_contours!
```

### Standard Contour Shapes

For standard shapes without a parameter vector, use these constructors. Each
returns a full `FeastContour` with normalized quadrature weights. See
[Built-in Circle, Ellipse, and Box](custom_contours.md#Built-in-Circle,-Ellipse,-and-Box)
for complete solves.

```@docs
feast_circle
feast_ellipse
feast_rectangle
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

Parallel FeastKit interfaces.

```julia
feast(A, interval; backend=:mpi, comm=MPI.COMM_WORLD, kwargs...)
```

**Parallel Options:**
- `backend=:serial`: Serial execution
- `backend=:auto`: Select an available backend
- `backend=:threads`: Shared-memory threading
- `backend=:distributed`: Distributed workers
- `backend=:mpi`: MPI parallelization

`parallel=:threads` and similar values remain available for compatibility.
Use `strict_backend=true` when fallback to serial should be treated as an error.

### mpi_feast

Direct MPI interface.

```julia
mpi_feast(A, B, interval; comm=comm, kwargs...)
```

### ParallelFeastState

State for the threaded/distributed reverse-communication API `pfeast_srci!`,
not an MPI communicator or a standalone solver. Construct it with the number
of contour points and the subspace size. For an automatic solve, use
`feast_parallel`:

```@example api_parallel_state
using FeastKit, LinearAlgebra
A = Matrix(Diagonal(collect(1.0:40.0)))
B = Matrix{Float64}(I, 40, 40)
interval = (0.5, 3.5)
fpm = feastinit().fpm
fpm[2] = 8  # Set the point count before constructing manual RCI state
M0 = 10
state = ParallelFeastState{Float64}(fpm[2], M0, true, true)
# Manual RCI callers pass state to pfeast_srci! and service its requested jobs.
# Automatic solve (manages its own state):
result = feast_parallel(A, B, interval; M0=M0, fpm=fpm)
@assert result.converged && result.M == 3
```

```@docs
FeastKit.pfeast_show_distribution
FeastKit.feast_parallel_comparison
FeastKit.MPIFeastState
```

---

## Types and Structures

### FeastResult

Result structure returned by FeastKit calculations.

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

For linear eigenproblems, each residual is
`norm(A*q - λ*B*q) / norm(B*q) / max(abs(λ), σ)`, with `B = I` for standard
problems. `σ` is the spectral scale of the pencil, `‖A R‖ / ‖B R‖` for a block
of random probes `R`, measured once before the first contour sweep. Pairs with
`|λ| ≥ σ` are thus judged relative to their eigenvalue and smaller ones by
backward error. Scaling both matrices by a common factor, or `A` alone (a
change of units), leaves every residual and convergence decision unchanged. A
zero `B*q` yields an infinite residual.

**Access patterns:**
```@example apiresult
using FeastKit, LinearAlgebra

A = Matrix(SymTridiagonal(2.0 * ones(100), -1.0 * ones(99)))
result = feast(A, (0, 0.05), M0=10)   # interval endpoints may be any Real

eigenvalues = result.lambda[1:result.M]
eigenvectors = result.q[:, 1:result.M]
success = (result.info == 0)
```

```@docs
FeastKit.FeastResult
FeastKit.FeastGeneralResult
```

### FeastContour

Integration contour structure (`FeastKit.FeastContour`; not exported). The
two-vector constructor remains supported. Rectangles store their actual corners
separately from quadrature nodes for eigenvalue selection.

```julia
struct FeastContour{T<:Real}
    Zne::Vector{Complex{T}}  # Integration nodes
    Wne::Vector{Complex{T}}  # Integration weights
    vertices::Union{Nothing,Vector{Complex{T}}}  # Optional polygon boundary
end
```

### FeastParameters

FeastKit parameter structure.

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

### RCI State Types

Explicit state objects for Reverse Communication Interface (RCI) kernels. These must be created once before the RCI loop and reused across all iterations — the
kernel keeps the contour, the trial subspace, and the outstanding sub-phase in
them. Passing a fresh state mid-loop now raises an `ArgumentError` instead of
silently restarting and returning `M = 0`.

#### Jobs a caller must handle

Once, right after initialization, the kernels issue `Feast_RCI_MULT_A` then
`Feast_RCI_MULT_B` on `q[:, 1:mode[]]`, which then holds random probes; they
measure the spectral scale that residuals are relative to. After that, one
refinement loop of `feast_srci!` / `feast_hrci!` issues:

1. `Feast_RCI_FACTORIZE` / `Feast_RCI_SOLVE`, once per contour point — factorize
   `Ze*B - A` and solve it against `B * work`.
2. `Feast_RCI_MULT_A` then `Feast_RCI_MULT_B` — multiply `q[:, 1:mode[]]` to
   build the reduced Rayleigh–Ritz pencil.
3. `Feast_RCI_MULT_A` then `Feast_RCI_MULT_B` again — this time on the Ritz
   vectors, so the kernel can form the true generalized residual
   `‖A q - λ B q‖`.
4. `Feast_RCI_DONE`.

`Feast_RCI_MULT_B` is required. Handling only `MULT_A` computes `‖A q - λ q‖`,
which is wrong for any problem with `B ≠ I`. The kernel tracks which of the two
`MULT_A`/`MULT_B` pairs is outstanding, so a caller simply multiplies the first
`mode[]` columns of `q` every time.

On exit, `info` is `Feast_SUCCESS` only when `epsout` met the tolerance and
the subspace was not saturated (except a complete full-space solve);
exhausting `fpm[4]` refinement loops reports `Feast_ERROR_NO_CONVERGENCE`. A
region with no eigenvalues ends with `Feast_SUCCESS` and `mode[] == 0`. A loop
can also end directly after a contour sweep, when the sweep shows that every
unconverged Ritz pair is spurious; `Feast_RCI_DONE` then reports the converged
pairs, and callers need no special handling.

The `ifeast_srci!`, `ifeast_hrci!`, and `ifeast_grci!` entry points expose
IFEAST-compatible RCI names. They are solver-neutral wrappers around the base
RCI kernels; callers still provide the shifted-system solve when the RCI job
code requests it, using either direct or iterative linear solvers.

```@docs
FeastKit.ifeast_srci!
FeastKit.ifeast_hrci!
FeastKit.ifeast_grci!
```

```@docs
FeastKit.FeastSRCIState
FeastKit.FeastHRCIState
FeastKit.FeastGRCIState
FeastKit.FeastPolyRCIState
```

---

## Utility Functions

### Estimating the enclosed eigenvalue count

Use this estimate to help choose `M0`, leaving a safety margin; a stochastic
estimate is not a proof of the exact count.

```@docs
FeastKit.feast_estimate_count
```

### feastinit!

Initialize FeastKit parameter array.

```julia
feastinit!(fpm::Vector{Int})
```

Sets all 64 entries to the unset sentinel `-111`. Solvers apply defaults later.
Call this before assigning custom entries; `feastinit()` returns a wrapper
with the same unset entries.

### feastdefault!

Validate configured entries and fill unset entries with defaults. Existing
explicit settings are preserved; call `feastinit!` first to reset all settings.

```julia
feastdefault!(fpm::Vector{Int})
```

### feast_set_defaults!

Set common FeastKit parameters with user-friendly names.

```julia
feast_set_defaults!(fpm; print_level=1, integration_points=8, 
                   tolerance_exp=12, max_refinement=20)
```

### feast_validate_interval

Validate search interval and estimate eigenvalue bounds.

```julia
feast_validate_interval(A, interval)
feast_validate_interval(A, B, interval)   # generalized problem A x = λ B x
```

**Returns:**
- `Tuple{Real,Real}`: Estimated eigenvalue bounds using Gershgorin circles

```@docs
FeastKit.feast_validate_interval(::AbstractMatrix, ::AbstractMatrix, ::Tuple{Real,Real})
```

### feast_summary

Print summary of FeastKit results.

```julia
feast_summary(result::FeastResult)
feast_summary(result::FeastGeneralResult)
feast_summary(io::IO, result)            # write to any IO instead of stdout
```

### eigvals_feast

```@docs
FeastKit.eigvals_feast
```

Extract only eigenvalues from FeastKit calculation.

```julia
eigvals_feast(A, interval; check=false, kwargs...)
eigvals_feast(A, B, interval; check=false, kwargs...)
```

Set `check=true` to throw an `ErrorException` for any nonzero FEAST status,
including a saturated subspace even when residuals are small. The error
includes the status code and recovery guidance. The default `check=false`
preserves the existing behavior of returning eigenvalues after an unsuccessful
solve. It does not suppress input-validation or other exceptions from `feast`.
Use `feast` directly when you need to inspect status, residuals, or partial results.
All other keywords are forwarded to `feast`.

**Returns:**
- `Vector`: Eigenvalues found

### eigen_feast

```@docs
FeastKit.eigen_feast
```

Return Eigen object from FeastKit calculation.

```julia
eigen_feast(A, interval; check=false, kwargs...)
eigen_feast(A, B, interval; check=false, kwargs...)
```

`check` has the same behavior as in `eigvals_feast`: `true` requires `info == 0`,
and the default `false` preserves the returned eigenpairs without checking the
status. Other keywords are forwarded to `feast`.

```@example checked_wrappers
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
B = Matrix{Float64}(I, 4, 4)
values = eigvals_feast(A, (0.5, 2.5); subspace_size=3, check=true)
decomposition = eigen_feast(A, B, (0.5, 2.5); subspace_size=3, check=true)
@assert values ≈ [1.0, 2.0]
@assert decomposition.values ≈ values
decomposition.values
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

FeastKit functions return status codes in `result.info`:

| Code | Name | Description |
|------|------|-------------|
| 0 | `Feast_SUCCESS` | Success |
| 1 | `Feast_ERROR_N` | Invalid matrix size N |
| 2 | `Feast_ERROR_M0` | Invalid or saturated subspace; increase M0 or narrow the region |
| 3 | `Feast_ERROR_EMIN_EMAX` | Invalid search interval |
| 4 | `Feast_ERROR_EMID_R` | Invalid center/radius for complex problems |
| 5 | `Feast_ERROR_NO_CONVERGENCE` | No convergence achieved |
| 6 | `Feast_ERROR_MEMORY` | Memory allocation failed |
| 7 | `Feast_ERROR_INTERNAL` | Internal error |
| 8 | `Feast_ERROR_LAPACK` | Linear algebra error |
| 9 | `Feast_ERROR_FPM` | Invalid FeastKit parameters |

**Error handling:**
```julia
result = feast(A, interval)   # A and interval are yours

if result.info != 0
    error_name = ["Feast_SUCCESS", "Feast_ERROR_N", "Feast_ERROR_M0", 
                  "Feast_ERROR_EMIN_EMAX", "Feast_ERROR_EMID_R",
                  "Feast_ERROR_NO_CONVERGENCE", "Feast_ERROR_MEMORY",
                  "Feast_ERROR_INTERNAL", "Feast_ERROR_LAPACK", 
                  "Feast_ERROR_FPM"][result.info + 1]
    @warn "FeastKit failed with $error_name"
end
```

---

## Parameter Reference

The `fpm` parameter array controls FeastKit behavior:

| Index | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| `fpm[1]` | Print level | 0 | 0=silent, 1=summary, negative=write to file |
| `fpm[2]` | Integration points | 8 | Symmetric/Hermitian half-contour node count |
| `fpm[3]` | Tolerance exponent | 12 | Convergence: 10^(-fpm[3]) |
| `fpm[4]` | Max iterations | 20 | Maximum refinement loops |
| `fpm[5]` | Initial subspace | 0 | 0=random, 1=user-provided |
| `fpm[8]` | General integration points | 16 | General full-contour node count |
| `fpm[10]` | Factorization cache | 1 | 1=store direct factorizations, 0=recompute |
| `fpm[16]` | Integration type | 0 | 0=Gauss, 1=Trapezoidal, 2=Zolotarev |
| `fpm[18]` | Ellipse ratio | 100 | Aspect ratio × 100 |
| `fpm[42]` | Mixed precision | 0 | 1=serial dense Float64 residual inverse iteration with Float32 correction solves |

The precision-aware `feast_tolerance(fpm, Float32)` floors the target at
`sqrt(eps(Float32))`; `fpm[7]` is a legacy slot, not its active stopping
control. Interval solvers require at least 3 half-contour points. For
Gauss/Zolotarev, counts above 20 must be one of 24, 32, 40, 48, or 56. General
full contours require at least 2 points; Gauss counts above 40 must be one of
48, 64, 80, 96, or 112. See [Problem Setup](problem_setup.md) for the distinction
between outer parameters and inner-solver tolerances.

With `fpm[5]=1`, the real-symmetric, complex-Hermitian, and general RCI kernels
use the supplied initial subspace first. Before accepting converged pairs with
unused subspace capacity, they retain those pairs and fill the remaining columns
with deterministic random probes for a verification sweep. This helps recover enclosed eigendirections
absent from the initial guess. The sweep counts toward `fpm[4]`; exhausting that
budget before verification/refinement finishes reports non-convergence, not
success. As with a random initial subspace, this is not a certified eigenvalue
count.

**Setting parameters:**
```@example apifpm
using FeastKit, LinearAlgebra

A = Matrix(SymTridiagonal(2.0 * ones(100), -1.0 * ones(99)))
interval = (0.0, 0.05)

fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[1] = 1      # Summary output (0, 1, or negative for a file -- 2 is rejected)
fpm[2] = 16     # 16 integration points
fpm[3] = 14     # High precision (10^-14)
fpm[16] = 2     # Zolotarev integration

result = feast(A, interval, M0=10, fpm=fpm)
result.info, result.M
```

---

## Performance Guidelines

### Memory Usage

Subspace and projection workspaces scale as `O(N*M0 + M0^2)`, with several
real and complex buffers. Dense matrices add `O(N^2)` storage, and cached
complex LU factors can add `O(ne*N^2)`. Sparse LU storage depends on fill-in;
it cannot be inferred from `nnz(A)` alone. Matrix-free GMRES also stores a
Krylov basis and callback-owned buffers. See [Performance](performance.md).

### Choosing Controls

Choose `subspace_size` from the expected enclosed eigenvalue count, including
multiplicities, rather than from `N` alone. Choose outer `tol` from the required
accuracy and input precision. Increase quadrature points if the filter does
not adequately separate inside and outside eigenvalues.

### Solver Selection

Assembled solvers default to `:direct` and also accept `:gmres` on supported
backends. Matrix-free solvers default to `:gmres` and additionally support
`:bicgstab` or a callback. Iterative options are `rtol`, `maxiter`, and GMRES
`restart`; matrix-free calls also accept `preconditioner`. Neither `:cg` nor
a BiCGSTAB `l` option is supported. See [Matrix-Free Interface](matrix_free_interface.md).
