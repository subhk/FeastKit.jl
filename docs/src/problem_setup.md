# Setting Up a Problem

Use this checklist when translating a mathematical model into a FeastKit call.
The examples below are executed when the web documentation is built.

## 1. Install in your project

The current package requires Julia 1.10 or later (see `Project.toml`). Keep a
project environment so collaborators can reproduce your dependencies:

```julia
using Pkg
Pkg.activate("my-feast-problem")
Pkg.add("FeastKit")
# Optional: install only what your problem needs.
Pkg.add("Krylov")  # iterative shifted solves
Pkg.add("MPI")     # MPI execution
```

Run scripts with `julia --project=my-feast-problem solve.jl`. Installing an
optional dependency is not enough: `using Krylov` or `using MPI` activates its
FeastKit extension. Direct serial solves need neither. To use this repository's
checkout instead of a registry release, run
`Pkg.develop(path="/absolute/path/to/FeastKit.jl")` in your environment.

## 2. Choose the mathematical problem and entry point

| Your problem | Entry point | Required structure |
|:--|:--|:--|
| `A*x = λ*x`, real spectrum | `feast(A, (Emin, Emax); M0)` | Real symmetric or complex Hermitian `A` |
| `A*x = λ*B*x`, real spectrum | `feast(A, B, (Emin, Emax); M0)` | Symmetric/Hermitian `A`; matching positive-definite `B` |
| General standard or generalized pencil | `feast_general(A, center, radius; M0)` or `feast_general(A, B, center, radius; M0)` | Square matrices; positive radius and complex center |
| Real symmetric matrix-free problem | `feast(Aop, Bop, interval; M0, solver)` | Real operators implementing multiplication; SPD `Bop` |
| Complex matrix-free problem | `feast_general(Aop, center, radius; M0, solver)` or the overload with `Bop` | Complex operators; omitting `Bop` uses the identity |
| Polynomial `Σ λ^k Aₖ*x = 0` | `feast_polynomial(coeffs, center, radius; M0)` | Complex coefficients ordered `[A₀, A₁, …, Aₚ]` |

Use floating-point arrays (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`)
and matching precision for `A`, `B`, and the search bounds. Low-level drivers
have stricter type signatures than the convenience interfaces. `A` and `B`
must be square and have identical dimensions. Symmetry checks do not prove
positive definiteness: for a manageable assembled mass matrix, check
`isposdef(Hermitian(B))` yourself. A general pencil must be regular; singular
mass matrices and infinite eigenvalues require particular care.

Remove constrained degrees of freedom consistently from both stiffness and
mass matrices before solving.

### [Matrix types: detected or declared?](@id matrix-properties)

For assembled dense or sparse matrices, FeastKit gets **real versus complex
from `eltype(A)`**. You select an interval or a full contour; the interval
interface checks the required symmetry. There is no `matrix_type` keyword.

| Matrix properties | Interface | What the code does |
|:--|:--|:--|
| Real symmetric | `feast(A, (Emin, Emax))` | Checks `issymmetric(A)` and uses a real symmetric driver |
| Complex Hermitian | `feast(A, (Emin, Emax))` | Checks `ishermitian(A)` and uses a Hermitian driver |
| Real nonsymmetric | `feast(A, contour)` or `feast_general(A, center, radius)` | Converts to complex arithmetic and uses a general driver |
| Complex non-Hermitian, including complex symmetric | `feast(A, contour)` or `feast_general(A, center, radius)` | Uses a general driver |

Complex **symmetric** means `A == transpose(A)`. Complex **Hermitian** means
`A == adjoint(A)` (written `A'` in Julia); these are different properties.
Even a nonsymmetric matrix with entirely real eigenvalues needs the general
interface. A `ComplexF64` array uses the complex path even when every stored
imaginary part is zero.

An incompatible matrix passed to `feast(A, interval)` raises `ArgumentError`;
the call does **not** automatically switch to a general solver or choose a
complex search region. Conversely, `feast(A, contour)` always uses a general
driver, even for symmetric or Hermitian input, and returns complex eigenvalues.
For generalized interval problems, the same structural requirements apply to
`B`, which must also be positive definite.

Here are all four cases in one executable example:

```@example matrix_properties
using FeastKit, LinearAlgebra

S = [2.0 -1.0; -1.0 2.0]         # Real symmetric: eigenvalues 1 and 3
H = ComplexF64[2 im; -im 2]       # Complex Hermitian: eigenvalues 1 and 3
R = [0.0 -1.0; 1.0 0.0]          # Real nonsymmetric: eigenvalues ±im
C = ComplexF64[0 im; im 0]        # Complex symmetric, not Hermitian: ±im

interval = (0.5, 3.5)
contour = feast_circle(0, 1.5)
real_symmetric = feast(S, interval; subspace_size=2)
complex_hermitian = feast(H, interval; subspace_size=2)
real_general = feast(R, contour; subspace_size=2)
complex_general = feast(C, contour; subspace_size=2)

@assert issymmetric(C) && !ishermitian(C)
for result in (real_symmetric, complex_hermitian)
    @assert result.converged
    @assert isapprox(result.values, [1.0, 3.0]; atol=1e-9)
end
for result in (real_general, complex_general)
    @assert result.converged
    @assert isapprox(sort(result.values; by=imag), [-1.0im, 1.0im]; atol=1e-9)
end
(real_symmetric.values, complex_hermitian.values,
 real_general.values, complex_general.values)
```

**Explicit structure for assembled matrices.** Use Julia's `Symmetric` or
`Hermitian` wrappers when your model defines that structure:

```@example matrix_properties
S_declared = Symmetric(S, :U)
H_declared = Hermitian(H, :U)
@assert Matrix(S_declared) == S && Matrix(H_declared) == H
(issymmetric(S_declared), ishermitian(H_declared))
```

`:U` makes the upper triangle authoritative; `:L` uses the lower triangle.
These wrappers construct the represented symmetric/Hermitian matrix from that
triangle, rather than checking that both stored triangles agree. Wrapping a
nonsymmetric model therefore changes it. Plain-array symmetry checks use exact
entry comparisons, so small assembly differences can fail them; decide from
your model whether declaring one triangle authoritative is appropriate.

**Explicit properties for matrix-free operators.** FeastKit cannot inspect
the entries behind a multiplication callback. Declare the element type and
known properties yourself, for example
`LinearOperator{Float64}(A_mul!, (n, n); issymmetric=true)`.
The `issymmetric`, `ishermitian`, and `isposdef` flags are declarations, not
properties inferred or verified from the callback; they default to `false`.
The current matrix-free interval interface accepts real symmetric operators.
Use `LinearOperator{ComplexF64}` with `feast_general` or `feast(Aop, contour)`
for complex operators, including complex Hermitian ones. See the
[Matrix-Free Interface](matrix_free_interface.md) for complete callback examples.

### Generalized symmetric example

```@example setup_generalized
using FeastKit, LinearAlgebra
A = [2.0 1 0; 1 3 1; 0 1 4]
B = [2.0 0.5 0; 0.5 1 0; 0 0 1]
@assert issymmetric(A) && isposdef(Symmetric(B))
result = feast(A, B, (0.1, 5.0); M0=3, backend=:serial)
@assert result.info == 0
@assert sort(result.lambda) ≈ eigvals(Symmetric(A), Symmetric(B))
result.lambda
```

`A` and `B` do not need to commute. For a standard problem omit `B`; do not
construct a large dense identity unnecessarily.

### General complex example

```@example setup_complex
using FeastKit, LinearAlgebra
A = ComplexF64[1+im 0.2 0; 0 2+im 0.3; 0 0 6]
result = feast_general(A, 1.5+1.0im, 0.8; M0=3, backend=:serial)
@assert result.info == 0 && result.M == 2
@assert sort(real.(result.lambda)) ≈ [1.0, 2.0]
result.lambda
```

Complex **Hermitian** matrices still have real eigenvalues and use `feast`
with a real interval. Complex entries alone do not imply a general problem.

## 3. Choose the region and subspace size

FEAST targets a region, not a requested count of "smallest" eigenvalues.
Choose finite `Emin < Emax`, or a complex center and positive radius. Keep
eigenvalues away from the contour itself: a contour node near an eigenvalue
makes the shifted system difficult or singular.

`M0` is the trial-subspace dimension, not the number guaranteed to be returned.
Choose it larger than the expected number of eigenvalues in the region
(including multiplicities), with `1 ≤ M0 ≤ N`. For small problems `M0=N`
is useful for verification. Assembled high-level calls clamp `M0` to `N`;
matrix-free and low-level calls require a valid size explicitly. Polynomial
interfaces differ: the assembled `feast_polynomial` call expands `M0` to
`degree*M0` internally, so its input still obeys `M0 ≤ N`. The matrix-free
polynomial call passes `M0` directly to its `degree*N` companion problem.

A converged but saturated subspace with `M == M0 < N` returns
`Feast_ERROR_M0` (2): it cannot certify that no eigenvalues were missed.
Increase `M0` or split the region. Too few directions can also manifest as
non-convergence (5). Small residuals alone do not establish completeness.
`feast_validate_interval` supplies bounds for `A`, not an exact eigenvalue
count, and those bounds are not bounds for a general `A*x = λ*B*x` pencil.
For symmetric/Hermitian problems, `feast_estimate_count(A, interval; B=B,
nprobe=16)` provides a stochastic count estimate to help choose `M0`; leave
a safety margin and validate with a larger subspace. It is not an exact count.

## 4. Select storage and shifted solves

`Matrix` uses dense drivers and `SparseMatrixCSC` uses sparse drivers.
Other structured arrays may be materialized by the convenience API; choose
`sparse(A)` explicitly for sparse storage. For banded storage, use
`full_to_banded` with `feast_banded`; consult the
[API reference](api_reference.md) for bandwidth and triangle conventions.

Every solve needs `(z*B - A)*Y = X` at contour nodes. High-level `feast`,
`feast_general`, and `feast_banded` accept `solver=:direct` (the default) or
`:gmres`, with `solver_opts=(rtol=..., maxiter=..., restart=...)` for iterative
shifted solves. Set outer controls with `tol`, `maxiter`, `subspace_size`
(alias `M0`), and `quadrature_points`, or retain `fpm` for advanced settings:

```@example setup_sparse_iterative
using FeastKit, Krylov, LinearAlgebra, SparseArrays
A = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
B = spdiagm(0 => ones(4))
result = feast(A, B, (0.5, 2.5); subspace_size=3, tol=1e-12,
               solver=:gmres, solver_opts=(rtol=1e-14, maxiter=200, restart=20))
@assert result.converged && result.M == 2
result.values
```

Direct factorization is a useful baseline. Sparse factorization can have
substantial fill-in; storing all contour factorizations trades memory for
speed. Iterative solves need sufficiently accurate inner solutions, especially
near clustered eigenvalues. Here the inner tolerance is `1e-14`, tighter than
the default outer residual target `1e-12`, to leave room for
error in the shifted solves. Tightening the outer tolerance alone is not enough.

Named settings that conflict with an explicitly set `fpm` entry raise an
`ArgumentError`. Named overrides use a copy of `fpm`. `tol` is rounded down to
a decimal power in `[1e-16, 1]`; Float32 still uses a minimum effective tolerance
of `sqrt(eps(Float32))`. `quadrature_points` controls the half-contour count for
interval solves and the full-contour count for general solves, subject to the
chosen integration rule's valid counts.

GMRES is supported by serial assembled drivers and complex MPI drivers.
Explicit unsupported backend requests raise an error; `backend=:auto` may
fall back to serial. Matrix-free calls support `:gmres`, `:bicgstab`, or a
callback, with the same inner option names and an additional `preconditioner`.

### Matrix-free callback contract

`mul!(y, Aop, x)` must overwrite `y` with `A*x` without modifying `x`.
A custom `solver(Y, z, X)` must overwrite **all columns** of `Y` with the
solution of `(z*B-A)*Y=X`; it must not multiply the supplied RHS by `B` again.
Shifted solve buffers are complex even for real symmetric problems.

```@example setup_matrix_free
using FeastKit, LinearAlgebra
A = [2.0 1 0; 1 3 1; 0 1 4]
B = [2.0 0.5 0; 0.5 1 0; 0 0 1]
Aop = LinearOperator{Float64}((y,x)->mul!(y,A,x), (3,3); issymmetric=true)
Bop = LinearOperator{Float64}((y,x)->mul!(y,B,x), (3,3);
                               issymmetric=true, isposdef=true)
# A small reference implementation; replace with your structured solver.
shifted_solve! = (Y,z,X)->copyto!(Y, (z*B-A) \ X)
result = feast(Aop, Bop, (0.1,5.0); M0=3, solver=shifted_solve!)
@assert result.info == 0
@assert sort(result.lambda) ≈ eigvals(Symmetric(A), Symmetric(B))
result.lambda
```

For built-in iteration, load `Krylov` and use `solver=:gmres` or `:bicgstab`,
with e.g. `solver_opts=(rtol=1e-10, maxiter=500, restart=30)`.
An optional `preconditioner` in that tuple is a **left inverse action**
applied through `mul!` on complex vectors, not a matrix to be factorized by
FeastKit. CG is not supported: complex shifted systems are not SPD.
See [Matrix-Free Interface](matrix_free_interface.md) for complex operators,
preconditioning, reusable workspaces, and the simpler `feast_matvec` interface.

## 5. Set outer parameters and custom contours

Start with `fpm = feastinit().fpm`; do not leave a zero-filled vector
uninitialized. Pass it as `fpm=fpm` to the solver.

| Parameter | Meaning |
|:--|:--|
| `fpm[1]` | Logging: 0 or 1 (not 2) |
| `fpm[2]` | Symmetric/Hermitian half-contour node count; default 8 |
| `fpm[8]` | General full-contour node count; default 16 |
| `fpm[3]` | Outer tolerance exponent; default 12; precision-aware kernels floor Float32 tolerance at `sqrt(eps(Float32))` |
| `fpm[7]` | Legacy single-precision slot (default 5); not the active tolerance control in `feast_tolerance` |
| `fpm[4]` | Maximum refinement loops |
| `fpm[10]` | Cache direct factorizations: 1 stores, 0 recomputes |
| `fpm[16]` | Integration: 0 Gauss, 1 trapezoidal, 2 Zolotarev (not general problems) |

High-level named controls must agree with explicit `fpm` entries. For the
lower-level `feast_matfree_srci!` and `feast_matfree_grci!` functions only,
a supplied `fpm` takes precedence over their `tol` and `maxiter` keywords. `solver_opts` controls
the inner solver separately. Use precision-appropriate tolerances and consult
the [parameter reference](api_reference.md) for supported quadrature counts.

Pass full contours directly with `feast(A, contour)` or `feast(A, B, contour)`.
Half-contours for interval problems use the advanced registration workflow
below. There is no `contour=` keyword:

```@example setup_contour
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
interval = (0.5, 2.5)
contour = feast_contour_expert(interval..., 16, 0, 100)
fpm = feastinit().fpm
result = FeastKit.with_custom_contour(fpm, contour) do
    feast(A, interval; M0=3, fpm=fpm, backend=:serial)
end
@assert result.info == 0 && result.M == 2
result.lambda
```

Use the **same raw parameter vector** in both calls, and a separate vector for
each concurrent solve. Symmetric/Hermitian paths expect a half-contour;
do not append conjugates or double weights yourself. General problems expect
a full closed contour with weights including `dz/(2π*im)`.
See [Custom Contours](custom_contours.md) for full-contour construction and
selection semantics. These setup examples explicitly use the serial backend.

## 6. Polynomial setup

For `(K + λ*C + λ²*M)*x=0`, supply `[K,C,M]`, not the reverse order.
Even symmetric coefficients can produce complex roots. The assembled
`feast_polynomial` convenience interface requires complex coefficients and
builds a dense companion pencil, including when coefficients are sparse.

```@example setup_polynomial
using FeastKit, LinearAlgebra
K = -Matrix(Diagonal(ComplexF64[1,4]))
C = zeros(ComplexF64,2,2)
M = Matrix{ComplexF64}(I,2,2)
result = feast_polynomial([K,C,M], 1.5+0.0im, 0.75; M0=2)
@assert result.info == 0 && result.M == 2
@assert sort(real.(result.lambda)) ≈ [1.0,2.0]
result.lambda
```

Check residuals against the original polynomial, not just the companion
pencil. For large problems consult [Polynomial Problems](polynomial_problems.md)
for matrix-free companions and direct polynomial RCI alternatives.

## 7. Verify the result before scaling up

Check `result.converged` and `result.message` first, then `result.M`, `result.lambda[1:result.M]`,
`result.q[:,1:result.M]`, and `result.res[1:result.M]`. Eigenvectors are columns.
`epsout` and `loop` report the outer convergence metric and refinement progress.
Some invalid inputs throw exceptions before a result is returned.

| Status | Action |
|:--|:--|
| 0 (`Feast_SUCCESS`) | Check the returned spectrum and physical residuals |
| 2 (`Feast_ERROR_M0`) | Check subspace size; increase it or split the region |
| 5 (`Feast_ERROR_NO_CONVERGENCE`) | Check region, `M0`, quadrature, loop budget, and inner solves |
| 8 (`Feast_ERROR_LAPACK`) | Inspect warnings for shifted-solve or reduced-pencil failure; check `B` and contour placement |

Do not assume partial results with nonzero status are a complete solution.
For each returned generalized eigenpair a useful independent check is
`norm(A*x-λ*B*x)/(norm(A*x)+abs(λ)*norm(B*x))`, handling a zero denominator
explicitly. Compare small problems against Julia's `eigen(A,B)`, and repeat
with a larger subspace or more contour nodes to check completeness/stability.

## 8. Choose execution resources last

First obtain a verified `backend=:serial` baseline. Then use the
[Parallel Computing](parallel_computing.md) guide for launchable examples:

- Threads: launch Julia with `--threads=N`; high-level support is real symmetric
  dense/sparse problems, not general or complex Hermitian problems.
- Distributed: add workers and load the project on them; high-level support
  is sparse real symmetric problems.
- MPI: install/load `MPI`, initialize it, and have **all communicator ranks**
  enter the collective solve under an MPI launcher. These are contour-parallel
  solvers, not a distributed-row matrix interface; budget matrix storage per rank.
- `backend=:auto` may fall back. An explicit unsupported backend request throws;
  inspect `feast_parallel_capabilities()` and do not infer MPI from installation alone.

Avoid oversubscribing BLAS threads across workers or MPI ranks. Keep your
project/manifest, matrix construction, region, `M0`, `fpm`, solver options,
and backend settings with the result so the computation can be reproduced.

## Repeated Solves and Unknown Eigenvalue Counts

For related problems, use `initial_subspace=previous.vectors` with the serial
assembled or matrix-free API. To choose a width automatically, use
`subspace_size=:auto` and optionally `max_subspace_size=cap`. Dense serial
Float64/ComplexF64 direct solves can opt into `mixed_precision=true`.
See the [Performance Guide](performance.md) for executable examples, supported
backends, estimator costs, accuracy checks, and fallback behavior.
