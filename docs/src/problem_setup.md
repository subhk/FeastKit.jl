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
| Complex matrix-free problem | `feast_general(Aop, Bop, center, radius; M0, solver)` | Complex operators, including an explicit identity `Bop` for standard problems |
| Polynomial `Σ λ^k Aₖ*x = 0` | `feast_polynomial(coeffs, center, radius; M0)` | Complex coefficients ordered `[A₀, A₁, …, Aₚ]` |

Use floating-point arrays (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`)
and matching precision for `A`, `B`, and the search bounds. Low-level drivers
have stricter type signatures than the convenience interfaces. `A` and `B`
must be square and have identical dimensions. Symmetry checks do not prove
positive definiteness: for a manageable assembled mass matrix, check
`isposdef(Hermitian(B))` yourself. A general pencil must be regular; singular
mass matrices and infinite eigenvalues require particular care.

Do not make a nonsymmetric model "symmetric" by wrapping it in `Symmetric`:
that wrapper treats one triangle as authoritative and changes the represented
matrix. Use `feast_general` instead. Remove constrained degrees of freedom
consistently from both stiffness and mass matrices before solving.

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

Every solve needs `(z*B - A)*Y = X` at contour nodes. Assembled high-level
`feast`/`feast_general` calls do **not** accept `solver`, `solver_opts`,
`tolerance`, or `integration_points` keywords. Configure `fpm`, or select a
low-level driver when you need explicit inner-solver controls:

```@example setup_sparse_iterative
using FeastKit, Krylov, LinearAlgebra, SparseArrays
A = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
B = spdiagm(0 => ones(4))
fpm = feastinit().fpm
result = feast_scsrgv!(A, B, 0.5, 2.5, 3, fpm;
                       solver=:gmres, solver_tol=1e-12,
                       solver_maxiter=200, solver_restart=20)
@assert result.info == 0 && result.M == 2
result.lambda
```

Direct factorization is a useful baseline. Sparse factorization can have
substantial fill-in; storing all contour factorizations trades memory for
speed. Iterative solves need sufficiently accurate inner solutions, especially
near clustered eigenvalues. Tightening the outer tolerance alone is not enough.

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

For matrix-free calls, `tol` and `maxiter` initialize outer parameters only
when `fpm` is omitted; a supplied `fpm` takes precedence. `solver_opts` controls
the inner solver separately. Use precision-appropriate tolerances and consult
the [parameter reference](api_reference.md) for supported quadrature counts.

Creating a contour does not attach it to a solve. There is no `contour=` keyword:

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

Read `result.info` first, then `result.M`, `result.lambda[1:result.M]`,
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
