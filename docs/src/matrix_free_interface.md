# Matrix-Free FeastKit Interface

Matrix-free solves replace stored matrices with multiplication callbacks. FEAST
still allocates subspace, projection, and linear-solver workspaces; avoiding a
stored matrix does not make a solve allocation-free.

## Overview

Provide operations that overwrite their outputs without changing their inputs:

- `A_mul!(y, x)` computes `y = A*x`.
- `B_mul!(y, x)` computes `y = B*x`.
- A custom `solver(Y, z, X)` overwrites every column of `Y` with the solution of
  `(z*B - A)*Y = X`. Shifted systems and their right-hand sides are complex even
  when `A` and `B` are real.

For generalized projection, FEAST supplies `X = B*Q`. The solver callback must
not multiply that supplied right-hand side by `B` again.

The high-level interval overload supports **real symmetric operators**. For
complex operators, use `feast_general` or a full contour. Complex Hermitian
assembled matrices also have the interval interface described in the
[API reference](api_reference.md).

## Matrix-Free Operator Types

### LinearOperator

`LinearOperator{T}(A_mul!, (n, n); issymmetric=false, ishermitian=false,
isposdef=false, At_mul! = nothing, Ac_mul! = nothing, solve! = nothing)` stores
multiplication callbacks and structural flags. In Julia calls, put a space
before `=` for keywords ending in `!`, for example `At_mul! = callback`.
`T` declares the element type. Structural flags are supplied by you; FeastKit
does not infer or verify symmetry or positive definiteness from the callback.
For the real interval interface, declare the known symmetry explicitly. Complex
operators, including Hermitian ones, use the general/full-contour interface.
See [Matrix types: detected or declared?](@ref matrix-properties) for the
difference between operator declarations and checks on assembled matrices.

`At_mul!` computes the transpose product, while `Ac_mul!` computes the adjoint.
The high-level solvers use their explicit `solver` keyword; storing `solve!`
in an operator does not automatically select it.

LinearOperators.jl, often loaded alongside Krylov.jl, exports its own
`LinearOperator`. With both packages loaded the unqualified name is ambiguous;
use the alias `FeastLinearOperator` (or `FeastKit.LinearOperator`) instead.

### MatrixVecFunction

`MatrixVecFunction{T}(callback, (n, n); issymmetric=false, ...)` calls
`callback(y, op, x)`. It has no `data` field. Capture additional data in the
callback's closure or use a callable object:

```@example matfree_payload
using FeastKit, LinearAlgebra
weights = [1.0, 2.0, 3.0]
op = MatrixVecFunction{Float64}(
    (y, op, x) -> (y .= weights .* x), (3, 3); issymmetric=true)
y = zeros(3)
mul!(y, op, ones(3))
@assert y == weights
y
```

## Basic Usage

### Standard Eigenvalue Problem

This example only stores a vector of diagonal entries. Load the optional
Krylov dependency for the built-in iterative solvers.

```@example matfree_basic
using FeastKit, Krylov, LinearAlgebra
entries = collect(1.0:12.0)
n = length(entries)
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x),
                              (n, n); issymmetric=true)
result = feast(A_op, (0.5, 2.5); subspace_size=4, tol=1e-10,
               solver=:gmres, solver_opts=(rtol=1e-13, maxiter=200, restart=16))
@assert result.converged result.message
@assert result.values ≈ [1.0, 2.0]
result.values
```

### Generalized Eigenvalue Problem and Custom Solver

A custom callback can exploit structure without loading Krylov:

```@example matfree_generalized
using FeastKit, LinearAlgebra
mass = [2.0, 3.0, 4.0, 5.0]
stiffness = mass .* [1.0, 2.0, 3.0, 4.0]
A_op = LinearOperator{Float64}((y, x) -> (y .= stiffness .* x),
                              (4, 4); issymmetric=true)
B_op = LinearOperator{Float64}((y, x) -> (y .= mass .* x),
                              (4, 4); issymmetric=true, isposdef=true)
function shifted_solve!(Y, z, X)
    for j in axes(X, 2), i in axes(X, 1)
        Y[i, j] = X[i, j] / (z * mass[i] - stiffness[i])
    end
    return Y
end
result = feast(A_op, B_op, (0.5, 2.5); subspace_size=3, solver=shifted_solve!)
@assert result.converged result.message
@assert result.values ≈ [1.0, 2.0]
result.values
```

## Linear Solvers

| Solver | Requirements and options |
|:--|:--|
| `:gmres` (default) | Load `Krylov`; `rtol`, `maxiter`, `restart`, `preconditioner` |
| `:bicgstab` | Load `Krylov`; `rtol`, `maxiter`, `preconditioner` |
| Callback | Pass `solver=shifted_solve!`; configure the callback directly |

`solver_opts` is a named tuple. Defaults are `rtol=1e-6`, `maxiter=1000`, and
GMRES `restart=30`. Tight outer targets may require a smaller inner `rtol`.
`solver_opts.maxiter` limits each shifted solve; outer `maxiter` limits FEAST
refinement. `restart` is accepted for BiCGSTAB but only affects GMRES.
There is no `l` option or built-in BiCGSTAB(l) variant. `:direct` requires
assembled matrices; `:cg` is rejected because complex shifts do not preserve
symmetric positive definiteness.

### Preconditioning

The optional `solver_opts.preconditioner` is a **left inverse-action** operator:
`mul!(y, preconditioner, x)` must apply the preconditioning action to complex
vectors. A factorization intended for `ldiv!` must be wrapped in an appropriate
multiplication callback; `Pl` is not a supported option. A preconditioner for
`A` alone may be ineffective for the family of shifted systems `z*B - A`.

## Advanced Features

### Named Options and Parameters

`subspace_size` aliases `M0` (default 10, clamped to the operator dimension).
Use `initial_subspace=previous.vectors` to seed a related solve, or
`subspace_size=:auto, max_subspace_size=cap` for bounded growth from a small
heuristic width. Matrix-free sizing does not run a direct count estimator.
Built-in solvers adapt their default inner tolerance to the outer residual;
set `solver_opts.rtol` to keep it fixed. Custom callbacks retain control over
their own tolerances. Mixed precision requires assembled dense matrices.
`tol`, `maxiter`, and `quadrature_points` have the same meaning as in assembled
solves. Conflicting named controls and explicitly set `fpm` entries raise an
`ArgumentError`. Named overrides use a copy of the supplied parameter vector.
The tolerance is rounded down to a decimal power; Float32 retains its precision
floor. See [Problem Setup](problem_setup.md) for details.

### Custom Contour Integration and General Problems

Use complex operators for a full contour. The contour determines which
values are selected; the wrapper manages registration and cleanup.

```@example matfree_contour
using FeastKit, LinearAlgebra
entries = ComplexF64[-0.3+0.2im, 0.4-0.1im, 2.5]
A_op = LinearOperator{ComplexF64}((y, x) -> (y .= entries .* x), (3, 3))
shifted_solve!(Y, z, X) = (Y .= X ./ (z .- entries))
contour = feast_rectangle(-1, 1, -1, 1)
result = feast(A_op, contour; subspace_size=3, solver=shifted_solve!)
@assert result.converged result.message
@assert sort(result.values; by=real) ≈ entries[1:2]
result.values
```

For a circle, `feast_general(A_op, center, radius; ...)` and
`feast_general(A_op, B_op, center, radius; ...)` are also available. Half-contours
from `feast_contour_expert` use the advanced registration workflow in
[Custom Contours](custom_contours.md).

### Workspace Reuse

The lower-level real interface accepts an allocated workspace and an explicit
linear-solver callback. Reuse is valid for the same dimension, subspace size,
and precision. It does not eliminate every temporary allocation.

```@example matfree_generalized
workspace = allocate_matfree_workspace(Float64, 4, 3)
result = feast_matfree_srci!(A_op, B_op, (0.5, 2.5), 3;
                             workspace=workspace, linear_solver=shifted_solve!)
@assert result.converged result.message
result.values
```

Unlike the high-level named-option interface, low-level `feast_matfree_srci!`
and `feast_matfree_grci!` use their `tol`/`maxiter` keywords only when `fpm` is
omitted. Set the corresponding entries when supplying `fpm` at that level.

### Polynomial Eigenvalue Problems

`feast_polynomial(coeffs_ops, center, radius; ...)` accepts complex coefficient
operators in increasing powers of λ. It builds matrix-free companion operators
of dimension `degree*n`; a solver callback receives right-hand sides of that
larger dimension. See [Polynomial Problems](polynomial_problems.md).

## Performance and Error Handling

Check `result.converged` and `result.message` before using eigenpairs. A small
residual does not establish completeness when the subspace is saturated.
Choose a region and subspace that can hold every target eigenvalue, and tighten
inner solves when outer refinement stalls. GMRES convergence is not guaranteed
by choosing it, even for an invertible shifted system with a limited budget.

External multiplication operators can be wrapped in `LinearOperator`, and
external linear solvers can be adapted to the callback contract. FeastKit's
built-in iterative integration uses Krylov.jl. Its current workspaces are CPU
`Matrix`/`Vector` arrays; there is no built-in CUDA or GPU workspace backend.
