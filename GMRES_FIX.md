# GMRES API Fix

## Problem

The GMRES iterative solver was returning 0 eigenvalues when it should have returned 4, causing test failures with:
```
DimensionMismatch: dimensions must match: a has dims (Base.OneTo(0),), b has dims (Base.OneTo(4),), mismatch at 1
```

## Root Cause

The code was using an outdated Krylov.jl API for calling `gmres`. In Krylov.jl 0.10.1 (the version specified in `Project.toml`), the initial guess must be passed as a **keyword argument** `x=...`, not as a **positional argument**.

### Incorrect (Old API):
```julia
x_sol, stats = gmres(op, b, x_initial;  # ❌ x_initial as 3rd positional arg
                     restart=true,
                     memory=max(restart, 2),
                     rtol=tol,
                     atol=tol,
                     itmax=maxiter)
```

### Correct (Current API):
```julia
x_sol, stats = gmres(op, b;  # ✅ No 3rd positional arg
                     x=x_initial,  # ✅ x as keyword argument
                     restart=true,
                     memory=max(restart, 2),
                     rtol=tol,
                     atol=tol,
                     itmax=maxiter)
```

## Files Fixed

1. **src/sparse/feast_sparse.jl**:
   - `solve_shifted_iterative!` function (line 46)

2. **src/dense/feast_dense.jl**:
   - `dense_solve_shifted_iterative!` function (line 35)

## Changes Made

### src/sparse/feast_sparse.jl
Changed line 46 from:
```julia
x_sol, stats = gmres(op, b, x_initial;
```
to:
```julia
x_sol, stats = gmres(op, b;
                     x=x_initial,
```

### src/dense/feast_dense.jl
Changed line 35 from:
```julia
x_sol, stats = gmres(op, b, x0;
```
to:
```julia
x_sol, stats = gmres(op, b;
                     x=x0,
```

## Testing

### Quick Test
Run the provided test script:
```bash
julia --project=. test_gmres_fix.jl
```

This script:
1. Tests a 12×12 tridiagonal eigenvalue problem
2. Compares results from:
   - Direct solver (LU factorization)
   - GMRES solver
   - DiFEAST solver (alias for GMRES)
3. Verifies all methods find the same eigenvalues

### Full Test Suite
Run the complete test suite:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The "Sparse iterative FEAST (GMRES)" test at line 400 in `test/runtests.jl` should now pass.

## Expected Results

### Before Fix
- GMRES solver: M=0, info=5 (NO_CONVERGENCE) or M=0, info=0 (incorrect)
- Direct solver: M=4, info=0 (correct)
- Test failure: DimensionMismatch

### After Fix
- GMRES solver: M=4, info=0
- Direct solver: M=4, info=0
- DiFEAST solver: M=4, info=0
- All eigenvalues match within tolerance (atol=1e-6)
- All tests pass ✓

## Impact

This fix affects all FEAST functions that support iterative solvers:
- `feast_scsrgv!` (sparse real symmetric generalized)
- `feast_hcsrev!` (sparse complex Hermitian)
- `feast_hcsrgv!` (sparse complex Hermitian generalized)
- `feast_gcsrgv!` (sparse complex general)
- `feast_sygv!` (dense real symmetric generalized)
- All `zifeast_*` and `difeast_*` wrappers

## Related Issues

This fix resolves the test failure reported in the "Sparse iterative FEAST (GMRES)" test case and should enable all GMRES-based solvers to work correctly.

## Technical Details

The Krylov.jl 0.10.1 API for `gmres` is:
```julia
gmres(A, b; x = zeros(eltype(b), length(b)),
      restart = true,
      memory = 20,
      rtol = √eps(real(eltype(b))),
      atol = zero(real(eltype(b))),
      itmax = 0,
      kwargs...)
```

All parameters except `A` and `b` are keyword arguments. The old API (Krylov.jl < 0.9) allowed a third positional argument for the initial guess, but this was changed in later versions.
