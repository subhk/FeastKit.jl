# Zero to FEAST: A Practical Guide

New to FEAST or eigenvalue problems? This guide takes you from zero knowledge to running robust FEAST computations — step by step.

## What You’ll Learn

- What FEAST does and when to use it
- How to install and verify FEAST
- How to solve standard and generalized eigenvalue problems
- How to scale from dense to sparse to matrix-free
- How to target complex eigenvalues and custom contours
- How to tune performance and select parallel backends

---

## 1) Mental Model: What is FEAST?

- FEAST finds eigenvalues in a region you choose — not all eigenvalues at once.
- You provide a search interval `[Emin, Emax]` (real) or a circle `(center, radius)` (complex).
- FEAST integrates resolvent operators along a contour to filter eigenvalues inside the region.
- It scales well for large problems and works with dense, sparse, and matrix-free operators.

When to use FEAST:

- You know where the eigenvalues of interest live (e.g., “smallest 10”, “near 0”, or “inside this circle”).
- Your matrix is too large for full eigen-decomposition.
- You need a robust, parallel-friendly approach.

---

## 2) Installation and Verification

```julia
# In Julia REPL (press ] to enter Pkg mode)
pkg> add FEAST

# Back to Julia mode
julia> using FEAST, LinearAlgebra

# Quick verification
a = [2.0 -1.0; -1.0 2.0]
res = feast(a, (0.1, 5.0), M0=4)
@info "FEAST OK" M=res.M info=res.info lambda=res.lambda[1:res.M]
```

Expected: `info == 0` and two eigenvalues found near 1 and 3.

---

## 3) First Solve: Standard Symmetric Problem

Simple tridiagonal Laplacian (dense or sparse):

```julia
using FEAST, LinearAlgebra, SparseArrays

# 1D Laplacian on n points
n = 200
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Target low end of spectrum
Emin, Emax = 0.0, 0.2
res = feast(A, (Emin, Emax), M0=16)

@assert res.info == 0
@info "Found" M=res.M vals=res.lambda[1:res.M]
```

Tips:

- `M0` is the max desired eigenvalues; choose slightly larger than expected.
- If `res.M == 0`, broaden the interval or validate bounds (see Section 8).

---

## 4) Generalized Problem A x = λ B x

Common in structural dynamics and PDEs:

```julia
using FEAST, LinearAlgebra, SparseArrays

n = 1000
K = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))  # stiffness
M = spdiagm(0 => ones(n))                                        # mass

res = feast(K, M, (0.0, 0.5), M0=20)
@info "Generalized" info=res.info M=res.M
```

---

## 5) Matrix-Free: No Matrix Stored

When storing A is infeasible, supply y = A*x as a function:

```julia
using FEAST

function lap1d_matvec!(y, x)
    n = length(x)
    y[1] = 2x[1] - x[2]
    @inbounds for i in 2:n-1
        y[i] = -x[i-1] + 2x[i] - x[i+1]
    end
    y[n] = -x[n-1] + 2x[n]
end

n = 200_000
Aop = LinearOperator{Float64}(lap1d_matvec!, (n, n), issymmetric=true)
res = feast(Aop, (0.0, 0.2), M0=12)  # internally uses iterative solves
@info "Matrix-free" info=res.info M=res.M
```

Notes:

- Use efficient BLAS/threading inside `matvec!` for speed.
- You can choose iterative solvers and tolerances via kwargs if exposed in your version.

---

## 6) Complex Eigenvalues: Non-Hermitian Problems

Search in a circular region of the complex plane:

```julia
using FEAST

A = [2.0  5.0; -3.0  1.0]  # non-symmetric
I2 = Matrix{Float64}(I, 2, 2)
center, radius = 1.0 + 1.0im, 3.0
res = feast_general(A, I2, center, radius, M0=8)
@info "Complex region" info=res.info M=res.M λ=res.lambda[1:res.M]
```

---

## 7) From Defaults to Expert Parameters

Parameters live in `fpm::Vector{Int}` (length ≥ 64):

```julia
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[1] = 1     # print level (0=silent, 1=summary)
fpm[2] = 16    # integration points (8–32 typical)
fpm[3] = 12    # tolerance exponent (target ~ 1e-12)
fpm[4] = 30    # max refinement loops

res = feast(A, (Emin, Emax), M0=20, fpm=fpm)
```

Expert controls (see docs for details):

- Integration type: Gauss/Trapezoidal/Zolotarev via `fpm[16]`
- Ellipse aspect ratio via `fpm[18]`
- Custom contours: `feast_contour_expert`, `feast_contour_custom_weights!`

---

## 8) Validating and Troubleshooting

Check that the interval contains eigenvalues:

```julia
using FEAST, LinearAlgebra

A = diagm(0 => [1.0, 2.0, 3.0, 4.0])
bounds = feast_validate_interval(A, (1.5, 3.5))
@info "Gershgorin bounds" bounds
```

Common issues and fixes:

- Found zero eigenvalues: widen the interval or increase M0.
- Slow/No convergence: increase `fpm[2]`, adjust `fpm[3]`, or use different integration type.
- Generalized problem instability: ensure B is SPD for symmetric problems.
- Matrix-free solves: ensure matvec is correct and efficient; tune iterative solver options.

---

## 9) Parallel Backends

FEAST supports threads, distributed workers, and MPI (if available):

```julia
using FEAST
cap = feast_parallel_capabilities()
@info "Parallel capabilities" cap

# Use threads if available
res_t = feast(A, (Emin, Emax), M0=20, parallel=:threads)

# Or use distributed workers (addprocs required)
# res_d = feast(A, (Emin, Emax), M0=20, parallel=:distributed)
```

MPI paths require MPI.jl and a proper MPI environment.

---

## 10) Quick Checklist

- [ ] Install and verify FEAST
- [ ] Choose search region and M0
- [ ] Start dense/sparse; switch to matrix-free if needed
- [ ] Tune parameters (fpm) for accuracy/performance
- [ ] Scale up with threads/distributed/MPI
- [ ] Validate results and residuals

---

## Where to Go Next

- Getting Started: basics and common workflows
- Matrix-Free Interface: large-scale problems without assembling matrices
- Custom Contours: Gauss/Zolotarev and user-defined nodes/weights
- Performance Guide: tuning and resource selection

Happy computing!

