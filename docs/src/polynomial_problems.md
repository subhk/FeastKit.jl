# Polynomial Eigenvalue Problems

FeastKit.jl supports polynomial eigenvalue problems (PEPs) of the form:

**P(λ) x = (A₀ + λ A₁ + λ² A₂ + ... + λᵖ Aₚ) x = 0**

This guide covers the theory, implementation, and practical usage of FeastKit's polynomial eigenvalue solvers.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Basic Usage](#basic-usage)
- [Problem Types](#problem-types)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Overview

Polynomial eigenvalue problems arise in many applications:

| Application | Problem Type | Degree |
|-------------|--------------|--------|
| Structural dynamics with damping | Quadratic (QEP) | 2 |
| Gyroscopic systems | Quadratic | 2 |
| Delay differential equations | Higher-order | Variable |
| Acoustic waveguides | Quadratic | 2 |
| Viscoelastic materials | Rational/Polynomial | Variable |

### Choose the Polynomial Interface

The assembled `feast_polynomial` convenience interface and the `*pev!` drivers
use a companion linearization of dimension `p*n`. The convenience interface
materializes coefficients as dense matrices, even for sparse input. The
matrix-free convenience interface applies a companion operator without
assembling that enlarged matrix. Direct polynomial RCI is a separate interface;
see the implementation notes below. Check residuals in the original polynomial
and account for possible infinite eigenvalues when the leading coefficient is singular.

See [Problem Setup](problem_setup.md) for an executable minimal example and
coefficient-order, search-region, and subspace-size guidance.

---

## Mathematical Background

### Standard Form

A polynomial eigenvalue problem of degree p:

**P(λ) = A₀ + λ A₁ + λ² A₂ + ... + λᵖ Aₚ**

where Aᵢ are n×n matrices. We seek λ and x such that P(λ)x = 0.

### Quadratic Eigenvalue Problem (QEP)

The most common case (p = 2):

**(λ² M + λ C + K) x = 0**

- **M**: Mass matrix
- **C**: Damping matrix
- **K**: Stiffness matrix

### FEAST for Polynomial Problems

FEAST uses contour integration to project onto the eigenspace:

**Q̃ = (1/2πi) ∮_C P(z)⁻¹ Q dz**

The contour C encloses the eigenvalues of interest.

---

## Basic Usage

### Using `feast_polynomial`

Coefficient matrices may use dense, sparse, or structured storage. This
convenience interface materializes them as dense matrices and builds a dense
companion problem; sparse input does not make this particular solver sparse.
For large problems, use the matrix-free polynomial interface.

```julia
using FeastKit, LinearAlgebra

# Define polynomial matrices [A₀, A₁, A₂, ...]
n = 50
A0 = randn(ComplexF64, n, n)
A1 = randn(ComplexF64, n, n)
A2 = randn(ComplexF64, n, n)
coeffs = [A0, A1, A2]  # Quadratic: A0 + λ*A1 + λ²*A2

# Search region (circular contour)
center = 0.0 + 0.0im
radius = 2.0

# Solve
result = feast_polynomial(coeffs, center, radius, M0=30)

println("Found $(result.M) eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
```

### Result Structure

Returns `FeastGeneralResult` with complex eigenvalues:

```julia
result.lambda  # Complex eigenvalues
result.q       # Eigenvectors
result.M       # Number found
result.info    # Status code
result.epsout  # Residual
```

---

## Problem Types

### Symmetric/Hermitian Problems

For real symmetric or complex Hermitian coefficient matrices:

```julia
# Symmetric coefficient matrices. The drivers dispatch on Vector{Matrix{T}},
# so materialize the Symmetric wrappers.
K = Matrix(Symmetric(randn(n, n)))
C = Matrix(Symmetric(randn(n, n)))
M = Matrix(Symmetric(randn(n, n) + 5I))

coeffs = [K, C, M]

# Use symmetric solver. Even with symmetric coefficients the search region is
# a disc in the complex plane, because a quadratic in λ has complex roots.
result = feast_sypev!(coeffs, 2, center, radius, M0, fpm)
```

### Real Symmetric with Real Eigenvalues

When M, C, K are real symmetric and eigenvalues are real:

```julia
using FeastKit

# Undamped system: with C = 0 the eigenvalues come in ± pairs on the real axis
K = Matrix(Symmetric(randn(n, n) + 10I))
C = zeros(n, n)  # No damping
M = Matrix(Symmetric(randn(n, n) + 5I))

# A disc covering the positive real eigenvalues. Centre it away from the
# origin so it does not enclose a root and its negative at the same time.
result = feast_sypev!([K, C, M], 2, 1.05 + 0.0im, 0.95, M0, fpm)
```

### General Complex Problems

For non-Hermitian coefficient matrices:

```julia
# General complex matrices
A0 = randn(ComplexF64, n, n)
A1 = randn(ComplexF64, n, n)
A2 = randn(ComplexF64, n, n)

# Circular contour search
result = feast_gepev!([A0, A1, A2], 2, center, radius, M0, fpm)
```

### Sparse Polynomial Problems

For large-scale problems with sparse coefficient matrices:

```julia
using SparseArrays

# Sparse coefficient matrices
K = sprandn(n, n, 0.01) + 10I
C = sprandn(n, n, 0.01)
M = sprandn(n, n, 0.01) + 5I

# Sparse solver
result = feast_scsrpev!([K, C, M], 2, center, radius, M0, fpm)
```

---

## Examples

### Example 1: Damped Vibration Problem

```julia
using FeastKit, LinearAlgebra

# Damped vibration: (λ²M + λC + K)x = 0
n = 100

# Physical matrices
K = SymTridiagonal(4*ones(n), -ones(n-1))  # Stiffness
M = I(n)                                    # Mass
C = 0.1 * K                                 # Proportional damping

# Convert to complex for general PEP
K_c = Complex.(Matrix(K))
M_c = Complex.(Matrix(M))
C_c = Complex.(Matrix(C))

# The quadratic λ²+0.1kλ+k = 0 puts every eigenvalue on |λ| = √k with k in
# [2, 6], so there is nothing near the origin at all and radius 3 would enclose
# all 2n = 200 of them. Target a small disc on that ring instead.
result = feast_polynomial([K_c, C_c, M_c], -0.15+2.0im, 0.08, M0=18)

# Analyze results
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    ω = sqrt(-λ)  # Natural frequency (approximately)
    ζ = real(λ) / (2 * abs(imag(ω)))  # Damping ratio
    println("Mode $i: λ = $λ")
end
```

### Example 2: Gyroscopic System

```julia
using FeastKit, LinearAlgebra

# Gyroscopic system: (λ²M + λG + K)x = 0
# G is skew-symmetric (gyroscopic matrix)

n = 50
K = Symmetric(randn(n, n) + 10I)
M = Symmetric(randn(n, n) + 5I)

# Skew-symmetric gyroscopic matrix
G = randn(n, n)
G = G - G'  # Make skew-symmetric

# Convert to complex
coeffs = [Complex.(K), Complex.(G), Complex.(M)]

# Eigenvalues are (nearly) imaginary for undamped gyroscopic systems and spread
# over |λ| up to ~14, so pick a disc on the imaginary axis rather than one large
# enough to swallow the whole spectrum.
result = feast_polynomial(coeffs, 0.0+1.0im, 0.3, M0=20)

println("Gyroscopic eigenvalues (should be imaginary):")
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    println("  λ[$i] = $(real(λ)) + $(imag(λ))im")
end
```

### Example 3: Acoustic Waveguide

```julia
using FeastKit, SparseArrays

# Helmholtz equation in waveguide: (k² M + K)x = 0
# k is wavenumber (eigenvalue)

n = 200
h = 1.0 / (n + 1)

# Finite element matrices
K = spdiagm(-1 => -ones(n-1)/h, 0 => 2*ones(n)/h, 1 => -ones(n-1)/h)
M = spdiagm(0 => h * ones(n))

# This is quadratic in k: (k²M + 0*k + K)x = 0
K_c = sparse(Complex.(K))
M_c = sparse(Complex.(M))
zero_mat = sparse(zeros(ComplexF64, n, n))

# Find propagating wavenumbers
result = feast_hcsrpev!([K_c, zero_mat, M_c], 2, 25.0 + 0.0im, 25.0, 20, fpm)

println("Propagating wavenumbers:")
for i in 1:result.M
    k = sqrt(result.lambda[i])
    println("  k[$i] = $k")
end
```

### Example 4: Delay System Approximation

```julia
using FeastKit, LinearAlgebra

# Delay differential equation approximated as polynomial
# ẋ(t) = A₀x(t) + A₁x(t-τ)
# Characteristic equation involves exponentials, approximate with polynomials

n = 20
τ = 1.0

A0 = -2.0 * I(n) + 0.5 * randn(n, n)
A1 = 0.3 * randn(n, n)

# Padé approximation of e^{-λτ} gives polynomial eigenvalue problem
# For simplicity, use Taylor expansion: e^{-λτ} ≈ 1 - λτ + (λτ)²/2 - ...

# Resulting polynomial (truncated)
P0 = Complex.(A0 + A1)
P1 = Complex.(-τ * A1)
P2 = Complex.(τ^2/2 * A1)

result = feast_polynomial([P0, P1, P2], 0.0+0.0im, 5.0, M0=30)

println("Approximate delay system eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
```

---

## Advanced Topics

### Scaling and Conditioning

Polynomial problems are often ill-conditioned. Use scaling:

```julia
# Fan-Patel-Zhou scaling
function scale_pep(coeffs)
    p = length(coeffs) - 1
    norms = [norm(A) for A in coeffs]

    # Compute optimal scaling factors
    γ = (norms[1] / norms[end])^(1/(2p))
    δ = 2 / (norms[1] + norms[end] * γ^(2p))

    # Scale coefficient matrices
    scaled = similar(coeffs)
    for k in 0:p
        scaled[k+1] = δ * γ^k * coeffs[k+1]
    end

    return scaled, γ, δ
end

# Apply scaling
scaled_coeffs, γ, δ = scale_pep(coeffs)
result = feast_polynomial(scaled_coeffs, center*γ, radius*γ, M0=M0)

# Recover original eigenvalues
result.lambda ./= γ
```

### Infinite Eigenvalues

Polynomial problems may have eigenvalues at infinity:

```julia
# If Aₚ is singular, there are infinite eigenvalues
# These are not found by FEAST (which searches finite regions)

# Check for infinite eigenvalues:
if rank(coeffs[end]) < size(coeffs[end], 1)
    @warn "Leading coefficient is singular - infinite eigenvalues exist"
end
```

### Integration Parameters

```julia
fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[2] = 16   # Integration points (more for polynomials)
fpm[3] = 12   # Tolerance
fpm[4] = 30   # Iterations

result = feast_polynomial(coeffs, center, radius, M0=M0, fpm=fpm)
```

### Companion Linearization (Comparison)

For debugging, compare with linearized problem:

```julia
using LinearAlgebra

function companion_linearize(coeffs)
    p = length(coeffs) - 1
    n = size(coeffs[1], 1)

    # Block companion form
    A = zeros(ComplexF64, p*n, p*n)
    B = zeros(ComplexF64, p*n, p*n)

    # Build companion matrices
    for k in 1:p-1
        A[(k-1)*n+1:k*n, k*n+1:(k+1)*n] = I(n)
        B[(k-1)*n+1:k*n, (k-1)*n+1:k*n] = I(n)
    end

    for k in 1:p
        A[(p-1)*n+1:p*n, (k-1)*n+1:k*n] = -coeffs[k]
    end
    B[(p-1)*n+1:p*n, (p-1)*n+1:p*n] = coeffs[end]

    return A, B
end

# Compare results
A_lin, B_lin = companion_linearize(coeffs)
λ_linearized = eigvals(A_lin, B_lin)
```

---

## Troubleshooting

### No Eigenvalues Found

1. **Check search region**: Eigenvalues may be elsewhere
```julia
# Linearize and find all eigenvalues for reference
A_lin, B_lin = companion_linearize(coeffs)
λ_all = eigvals(A_lin, B_lin)
println("All eigenvalues: $λ_all")
```

2. **Expand search region**
```julia
result = feast_polynomial(coeffs, center, 2*radius, M0=M0)
```

### Poor Accuracy

1. **Scale the problem**
```julia
scaled_coeffs, γ, δ = scale_pep(coeffs)
```

2. **Increase integration points**
```julia
fpm[2] = 24  # More quadrature points
```

3. **Check condition number**
```julia
for (i, A) in enumerate(coeffs)
    println("Condition of A[$i]: $(cond(A))")
end
```

### Spurious Eigenvalues

If eigenvalues appear that shouldn't exist:

1. **Check residual**: True eigenvalues have small residuals
```julia
for i in 1:result.M
    λ = result.lambda[i]
    x = result.q[:, i]
    P_λ = sum(λ^k * coeffs[k+1] for k in 0:length(coeffs)-1)
    residual = norm(P_λ * x)
    println("λ[$i]: residual = $residual")
end
```

2. **Increase M0**: May be picking up nearby eigenvalues
3. **Refine search region**: Use smaller radius

---

## API Reference

### Main Functions

A polynomial eigenvalue problem is always searched over a **disc in the complex
plane**, never a real interval: every driver takes a complex `center` and a real
`radius`, and every one takes the degree `degree` explicitly.

```julia
# High-level interface (keyword arguments)
feast_polynomial(coeffs, center, radius; M0=10, fpm=nothing)

# Dense drivers (positional; `degree` is coeffs length - 1)
feast_sypev!(coeffs, degree, center, radius, M0, fpm)   # real coefficients
feast_hepev!(coeffs, degree, center, radius, M0, fpm)   # complex coefficients
feast_gepev!(coeffs, degree, center, radius, M0, fpm)   # complex coefficients

# Sparse drivers
feast_scsrpev!(coeffs, degree, center, radius, M0, fpm)  # real coefficients
feast_hcsrpev!(coeffs, degree, center, radius, M0, fpm)  # complex coefficients
feast_gcsrpev!(coeffs, degree, center, radius, M0, fpm)  # complex coefficients

# Moment-based drivers
feast_srcipev!(coeffs, degree, center, radius, M0, fpm)  # real coefficients
feast_grcipev!(coeffs, degree, center, radius, M0, fpm)  # complex coefficients

# IFEAST-compatible precision aliases
difeast_srcipev!(coeffs, degree, center, radius, M0, fpm)
zifeast_grcipev!(coeffs, degree, center, radius, M0, fpm)
difeast_scsrpev!(coeffs, degree, center, radius, M0, fpm)
zifeast_gcsrpev!(coeffs, degree, center, radius, M0, fpm)
```

Every driver above has an `x`-suffixed twin taking an explicit contour as two
extra trailing arguments, `Zne` and `Wne`:

```julia
feast_gepevx!(coeffs, degree, center, radius, M0, fpm, Zne, Wne)
feast_scsrpevx!(coeffs, degree, center, radius, M0, fpm, Zne, Wne)
```

The `*pev!` and `*csrpev!` drivers linearize the polynomial into a companion
pencil of size `degree * N` and hand it to the general solver. The
`feast_srcipev!` / `feast_grcipev!` pair instead works directly on the
polynomial through its contour moments, so it never forms the larger pencil --
at the cost of needing a contour that is not symmetric about the origin (see
*Choosing a contour* below).

### RCI Interface

`feast_srcipev!`, `feast_grcipev!` and their `x` twins are each **two** methods.
Passing the coefficient vector first calls the driver shown above; passing an
`ijob` reference first calls the reverse-communication kernel, which returns to
you for each factorization, solve and multiply:

```julia
feast_grcipev!(ijob, degree, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
               center, radius, M0, lambda, q, mode, res, info; state)

feast_grcipevx!(ijob, degree, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                center, radius, M0, lambda, q, mode, res, info, Zne, Wne; state)
```

`state` is a `FeastPolyRCIState{T}` and must be the same object for every call
in one RCI loop. The kernel emits `Feast_RCI_FACTORIZE`, `Feast_RCI_SOLVE`,
`Feast_RCI_MULT_A` and `Feast_RCI_DONE`; on `MULT_A` write `P(lambda[j]) * q[:, j]`
into `workc[:, j]` for `j = 1:mode[]`.

### Choosing a contour

`feast_srcipev!` and `feast_grcipev!` recover eigenvalues from the contour
moments `A0 = ∮ P(z)⁻¹ dz` and `A1 = ∮ z P(z)⁻¹ dz`. A disc placed so that it
encloses a root and its negative in equal measure makes their residues cancel:
`A0` vanishes, the reduced pencil is singular, and nothing is found. Offset the
disc instead.

```@example poly-contour
using FeastKit, LinearAlgebra

# P(λ) = λ²I - diag(1, 4, 9), so the spectrum is ±1, ±2, ±3.
coeffs = [Matrix(Diagonal([-1.0, -4.0, -9.0])), zeros(3, 3),
          Matrix{Float64}(I, 3, 3)]

fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[8] = 32     # full-contour integration points
fpm[16] = 1     # trapezoidal, the accurate rule on a circle

# A disc holding 2 and 3 but neither -2 nor -3.
good = feast_srcipev!(coeffs, 2, 2.5 + 0.0im, 1.0, 3, copy(fpm))
sort(real.(good.lambda[1:good.M]))
```

Centred on the origin the same disc holds every root together with its
negative, and the method has nothing to work with:

```@example poly-contour
bad = feast_srcipev!(coeffs, 2, 0.0 + 0.0im, 4.0, 3, copy(fpm))
bad.M   # 0
```

The linearizing drivers (`feast_gepev!`, `feast_scsrpev!`, …) build a companion
pencil instead of contour moments and are not subject to this restriction.

`M0` may exceed the number of eigenvalues inside the contour — which is the
normal case, since the count is what you are trying to find out. The kernel
truncates the moment `S0 = U Σ Wᴴ` at its numerical rank before forming the
reduced matrix, so the extra width costs work but not accuracy.

If every available probe direction yields an enclosed eigenpair
(`M == min(N, M0)`), the kernel returns `Feast_ERROR_M0` even when the
reported residuals are small: additional enclosed roots cannot be ruled out.
Increase `M0` when it is smaller than `N`. If the probe already spans `N`
directions, use a smaller contour or a companion-linearization driver such as
`feast_gepev!`, which works in the full `degree*N` space. A degree-one problem
with all `N` eigenpairs is complete and can still return success.

Accuracy is governed by how well the quadrature resolves the contour integral
rather than by refinement loops. If the residual is too large, add contour
points (`fpm[8]`) before raising the loop count (`fpm[4]`).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `coeffs` | `Vector{AbstractMatrix}` | Coefficient matrices [A₀, A₁, ..., Aₚ] |
| `center` | `Complex` | Center of search circle |
| `radius` | `Real` | Radius of search circle |
| `Emin, Emax` | `Real` | Search interval (symmetric problems) |
| `M0` | `Int` | Maximum eigenvalues to find |
| `fpm` | `Vector{Int}` | FEAST parameters |

---

## See Also

- [Complex Eigenvalues](complex_eigenvalues.md) - Non-Hermitian standard problems
- [Custom Contours](custom_contours.md) - Advanced contour integration
- [API Reference](api_reference.md) - Complete function documentation

---

<div align="center">
  <p><strong>Solving polynomial eigenvalue problems with FeastKit.jl</strong></p>
  <a href="examples.md">Examples</a> · <a href="complex_eigenvalues.md">Complex Eigenvalues</a> · <a href="api_reference.md">API Reference</a>
</div>
