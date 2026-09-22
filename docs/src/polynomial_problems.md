# Polynomial Eigenvalue Problems

FeastKit.jl supports polynomial eigenvalue problems (PEPs) of the form:

**P(λ) x = (A₀ + λ A₁ + λ² A₂ + ... + λᵖ Aₚ) x = 0**

This guide covers the theory, implementation, and practical usage of FeastKit's polynomial eigenvalue solvers.

```@contents
Pages = ["polynomial_problems.md"]
Depth = 2
```

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
For large problems, use the matrix-free polynomial interface. Coefficients
must have complex floating-point elements for this high-level assembled call.
Its `M0` is multiplied by the degree when sizing the companion subspace;
choose `M0 ≤ n`. The matrix-free convenience call uses `M0` (or
`subspace_size`) directly in the `degree*n` companion space. The assembled
polynomial wrapper accepts `M0` and `fpm`; it does not accept the named
linear-solver controls added to `feast` and `feast_general`.

```@example polynomial_basic
using FeastKit, LinearAlgebra

# P(λ) = λ²I - diag(1, 4, 9), with roots ±1, ±2, ±3.
coeffs = [Matrix(Diagonal(ComplexF64[-1, -4, -9])),
          zeros(ComplexF64, 3, 3), Matrix{ComplexF64}(I, 3, 3)]

# Select the positive roots 1 and 2.
center, radius = 1.5 + 0.0im, 0.75
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[8] = 32
fpm[16] = 1

result = feast_polynomial(coeffs, center, radius; M0=3, fpm=fpm)
@assert result.converged
@assert isapprox(sort(real.(result.values)), [1.0, 2.0]; atol=1e-9)

println("Found $(result.M) eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
```

### Result Structure

Returns `FeastGeneralResult` with complex eigenvalues:

```@example polynomial_basic
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

```@example polynomial_symmetric_drivers
using FeastKit, LinearAlgebra
coeffs = [Matrix(Diagonal([-1.0, -4.0, -9.0])), zeros(3, 3),
          Matrix{Float64}(I, 3, 3)]
fpm = feastinit().fpm
fpm[8] = 32
fpm[16] = 1
real_result = feast_sypev!(coeffs, 2, 1.5+0.0im, 0.75, 3, copy(fpm))
complex_coeffs = [ComplexF64.(C) for C in coeffs]
hermitian_result = feast_hepev!(complex_coeffs, 2, 1.5+0.0im, 0.75, 3, copy(fpm))
for result in (real_result, hermitian_result)
    @assert result.converged && result.M == 2
    @assert isapprox(sort(real.(result.values)), [1.0, 2.0]; atol=1e-9)
end
real_result.values
```

### Undamped Vibrations

For positive definite stiffness `K` and mass `M`, the undamped polynomial
`K + λ²M` has imaginary roots `λ = ±iω`. Real symmetric coefficients alone
do not imply real polynomial eigenvalues. To solve for real frequencies
instead, formulate `K - ω²M`.

```@example polynomial_undamped
using FeastKit, LinearAlgebra

K = Matrix(Diagonal([1.0, 4.0, 9.0]))
C = zeros(3, 3)
M = Matrix{Float64}(I, 3, 3)
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[8] = 32
fpm[16] = 1

# Select the roots i and 2i.
result = feast_sypev!([K, C, M], 2, 0.0 + 1.5im, 0.75, 3, fpm)
@assert result.converged
@assert isapprox(sort(result.values; by=imag), [1.0im, 2.0im]; atol=1e-9)
sort(imag.(result.values))  # Angular frequencies in this example
```

### General Complex Problems

For non-Hermitian coefficient matrices:

```@example polynomial_general_driver
using FeastKit, LinearAlgebra
# Upper-triangular P(λ) has diagonal entries λ²-1, λ²-4, λ²-9.
A0 = ComplexF64[-1 0.1im 0; 0 -4 0.2; 0 0 -9]
A1 = zeros(ComplexF64, 3, 3)
A2 = Matrix{ComplexF64}(I, 3, 3)
fpm = feastinit().fpm
fpm[8] = 32
fpm[16] = 1
result = feast_gepev!([A0, A1, A2], 2, 1.5+0.0im, 0.75, 3, fpm)
@assert result.converged && result.M == 2
@assert isapprox(sort(real.(result.values)), [1.0, 2.0]; atol=1e-9)
result.values
```

### Sparse Polynomial Problems

For large-scale problems with sparse coefficient matrices:

```@example polynomial_basic
using SparseArrays
# Continue with the diagonal coefficients from Basic Usage, stored sparsely.
sparse_coeffs = [sparse(real.(C)) for C in coeffs]
sparse_result = feast_scsrpev!(sparse_coeffs, 2, center, radius, 3, copy(fpm))
@assert sparse_result.converged && sparse_result.M == 2
@assert isapprox(sort(real.(sparse_result.values)), [1.0, 2.0]; atol=1e-9)
sparse_result.values
```

---

## Examples

### Example 1: Damped Vibration Problem

```@example polynomial_damped
using FeastKit, LinearAlgebra

# Damped vibration: (λ²M + λC + K)x = 0
n = 8

# Physical matrices
K = SymTridiagonal(4*ones(n), -ones(n-1))  # Stiffness
M = I(n)                                    # Mass
C = 0.1 * K                                 # Proportional damping

# Convert to complex for general PEP
K_c = Complex.(Matrix(K))
M_c = Complex.(Matrix(M))
C_c = Complex.(Matrix(C))

# Select the eight roots in the upper half-plane.
result = feast_polynomial([K_c, C_c, M_c], 0.0+2.0im, 1.0; M0=n)
@assert result.converged
k = eigvals(K)
expected = -0.05 .* k .+ im .* sqrt.(k .- 0.0025 .* k.^2)
@assert isapprox(sort(result.values; by=imag), expected; atol=1e-8)

# Analyze results
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    ω_n = abs(λ)              # Undamped natural angular frequency
    ω_d = abs(imag(λ))        # Damped angular frequency
    ζ = -real(λ) / abs(λ)     # Damping ratio for this proportional damping
    println("Mode $i: ω_n = $ω_n, ω_d = $ω_d, ζ = $ζ")
end
```

### Example 2: Gyroscopic System

```@example polynomial_gyroscopic
using FeastKit, LinearAlgebra

# Gyroscopic system: (λ²M + λG + K)x = 0
# G is skew-symmetric (gyroscopic matrix)

n = 2
K = Matrix(Diagonal([1.0, 4.0]))
M = Matrix{Float64}(I, n, n)

# Skew-symmetric gyroscopic matrix
G = [0.0 -0.2; 0.2 0.0]

# Convert to complex
coeffs = [Complex.(K), Complex.(G), Complex.(M)]

# Positive definite K and M with skew-symmetric G give imaginary roots here.
result = feast_polynomial(coeffs, 0.0+1.5im, 0.8; M0=n)
@assert result.converged
# det(P(λ)) = λ⁴ + 5.04λ² + 4.
expected = im .* sqrt.([(5.04 - sqrt(5.04^2 - 16))/2,
                       (5.04 + sqrt(5.04^2 - 16))/2])
@assert isapprox(sort(result.values; by=imag), expected; atol=1e-8)

println("Gyroscopic eigenvalues (should be imaginary):")
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    println("  λ[$i] = $(real(λ)) + $(imag(λ))im")
end
```

### Example 3: Spatial Wavenumbers

The one-dimensional problem `-u'' = k²u` with zero endpoint values gives
`(K - k²M)x = 0` after discretization. The polynomial eigenvalue is already
the wavenumber `k`.

```@example polynomial_wavenumbers
using FeastKit, SparseArrays, LinearAlgebra

n = 16
h = 1.0 / (n + 1)

# Difference stiffness and lumped mass on a unit interval
K = spdiagm(-1 => -ones(n-1)/h, 0 => 2*ones(n)/h, 1 => -ones(n-1)/h)
M = spdiagm(0 => h * ones(n))

# This is quadratic in k: (K - k²M)x = 0.
K_c = sparse(Complex.(K))
M_c = sparse(Complex.(-M))
zero_mat = spzeros(ComplexF64, n, n)
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[8] = 32
fpm[16] = 1

# Find the first two positive wavenumbers using a sparse companion pencil.
result = feast_hcsrpev!([K_c, zero_mat, M_c], 2, 4.5 + 0.0im, 2.5, n, fpm)
@assert result.converged
expected = [2/h * sin(j*π/(2*(n+1))) for j in 1:2]
@assert isapprox(sort(real.(result.values)), expected; atol=1e-8)

println("Propagating wavenumbers:")
for i in 1:result.M
    k = result.lambda[i]
    println("  k[$i] = $k")
end
```

### Example 4: Delay System Approximation

This is a local Taylor approximation of the exponential near `λτ = 0`;
its roots are not automatically accurate eigenvalues of the delay equation.
Check them against the original characteristic equation for the application.

```@example polynomial_delay
using FeastKit, LinearAlgebra

# Delay differential equation approximated as polynomial
# ẋ(t) = A₀x(t) + A₁x(t-τ)
# Characteristic equation involves exponentials, approximate with polynomials

n = 1
τ = 0.1

A0 = fill(-2.0, n, n)
A1 = fill(0.3, n, n)

# Taylor expansion: e^{-λτ} ≈ 1 - λτ + (λτ)²/2

# From (A0 + exp(-λ*τ)*A1 - λ*I)x = 0, truncated at order two
P0 = Complex.(A0 + A1)
P1 = Complex.(-Matrix{Float64}(I, n, n) - τ * A1)
P2 = Complex.(τ^2/2 * A1)

result = feast_polynomial([P0, P1, P2], -1.0+0.0im, 2.0; M0=n)
@assert result.converged
expected = (1.03 - sqrt(1.03^2 + 4*0.0015*1.7)) / (2*0.0015)
@assert isapprox(result.values, [expected]; atol=1e-8)

println("Approximate delay system eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
```

---

## Advanced Topics

### Scaling and Conditioning

Balancing coefficient norms can improve numerical scaling, but does not
guarantee a well-conditioned polynomial. Transform the search region as well
as the coefficients, and recover eigenvalues in the original units:

```@example polynomial_scaling
using FeastKit, LinearAlgebra

# Simple endpoint-norm scaling: Q(μ) = δ * P(γ*μ)
function scale_pep(coeffs)
    p = length(coeffs) - 1
    norms = [norm(A) for A in coeffs]

    # Balance the constant and highest-degree coefficient norms.
    p >= 1 && norms[1] > 0 && norms[end] > 0 ||
        throw(ArgumentError("Scaling requires nonzero endpoint coefficients"))
    γ = (norms[1] / norms[end])^(1/p)
    δ = 1 / norms[1]

    # Scale coefficient matrices
    scaled = similar(coeffs)
    for k in 0:p
        scaled[k+1] = δ * γ^k * coeffs[k+1]
    end

    return scaled, γ, δ
end

# P(λ) = λ²I - diag(100, 400, 900), with positive roots 10, 20, 30.
coeffs = [Matrix(Diagonal(ComplexF64[-100, -400, -900])),
          zeros(ComplexF64, 3, 3), Matrix{ComplexF64}(I, 3, 3)]
center, radius = 15.0 + 0.0im, 7.5
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[8] = 32
fpm[16] = 1
scaled_coeffs, γ, δ = scale_pep(coeffs)
result = feast_polynomial(scaled_coeffs, center/γ, radius/γ; M0=3, fpm=fpm)
@assert result.converged

# Recover original eigenvalues
original_values = γ .* result.values
@assert isapprox(sort(real.(original_values)), [10.0, 20.0]; atol=1e-8)
for (j, λ) in enumerate(original_values)
    Pλ = sum(λ^k * coeffs[k+1] for k in 0:2)
    @assert norm(Pλ * result.vectors[:, j]) < 1e-7
end
original_values
```

### Infinite Eigenvalues

A regular polynomial with a singular leading coefficient has eigenvalues at
infinity, which a finite FEAST contour does not target. If the polynomial is
singular for every λ, the ordinary eigenvalue interpretation does not apply:

```@example polynomial_infinite
using LinearAlgebra
# A regular degree-one polynomial P(λ) = diag(λ, 1).
# The leading coefficient is singular: one root is finite, one is at infinity.
A0 = Matrix(Diagonal([0.0, 1.0]))
A1 = Matrix(Diagonal([1.0, 0.0]))
reference = eigvals(-A0, A1)
@assert rank(A1) == 1
@assert count(isfinite, reference) == 1 && count(isinf, reference) == 1
reference
```

### Integration Parameters

```@example polynomial_basic
fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[8] = 32   # Full-contour integration points
fpm[3] = 12   # Tolerance
fpm[4] = 30   # Iterations

result = feast_polynomial(coeffs, center, radius, M0=3, fpm=fpm)
@assert result.converged && result.M == 2
```

### Companion Linearization (Comparison)

For debugging, compare with linearized problem:

```@example polynomial_basic
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
@assert isapprox(sort(real.(λ_linearized)), [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]; atol=1e-9)
```

---

## Troubleshooting

The following examples continue the coefficients and search region from Basic
Usage. The scaling snippet continues the separate Scaling and Conditioning example.

### No Eigenvalues Found

1. **Check search region**: Eigenvalues may be elsewhere
```@example polynomial_basic
# Linearize and find all eigenvalues for reference
A_lin, B_lin = companion_linearize(coeffs)
λ_all = eigvals(A_lin, B_lin)
println("All eigenvalues: $λ_all")
```

2. **Expand search region**
```@example polynomial_basic
# Increase the radius without putting a root on the boundary.
result = feast_polynomial(coeffs, center, 1.75; M0=3)
@assert result.converged && result.M == 3
@assert isapprox(sort(real.(result.values)), [1.0, 2.0, 3.0]; atol=1e-9)
result.values
```

### Poor Accuracy

1. **Scale the problem**
```@example polynomial_scaling
scaled_coeffs, γ, δ = scale_pep(coeffs)
```

2. **Increase integration points**
```@example polynomial_basic
fpm[8] = 48  # Full-contour quadrature points
```

3. **Check condition number**
```@example polynomial_basic
for (i, A) in enumerate(coeffs)
    println("Condition of A[$i]: $(cond(A))")
end
```

### Spurious Eigenvalues

If eigenvalues appear that shouldn't exist:

1. **Check residual**: True eigenvalues have small residuals
```@example polynomial_basic
for i in 1:result.M
    λ = result.lambda[i]
    x = result.q[:, i]
    P_λ = sum(λ^k * coeffs[k+1] for k in 0:length(coeffs)-1)
    residual = norm(P_λ * x)
    @assert residual < 1e-8
    println("λ[$i]: residual = $residual")
end
```

2. **Check convergence**: Increase `M0` if the search subspace is saturated
3. **Refine search region**: Use smaller radius

---

## API Reference

### Main Functions

A polynomial eigenvalue problem is always searched over a **disc in the complex
plane**, never a real interval: the calls below take a complex `center` and a
real `radius`. Low-level drivers also take `degree` explicitly; the high-level
wrapper infers it from the coefficient vector.

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
@assert good.converged
@assert isapprox(sort(real.(good.values)), [2.0, 3.0]; atol=1e-8)
sort(real.(good.lambda[1:good.M]))
```

Centred on the origin the same disc holds every root together with its
negative, and the method has nothing to work with:

```@example poly-contour
bad = feast_srcipev!(coeffs, 2, 0.0 + 0.0im, 4.0, 3, copy(fpm))
@assert !bad.converged
bad.M   # Symmetric-root cancellation can leave no usable moment directions
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
| `M0` | `Int` | Probe width; companion drivers use `degree*M0` columns |
| `fpm` | `Vector{Int}` | FEAST parameters |

---

## See Also

- [Complex Eigenvalues](complex_eigenvalues.md) - Non-Hermitian standard problems
- [Custom Contours](custom_contours.md) - Advanced contour integration
- [API Reference](api_reference.md) - Complete function documentation

---

**Solving polynomial eigenvalue problems with FeastKit.jl**

[Examples](examples.md) · [Complex Eigenvalues](complex_eigenvalues.md) · [API Reference](api_reference.md)
