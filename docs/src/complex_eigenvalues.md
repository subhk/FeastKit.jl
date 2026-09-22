# Complex Eigenvalues (Non-Hermitian Problems)

FeastKit.jl supports general non-Hermitian eigenvalue problems where eigenvalues can be complex. Full shapes are also accepted by `feast(A, contour)`. This guide covers the theory, usage, and best practices for finding complex eigenvalues.

```@contents
Pages = ["complex_eigenvalues.md"]
Depth = 2
```

---

## Overview

For **non-symmetric** or **non-Hermitian** matrices, eigenvalues are generally complex numbers. FeastKit uses circular contours in the complex plane to find eigenvalues within specified regions.

### When to Use `feast_general`

| Matrix Type | Eigenvalues | Use |
|-------------|-------------|-----|
| Real symmetric | Real | `feast()` |
| Complex Hermitian | Real | `feast()` |
| Real non-symmetric | Complex (conjugate pairs) | `feast_general()` |
| Complex non-Hermitian | Complex | `feast_general()` |

These are interface choices, not automatic switching rules. For the distinction
between real/complex type detection, symmetry checks, and explicit declarations,
see [Matrix types: detected or declared?](@ref matrix-properties). In particular,
complex symmetric matrices are not necessarily Hermitian and can have complex
eigenvalues.

---

## Mathematical Background

### The General Eigenvalue Problem

For a general (non-Hermitian) matrix pencil:

**A x = λ B x**

where A and B are complex matrices, the eigenvalues λ can be anywhere in the complex plane.

### FEAST for Complex Eigenvalues

FEAST finds eigenvalues inside a circular contour:

```
           Im(λ)
             ↑
             │     ●  λ₂
     ╭───────●─────────╮
     │   λ₁  │  center │
─────●───────●─────────●───→ Re(λ)
     │       │    r    │
     │    λ₃ ●         │
     ╰─────────────────╯
           λ₄ ●  (outside - not found)
```

The algorithm uses contour integration:

**Q̃ = (1/2πi) ∮_C (z B - A)⁻¹ B Q dz**

where C is the circular contour.

---

## Basic Usage

### Simple Example

```@example cx
using FeastKit, LinearAlgebra

# A non-Hermitian triangular matrix has its eigenvalues on the diagonal.
A = ComplexF64[0.5+0.1im 0.2 0; 0 1.0-0.2im 0.3; 0 0 3.0+1.0im]
B = Matrix{ComplexF64}(I, 3, 3)

# Define circular search region
center = 0.0 + 0.0im  # Center of circle
radius = 2.0          # Radius of circle

# Find eigenvalues
result = feast_general(A, B, center, radius; subspace_size=3, tol=1e-10)
@assert result.converged
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 1.0-0.2im]; atol=1e-9)

# Display results
println("Found $(result.M) eigenvalues:")
for i in 1:result.M
    λ = result.lambda[i]
    println("  λ[$i] = $(real(λ)) + $(imag(λ))im")
end
```

### Result Structure

`feast_general` returns a `FeastGeneralResult`:

```julia
struct FeastGeneralResult{T<:Real}
    lambda::Vector{Complex{T}}  # Complex eigenvalues
    q::Matrix{Complex{T}}       # Complex eigenvectors
    M::Int                      # Number found
    res::Vector{T}              # Residuals
    info::Int                   # Status code
    epsout::T                   # Final residual
    loop::Int                   # Iterations
end
```

---

## Search Regions

### Circular Contours

The standard search region is a circle defined by center and radius:

```@example cx
# Continue with A and B from the simple example.
near_origin = feast_general(A, B, 0.0+0.0im, 0.75; subspace_size=3)
near_second = feast_general(A, B, 1.0-0.2im, 0.25; subspace_size=3)
wide = feast_general(A, B, 1.5+0.0im, 2.5; subspace_size=3)
@assert near_origin.converged && near_origin.M == 1
@assert isapprox(only(near_origin.values), 0.5+0.1im; atol=1e-9)
@assert near_second.converged && near_second.M == 1
@assert isapprox(only(near_second.values), 1.0-0.2im; atol=1e-9)
@assert wide.converged && wide.M == 3
wide.values
```

### Choosing Center and Radius

**Strategy 1: Estimate from Gershgorin circles**
```@example cx
# Gershgorin bounds give rough eigenvalue locations
function gershgorin_bounds(A)
    n = size(A, 1)
    radii = [sum(abs.(A[i, :])) - abs(A[i,i]) for i in 1:n]
    centers = diag(A)
    return centers, radii
end

centers, radii = gershgorin_bounds(A)
# Use these to choose search region
@assert all(λ -> any(abs.(λ .- centers) .<= radii .+ 1e-12), eigvals(A))
```

**Strategy 2: Use independent spectral estimates**

For a large sparse problem, a separate extremal eigensolver can help locate
part of the spectrum. Such estimates do not establish the count inside an
arbitrary circle; choose a subspace with headroom and check convergence.

**Strategy 3: Physical knowledge**
- For stability analysis: search near the imaginary axis
- For resonances: search near expected frequencies

---

## Examples

### Example 1: Convection-Diffusion Operator

```@example cxcd
using FeastKit, SparseArrays, LinearAlgebra

# 1D convection-diffusion: -εu'' + cu' = λu
# Non-symmetric due to convection term

n = 200
ε = 0.1      # Diffusion. Very small ε (large mesh Péclet number c*h/(2ε))
c = 1.0      # Convection. makes this operator so non-normal that even a dense
h = 1.0 / (n + 1)   # eigensolver loses accuracy -- keep the cell Péclet modest.

# Discretization (non-symmetric)
diag_main = 2ε/h^2 * ones(n)
diag_upper = (-ε/h^2 + c/(2h)) * ones(n-1)
diag_lower = (-ε/h^2 - c/(2h)) * ones(n-1)

A = spdiagm(-1 => diag_lower, 0 => diag_main, 1 => diag_upper)
A = Complex.(A)
B = sparse(Complex{Float64}(1.0)I, n, n)

# The spectrum sits on 2ε/h² ± 2√(bc) and runs from ~3.5 up to ~1.6e4, so a
# circle at the origin encloses nothing at all. Centre it on the low end,
# and give M0 headroom over the 8 eigenvalues the disc contains.
result = feast_general(A, B, 34.5+0.0im, 37.7, M0=20)

println("Convection-diffusion eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
expected = sort([2ε/h^2 + 2sqrt(diag_upper[1]*diag_lower[1])*cos(k*π/(n+1)) for k in 1:n])
expected = filter(λ -> abs(λ - (34.5+0im)) < 37.7, expected)
@assert result.converged && result.M == 8
@assert isapprox(sort(real.(result.values)), expected; atol=1e-7)
```

### Example 2: Orr-Sommerfeld Equation (Hydrodynamic Stability)

Template: supply your discretized matrices and boundary conditions. FeastKit
does not assemble this model; the example cannot run until `A` and `B` exist.
The disc below includes both signs of the imaginary part; the final filter
selects positive-imaginary modes within that disc, not every unstable mode.

```julia
using FeastKit, LinearAlgebra

# Simplified Orr-Sommerfeld for plane Poiseuille flow
# Eigenvalues determine flow stability

n = 100
Re = 5000  # Reynolds number
α = 1.0    # Wavenumber

# Build the Orr-Sommerfeld matrices (simplified)
# A = L, B = M where L*φ = c*M*φ

# Template: assemble your Orr–Sommerfeld A and B, including boundary conditions.
# These matrices are not constructed by FeastKit.

# Search for unstable modes (positive imaginary part)
center = 0.5 + 0.0im  # Wave speed
radius = 0.3

result = feast_general(A, B, center, radius, M0=30)
@assert result.converged result.message

# Check for unstable eigenvalues
unstable = [λ for λ in result.lambda[1:result.M] if imag(λ) > 0]
println("Unstable modes: $(length(unstable))")
```

### Example 3: Complex Symmetric Matrix

```@example complex_symmetric
using FeastKit, LinearAlgebra
A = ComplexF64[1 im 0; im 1 0; 0 0 4]
@assert issymmetric(A) && !ishermitian(A)
result = feast_general(A, 1.0+0im, 1.5; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=imag), [1-im, 1+im]; atol=1e-9)
result.values
```

### Example 4: Quadratic Eigenvalue Problem (Linearized)

```@example complex_linearized
using FeastKit, LinearAlgebra
# (K + λC + λ²M)x = 0 has roots ±im and ±2im.
n = 2
M = Matrix{ComplexF64}(I, n, n)
C = zeros(ComplexF64, n, n)
K = Matrix(Diagonal(ComplexF64[1, 4]))
Id = Matrix{ComplexF64}(I, n, n)
Z = zeros(ComplexF64, n, n)
A_lin = [Z Id; -K -C]
B_lin = [Id Z; Z M]
result = feast_general(A_lin, B_lin, 0.0+1.5im, 0.8; subspace_size=4)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=imag), [1.0im, 2.0im]; atol=1e-9)
for (j, λ) in enumerate(result.values)
    x = result.vectors[1:n, j]
    @assert norm((K + λ*C + λ^2*M)*x) < 1e-9
end
result.values
```

---

## Advanced Topics

### Custom Integration Parameters

```@example cx
fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[8] = 32   # Full-contour integration points
fpm[3] = 14   # Higher precision (10^-14)
fpm[4] = 30   # More iterations

result = feast_general(A, B, center, radius, M0=20, fpm=fpm)
@assert result.converged && result.M == 2
```

### Left and Right Eigenvectors

For non-Hermitian problems, left and right eigenvectors differ:

**Right**: A x = λ B x
**Left**: y^H A = λ y^H B

```@example cx
# FeastKit returns right eigenvectors in result.q
# For left eigenvectors, solve the adjoint problem:
result_left = feast_general(A', B', conj(center), radius, M0=20)
# Columns of result_left.q are the left eigenvectors y themselves.
# Match conj.(result_left.values) to the right eigenvalues before pairing them.
@assert result_left.converged && result_left.M == result.M
for λ in result.values
    j = argmin(abs.(conj.(result_left.values) .- λ))
    y = result_left.vectors[:, j]
    @assert norm(y' * A - λ * y' * B) < 1e-8
end
```

### Defective Matrices

Matrices with repeated eigenvalues and incomplete eigenvectors:

```@example complex_defective
using FeastKit, LinearAlgebra
# Jordan block - defective matrix
A = [2.0+0im 1.0+0im 0.0+0im;
     0.0+0im 2.0+0im 1.0+0im;
     0.0+0im 0.0+0im 2.0+0im]
B = Matrix{ComplexF64}(I, 3, 3)

# FEAST may have trouble with defective matrices
# Use the full matrix dimension; larger requests would be clamped to 3.
result = feast_general(A, B, 2.0+0.0im, 0.1, M0=3)
result.converged, result.message
```

A defective matrix has fewer independent eigenvectors than its algebraic
multiplicity. Increasing the subspace cannot create the missing eigenvectors;
inspect the status and residuals before using the result.

### Sparse Matrices

```@example complex_sparse
using FeastKit, SparseArrays, LinearAlgebra
A = spdiagm(0 => ComplexF64[0.5+0.1im, 1.0+0.2im, 4.0],
            1 => ComplexF64[0.2, 0.3])
B = spdiagm(0 => ones(ComplexF64, 3))
fpm = feastinit().fpm
result = feast_gcsrgv!(A, B, 1.0+0.1im, 1.0, 3, fpm)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 1.0+0.2im]; atol=1e-9)
result.values
```

---

## Troubleshooting

### No Eigenvalues Found

**Problem**: `result.M == 0`

**Solutions**:
1. Check if eigenvalues exist in the region
```@example cx
# Compute a reference spectrum with a dense solver. Only do this for small A:
# Matrix(A) is dense and eigvals is O(n^3), so at n = 10⁴ this allocates ~1.6 GB
# per matrix and will not finish.
@assert size(A, 1) <= 2000 "densify only for small problems"
λ_all = eigvals(Matrix(A), Matrix(B))
in_region = [λ for λ in λ_all if abs(λ - center) < radius]
println("Expected eigenvalues: $(length(in_region))")
```

2. Expand search region
```@example cx
result = feast_general(A, B, center, 2*radius, M0=size(A, 1))
```

3. Increase M0
```@example cx
result = feast_general(A, B, center, radius, M0=size(A, 1))
```

### Poor Convergence

**Problem**: `result.epsout` is large

**Solutions**:
1. Increase integration points
```@example cx
fpm[8] = 48  # Full-contour quadrature points
```

2. Increase iterations
```@example cx
fpm[4] = 50  # More refinement iterations
```

3. Check matrix conditioning
```@example cx
# cond has no sparse method; densify the (small) shifted matrix explicitly.
κ = cond(Array(A - center*B))
println("Condition number: $κ")
# High condition number → numerical difficulties
```

### Eigenvalues on the Contour

**Problem**: Eigenvalues exactly on the circle boundary

**Solution**: Move the center or radius so target eigenvalues have a margin
inside the boundary. Recheck which eigenvalues the revised region encloses.
```@example boundary_adjustment
using FeastKit, LinearAlgebra
A = Matrix(Diagonal(ComplexF64[1, 2, 4]))
# Radius 1 would put λ=1 on the boundary. Give it a margin instead.
result = feast_general(A, 0.0+0.0im, 1.2; subspace_size=3)
@assert result.converged && result.M == 1
@assert isapprox(only(result.values), 1; atol=1e-9)
result.values
```

### Memory Issues

**Problem**: Out of memory for large problems

**Solutions**:
1. Use sparse matrices
2. Reduce M0
3. Use complex operators with the [matrix-free interface](matrix_free_interface.md)

---

## API Reference

### Main Functions

```julia
# High-level interface
feast_general(A, B, center, radius; M0=10, fpm=nothing)

# Dense solver
feast_gegv!(A, B, center, radius, M0, fpm)
feast_geev!(A, center, radius, M0, fpm)  # Standard problem

# Sparse solver
feast_gcsrgv!(A, B, center, radius, M0, fpm)
feast_gcsrev!(A, center, radius, M0, fpm)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `AbstractMatrix` | System matrix; the high-level call promotes real input to complex |
| `B` | `AbstractMatrix` | Mass matrix with matching dimensions |
| `center` | `Complex` | Center of search circle |
| `radius` | `Real` | Radius of search circle |
| `M0` / `subspace_size` | `Int` | Search-subspace size, preferably larger than the enclosed count |
| `fpm` | `Vector{Int}`, `FeastParameters`, or `nothing` | FEAST parameters for the high-level call |

---

## See Also

- [Custom Contours](custom_contours.md) - Advanced contour integration
- [Polynomial Problems](polynomial_problems.md) - Higher-order eigenvalue problems
- [API Reference](api_reference.md) - Complete function documentation

---

**Solving non-Hermitian eigenvalue problems with FeastKit.jl**

[Examples](examples.md) · [Custom Contours](custom_contours.md) · [API Reference](api_reference.md)
