# Complex Eigenvalues (Non-Hermitian Problems)

FeastKit.jl supports general non-Hermitian eigenvalue problems where eigenvalues can be complex. This guide covers the theory, usage, and best practices for finding complex eigenvalues.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Basic Usage](#basic-usage)
- [Search Regions](#search-regions)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

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

```julia
using FeastKit, LinearAlgebra

# Create a non-symmetric matrix
n = 100
A = randn(ComplexF64, n, n)
B = Matrix{ComplexF64}(I, n, n)

# Define circular search region
center = 0.0 + 0.0im  # Center of circle
radius = 2.0          # Radius of circle

# Find eigenvalues
result = feast_general(A, B, center, radius, M0=15)

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

```julia
# Eigenvalues near the origin
result = feast_general(A, B, 0.0+0.0im, 1.0, M0=10)

# Eigenvalues near (2 + 3i)
result = feast_general(A, B, 2.0+3.0im, 0.5, M0=10)

# Large search region
result = feast_general(A, B, 0.0+0.0im, 10.0, M0=50)
```

### Choosing Center and Radius

**Strategy 1: Estimate from Gershgorin circles**
```julia
# Gershgorin bounds give rough eigenvalue locations
function gershgorin_bounds(A)
    n = size(A, 1)
    radii = [sum(abs.(A[i, :])) - abs(A[i,i]) for i in 1:n]
    centers = diag(A)
    return centers, radii
end

centers, radii = gershgorin_bounds(A)
# Use these to choose search region
```

**Strategy 2: Use sparse eigenvalue solver for initial estimate**
```julia
using Arpack  # or other iterative solver
λ_approx = eigs(A, nev=5)[1]  # Get a few eigenvalues
# Center search around these
```

**Strategy 3: Physical knowledge**
- For stability analysis: search near the imaginary axis
- For resonances: search near expected frequencies

---

## Examples

### Example 1: Convection-Diffusion Operator

```julia
using FeastKit, SparseArrays

# 1D convection-diffusion: -εu'' + cu' = λu
# Non-symmetric due to convection term

n = 500
ε = 0.01
c = 1.0
h = 1.0 / (n + 1)

# Discretization (non-symmetric)
diag_main = 2ε/h^2 * ones(n)
diag_upper = (-ε/h^2 + c/(2h)) * ones(n-1)
diag_lower = (-ε/h^2 - c/(2h)) * ones(n-1)

A = spdiagm(-1 => diag_lower, 0 => diag_main, 1 => diag_upper)
A = Complex.(A)
B = sparse(Complex{Float64}(1.0)I, n, n)

# Find eigenvalues near the origin
result = feast_general(A, B, 0.0+0.0im, 50.0, M0=20)

println("Convection-diffusion eigenvalues:")
for i in 1:min(5, result.M)
    println("  λ[$i] = $(result.lambda[i])")
end
```

### Example 2: Orr-Sommerfeld Equation (Hydrodynamic Stability)

```julia
using FeastKit, LinearAlgebra

# Simplified Orr-Sommerfeld for plane Poiseuille flow
# Eigenvalues determine flow stability

n = 100
Re = 5000  # Reynolds number
α = 1.0    # Wavenumber

# Build the Orr-Sommerfeld matrices (simplified)
# A = L, B = M where L*φ = c*M*φ

# ... (matrix construction details)

# Search for unstable modes (positive imaginary part)
center = 0.5 + 0.0im  # Wave speed
radius = 0.3

result = feast_general(A, B, center, radius, M0=30)

# Check for unstable eigenvalues
unstable = [λ for λ in result.lambda[1:result.M] if imag(λ) > 0]
println("Unstable modes: $(length(unstable))")
```

### Example 3: Complex Symmetric Matrix

```julia
using FeastKit

# Complex symmetric (not Hermitian!)
# Arises in electromagnetics, quantum mechanics

n = 200
A = randn(ComplexF64, n, n)
A = (A + transpose(A)) / 2  # Symmetric, not Hermitian!
B = Matrix{ComplexF64}(I, n, n)

# Eigenvalues are complex (not necessarily real)
result = feast_general(A, B, 0.0+0.0im, 5.0, M0=20)

println("Complex symmetric eigenvalues:")
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    println("  λ[$i] = $(round(real(λ), digits=6)) + $(round(imag(λ), digits=6))im")
end
```

### Example 4: Quadratic Eigenvalue Problem (Linearized)

```julia
using FeastKit, LinearAlgebra

# Quadratic eigenvalue problem: (λ²M + λC + K)x = 0
# Linearize to standard form

n = 50
M = rand(ComplexF64, n, n); M = M'M + I  # Mass
C = rand(ComplexF64, n, n)                 # Damping
K = rand(ComplexF64, n, n); K = K'K + I  # Stiffness

# Linearize: [0 I; -K -C] [x; λx] = λ [I 0; 0 M] [x; λx]
A_lin = [zeros(n,n) I; -K -C]
B_lin = [I zeros(n,n); zeros(n,n) M]

result = feast_general(A_lin, B_lin, 0.0+0.0im, 3.0, M0=30)

println("Quadratic eigenvalue problem:")
println("Found $(result.M) eigenvalues")
```

---

## Advanced Topics

### Custom Integration Parameters

```julia
fpm = zeros(Int, 64)
feastinit!(fpm)

fpm[2] = 16   # More integration points for accuracy
fpm[3] = 14   # Higher precision (10^-14)
fpm[4] = 30   # More iterations

result = feast_general(A, B, center, radius, M0=20, fpm=fpm)
```

### Left and Right Eigenvectors

For non-Hermitian problems, left and right eigenvectors differ:

**Right**: A x = λ B x
**Left**: y^H A = λ y^H B

```julia
# FeastKit returns right eigenvectors in result.q
# For left eigenvectors, solve the adjoint problem:
result_left = feast_general(A', B', conj(center), radius, M0=20)
# Left eigenvectors are conjugates of result_left.q
```

### Defective Matrices

Matrices with repeated eigenvalues and incomplete eigenvectors:

```julia
# Jordan block - defective matrix
A = [2.0+0im 1.0+0im 0.0+0im;
     0.0+0im 2.0+0im 1.0+0im;
     0.0+0im 0.0+0im 2.0+0im]
B = Matrix{ComplexF64}(I, 3, 3)

# FEAST may have trouble with defective matrices
# Increase M0 beyond expected multiplicity
result = feast_general(A, B, 2.0+0.0im, 0.1, M0=5)
```

### Sparse Matrices

```julia
using SparseArrays

# Large sparse non-Hermitian problem
n = 10000
A = sprandn(ComplexF64, n, n, 0.001)
B = sparse(ComplexF64(1.0)I, n, n)

# Use sparse solver
result = feast_gcsrgv!(A, B, center, radius, M0, fpm)
```

---

## Troubleshooting

### No Eigenvalues Found

**Problem**: `result.M == 0`

**Solutions**:
1. Check if eigenvalues exist in the region
```julia
# Compute a few eigenvalues with dense solver
λ_all = eigvals(Matrix(A), Matrix(B))
in_region = [λ for λ in λ_all if abs(λ - center) < radius]
println("Expected eigenvalues: $(length(in_region))")
```

2. Expand search region
```julia
result = feast_general(A, B, center, 2*radius, M0=M0)
```

3. Increase M0
```julia
result = feast_general(A, B, center, radius, M0=2*M0)
```

### Poor Convergence

**Problem**: `result.epsout` is large

**Solutions**:
1. Increase integration points
```julia
fpm[2] = 24  # More quadrature points
```

2. Increase iterations
```julia
fpm[4] = 50  # More refinement iterations
```

3. Check matrix conditioning
```julia
κ = cond(A - center*B)
println("Condition number: $κ")
# High condition number → numerical difficulties
```

### Eigenvalues on the Contour

**Problem**: Eigenvalues exactly on the circle boundary

**Solution**: Shift the center slightly
```julia
# Original
result = feast_general(A, B, 0.0+0.0im, 1.0, M0=10)

# Shifted center
result = feast_general(A, B, 0.01+0.01im, 1.0, M0=10)
```

### Memory Issues

**Problem**: Out of memory for large problems

**Solutions**:
1. Use sparse matrices
2. Reduce M0
3. Use matrix-free interface (if available for general problems)

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
| `A` | `AbstractMatrix{Complex}` | System matrix |
| `B` | `AbstractMatrix{Complex}` | Mass matrix |
| `center` | `Complex` | Center of search circle |
| `radius` | `Real` | Radius of search circle |
| `M0` | `Int` | Maximum eigenvalues to find |
| `fpm` | `Vector{Int}` | FEAST parameters |

---

## See Also

- [Custom Contours](custom_contours.md) - Advanced contour integration
- [Polynomial Problems](polynomial_problems.md) - Higher-order eigenvalue problems
- [API Reference](api_reference.md) - Complete function documentation

---

<div align="center">
  <p><strong>Solving non-Hermitian eigenvalue problems with FeastKit.jl</strong></p>
  <a href="examples.md">Examples</a> · <a href="custom_contours.md">Custom Contours</a> · <a href="api_reference.md">API Reference</a>
</div>
