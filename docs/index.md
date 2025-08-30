# FeastKit.jl Documentation

<div align="center">
  <h1>🎯 FeastKit.jl</h1>
  <p><em>Fast Eigenvalue Algorithm using Spectral Transformations in Julia</em></p>
  
  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="#examples">Examples</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#advanced-features">Advanced</a>
  </p>
</div>

---

## What is Feast?

FeastKit.jl is a Julia implementation of the **FEAST eigenvalue algorithm**, a powerful numerical method for finding eigenvalues and eigenvectors of large sparse matrices within specified intervals or regions. Unlike traditional methods that compute all eigenvalues, FeastKit allows you to:

- 🎯 **Target specific eigenvalues** in intervals `[Emin, Emax]` or complex regions
- 🚀 **Handle very large problems** (millions of unknowns) efficiently  
- 🔧 **Work matrix-free** without storing explicit matrices
- ⚡ **Leverage parallelization** for high-performance computing
- 🎨 **Use custom contour integration** for optimal convergence

### Key Features

| Feature | Description |
|---------|-------------|
| **Interval Targeting** | Find eigenvalues only in `[Emin, Emax]` |
| **Matrix-Free** | Use callback functions instead of explicit matrices |
| **Parallel Computing** | MPI and shared-memory parallelization |
| **Multiple Matrix Types** | Dense, sparse, banded, custom operators |
| **Complex Eigenvalues** | General non-Hermitian problems |
| **Custom Integration** | Gauss-Legendre, Zolotarev, custom contours |

---

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("FeastKit")  # When available from registry

# Or for development:
Pkg.add(url="https://github.com/your-repo/FeastKit.jl")
```

### Your First FeastKit Calculation

```julia
using FeastKit, LinearAlgebra

# Create a test matrix (1000x1000 tridiagonal)
n = 1000
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))

# Find eigenvalues between 0.5 and 1.5
result = feast(A, (0.5, 1.5), M0=10)

println("Found $(result.M) eigenvalues:")
println(result.lambda[1:result.M])
```

**That's it!** Feast found the eigenvalues in your target interval.

### Matrix-Free Example

For very large problems, use matrix-free operations:

```julia
# Define matrix-vector multiplication (no explicit matrix needed!)
function A_mul!(y, x)
    n = length(x)
    y[1] = 2*x[1] - x[2]
    for i in 2:n-1
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    end
    y[n] = -x[n-1] + 2*x[n]
end

# Create matrix-free operator
A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)

# Solve the same way!
result = feast(A_op, (0.5, 1.5), M0=10)
```

---

## Examples Gallery

### Basic Examples

<details>
<summary><strong>📊 Dense Matrix Eigenvalues</strong></summary>

```julia
using FeastKit, LinearAlgebra

# Create a random symmetric matrix
n = 500
A = randn(n, n)
A = A + A'  # Make symmetric

# Find eigenvalues near zero
result = feast(A, (-1.0, 1.0), M0=20)

println("Eigenvalues near zero:")
for i in 1:result.M
    println("λ[$i] = $(result.lambda[i])")
end
```
</details>

<details>
<summary><strong>🕸️ Sparse Matrix Problems</strong></summary>

```julia
using FeastKit, SparseArrays

# Large sparse symmetric matrix
n = 10000
A = sprand(n, n, 0.001)  # 0.1% density
A = A + A' + 5*I         # Make symmetric positive definite

# Find largest eigenvalues
result = feast(A, (4.8, 5.2), M0=8)

println("Largest eigenvalues: $(result.lambda[1:result.M])")
```
</details>

<details>
<summary><strong>🔧 Generalized Eigenvalue Problem</strong></summary>

```julia
using FeastKit

# Create matrices A and B
n = 1000
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
B = SymTridiagonal(3.0 * ones(n), -0.5 * ones(n-1))

# Solve A*x = λ*B*x
result = feast(A, B, (0.1, 0.8), M0=15)

println("Generalized eigenvalues: $(result.lambda[1:result.M])")
```
</details>

### Advanced Examples

<details>
<summary><strong>🌐 2D Partial Differential Equation</strong></summary>

```julia
using FeastKit

# 2D Laplacian eigenvalue problem: -Δu = λu
nx, ny = 100, 100
n = nx * ny
h = 1.0 / (nx + 1)

# Matrix-free 2D Laplacian
function laplacian_2d!(y, x)
    fill!(y, 0)
    for j in 1:ny, i in 1:nx
        k = (j-1) * nx + i
        y[k] += 4 * x[k] / h^2
        
        # Neighbors with boundary conditions
        i > 1  && (y[k] -= x[k-1] / h^2)
        i < nx && (y[k] -= x[k+1] / h^2) 
        j > 1  && (y[k] -= x[k-nx] / h^2)
        j < ny && (y[k] -= x[k+nx] / h^2)
    end
end

A_op = LinearOperator{Float64}(laplacian_2d!, (n, n), 
                              issymmetric=true, isposdef=true)

# Find smallest eigenvalues (fundamental modes)
λ_min = 2π^2 * (1/nx^2 + 1/ny^2)
result = feast(A_op, (0.8*λ_min, 2.0*λ_min), M0=10, solver=:cg)

println("PDE eigenvalues: $(result.lambda[1:result.M])")
```
</details>

<details>
<summary><strong>🔄 Complex Non-Hermitian Problems</strong></summary>

```julia
using FeastKit

# Non-symmetric matrix with complex eigenvalues
n = 200
A = randn(ComplexF64, n, n)
B = Matrix{ComplexF64}(I, n, n)

# Search in circular region
center = 0.0 + 0.0im
radius = 2.0

result = feast_general(A, B, center, radius, M0=15)

println("Complex eigenvalues:")
for i in 1:result.M
    λ = result.lambda[i]
    println("λ[$i] = $(real(λ)) + $(imag(λ))im")
end
```
</details>

---

## Core Concepts

### The FEAST Algorithm

Feast uses **contour integration** in the complex plane to extract eigenvalues in specified regions. The key idea:

1. **Define a contour** around your region of interest
2. **Integrate along the contour** using spectral projectors  
3. **Extract eigenvalues** inside the contour via reduced eigenvalue problems

```
     Im(z)
       ↑
   ┌───●───●───┐  ← Integration contour
   │   ●   ●   │    (eigenvalues inside)
───●───●───●───●──→ Re(z)
   │   ●   ●   │
   └───●───●───┘
```

### Search Regions

**Real Intervals**: For symmetric/Hermitian matrices
```julia
result = feast(A, (Emin, Emax), M0=10)
```

**Complex Regions**: For general matrices
```julia
result = feast_general(A, B, center, radius, M0=10)
```

**Custom Contours**: For advanced users
```julia
contour = feast_contour_expert(Emin, Emax, 16, 2, 100)  # Zolotarev integration
```

---

## Performance Guide

### Choosing Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `M0` | Max eigenvalues to find | 10-50 | Memory usage, accuracy |
| `ne` | Integration points | 8-32 | Accuracy vs speed |
| `tol` | Convergence tolerance | 1e-12 | Accuracy vs iterations |
| `maxiter` | Max refinement loops | 20-100 | Convergence robustness |

### Memory Usage

| Problem Size | Standard FeastKit | Matrix-Free FeastKit |
|--------------|----------------|-------------------|
| 1,000 × 1,000 | ~24 MB | ~1 MB |
| 10,000 × 10,000 | ~2.4 GB | ~10 MB |
| 100,000 × 100,000 | ~240 GB | ~100 MB |

### Performance Tips

✅ **Use matrix-free** for large problems  
✅ **Choose appropriate solvers**: CG for SPD, GMRES for general  
✅ **Tune integration points**: More points = better accuracy, slower  
✅ **Enable parallelization** for very large problems  
✅ **Use custom contours** for challenging geometries  

---

## Troubleshooting

### Common Issues

<details>
<summary><strong>❌ "No eigenvalues found"</strong></summary>

**Cause**: Search interval doesn't contain eigenvalues

**Solutions**:
```julia
# Check eigenvalue bounds first
bounds = feast_validate_interval(A, (Emin, Emax))
println("Estimated eigenvalue range: $bounds")

# Use a broader interval
result = feast(A, (bounds[1], bounds[2]), M0=10)
```
</details>

<details>
<summary><strong>❌ "Linear solver not converging"</strong></summary>

**Cause**: Iterative solver issues in matrix-free mode

**Solutions**:
```julia
# Increase solver tolerance and iterations
result = feast(A_op, interval, 
              solver=:gmres,
              solver_opts=(rtol=1e-4, maxiter=2000, restart=50))

# Try different solver
result = feast(A_op, interval, solver=:bicgstab)

# Use custom preconditioner
P = create_preconditioner(A_op)  # Your preconditioner
result = feast(A_op, interval, 
              solver_opts=(Pl=P, rtol=1e-6))
```
</details>

<details>
<summary><strong>❌ "Memory allocation failed"</strong></summary>

**Cause**: Problem too large for available memory

**Solutions**:
```julia
# Switch to matrix-free interface
A_op = LinearOperator{Float64}(A_mul!, size(A))
result = feast(A_op, interval, M0=10)

# Reduce M0 (max eigenvalues)
result = feast(A, interval, M0=5)  # Instead of M0=20

# Use iterative refinement
fpm = zeros(Int, 64)
fpm[4] = 5  # Fewer refinement iterations
result = feast(A, interval, M0=10, fpm=fpm)
```
</details>

### Getting Help

- 📖 **Documentation**: Check [API Reference](#api-reference)
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/your-repo/FeastKit.jl/issues)
- 💬 **Discussions**: Ask questions on [GitHub Discussions](https://github.com/your-repo/FeastKit.jl/discussions)
- 📧 **Email**: Contact developers at your-email@domain.com

---

## What's Next?

Ready to dive deeper? Explore these advanced topics:

- 🔧 [Matrix-Free Interface](matrix_free_interface.md) - For large-scale problems
- ⚡ [Parallel Computing](parallel_computing.md) - MPI and threading  
- 🎨 [Custom Contours](custom_contours.md) - Advanced integration methods
- 📊 [Performance Optimization](performance.md) - Speed and memory tips
- 🧪 [Examples](examples.md) - Real-world applications

### Quick Navigation

| I want to... | Go to... |
|---------------|----------|
| **Get started immediately** | [Quick Start](#quick-start) |
| **See working examples** | [Examples Gallery](#examples) |
| **Find function documentation** | [API Reference](api_reference.md) |
| **Solve very large problems** | [Matrix-Free Guide](matrix_free_interface.md) |
| **Use multiple processors** | [Parallel Computing](parallel_computing.md) |
| **Optimize performance** | [Performance Guide](performance.md) |

---

<div align="center">
  <p><strong>Ready to solve your eigenvalue problems with FeastKit.jl?</strong></p>
  <p><a href="#quick-start">Start Computing →</a></p>
</div>

---

<small>
<em>FeastKit.jl</em> | <a href="https://github.com/your-repo/FeastKit.jl">GitHub</a> | <a href="https://github.com/your-repo/FeastKit.jl/releases">Releases</a> | <a href="LICENSE.html">License</a>
</small>
