# FeastKit.jl

*Fast Eigenvalue Algorithm using Spectral Transformations in Julia*

[Quick Start](@ref quick-start) | [Examples](@ref examples-gallery) | [API Reference](api_reference.md) | [Advanced Features](@ref core-concepts)

---

## What is FEAST?

FeastKit.jl is a Julia implementation of the **FEAST eigenvalue algorithm**, a powerful numerical method for finding eigenvalues and eigenvectors of large sparse matrices within specified intervals or regions. Unlike traditional methods that compute all eigenvalues, FeastKit allows you to:

- Target specific eigenvalues in intervals `[Emin, Emax]` or complex regions
- Handle very large problems (millions of unknowns) efficiently
- Work matrix-free without storing explicit matrices
- Leverage parallelization for high-performance computing
- Use custom contour integration for optimal convergence

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

## [Quick Start](@id quick-start)

### Installation

```julia
using Pkg
Pkg.add("FeastKit")  # When available from registry

# Or for development:
Pkg.add(url="https://github.com/subhk/FeastKit.jl")
```

### Your First FeastKit Calculation

```@example quickstart
using FeastKit, LinearAlgebra

# Create a test matrix (1000x1000 tridiagonal)
n = 1000
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))

# Find eigenvalues near λ = 1. The eigenvalues here are 2 - 2cos(kπ/(n+1)),
# spaced about 3e-3 apart near λ = 1, so this window holds 8 of them. M0 must
# be at least the number of eigenvalues in the interval, or FEAST cannot
# converge and returns info = 5.
result = feast(A, (0.9801, 1.0182), M0=10)

println("Found $(result.M) eigenvalues:")   # 8
println(result.lambda[1:result.M])
```

**That's it!** FeastKit found the eigenvalues in your target interval.

### Matrix-Free Example

For very large problems, use matrix-free operations:

```@example quickstart
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

# Supply the shifted solve too. Without a `solver=`, FEAST falls back to
# unpreconditioned GMRES, which stalls when the contour hugs a dense stretch of
# the spectrum. Here z*I - A is tridiagonal, so one O(n) Thomas sweep per
# right-hand side is both exact and fast -- that is the real payoff of the
# matrix-free interface.
function tridiagonal_solve!(Y, z, X)
    d0 = ComplexF64(z) - 2          # diagonal of z*I - A
    c = Vector{ComplexF64}(undef, n - 1)
    d = Vector{ComplexF64}(undef, n)
    for j in axes(X, 2)
        c[1] = 1 / d0
        d[1] = X[1, j] / d0
        for i in 2:n
            m = d0 - c[i - 1]
            i < n && (c[i] = 1 / m)
            d[i] = (X[i, j] - d[i - 1]) / m
        end
        Y[n, j] = d[n]
        for i in (n - 1):-1:1
            Y[i, j] = d[i] - c[i] * Y[i + 1, j]
        end
    end
    return Y
end

# Solve the same way!
result = feast(A_op, (0.9801, 1.0182), M0=10, solver=tridiagonal_solve!)

println("Found $(result.M) eigenvalues, info = $(result.info)")
```

---

## [Examples Gallery](@id examples-gallery)

### Dense Matrix Eigenvalues

```@example dense
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

### Sparse Matrix Problems

```@example sparse
using FeastKit, SparseArrays, LinearAlgebra

# Large sparse symmetric matrix
n = 5000
A = sprand(n, n, 0.001)  # 0.1% density
A = A + A' + 5*I         # Make symmetric positive definite

# Bracket the top of the spectrum, which is well separated from the bulk near
# 5. Estimate it first rather than guessing an interval width.
function power_iteration_max(S, N)
    v = randn(N)
    for _ in 1:100
        v = S * v
        v ./= norm(v)
    end
    return dot(v, S * v)
end
λ_max = power_iteration_max(A, n)

result = feast(A, (λ_max - 0.1, λ_max + 0.1), M0=8)

println("Largest eigenvalue: $(result.lambda[1:result.M])")
```

### Generalized Eigenvalue Problem

```@example generalized
using FeastKit, LinearAlgebra

# Create matrices A and B
n = 1000
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
B = SymTridiagonal(3.0 * ones(n), -0.5 * ones(n-1))

# Solve A*x = λ*B*x. The pencil's spectrum runs from ~0 to 1 across 1000
# eigenvalues, so target the low end rather than a broad slice: (0.1, 0.8)
# would contain 461 of them and no M0 of 15 could resolve that.
result = feast(A, B, (0.0, 0.00032), M0=15)

println("Generalized eigenvalues: $(result.lambda[1:result.M])")   # 8
```

### Complex Non-Hermitian Problems

```@example general
using FeastKit, LinearAlgebra

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

---

## [Core Concepts](@id core-concepts)

### The FEAST Algorithm

The FEAST algorithm uses **contour integration** in the complex plane to extract eigenvalues in specified regions. The key idea:

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

- Use matrix-free for large problems
- Choose appropriate solvers: CG for SPD, GMRES for general
- Tune integration points: More points = better accuracy, slower
- Enable parallelization for very large problems
- Use custom contours for challenging geometries

---

## Troubleshooting

### No eigenvalues found

**Cause**: Search interval doesn't contain eigenvalues

**Solutions**:
```julia
# Check eigenvalue bounds first
bounds = feast_validate_interval(A, (Emin, Emax))
println("Estimated eigenvalue range: $bounds")

# Use a broader interval
result = feast(A, (bounds[1], bounds[2]), M0=10)
```

### Linear solver not converging

**Cause**: Iterative solver issues in matrix-free mode

**Solutions**:
```julia
# Increase solver tolerance and iterations
result = feast(A_op, interval,
              solver=:gmres,
              solver_opts=(rtol=1e-4, maxiter=2000, restart=50))

# Try different solver
result = feast(A_op, interval, solver=:bicgstab)
```

### Memory allocation failed

**Cause**: Problem too large for available memory

**Solutions**:
```julia
# Switch to matrix-free interface
A_op = LinearOperator{Float64}(A_mul!, size(A))
result = feast(A_op, interval, M0=10)

# Reduce M0 (max eigenvalues)
result = feast(A, interval, M0=5)  # Instead of M0=20
```

### Getting Help

- **Documentation**: Check [API Reference](api_reference.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/subhk/FeastKit.jl/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/subhk/FeastKit.jl/discussions)

---

## What's Next?

Ready to dive deeper? Explore these advanced topics:

- [Matrix-Free Interface](matrix_free_interface.md) - For large-scale problems
- [Parallel Computing](parallel_computing.md) - MPI and threading
- [Custom Contours](custom_contours.md) - Advanced integration methods
- [Performance Optimization](performance.md) - Speed and memory tips
- [Examples](examples.md) - Real-world applications

### Quick Navigation

| I want to... | Go to... |
|---------------|----------|
| Get started immediately | [Quick Start](@ref quick-start) |
| See working examples | [Examples Gallery](@ref examples-gallery) |
| Find function documentation | [API Reference](api_reference.md) |
| Solve very large problems | [Matrix-Free Guide](matrix_free_interface.md) |
| Use multiple processors | [Parallel Computing](parallel_computing.md) |
| Optimize performance | [Performance Guide](performance.md) |

---

**Ready to solve your eigenvalue problems with FeastKit.jl?** [Start Computing →](@ref quick-start)

---

*FeastKit.jl* | [GitHub](https://github.com/subhk/FeastKit.jl) | [Releases](https://github.com/subhk/FeastKit.jl/releases) | [License](license.md)
