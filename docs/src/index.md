# FeastKit.jl

*Fast Eigenvalue Algorithm using Spectral Transformations in Julia*

[Problem Setup](problem_setup.md) | [Quick Start](@ref quick-start) | [Examples](@ref examples-gallery) | [API Reference](api_reference.md) | [Advanced Features](@ref core-concepts)

---

## What is FEAST?

FeastKit.jl is a Julia implementation of the **FEAST eigenvalue algorithm**, a powerful numerical method for finding eigenvalues and eigenvectors of large sparse matrices within specified intervals or regions. Unlike traditional methods that compute all eigenvalues, FeastKit allows you to:

- Target specific eigenvalues in intervals `[Emin, Emax]` or complex regions
- Work with dense, sparse, banded, and matrix-free inputs
- Work matrix-free without storing explicit matrices
- Leverage parallelization for high-performance computing
- Use custom contour integration for optimal convergence

Use the site's version selector to match your installed package. The development
site follows `main`; stable documentation follows published version tags.

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
Pkg.add("FeastKit")

# Or for development:
Pkg.add(url="https://github.com/subhk/FeastKit.jl")
```

### Your First FeastKit Calculation

```@example quickstart
using FeastKit, LinearAlgebra, SparseArrays

# Create a sparse 1000x1000 tridiagonal matrix
n = 1000
A = sparse(SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1)))

# Find eigenvalues near λ = 1. The eigenvalues here are 2 - 2cos(kπ/(n+1)),
# spaced about 3e-3 apart near λ = 1, so this window holds 8 of them. M0 must
# exceed the number of eigenvalues in the interval unless using the full space;
# saturation or non-convergence means the subspace/region needs adjustment.
result = feast(A, (0.9801, 1.0182); subspace_size=10, tol=1e-10)
@assert result.converged result.message
@assert result.M == 8

println("Found $(result.M) eigenvalues:")   # 8
println(result.values)
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
result = feast(A_op, (0.9801, 1.0182); subspace_size=10, solver=tridiagonal_solve!)
@assert result.converged result.message
@assert result.M == 8

println("Found $(result.M) eigenvalues, info = $(result.info)")
```

---

## [Examples Gallery](@id examples-gallery)

### Dense Matrix Eigenvalues

```@example dense
using FeastKit, LinearAlgebra
n = 40
A = Matrix(SymTridiagonal(2.0 * ones(n), -ones(n-1)))
interval = (0.0, 0.2)
expected = filter(λ -> interval[1] < λ < interval[2], eigvals(Symmetric(A)))
result = feast(A, interval; subspace_size=8)
@assert result.converged && result.M == length(expected)
@assert isapprox(result.values, expected; atol=1e-10)
result.values
```

### Sparse Matrix Problems

```@example sparse
using FeastKit, SparseArrays, LinearAlgebra
n = 5000
A = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
# The exact discrete-Laplacian spectrum lets us isolate the largest eigenvalue.
λ_max = 2 - 2cos(n*π/(n+1))
λ_next = 2 - 2cos((n-1)*π/(n+1))
interval = ((λ_max + λ_next)/2, λ_max + (λ_max - λ_next)/2)
result = feast(A, interval; subspace_size=4)
@assert result.converged && result.M == 1
@assert isapprox(only(result.values), λ_max; atol=1e-9)
result.values
```

### Generalized Eigenvalue Problem

```@example generalized
using FeastKit, LinearAlgebra, SparseArrays
n = 200
A = sparse(SymTridiagonal(2.0 * ones(n), -ones(n-1)))
B = sparse(SymTridiagonal(3.0 * ones(n), -0.5 * ones(n-1)))
# Both tridiagonal matrices share the discrete sine eigenvectors.
λ(k) = (2 - 2cos(k*π/(n+1))) / (3 - cos(k*π/(n+1)))
interval = (0.0, (λ(8) + λ(9))/2)
result = feast(A, B, interval; subspace_size=12)
@assert result.converged && result.M == 8
@assert isapprox(result.values, λ.(1:8); atol=1e-10)
result.values
```

### Complex Non-Hermitian Problems

```@example general
using FeastKit, LinearAlgebra
A = ComplexF64[0.5+0.1im 0.2 0; 0 1.0-0.2im 0.3; 0 0 3.0+1.0im]
result = feast_general(A, 0.0+0.0im, 2.0; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 1.0-0.2im]; atol=1e-9)
result.values
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

Signature templates using your matrices and search bounds:

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
contour = feast_rectangle(-1, 1, -0.5, 0.5)
result = feast(A, contour; subspace_size=10)
```

---

## Performance Guide

### Choosing Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|---------|
| `subspace_size` (`M0`) | Trial-subspace capacity | 10-50 | Memory usage, accuracy |
| `quadrature_points` | Half-contour count for intervals; full count for general solves | 8-32 | Accuracy vs speed |
| `tol` | Convergence tolerance | 1e-12 | Accuracy vs iterations |
| `maxiter` | Max refinement loops | 20-100 | Convergence robustness |

### Memory Usage

FEAST stores several `N × subspace_size` workspaces and smaller projected
matrices. Dense matrix storage and cached shifted factorizations add quadratic
memory; sparse LU factors may have substantial fill-in. Matrix-free iterative
solves add Krylov-basis and callback storage. There is no fixed memory total
based on `N` alone. See the [performance guide](performance.md).

### Performance Tips

- Use matrix-free for large problems
- Use direct factorization or GMRES for assembled inputs; matrix-free also supports BiCGSTAB or a callback. CG is unsupported for complex shifted systems.
- Tune integration points: More points = better accuracy, slower
- Enable parallelization for very large problems
- Use custom contours for challenging geometries

---

## Troubleshooting

### No eigenvalues found

**Cause**: Search interval doesn't contain eigenvalues

**Solutions**:
```@example home_bounds
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
Emin, Emax = 0.5, 2.5
# Check eigenvalue bounds first
bounds = feast_validate_interval(A, (Emin, Emax))
println("Estimated eigenvalue range: $bounds")

# For a small reference problem, search the full range with a full subspace.
# For large problems, split the range and estimate each enclosed count.
result = feast(A, (bounds[1] - 0.1, bounds[2] + 0.1); subspace_size=size(A, 1))
@assert result.converged && result.M == 4
```

### Linear solver not converging

**Cause**: Iterative solver issues in matrix-free mode

**Solutions**:
```@example home_inner_solvers
using FeastKit, Krylov, LinearAlgebra
entries = collect(1.0:12.0)
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x), (12, 12);
                              issymmetric=true)
interval = (0.5, 2.5)
for solver in (:gmres, :bicgstab)
    result = feast(A_op, interval; subspace_size=4, tol=1e-9, solver=solver,
                   solver_opts=(rtol=1e-12, maxiter=500, restart=16))
    @assert result.converged && result.M == 2
    @assert isapprox(result.values, [1.0, 2.0]; atol=1e-8)
end
```

### Memory allocation failed

**Cause**: Problem too large for available memory

**Solutions**:
```@example home_memory
using FeastKit, Krylov, LinearAlgebra
# Check the callback on a small problem before scaling it up.
entries = collect(1.0:12.0)
A = Matrix(Diagonal(entries))
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x), size(A);
                              issymmetric=true)
interval = (0.5, 2.5)
result = feast(A_op, interval; subspace_size=4, tol=1e-9,
               solver_opts=(rtol=1e-12, maxiter=500, restart=16))
@assert result.converged && result.M == 2
# A smaller subspace must still hold every targeted eigenvalue.
result_small = feast(A, interval; subspace_size=3)
@assert result_small.converged && result_small.M == 2
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
- [Custom Contours](custom_contours.md) - Circle, ellipse, and box constructors with complete solves
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
