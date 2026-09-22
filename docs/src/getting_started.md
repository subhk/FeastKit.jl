# Getting Started with FeastKit.jl

This guide will get you up and running with FeastKit.jl in minutes. Whether you're new to eigenvalue problems or an experienced user, this tutorial covers everything you need to know.

For a code-aligned checklist covering matrix assumptions, storage, contours,
solver options, and result validation, start with [Problem Setup](problem_setup.md).

```@contents
Pages = ["getting_started.md"]
Depth = 2
```

---

## Installation

### Prerequisites

FeastKit.jl requires Julia 1.10 or later, as specified in `Project.toml`. Install Julia from [julialang.org](https://julialang.org/downloads/).

### Installing FeastKit.jl

This is a REPL session, not a script — `]` switches to package mode and
backspace returns to Julia mode.

```julia-repl
julia> ]

pkg> add FeastKit

pkg> add https://github.com/subhk/FeastKit.jl.git   # development version

julia> using FeastKit
```

### Verify Installation

```@example verify
using FeastKit, LinearAlgebra

# Create a small test problem. Its eigenvalues are 1 and 3, so this interval
# has to reach past 3 to contain both.
A = [2.0 -1.0; -1.0 2.0]
result = feast(A, (0.5, 3.5))
@assert result.converged && result.M == 2

println("Installation successful! Found $(result.M) eigenvalues.")
```

Expected output: `Installation successful! Found 2 eigenvalues.`

---

## First Steps

### Your First Eigenvalue Calculation

Let's solve a classic eigenvalue problem step by step:

```@example first
using FeastKit, LinearAlgebra

# Step 1: Create a matrix
# (This is a 1D discrete Laplacian - common in scientific computing)
n = 100
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))

println("Created $(n)×$(n) tridiagonal matrix")
println("Matrix A has eigenvalues between $(2-2) and $(2+2)")

# Exact eigenvalues, so we can say up front how many are in any interval
exact(k) = 2 - 2cos(k * π / (n + 1))
```

```@example first
# Step 2: Define search interval
# This matrix's eigenvalues are 2 - 2cos(kπ/(n+1)). Near λ = 1 they are spaced
# about 0.06 apart, so this window holds 6 of them -- comfortably under the
# M0 = 10 we ask for below. Choosing an interval without checking how many
# eigenvalues it contains is the most common way to make FEAST fail to
# converge: M0 must be at least that count.
Emin, Emax = 0.8094, 1.1846

println("Searching for eigenvalues in [$Emin, $Emax]")
```

```@example first
# Step 3: Run FeastKit
# M0 = trial-subspace size; leave room beyond the expected eigenvalue count
result = feast(A, (Emin, Emax), M0=10)
@assert result.converged && result.M == 6
@assert isapprox(result.values, filter(λ -> Emin < λ < Emax, exact.(1:n)); atol=1e-9)

println("FeastKit completed:")
println("  Status: $(result.info == 0 ? "Success" : "Error")")
println("  Expected: $(count(k -> Emin <= exact(k) <= Emax, 1:n)) eigenvalues")
println("  Found: $(result.M) eigenvalues")
println("  Iterations: $(result.loop)")
```

```@example first
# Step 4: Examine results
if result.M > 0
    println("\nEigenvalues found:")
    for i in 1:result.M
        println("  λ[$i] = $(result.lambda[i])")
    end
end
```

### Understanding What Happened

FeastKit searched in `[0.8094, 1.1846]` and returned the six enclosed
eigenvalues. The example checks them against the known spectrum and verifies
`result.converged`. The corresponding eigenvectors are columns of `result.q`.

---

## Basic Usage Patterns

### Pattern 1: Standard Eigenvalue Problem

For problems of the form **A⋅x = λ⋅x**:

```@example pattern_standard
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
result = feast(A, (0.5, 2.5); subspace_size=3)
@assert result.converged && result.M == 2
eigenvalues, eigenvectors = result.values, result.vectors
@assert isapprox(eigenvalues, [1.0, 2.0]; atol=1e-10)
eigenvalues
```

**Real-world example:**
```@example pattern_sparse
using FeastKit, SparseArrays

# Large sparse matrix from discretized PDE
n = 10000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Find the 10 smallest eigenvalues. They are 2 - 2cos(kπ/(n+1)), so λ₁₀ is
# about 9.9e-6 -- an interval like (0.001, 0.1) would sit *above* all ten and
# still contain ~900 others.
result = feast(A, (0.0, 1.04e-5), M0=12)
println("Smallest eigenvalues: $(result.lambda[1:result.M])")
@assert result.converged && result.M == 10
@assert isapprox(result.values, [2-2cos(k*π/(n+1)) for k in 1:10]; atol=1e-9)
```

### Pattern 2: Generalized Eigenvalue Problem

For problems of the form **A⋅x = λ⋅B⋅x**:

```@example pattern_generalized
using FeastKit, LinearAlgebra
B = Matrix(Diagonal([2.0, 3.0, 4.0]))
A = B * Diagonal([1.0, 2.0, 4.0])
result = feast(A, B, (0.5, 2.5); subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(result.values, [1.0, 2.0]; atol=1e-10)
result.values
```

**Real-world example:**
```@example pattern_frequencies
using FeastKit, LinearAlgebra

# Structural dynamics: K⋅u = ω²⋅M⋅u
M = Matrix(Diagonal([2.0, 3.0, 4.0]))
K = M * Diagonal((2π .* [20.0, 50.0, 150.0]).^2)

# Find natural frequencies between 10 and 100 Hz
ω²_min, ω²_max = (2π*10)^2, (2π*100)^2
result = feast(K, M, (ω²_min, ω²_max); subspace_size=3)
@assert result.converged && result.M == 2

frequencies_Hz = sqrt.(result.lambda[1:result.M]) / (2π)
@assert isapprox(frequencies_Hz, [20.0, 50.0]; atol=1e-8)
println("Natural frequencies: $frequencies_Hz Hz")
```

### Pattern 3: Matrix-Free for Large Problems

Load `Krylov` for built-in iterative solves. This small stencil example checks
the operator against its known spectrum. For larger problems, provide a
structured shifted solve or a suitable preconditioner; see the
[matrix-free guide](matrix_free_interface.md).

```@example pattern_matfree
using FeastKit, Krylov, LinearAlgebra
n = 12
function A_multiply!(y, x)
    y .= 2 .* x
    y[1:end-1] .-= x[2:end]
    y[2:end] .-= x[1:end-1]
    return y
end
A_op = LinearOperator{Float64}(A_multiply!, (n, n); issymmetric=true)
result = feast(A_op, (0.0, 0.6); subspace_size=4, tol=1e-9,
               solver=:gmres, solver_opts=(rtol=1e-12, maxiter=200, restart=16))
@assert result.converged && result.M == 3
@assert isapprox(result.values, [2-2cos(k*π/(n+1)) for k in 1:3]; atol=1e-8)
result.values
```

### Pattern 4: Complex Eigenvalues

For non-symmetric matrices with complex eigenvalues:

```@example pattern_complex
using FeastKit, LinearAlgebra
A = [0.0 -1.0; 1.0 0.0]  # Real nonsymmetric, with eigenvalues ±im
result = feast_general(A, 0.0+1.0im, 0.5; subspace_size=2)
@assert result.converged && result.M == 1
@assert isapprox(only(result.values), 1im; atol=1e-9)
result.values
```

---

## Understanding Results

### The FeastResult Structure

Symmetric/Hermitian solves return `FeastResult`; general solves return
`FeastGeneralResult`. Both expose these fields:

```@example first
result = feast(A, (Emin, Emax), M0=10)

# Eigenvalues and eigenvectors
result.lambda    # Vector of eigenvalues
result.q         # Matrix of eigenvectors (columns)
result.M         # Number of eigenvalues found

# Convergence information  
result.info      # 0 = success, >0 = error code
result.epsout    # Final residual
result.loop      # Number of refinement iterations  
result.res       # Individual residuals
```

### Interpreting Status Codes

```@example first
if result.info == 0
    println("Success! Found $(result.M) eigenvalues")
elseif result.info == 1
    println("Error: Invalid matrix size")
elseif result.info == 2  
    println("Error: Invalid or saturated M0; increase the subspace or narrow the region")
elseif result.info == 3
    println("Error: Invalid search interval")
else
    println("Error: Code $(result.info)")
end
```

### Quality Assessment

```@example first
@assert result.converged && result.M > 0
# Check convergence quality
println("Final residual: $(result.epsout)")
println("Max individual residual: $(maximum(result.res; init=0.0))")

# Verify eigenvalues are in target interval
in_interval = [Emin <= λ <= Emax for λ in result.lambda[1:result.M]]
println("All eigenvalues in interval: $(all(in_interval))")

# Check orthogonality of eigenvectors (if B = I)
Q = result.q[:, 1:result.M]
orthogonality_error = norm(Q'*Q - I)
println("Orthogonality error: $orthogonality_error")
@assert all(in_interval) && orthogonality_error < 1e-8
```

---

## Common Workflows

### Workflow 1: Finding Specific Eigenvalues

**Problem**: You need the 10 eigenvalues closest to 5.0

```@example closest_eigenvalues
using FeastKit, LinearAlgebra
W = Matrix(Diagonal(collect(range(0.0, 10.0; length=200))))
center = 5.0
# This example has a known spectrum, so the search can be bounded in advance.
widths = [0.1 * 1.5^k for k in 0:20]
inside(w) = count(λ -> center-w < λ < center+w, diag(W))
index = findfirst(w -> inside(w) >= 10, widths)
index === nothing && error("No candidate interval holds ten eigenvalues")
width = widths[index]
result = feast(W, (center-width, center+width); subspace_size=2*inside(width))
@assert result.converged && result.M >= 10
closest_10 = result.values[sortperm(abs.(result.values .- center))[1:10]]
expected = diag(W)[sortperm(abs.(diag(W) .- center))[1:10]]
@assert isapprox(sort(closest_10), sort(expected); atol=1e-9)
closest_10
```

### Workflow 2: Eigenvalue Counting

**Problem**: How many eigenvalues are in `[0, 1]`?

```@example count_eigenvalues
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([0.1, 0.4, 0.9, 2.0]))
# Use FeastKit to count eigenvalues
result = feast(A, (0.0, 1.0), M0=4)  # Large M0 for counting

println("Number of eigenvalues in [0,1]: $(result.M)")

# A stochastic estimate can help size the subspace before solving.
estimate = feast_estimate_count(A, (0.0, 1.0); nprobe=16)
# The rational filter alone does not see A and cannot count its eigenvalues.
# A reported count from a solve is meaningful only after checking convergence.
@assert result.converged result.message
@assert result.M == 3
```

### Workflow 3: Parameter Tuning

**Problem**: FeastKit isn't converging well

```@example tuning_workflow
using FeastKit, LinearAlgebra
A = Matrix(Diagonal(collect(1.0:40.0)))
Emin, Emax = 0.5, 3.5
# Step 1: Check if eigenvalues exist in your interval
bounds = feast_validate_interval(A, (Emin, Emax))
println("Estimated eigenvalue range: $bounds")

# Step 2: Adjust FeastKit parameters
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[1] = 1      # Print level (0=silent, 1=on)
fpm[2] = 16     # Integration points (8-32 typical)
fpm[3] = 12     # Tolerance: 10^(-fpm[3])
fpm[4] = 50     # Max refinement iterations

result = feast(A, (Emin, Emax), M0=20, fpm=fpm)

# Step 3: Try different integration methods
fpm[16] = 2     # 0=Gauss-Legendre, 1=trapezoidal, 2=Zolotarev
fpm[2] = 12     # Half-contour integration points
result_zolotarev = feast(A, (Emin, Emax); M0=20, fpm=fpm)
@assert result.converged && result_zolotarev.converged
@assert result.M == result_zolotarev.M == 3
```

### Workflow 4: Large-Scale Problems

**Problem**: You need to understand storage before scaling up

Exercise a small instance first. The following continuation measures only the
returned result; it does not measure peak memory during a large solve.

```@example pattern_matfree
# Continue with the operator from Pattern 3. Measure result storage separately
# from solver workspaces, factors, and peak process memory.
result = feast(A_op, (0.0, 0.6); subspace_size=4, tol=1e-9,
               solver=:gmres, solver_opts=(rtol=1e-12, maxiter=200, restart=16))
@assert result.converged && result.M == 3
println("Returned result storage: $(Base.summarysize(result) / 1e6) MB")
```

---

## Next Steps

### Ready for More?

Now that you understand the basics, explore these advanced topics:

#### Performance Optimization
- [Matrix-Free Interface](matrix_free_interface.md) - Handle matrices too large for memory
- [Parallel Computing](parallel_computing.md) - Use multiple cores and nodes
- [Performance Tips](performance.md) - Speed and memory optimization

#### Advanced Features  
- [Custom Contour Integration](custom_contours.md) - Circle, ellipse, and box solves; Zolotarev and Gauss-Legendre methods
- [Complex Eigenvalues](complex_eigenvalues.md) - Non-Hermitian problems
- [Polynomial Eigenvalue Problems](polynomial_problems.md) - Quadratic and higher-order

#### Real Applications
- [Structural Dynamics](examples.md) - Vibration analysis
- [Quantum Mechanics](examples.md) - Electronic structure
- [Convection–Diffusion](complex_eigenvalues.md) - A nonsymmetric discretized operator

### Quick Reference Card

Signature templates: supply `A`, `B`, the region, and your `matvec!` callback.
Load `Krylov` before using a built-in iterative solver.

```julia
# Basic usage
result = feast(A, (Emin, Emax), M0=10)

# Generalized problem  
result = feast(A, B, (Emin, Emax), M0=10)

# Matrix-free
A_op = LinearOperator{Float64}(matvec!, (n,n), issymmetric=true)
result = feast(A_op, (Emin, Emax), M0=10, solver=:gmres)

# Complex eigenvalues
result = feast_general(A, B, center, radius, M0=10)

# Check results
if result.info == 0
    eigenvalues = result.lambda[1:result.M]
    eigenvectors = result.q[:, 1:result.M]
end
```

### Getting Help

- Full API: [API Reference](api_reference.md)
- Issues: [GitHub Issues](https://github.com/your-repo/FeastKit.jl/issues)  
- Community: [GitHub Discussions](https://github.com/your-repo/FeastKit.jl/discussions)
- Contact: your-email@domain.com

---

**Congratulations! You're now ready to use FeastKit.jl effectively.**

Explore the [API Reference](api_reference.md) · See more [Examples](examples.md)
