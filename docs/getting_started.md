# Getting Started with FeastKit.jl {#getting_started}

```@id getting_started
```


This guide will get you up and running with FeastKit.jl in minutes. Whether you're new to eigenvalue problems or an experienced user, this tutorial covers everything you need to know.

## Table of Contents

1. [Installation](#installation)
2. [First Steps](#first-steps)  
3. [Basic Usage Patterns](#basic-usage-patterns)
4. [Understanding Results](#understanding-results)
5. [Common Workflows](#common-workflows)
6. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

FeastKit.jl requires Julia 1.6 or later. Install Julia from [julialang.org](https://julialang.org/downloads/).

### Installing FeastKit.jl

```julia
# Start Julia and enter package mode with ]
julia> ]

# Install FeastKit.jl
pkg> add FeastKit

# Or for the development version:
pkg> add https://github.com/your-repo/FeastKit.jl.git

# Return to Julia mode
pkg> <backspace>

# Load the package
julia> using FeastKit
```

### Verify Installation

```julia
using FeastKit, LinearAlgebra

# Create a small test problem
A = [2.0 -1.0; -1.0 2.0]
result = feast(A, (0.5, 2.5))

println("Installation successful! Found $(result.M) eigenvalues.")
```

Expected output: `Installation successful! Found 2 eigenvalues.`

---

## First Steps

### Your First Eigenvalue Calculation

Let's solve a classic eigenvalue problem step by step:

```julia
using FeastKit, LinearAlgebra

# Step 1: Create a matrix
# (This is a 1D discrete Laplacian - common in scientific computing)
n = 100
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))

println("Created $(n)×$(n) tridiagonal matrix")
println("Matrix A has eigenvalues between $(2-2) and $(2+2)")
```

```julia
# Step 2: Define search interval
# We want eigenvalues between 0.5 and 1.5
Emin, Emax = 0.5, 1.5

println("Searching for eigenvalues in [$Emin, $Emax]")
```

```julia
# Step 3: Run FeastKit
# M0 = maximum number of eigenvalues to find
result = feast(A, (Emin, Emax), M0=10)

println("FeastKit completed:")
println("  Status: $(result.info == 0 ? "Success" : "Error")")
println("  Found: $(result.M) eigenvalues")
println("  Iterations: $(result.loop)")
```

```julia
# Step 4: Examine results
if result.M > 0
    println("\\nEigenvalues found:")
    for i in 1:result.M
        println("  λ[$i] = $(result.lambda[i])")
    end
end
```

### Understanding What Happened

FeastKit searched only in `[0.5, 1.5]` — it didn't compute all eigenvalues  
Found eigenvalues efficiently using contour integration  
Verified convergence — `result.info == 0` means success  
Provided eigenvectors too — stored in `result.q`

---

## Basic Usage Patterns

### Pattern 1: Standard Eigenvalue Problem

For problems of the form **A⋅x = λ⋅x**:

```julia
using FeastKit

# Your matrix (dense, sparse, whatever!)
A = your_matrix()

# Find eigenvalues in an interval
result = feast(A, (Emin, Emax), M0=20)

# Access results
eigenvalues = result.lambda[1:result.M]
eigenvectors = result.q[:, 1:result.M]
```

**Real-world example:**
```julia
using FeastKit, SparseArrays

# Large sparse matrix from discretized PDE
n = 10000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Find the 10 smallest eigenvalues  
result = feast(A, (0.001, 0.1), M0=10)
println("Smallest eigenvalues: $(result.lambda[1:result.M])")
```

### Pattern 2: Generalized Eigenvalue Problem

For problems of the form **A⋅x = λ⋅B⋅x**:

```julia
# Two matrices A and B
A = your_stiffness_matrix()
B = your_mass_matrix()  

# Solve generalized problem
result = feast(A, B, (Emin, Emax), M0=15)
```

**Real-world example:**
```julia
using FeastKit

# Structural dynamics: K⋅u = ω²⋅M⋅u
K = stiffness_matrix()  # Stiffness
M = mass_matrix()       # Mass

# Find natural frequencies between 10 and 100 Hz
ω²_min, ω²_max = (2π*10)^2, (2π*100)^2
result = feast(K, M, (ω²_min, ω²_max), M0=20)

frequencies_Hz = sqrt.(result.lambda[1:result.M]) / (2π)
println("Natural frequencies: $frequencies_Hz Hz")
```

### Pattern 3: Matrix-Free for Large Problems

When your matrix is too large to store:

```julia
# Define matrix-vector multiplication
function A_multiply!(y, x)
    # Your custom A*x computation here
    # Example: finite difference stencil
    n = length(x)
    y[1] = 2*x[1] - x[2]
    for i in 2:n-1
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    end
    y[n] = -x[n-1] + 2*x[n]
end

# Create operator
n = 1_000_000  # Very large!
A_op = LinearOperator{Float64}(A_multiply!, (n, n), issymmetric=true)

# Solve exactly the same way
result = feast(A_op, (Emin, Emax), M0=10, solver=:cg)
```

### Pattern 4: Complex Eigenvalues

For non-symmetric matrices with complex eigenvalues:

```julia
using FeastKit

# Non-symmetric matrix
A = your_nonsymmetric_matrix()
B = Matrix(I, size(A)...)  # Identity matrix

# Define circular search region in complex plane
center = 1.0 + 0.5im     # Center point  
radius = 2.0             # Search radius

result = feast_general(A, B, center, radius, M0=15)

# Complex eigenvalues
complex_eigenvalues = result.lambda[1:result.M]
```

---

## Understanding Results

### The FeastResult Structure

Every FeastKit calculation returns a `FeastResult` with these fields:

```julia
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

```julia
if result.info == 0
    println("Success! Found $(result.M) eigenvalues")
elseif result.info == 1
    println("Error: Invalid matrix size")
elseif result.info == 2  
    println("Error: Invalid M0 parameter")
elseif result.info == 3
    println("Error: Invalid search interval")
else
    println("Error: Code $(result.info)")
end
```

### Quality Assessment

```julia
# Check convergence quality
println("Final residual: $(result.epsout)")
println("Max individual residual: $(maximum(result.res[1:result.M]))")

# Verify eigenvalues are in target interval
in_interval = [Emin <= λ <= Emax for λ in result.lambda[1:result.M]]
println("All eigenvalues in interval: $(all(in_interval))")

# Check orthogonality of eigenvectors (if B = I)
Q = result.q[:, 1:result.M]
orthogonality_error = norm(Q'*Q - I)
println("Orthogonality error: $orthogonality_error")
```

---

## Common Workflows

### Workflow 1: Finding Specific Eigenvalues

**Problem**: You need the 10 eigenvalues closest to 5.0

```julia
using FeastKit

# Strategy: Search in a small interval around 5.0
center = 5.0
width = 0.1
result = feast(A, (center - width, center + width), M0=15)

if result.M >= 10
    closest_10 = result.lambda[1:10]
    println("10 eigenvalues closest to 5.0: $closest_10")
else
    println("Found only $(result.M) eigenvalues, try wider interval")
end
```

### Workflow 2: Eigenvalue Counting

**Problem**: How many eigenvalues are in `[0, 1]`?

```julia
# Use FeastKit to count eigenvalues
result = feast(A, (0.0, 1.0), M0=100)  # Large M0 for counting

println("Number of eigenvalues in [0,1]: $(result.M)")

# For more precise counting, use the rational function
rational_values = feast_rational(test_points, 0.0, 1.0, fpm)
# Values near 1.0 indicate eigenvalues nearby
```

### Workflow 3: Parameter Tuning

**Problem**: FeastKit isn't converging well

```julia
# Step 1: Check if eigenvalues exist in your interval
bounds = feast_validate_interval(A, (Emin, Emax))
println("Estimated eigenvalue range: $bounds")

# Step 2: Adjust FeastKit parameters
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[1] = 1      # Print level (0=silent, 1=summary, 2=detailed)
fpm[2] = 16     # Integration points (8-32 typical)
fmp[3] = 12     # Tolerance: 10^(-fmp[3])
fpm[4] = 50     # Max refinement iterations

result = feast(A, (Emin, Emax), M0=20, fpm=fpm)

# Step 3: Try different integration methods
result_zolotarev = feast(A, (Emin, Emax), M0=20, 
                        integration_method=:zolotarev, 
                        integration_points=12)
```

### Workflow 4: Large-Scale Problems

**Problem**: Matrix has millions of unknowns

```julia
# Convert to matrix-free 
function matvec!(y, x)
    # Your efficient A*x implementation
    # Use BLAS, threading, GPU, etc.
end

A_op = LinearOperator{Float64}(matvec!, (n, n), issymmetric=true)

# Use appropriate iterative solver
result = feast(A_op, (Emin, Emax), M0=10,
              solver=:cg,  # or :gmres, :bicgstab
              solver_opts=(rtol=1e-6, maxiter=1000))

# Monitor memory usage
println("Memory used: $(Base.summarysize(result) / 1e6) MB")
```

---

## Next Steps

### Ready for More?

Now that you understand the basics, explore these advanced topics:

#### Performance Optimization
- [Matrix-Free Interface](@ref "matrix_free_interface") - Handle matrices too large for memory
- [Parallel Computing](@ref "parallel_computing") - Use multiple cores and nodes
- [Performance Tips](@ref "performance_guide") - Speed and memory optimization

#### Advanced Features  
- [Custom Contour Integration](@ref "custom_contours") - Zolotarev, Gauss-Legendre methods
- [Complex Eigenvalues](@ref "complex_eigenvalues") - Non-Hermitian problems
- [Polynomial Eigenvalue Problems](@ref "polynomial_problems") - Quadratic and higher-order

#### Real Applications
- [Structural Dynamics](@ref "examples_page") - Vibration analysis
- [Quantum Mechanics](@ref "examples_page") - Electronic structure
- [Fluid Dynamics](@ref "examples_page") - Stability analysis
- [Network Analysis](@ref "examples_page") - Graph eigenvalues

### Quick Reference Card

Keep this handy while coding:

```julia
# Basic usage
result = feast(A, (Emin, Emax), M0=10)

# Generalized problem  
result = feast(A, B, (Emin, Emax), M0=10)

# Matrix-free
A_op = LinearOperator{Float64}(matvec!, (n,n), issymmetric=true)
result = feast(A_op, (Emin, Emax), M0=10, solver=:cg)

# Complex eigenvalues
result = feast_general(A, B, center, radius, M0=10)

# Check results
if result.info == 0
    eigenvalues = result.lambda[1:result.M]
    eigenvectors = result.q[:, 1:result.M]
end
```

### Getting Help

- Full API: [API Reference](@ref "api_reference")
- Issues: [GitHub Issues](https://github.com/your-repo/FeastKit.jl/issues)  
- Community: [GitHub Discussions](https://github.com/your-repo/FeastKit.jl/discussions)
- Contact: your-email@domain.com

---

<div align="center">
  <p><strong>Congratulations! You're now ready to use FeastKit.jl effectively.</strong></p>
  Explore the [API Reference](@ref "api_reference") · See more [Examples](@ref "examples_page")
</div>
