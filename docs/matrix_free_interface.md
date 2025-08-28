# Matrix-Free Feast Interface

The matrix-free interface allows you to use Feast without explicitly storing matrices, making it ideal for large-scale problems where memory is limited or when matrices are too expensive to construct.

## Overview

Instead of providing explicit matrices `A` and `B`, you provide functions that compute matrix-vector products:
- `A_mul!(y, x)` computes `y = A*x`
- `B_mul!(y, x)` computes `y = B*x`
- `linear_solver(Y, z, X)` solves `(z*B - A)*Y = X`

## Matrix-Free Operator Types

### LinearOperator

The main interface for defining matrix-free operators:

```julia
using Feast

# Define matrix-vector multiplication function
function A_mul!(y, x)
    # Your custom matrix-vector product code here
    # Example: y = A*x for some implicit matrix A
end

# Create operator
n = 1000  # Matrix size
A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
```

**Properties you can specify:**
- `issymmetric=true/false`: Matrix is symmetric
- `ishermitian=true/false`: Matrix is Hermitian  
- `isposdef=true/false`: Matrix is positive definite

### MatrixVecFunction

Alternative interface for operators that need additional data:

```julia
function A_mul!(y, op, x)
    # Access operator data via op.data
    mul!(y, op.data, x)
end

A_op = MatrixVecFunction{Float64}(A_mul!, (n, n), issymmetric=true)
```

## Basic Usage

### Standard Eigenvalue Problem (A*x = λ*x)

```julia
using Feast

n = 10000

# Define A*x operation
function laplacian_1d!(y, x)
    # 1D discrete Laplacian: [-1 2 -1] stencil
    y[1] = 2*x[1] - x[2]
    for i in 2:n-1
        y[i] = -x[i-1] + 2*x[i] - x[i+1]
    end  
    y[n] = -x[n-1] + 2*x[n]
end

# Create operator
A_op = LinearOperator{Float64}(laplacian_1d!, (n, n), issymmetric=true)

# Solve eigenvalue problem
result = feast(A_op, (0.1, 1.0), M0=10, solver=:cg)

println("Found $(result.M) eigenvalues")
println("Eigenvalues: $(result.lambda[1:result.M])")
```

### Generalized Eigenvalue Problem (A*x = λ*B*x)

```julia
# Define both A and B operations
function A_mul!(y, x)
    # Your A*x computation
end

function B_mul!(y, x) 
    # Your B*x computation
end

A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
B_op = LinearOperator{Float64}(B_mul!, (n, n), issymmetric=true, isposdef=true)

result = feast(A_op, B_op, (emin, emax), M0=10)
```

## Linear Solvers

Feast requires solving linear systems `(z*B - A)*Y = X` for various values of `z`. You have several options:

### Built-in Iterative Solvers

```julia
# Use GMRES (default, works for general problems)
result = feast(A_op, B_op, interval, solver=:gmres, 
              solver_opts=(rtol=1e-6, restart=30, maxiter=1000))

# Use CG (for symmetric positive definite systems)
result = feast(A_op, B_op, interval, solver=:cg,
              solver_opts=(rtol=1e-8, maxiter=500))

# Use BiCGSTAB(l) 
result = feast(A_op, B_op, interval, solver=:bicgstab,
              solver_opts=(l=2, rtol=1e-6, maxiter=800))
```

### Custom Linear Solver

For specialized problems, provide your own solver:

```julia
function my_custom_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
    # Solve (z*B - A)*Y = X for each column of X
    # Store results in corresponding columns of Y
    
    M0 = size(X, 2)
    for j in 1:M0
        # Your custom solution method for column j
        Y[:, j] = solve_linear_system(z, X[:, j])
    end
end

result = feast(A_op, B_op, interval, solver=my_custom_solver)
```

## Advanced Features

### Custom Contour Integration

Use advanced contour integration methods from the original Fortran Feast:

```julia
# Gauss-Legendre integration (high accuracy)
contour = feast_contour_expert(emin, emax, 8, 0, 100)

# Zolotarev integration (optimal for ellipses) 
contour = feast_contour_expert(emin, emax, 12, 2, 100)

# Custom ellipse aspect ratio (a/b = 0.5, flatter ellipse)
contour = feast_contour_expert(emin, emax, 10, 0, 50)
```

### General (Non-Hermitian) Problems

For non-symmetric matrices with complex eigenvalues:

```julia
# Complex operators
A_op = LinearOperator{ComplexF64}(A_mul!, (n, n))
B_op = LinearOperator{ComplexF64}(B_mul!, (n, n))

# Circular search region in complex plane
center = 1.0 + 0.5im
radius = 2.0

result = feast_general(A_op, B_op, center, radius, M0=10)
```

### Polynomial Eigenvalue Problems

For polynomial eigenvalue problems P(λ)x = 0:

```julia
# Define coefficient operators for P(λ) = A₀ + λ*A₁ + λ²*A₂
A0_op = LinearOperator{ComplexF64}(A0_mul!, (n, n))
A1_op = LinearOperator{ComplexF64}(A1_mul!, (n, n))  
A2_op = LinearOperator{ComplexF64}(A2_mul!, (n, n))

coeffs = [A0_op, A1_op, A2_op]

result = feast_polynomial(coeffs, center, radius, M0=15)
```

## Complete Examples

### 2D Discrete Laplacian

```julia
using Feast

# Parameters
nx, ny = 200, 200
n = nx * ny
h = 1.0 / (nx + 1)

# Index mapping
idx(i, j) = (j-1) * nx + i

# Matrix-free 2D Laplacian
function laplacian_2d!(y, x)
    fill!(y, 0)
    for j in 1:ny, i in 1:nx
        k = idx(i, j)
        y[k] += 4 * x[k] / h^2
        
        # Neighbors
        i > 1  && (y[k] -= x[idx(i-1, j)] / h^2)
        i < nx && (y[k] -= x[idx(i+1, j)] / h^2)
        j > 1  && (y[k] -= x[idx(i, j-1)] / h^2)
        j < ny && (y[k] -= x[idx(i, j+1)] / h^2)
    end
end

A_op = LinearOperator{Float64}(laplacian_2d!, (n, n), 
                              issymmetric=true, isposdef=true)

# Find smallest eigenvalues
λ_min_approx = 2π^2 * (1/nx^2 + 1/ny^2)
result = feast(A_op, (0.5*λ_min_approx, 2.0*λ_min_approx), 
              M0=20, solver=:cg)

println("Found $(result.M) eigenvalues:")
for i in 1:result.M
    println("  λ[$i] = $(result.lambda[i])")
end
```

### Large Sparse Matrix as Matrix-Free

Even when you have a sparse matrix, using the matrix-free interface can save memory for very large problems:

```julia
using SparseArrays

# Create large sparse matrix (don't store factorizations)
n = 100000
A_sparse = sprand(n, n, 0.0001)  # Very sparse
A_sparse = A_sparse + A_sparse' + 5*I  # Symmetric positive definite

# Matrix-free wrapper
A_mul!(y, x) = mul!(y, A_sparse, x)
A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true, isposdef=true)

# Find largest eigenvalues
result = feast(A_op, (4.8, 5.2), M0=8, solver=:cg)
```

## Performance Tips

1. **Choose appropriate solver**: Use `:cg` for symmetric positive definite systems, `:gmres` for general problems.

2. **Tune solver parameters**: Adjust `rtol`, `maxiter`, and `restart` based on your problem.

3. **Optimize matrix-vector products**: Make your `A_mul!` and `B_mul!` functions as efficient as possible.

4. **Use workspace reuse**: For repeated solves, pre-allocate workspace:
   ```julia
   workspace = allocate_matfree_workspace(Float64, n, M0)
   result = feast_matfree_srci!(A_op, B_op, interval, M0; workspace=workspace)
   ```

5. **Consider integration method**: Zolotarev integration often requires fewer points than Gauss-Legendre.

## Error Handling

Common issues and solutions:

- **Linear solver convergence**: Increase `maxiter`, decrease `rtol`, or try different solver
- **Feast not converging**: Increase `maxiter` in Feast parameters, adjust `tol` 
- **No eigenvalues found**: Check that search interval/region contains eigenvalues
- **Memory issues**: Use iterative solvers, increase sparsity, consider domain decomposition

## Integration with Other Packages

The matrix-free interface works well with:

- **IterativeSolvers.jl**: For advanced iterative methods
- **LinearMaps.jl**: Alternative operator interface
- **KrylovKit.jl**: High-performance Krylov methods  
- **Preconditioners.jl**: For preconditioning
- **CUDA.jl**: For GPU-accelerated operations

Example with LinearMaps.jl:
```julia
using LinearMaps, Feast

# Convert LinearMap to Feast operator
lmap = LinearMap(your_function!, n)
A_op = LinearOperator{Float64}((y,x) -> mul!(y, lmap, x), (n, n))

result = feast(A_op, interval)
```