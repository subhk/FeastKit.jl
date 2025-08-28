# Matrix-Free Feast Examples
# Demonstrates how to use Feast without storing explicit matrices

using Feast
using LinearAlgebra, SparseArrays
using Random

"""
Example 1: Large Tridiagonal Matrix (Matrix-Free)

Solve eigenvalue problem for a large symmetric tridiagonal matrix
without storing the full matrix. This is ideal for very large problems
where memory is a constraint.
"""
function example_tridiagonal_matfree()
    println("=== Example 1: Large Tridiagonal Matrix (Matrix-Free) ===")
    
    # Problem size
    n = 10000  # Large size that would use significant memory if stored explicitly
    
    # Tridiagonal matrix: T[i,i] = 2, T[i,i±1] = -1
    # This gives eigenvalues λ_k = 2 - 2*cos(kπ/(n+1)) for k = 1,...,n
    
    # Matrix-free operator for A*x
    function A_mul!(y, x)
        # Compute y = T*x for tridiagonal matrix T
        y[1] = 2*x[1] - x[2]
        for i in 2:n-1
            y[i] = -x[i-1] + 2*x[i] - x[i+1]
        end
        y[n] = -x[n-1] + 2*x[n]
    end
    
    # Matrix-free operator for B*x (identity)
    function B_mul!(y, x)
        copy!(y, x)
    end
    
    # Create matrix-free operators
    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                  issymmetric=true, ishermitian=true, isposdef=true)
    
    # Search interval - look for eigenvalues near λ = 1.0
    interval = (0.8, 1.2)
    
    println("Problem size: $n × $n")
    println("Search interval: $interval")
    println("Expected eigenvalues in interval: approximately 6-8")
    
    # Solve using matrix-free Feast
    println("\\nSolving with matrix-free Feast...")
    result = feast(A_op, B_op, interval, M0=10, 
                  solver=:cg,  # CG works well for symmetric positive definite systems
                  solver_opts=(rtol=1e-8, maxiter=1000))
    
    println("Eigenvalues found: $(result.M)")
    println("Feast convergence: $(result.info == 0 ? "Success" : "Failed")")
    println("Final residual: $(result.epsout)")
    println("Refinement loops: $(result.loop)")
    
    if result.M > 0
        println("\\nEigenvalues:")
        for i in 1:result.M
            # Analytical solution for comparison
            k_approx = round(Int, acos(1 - result.lambda[i]/2) * (n+1) / π)
            λ_exact = 2 - 2*cos(k_approx * π / (n+1))
            error = abs(result.lambda[i] - λ_exact)
            
            println("  λ[$i] = $(result.lambda[i]) (error: $(error), residual: $(result.res[i]))")
        end
    end
    
    return result
end

"""
Example 2: Finite Difference Laplacian (2D)

Matrix-free Feast for 2D discrete Laplacian arising from finite differences.
This is a common application in PDEs and scientific computing.
"""
function example_2d_laplacian_matfree()
    println("\\n=== Example 2: 2D Laplacian (Matrix-Free) ===")
    
    # Grid parameters
    nx, ny = 100, 100
    n = nx * ny  # Total degrees of freedom
    h = 1.0 / (nx + 1)  # Grid spacing
    
    println("Grid size: $nx × $ny (total DOFs: $n)")
    println("Grid spacing: h = $h")
    
    # Convert 2D index (i,j) to linear index
    idx(i, j) = (j-1) * nx + i
    
    # Matrix-free operator for -Δu (negative 2D Laplacian)
    function laplacian_mul!(y, x)
        fill!(y, 0)
        
        for j in 1:ny
            for i in 1:nx
                k = idx(i, j)
                
                # Central difference stencil: [-1 -1 4 -1 -1] / h²
                y[k] += 4 * x[k] / h^2
                
                if i > 1
                    y[k] -= x[idx(i-1, j)] / h^2
                end
                if i < nx
                    y[k] -= x[idx(i+1, j)] / h^2
                end
                if j > 1
                    y[k] -= x[idx(i, j-1)] / h^2
                end
                if j < ny
                    y[k] -= x[idx(i, j+1)] / h^2
                end
            end
        end
    end
    
    # Identity operator
    B_mul!(y, x) = copy!(y, x)
    
    # Create operators
    A_op = LinearOperator{Float64}(laplacian_mul!, (n, n), issymmetric=true, isposdef=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                  issymmetric=true, ishermitian=true, isposdef=true)
    
    # Search for smallest eigenvalues
    # For 2D Laplacian, smallest eigenvalue is approximately 2π²(1/nx² + 1/ny²)
    λ_min_approx = 2 * π^2 * (1/nx^2 + 1/ny^2)
    interval = (0.8 * λ_min_approx, 3.0 * λ_min_approx)
    
    println("Approximate smallest eigenvalue: $λ_min_approx")
    println("Search interval: $interval")
    
    # Solve with matrix-free Feast
    println("\\nSolving with matrix-free Feast...")
    result = feast(A_op, B_op, interval, M0=15,
                  solver=:cg,
                  solver_opts=(rtol=1e-6, maxiter=500))
    
    println("Eigenvalues found: $(result.M)")
    println("Feast status: $(result.info == 0 ? "Success" : "Error $(result.info)")")
    
    if result.M > 0
        println("\\nSmallest eigenvalues:")
        for i in 1:min(5, result.M)
            println("  λ[$i] = $(result.lambda[i]) (residual: $(result.res[i]))")
        end
    end
    
    return result
end

"""
Example 3: Custom Linear Solver

Demonstrates how to provide custom linear solver for specialized problems.
"""
function example_custom_solver()
    println("\\n=== Example 3: Custom Linear Solver ===")
    
    # Small test problem: 3x3 symmetric matrix
    n = 3
    A_matrix = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]
    
    # Matrix-free operators
    A_mul!(y, x) = mul!(y, A_matrix, x)
    B_mul!(y, x) = copy!(y, x)
    
    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                  issymmetric=true, ishermitian=true, isposdef=true)
    
    # Custom linear solver using direct factorization
    function custom_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
        println("    Custom solver called with z = $z")
        
        # Form shifted matrix: z*B - A = z*I - A
        shifted_matrix = z * I - A_matrix
        
        # Direct solve for each column
        M0 = size(X, 2)
        for j in 1:M0
            Y[:, j] = shifted_matrix \\ X[:, j]
        end
        
        println("    Linear systems solved successfully")
    end
    
    # Search interval
    interval = (0.5, 1.5)
    
    println("Matrix A:")
    display(A_matrix)
    println("Search interval: $interval")
    
    # Solve with custom solver
    println("\\nSolving with custom linear solver...")
    result = feast(A_op, B_op, interval, M0=3,
                  solver=custom_solver,
                  tol=1e-10)
    
    println("\\nResults:")
    println("Eigenvalues found: $(result.M)")
    
    if result.M > 0
        # Compare with Julia's built-in eigensolver
        true_eigvals = eigvals(A_matrix)
        println("\\nComparison with exact eigenvalues:")
        
        for i in 1:result.M
            closest_true = true_eigvals[argmin(abs.(true_eigvals .- result.lambda[i]))]
            error = abs(result.lambda[i] - closest_true)
            println("  Feast: λ[$i] = $(result.lambda[i])")
            println("  Exact:       λ = $closest_true")
            println("  Error:           $(error)")
            println()
        end
    end
    
    return result
end

"""
Example 4: Matrix-Free General (Non-Hermitian) Problem

Demonstrates Feast for general eigenvalue problems using matrix-free operators.
"""
function example_general_matfree()
    println("\\n=== Example 4: General (Non-Hermitian) Matrix-Free ===")
    
    # Create a non-symmetric matrix operator
    n = 8
    
    # Non-symmetric matrix with complex eigenvalues
    function A_mul!(y, x)
        # Upper Hessenberg matrix with some asymmetry
        fill!(y, 0)
        for i in 1:n
            y[i] += 2.0 * x[i]  # Diagonal
            if i > 1
                y[i] += -0.8 * x[i-1]  # Sub-diagonal
            end
            if i < n
                y[i] += -1.2 * x[i+1]  # Super-diagonal  
            end
        end
    end
    
    B_mul!(y, x) = copy!(y, x)
    
    # Create complex operators for general problem
    A_op = LinearOperator{ComplexF64}((y, x) -> A_mul!(y, real.(x)), (n, n))
    B_op = LinearOperator{ComplexF64}((y, x) -> B_mul!(y, real.(x)), (n, n))
    
    # Search region (circular contour in complex plane)
    center = 1.0 + 0.0im
    radius = 1.5
    
    println("Problem size: $n × $n")
    println("Search region: circle centered at $center with radius $radius")
    
    # Solve general eigenvalue problem
    println("\\nSolving general eigenvalue problem...")
    result = feast_general(A_op, B_op, center, radius, M0=6,
                          solver=:gmres,
                          solver_opts=(restart=20, rtol=1e-8))
    
    println("Eigenvalues found: $(result.M)")
    println("Feast status: $(result.info == 0 ? "Success" : "Error $(result.info)")")
    
    if result.M > 0
        println("\\nEigenvalues in search region:")
        for i in 1:result.M
            λ = result.lambda[i]
            distance_from_center = abs(λ - center)
            println("  λ[$i] = $λ (distance from center: $distance_from_center)")
        end
    end
    
    return result
end

"""
Example 5: Large Sparse Matrix via Matrix-Free Interface

Shows how to use matrix-free interface even when you have a sparse matrix,
which can be useful for very large problems or when you want to avoid
storing factorizations.
"""
function example_sparse_as_matfree()
    println("\\n=== Example 5: Sparse Matrix via Matrix-Free Interface ===")
    
    # Create a large sparse matrix
    n = 5000
    Random.seed!(123)
    
    # Random sparse symmetric matrix
    A_sparse = sprand(n, n, 0.001)  # Very sparse: 0.1% nonzeros
    A_sparse = A_sparse + A_sparse'  # Make symmetric
    A_sparse += 10.0 * I  # Make positive definite
    
    nnz_A = nnz(A_sparse)
    density = nnz_A / n^2
    
    println("Sparse matrix size: $n × $n")
    println("Nonzeros: $nnz_A (density: $(100*density)%)")
    
    # Matrix-free operators using sparse matrix
    A_mul!(y, x) = mul!(y, A_sparse, x)
    B_mul!(y, x) = copy!(y, x)
    
    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true, isposdef=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                  issymmetric=true, ishermitian=true, isposdef=true)
    
    # Search for largest eigenvalues
    # Rough estimate using power iteration for comparison
    temp_v = randn(n)
    for _ in 1:10
        temp_v = A_sparse * temp_v
        temp_v ./= norm(temp_v)
    end
    λ_max_approx = dot(temp_v, A_sparse * temp_v)
    
    interval = (0.9 * λ_max_approx, 1.1 * λ_max_approx)
    
    println("Approximate largest eigenvalue: $λ_max_approx")
    println("Search interval: $interval")
    
    # Solve with matrix-free Feast
    println("\\nSolving with matrix-free Feast...")
    result = feast(A_op, B_op, interval, M0=8,
                  solver=:gmres,
                  solver_opts=(restart=50, rtol=1e-6, maxiter=200))
    
    println("Eigenvalues found: $(result.M)")
    println("Feast status: $(result.info == 0 ? "Success" : "Error $(result.info)")")
    
    if result.M > 0
        println("\\nLargest eigenvalues:")
        for i in 1:result.M
            println("  λ[$i] = $(result.lambda[i]) (residual: $(result.res[i]))")
        end
    end
    
    return result
end

"""
Run all matrix-free examples
"""
function run_matfree_examples()
    println("Feast Matrix-Free Interface Examples")
    println("=====================================")
    
    try
        # Run examples
        result1 = example_tridiagonal_matfree()
        result2 = example_2d_laplacian_matfree() 
        result3 = example_custom_solver()
        result4 = example_general_matfree()
        result5 = example_sparse_as_matfree()
        
        println("\\n" * "="^50)
        println("All matrix-free examples completed successfully!")
        println("✅ Tridiagonal matrix: $(result1.M) eigenvalues found")
        println("✅ 2D Laplacian: $(result2.M) eigenvalues found") 
        println("✅ Custom solver: $(result3.M) eigenvalues found")
        println("✅ General problem: $(result4.M) eigenvalues found")
        println("✅ Sparse matrix: $(result5.M) eigenvalues found")
        
    catch e
        println("❌ Error running matrix-free examples: $e")
        rethrow(e)
    end
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_matfree_examples()
end