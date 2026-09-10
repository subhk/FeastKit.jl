# Matrix-Free FeastKit Examples
# Demonstrates how to use FeastKit without storing explicit matrices
#
# Every search interval below is derived from the problem's known spectrum, so
# each example prints how many eigenvalues it should find and then finds exactly
# that many. An example that reports `Error 5` is not demonstrating anything.

using FeastKit
# Iterative matrix-free solves go through the FeastKitKrylovExt extension.
using Krylov
using LinearAlgebra, SparseArrays
using Random

"""
Example 1: Large Tridiagonal Matrix (Matrix-Free)

Solve an eigenvalue problem for a large symmetric tridiagonal matrix without
storing it. The shifted systems are solved with a Thomas-algorithm sweep, which
is the point of the matrix-free interface: FEAST never needs the matrix, only
your fastest way of applying it and of inverting `z*B - A`.
"""
function example_tridiagonal_matfree()
    println("=== Example 1: Large Tridiagonal Matrix (Matrix-Free) ===")

    # Problem size
    n = 10_000  # Large size that would use significant memory if stored explicitly

    # Tridiagonal matrix: T[i,i] = 2, T[i,i±1] = -1
    # Eigenvalues are known exactly: λ_k = 2 - 2*cos(kπ/(n+1)) for k = 1,...,n
    exact_eigenvalue(k) = 2 - 2 * cos(k * π / (n + 1))

    # Matrix-free operator for A*x
    function A_mul!(y, x)
        y[1] = 2 * x[1] - x[2]
        for i in 2:(n - 1)
            y[i] = -x[i - 1] + 2 * x[i] - x[i + 1]
        end
        y[n] = -x[n - 1] + 2 * x[n]
        return y
    end

    # Matrix-free operator for B*x (identity)
    B_mul!(y, x) = copy!(y, x)

    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n),
                                   issymmetric=true, ishermitian=true, isposdef=true)

    # Bracket eigenvalues k = 3330..3337, which sit near λ = 1. Near the middle
    # of this spectrum the eigenvalues are spaced by only ~5e-4, so an interval
    # picked by eye (say 0.8 to 1.2) would contain ~700 of them and no modest M0
    # could ever converge.
    k_first, k_last = 3330, 3337
    gap = exact_eigenvalue(k_first + 1) - exact_eigenvalue(k_first)
    interval = (exact_eigenvalue(k_first) - 0.2 * gap,
                exact_eigenvalue(k_last) + 0.2 * gap)
    expected = count(k -> interval[1] <= exact_eigenvalue(k) <= interval[2], 1:n)

    println("Problem size: $n × $n")
    println("Search interval: $interval")
    println("Eigenvalues in interval (exact): $expected")

    # (z*I - A) is tridiagonal with diagonal z-2 and off-diagonals +1, so each
    # shifted solve is an O(n) Thomas sweep in complex arithmetic.
    function tridiagonal_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
        diag_val = ComplexF64(z) - 2
        c = Vector{ComplexF64}(undef, n - 1)
        d = Vector{ComplexF64}(undef, n)
        for j in axes(X, 2)
            c[1] = 1 / diag_val
            d[1] = X[1, j] / diag_val
            for i in 2:n
                m = diag_val - c[i - 1]
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

    println("\nSolving with matrix-free FeastKit...")
    result = feast(A_op, B_op, interval, M0=2 * expected,
                   solver=tridiagonal_solver, tol=1e-10)

    println("Eigenvalues found: $(result.M)")
    println("FeastKit status: $(result.info == 0 ? "Success" : "Error $(result.info)")")
    println("Final residual: $(result.epsout)")
    println("Refinement loops: $(result.loop)")

    if result.M > 0
        println("\nEigenvalues:")
        for i in 1:result.M
            k = round(Int, acos(1 - result.lambda[i] / 2) * (n + 1) / π)
            err = abs(result.lambda[i] - exact_eigenvalue(k))
            println("  λ[$i] = $(result.lambda[i]) (error: $err, residual: $(result.res[i]))")
        end
    end

    return result
end

"""
Example 2: Finite Difference Laplacian (2D)

Matrix-free Feast for the 2D discrete Laplacian from finite differences, a
common application in PDEs and scientific computing. The shifted systems go to
unpreconditioned GMRES, which is why the grid here is moderate.
"""
function example_2d_laplacian_matfree()
    println("\n=== Example 2: 2D Laplacian (Matrix-Free) ===")

    # Grid parameters
    nx, ny = 40, 40
    n = nx * ny          # Total degrees of freedom
    h = 1.0 / (nx + 1)   # Grid spacing

    println("Grid size: $nx × $ny (total DOFs: $n)")
    println("Grid spacing: h = $h")

    # Convert 2D index (i,j) to linear index
    idx(i, j) = (j - 1) * nx + i

    # Matrix-free operator for -Δu (negative 2D Laplacian)
    function laplacian_mul!(y, x)
        fill!(y, 0)
        for j in 1:ny, i in 1:nx
            k = idx(i, j)
            # Central difference stencil: [-1 -1 4 -1 -1] / h²
            y[k] += 4 * x[k] / h^2
            i > 1  && (y[k] -= x[idx(i - 1, j)] / h^2)
            i < nx && (y[k] -= x[idx(i + 1, j)] / h^2)
            j > 1  && (y[k] -= x[idx(i, j - 1)] / h^2)
            j < ny && (y[k] -= x[idx(i, j + 1)] / h^2)
        end
        return y
    end

    B_mul!(y, x) = copy!(y, x)

    A_op = LinearOperator{Float64}(laplacian_mul!, (n, n), issymmetric=true, isposdef=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n),
                                   issymmetric=true, ishermitian=true, isposdef=true)

    # The 5-point Laplacian scaled by 1/h² has the exact spectrum
    #   λ_{p,q} = (4 - 2cos(pπ/(nx+1)) - 2cos(qπ/(ny+1))) / h²
    # so the smallest eigenvalue tends to 2π² as the grid refines, not to
    # 2π²(1/nx² + 1/ny²).
    exact(p, q) = (4 - 2 * cos(p * π / (nx + 1)) - 2 * cos(q * π / (ny + 1))) / h^2
    spectrum = sort([exact(p, q) for p in 1:nx for q in 1:ny])
    interval = (spectrum[1] - 0.05 * (spectrum[4] - spectrum[1]),
                spectrum[3] + 0.05 * (spectrum[4] - spectrum[3]))
    expected = count(λ -> interval[1] <= λ <= interval[2], spectrum)

    println("Smallest exact eigenvalue: $(spectrum[1])  (2π² = $(2π^2))")
    println("Search interval: $interval")
    println("Eigenvalues in interval (exact): $expected")

    println("\nSolving with matrix-free FeastKit...")
    result = feast(A_op, B_op, interval, M0=2 * expected + 2,
                   solver=:gmres,
                   solver_opts=(rtol=1e-10, maxiter=4000, restart=60),
                   tol=1e-8)

    println("Eigenvalues found: $(result.M)")
    println("FeastKit status: $(result.info == 0 ? "Success" : "Error $(result.info)")")

    if result.M > 0
        println("\nSmallest eigenvalues:")
        for i in 1:min(5, result.M)
            println("  λ[$i] = $(result.lambda[i]) (residual: $(result.res[i]))")
        end
    end

    return result
end

"""
Example 3: Custom Linear Solver

Demonstrates how to provide a custom linear solver for specialized problems.
"""
function example_custom_solver()
    println("\n=== Example 3: Custom Linear Solver ===")

    # Small test problem: 3x3 symmetric matrix
    n = 3
    A_matrix = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]

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

        for j in axes(X, 2)
            Y[:, j] = shifted_matrix \ X[:, j]
        end

        println("    Linear systems solved successfully")
        return Y
    end

    interval = (0.5, 1.5)
    expected = count(λ -> interval[1] <= λ <= interval[2], eigvals(A_matrix))

    println("Matrix A:")
    display(A_matrix)
    println("Search interval: $interval")
    println("Eigenvalues in interval (exact): $expected")

    println("\nSolving with custom linear solver...")
    result = feast(A_op, B_op, interval, M0=n, solver=custom_solver, tol=1e-10)

    println("\nResults:")
    println("Eigenvalues found: $(result.M)")

    if result.M > 0
        true_eigvals = eigvals(A_matrix)
        println("\nComparison with exact eigenvalues:")
        for i in 1:result.M
            closest_true = true_eigvals[argmin(abs.(true_eigvals .- result.lambda[i]))]
            err = abs(result.lambda[i] - closest_true)
            println("  Feast: λ[$i] = $(result.lambda[i])")
            println("  Exact:       λ = $closest_true")
            println("  Error:           $err")
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
    println("\n=== Example 4: General (Non-Hermitian) Matrix-Free ===")

    n = 8

    # Non-symmetric tridiagonal operator. It must be applied to the vector it is
    # given: taking real.(x) first would make it non-linear over the complex
    # field, and FEAST's contour shifts are complex.
    function A_mul!(y, x)
        fill!(y, 0)
        for i in 1:n
            y[i] += 2.0 * x[i]                  # Diagonal
            i > 1 && (y[i] += -0.8 * x[i - 1])  # Sub-diagonal
            i < n && (y[i] += -1.2 * x[i + 1])  # Super-diagonal
        end
        return y
    end

    B_mul!(y, x) = copy!(y, x)

    A_op = LinearOperator{ComplexF64}(A_mul!, (n, n))
    B_op = LinearOperator{ComplexF64}(B_mul!, (n, n))

    # Search region (circular contour in complex plane)
    center = 1.0 + 0.0im
    radius = 1.5

    # Dense reference just to state the expected count for this small problem.
    A_dense = zeros(ComplexF64, n, n)
    for i in 1:n
        A_dense[i, i] = 2.0
        i > 1 && (A_dense[i, i - 1] = -0.8)
        i < n && (A_dense[i, i + 1] = -1.2)
    end
    expected = count(λ -> abs(λ - center) <= radius, eigvals(A_dense))

    println("Problem size: $n × $n")
    println("Search region: circle centered at $center with radius $radius")
    println("Eigenvalues in region (exact): $expected")

    println("\nSolving general eigenvalue problem...")
    result = feast_general(A_op, B_op, center, radius, M0=min(2 * expected, n),
                           solver=:gmres,
                           solver_opts=(restart=20, rtol=1e-10, maxiter=500),
                           tol=1e-10)

    println("Eigenvalues found: $(result.M)")
    println("FeastKit status: $(result.info == 0 ? "Success" : "Error $(result.info)")")

    if result.M > 0
        println("\nEigenvalues in search region:")
        for i in 1:result.M
            λ = result.lambda[i]
            println("  λ[$i] = $λ (distance from center: $(abs(λ - center)))")
        end
    end

    return result
end

"""
Example 5: Large Sparse Matrix via Matrix-Free Interface

Shows how to use the matrix-free interface even when you have a sparse matrix,
which is useful for very large problems or when you want to avoid storing
factorizations.
"""
function example_sparse_as_matfree()
    println("\n=== Example 5: Sparse Matrix via Matrix-Free Interface ===")

    n = 5000
    Random.seed!(123)

    A_sparse = sprand(n, n, 0.001)   # Very sparse: 0.1% nonzeros
    A_sparse = A_sparse + A_sparse'  # Make symmetric
    A_sparse += 10.0 * I             # Make positive definite

    nnz_A = nnz(A_sparse)
    println("Sparse matrix size: $n × $n")
    println("Nonzeros: $nnz_A (density: $(100 * nnz_A / n^2)%)")

    A_mul!(y, x) = mul!(y, A_sparse, x)
    B_mul!(y, x) = copy!(y, x)

    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true, isposdef=true)
    B_op = LinearOperator{Float64}(B_mul!, (n, n),
                                   issymmetric=true, ishermitian=true, isposdef=true)

    # Estimate the largest eigenvalue by power iteration, then bracket it. It is
    # well separated from the bulk here, so a narrow absolute window isolates it;
    # a window scaled as a percentage of λ_max would swallow part of the bulk.
    function power_iteration_max(S, N)
        v = randn(N)
        for _ in 1:200
            v = S * v
            v ./= norm(v)
        end
        return dot(v, S * v)
    end
    λ_max_approx = power_iteration_max(A_sparse, n)
    interval = (λ_max_approx - 0.1, λ_max_approx + 0.1)

    println("Approximate largest eigenvalue: $λ_max_approx")
    println("Search interval: $interval")

    println("\nSolving with matrix-free FeastKit...")
    result = feast(A_op, B_op, interval, M0=8,
                   solver=:gmres,
                   solver_opts=(rtol=1e-10, maxiter=2000, restart=60),
                   tol=1e-8)

    println("Eigenvalues found: $(result.M)")
    println("FeastKit status: $(result.info == 0 ? "Success" : "Error $(result.info)")")

    if result.M > 0
        println("\nLargest eigenvalues:")
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
    println("FeastKit Matrix-Free Interface Examples")
    println("=====================================")

    results = ["Tridiagonal matrix" => example_tridiagonal_matfree(),
               "2D Laplacian" => example_2d_laplacian_matfree(),
               "Custom solver" => example_custom_solver(),
               "General problem" => example_general_matfree(),
               "Sparse matrix" => example_sparse_as_matfree()]

    println("\n" * "="^50)
    for (name, r) in results
        status = r.info == 0 ? "converged" : "NOT converged (info $(r.info))"
        println("$name: $(r.M) eigenvalues found, $status")
    end

    failed = [name for (name, r) in results if r.info != 0]
    if isempty(failed)
        println("\nAll matrix-free examples converged.")
    else
        error("Did not converge: " * join(failed, ", "))
    end

    return results
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_matfree_examples()
end
