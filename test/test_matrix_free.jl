# Tests for Matrix-Free Feast Interface

using Test
using Feast
using LinearAlgebra, SparseArrays
using Random

@testset "Matrix-Free Feast Tests" begin
    
    @testset "MatrixFreeOperator Types" begin
        # Test MatrixVecFunction
        n = 10
        A_mul!(y, op, x) = mul!(y, op.data, x)
        A_data = randn(n, n)
        A_data = A_data + A_data'  # Make symmetric
        
        op = MatrixVecFunction{Float64}(A_mul!, (n, n), issymmetric=true)
        
        @test size(op) == (n, n)
        @test size(op, 1) == n
        @test size(op, 2) == n
        @test issymmetric(op) == true
        @test ishermitian(op) == false
        @test isposdef(op) == false
        
        # Test LinearOperator
        A_mul2!(y, x) = mul!(y, A_data, x)
        op2 = LinearOperator{Float64}(A_mul2!, (n, n), issymmetric=true)
        
        @test size(op2) == (n, n)
        @test issymmetric(op2) == true
        
        # Test matrix-vector multiplication
        x = randn(n)
        y = zeros(n)
        mul!(y, op2, x)
        @test y ≈ A_data * x
    end
    
    @testset "Small Symmetric Problem" begin
        # 3x3 tridiagonal matrix
        A = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]
        n = size(A, 1)
        
        # Matrix-free operators
        A_mul!(y, x) = mul!(y, A, x)
        B_mul!(y, x) = copy!(y, x)
        
        A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
        B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                      issymmetric=true, ishermitian=true, isposdef=true)
        
        # Custom linear solver for testing
        function test_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
            shifted = z * I - A
            for j in 1:size(X, 2)
                Y[:, j] = shifted \\ X[:, j]
            end
        end
        
        # Test feast_matfree_srci! directly
        result = feast(A_op, B_op, (0.5, 1.5), M0=3, solver=test_solver, tol=1e-12)
        
        # Compare with exact eigenvalues
        true_eigvals = sort(eigvals(A))
        found_eigvals = sort(result.lambda[1:result.M])
        
        @test result.info == 0  # Success
        @test result.M >= 1     # At least one eigenvalue found
        
        # Check that found eigenvalues are correct
        for i in 1:result.M
            @test any(abs.(found_eigvals[i] .- true_eigvals) .< 1e-10)
        end
        
        # Check residuals
        for i in 1:result.M
            @test result.res[i] < 1e-8
        end
    end
    
    @testset "Workspace Allocation" begin
        # Test workspace allocation for real problems
        T = Float64
        N, M0 = 100, 10
        
        ws = Feast.allocate_matfree_workspace(T, N, M0)
        
        @test size(ws.work) == (N, M0)
        @test size(ws.workc) == (N, M0)
        @test size(ws.Aq) == (M0, M0)
        @test size(ws.Sq) == (M0, M0)
        @test length(ws.lambda) == M0
        @test size(ws.q) == (N, M0)
        @test length(ws.res) == M0
        
        @test eltype(ws.work) == T
        @test eltype(ws.workc) == Complex{T}
        @test eltype(ws.lambda) == T
        @test eltype(ws.q) == T
        @test eltype(ws.res) == T
    end
    
    @testset "Tridiagonal Matrix-Free" begin
        # Larger tridiagonal test
        n = 100
        
        # Matrix-free tridiagonal operator: T[i,i] = 2, T[i,i±1] = -1
        function tridiag_mul!(y, x)
            y[1] = 2*x[1] - x[2]
            for i in 2:n-1
                y[i] = -x[i-1] + 2*x[i] - x[i+1]
            end
            y[n] = -x[n-1] + 2*x[n]
        end
        
        B_mul!(y, x) = copy!(y, x)
        
        A_op = LinearOperator{Float64}(tridiag_mul!, (n, n), issymmetric=true)
        B_op = LinearOperator{Float64}(B_mul!, (n, n), 
                                      issymmetric=true, ishermitian=true, isposdef=true)
        
        # Search for eigenvalues near λ = 1.0
        # For this tridiagonal matrix: λ_k = 2 - 2*cos(kπ/(n+1))
        interval = (0.8, 1.2)
        
        # Mock iterative solver for testing (very simplified)
        function mock_iterative_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
            # This is a placeholder - in practice you'd use a real iterative method
            # For testing purposes, we'll use a direct method on the small problem
            if n <= 200  # Only for small test cases
                # Build actual tridiagonal matrix for direct solve
                T_full = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
                shifted = z * I - T_full
                for j in 1:size(X, 2)
                    Y[:, j] = shifted \\ X[:, j]
                end
            else
                throw(ArgumentError("Mock solver only works for small problems"))
            end
        end
        
        result = feast(A_op, B_op, interval, M0=8, 
                      solver=mock_iterative_solver, tol=1e-8)
        
        @test result.info == 0
        @test result.M > 0
        
        # Verify eigenvalues are in the correct range
        for i in 1:result.M
            @test interval[1] <= result.lambda[i] <= interval[2]
        end
    end
    
    @testset "Identity Operator" begin
        # Test standard eigenvalue problem A*x = λ*x (B = I)
        n = 5
        A = diagm([1.0, 2.0, 3.0, 4.0, 5.0])  # Simple diagonal matrix
        
        A_mul!(y, x) = mul!(y, A, x)
        A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
        
        # Custom solver for diagonal matrix
        function diag_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
            for j in 1:size(X, 2)
                for i in 1:n
                    Y[i, j] = X[i, j] / (z - A[i, i])
                end
            end
        end
        
        # Should find eigenvalues 2.0, 3.0, 4.0
        result = feast(A_op, (1.5, 4.5), M0=5, solver=diag_solver)
        
        @test result.info == 0
        @test result.M == 3  # Should find exactly 3 eigenvalues
        
        found_vals = sort(result.lambda[1:result.M])
        @test found_vals ≈ [2.0, 3.0, 4.0] atol=1e-10
    end
    
    @testset "Error Handling" begin
        n = 5
        A_mul!(y, x) = mul!(y, ones(n, n), x)  # Not symmetric
        B_mul!(y, x) = copy!(y, x)
        
        A_op = LinearOperator{Float64}(A_mul!, (n, n))  # Not marked symmetric
        B_op = LinearOperator{Float64}(B_mul!, (n, n), issymmetric=true, isposdef=true)
        
        # Should throw error for non-symmetric operator
        @test_throws ArgumentError feast(A_op, B_op, (0.0, 1.0))
        
        # Test mismatched sizes
        A_op_5 = LinearOperator{Float64}(A_mul!, (5, 5), issymmetric=true)
        B_op_6 = LinearOperator{Float64}((y,x) -> copy!(y, x[1:5]), (6, 6), issymmetric=true)
        
        @test_throws DimensionMismatch feast(A_op_5, B_op_6, (0.0, 1.0))
        
        # Test invalid interval
        A_op_sym = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
        @test_throws ArgumentError feast(A_op_sym, B_op, (1.0, 0.5))  # Emin > Emax
    end
    
    @testset "Custom Contour Integration" begin
        # Test using custom contour with matrix-free operators
        n = 3
        A = [1.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 3.0]
        
        A_mul!(y, x) = mul!(y, A, x)
        B_mul!(y, x) = copy!(y, x)
        
        A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)
        B_op = LinearOperator{Float64}(B_mul!, (n, n), issymmetric=true, isposdef=true)
        
        # Use expert contour interface
        contour = feast_contour_expert(1.0, 3.0, 8, 0, 100)  # Gauss-Legendre, circular
        
        @test length(contour.Zne) == 8
        @test length(contour.Wne) == 8
        
        # Verify contour points are reasonable
        for i in 1:8
            z = contour.Zne[i]
            @test 1.0 <= real(z) <= 3.0  # Should be in interval
        end
    end
    
end