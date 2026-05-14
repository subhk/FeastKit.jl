# Tests for Matrix-Free Feast Interface

using Test
using FeastKit
using LinearAlgebra, SparseArrays
using Random

# Minimal callable wrapper used to verify that matrix-free operators can store
# arbitrary user data and still participate in LinearAlgebra.mul!.
struct DenseTestMul{M}
    A::M
end

(op::DenseTestMul)(y, x) = mul!(y, op.A, x)

function apply_repeated_test!(y, op, x, nrepeat)
    for _ in 1:nrepeat
        mul!(y, op, x)
    end
    return y
end

@testset "Matrix-Free Feast Tests" begin
    
    @testset "MatrixFreeOperator Types" begin
        # Exercise both callback styles: one with an operator payload and one
        # with a closure-only LinearOperator.
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
        
        # Custom linear solver for testing the RCI contract: FEAST provides a
        # complex contour shift and expects Y = (zB - A)^-1 X.
        function test_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
            shifted = z * I - A
            for j in 1:size(X, 2)
                Y[:, j] = shifted \ X[:, j]
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
        
        ws = allocate_matfree_workspace(T, N, M0)
        
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
        
        # Small deterministic solver used to exercise the matrix-free callback path.
        function mock_iterative_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
            # For testing purposes, use a direct method on this small problem.
            if n <= 200  # Only for small test cases
                # Build actual tridiagonal matrix for direct solve
                T_full = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
                shifted = z * I - T_full
                for j in 1:size(X, 2)
                    Y[:, j] = shifted \ X[:, j]
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

    @testset "Default Matrix-Free GMRES Solver" begin
        # No custom solver is supplied here; this verifies that the built-in
        # GMRES wrapper can operate from matrix-vector callbacks alone.
        n = 5
        A = diagm(0 => collect(1.0:n))
        A_mul!(y, x) = mul!(y, A, x)
        A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)

        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 4

        result = feast(A_op, (1.5, 4.5); M0=n, fpm=fpm,
                       solver_opts=(rtol=1e-10, maxiter=100, restart=10))

        @test result.info == 0
        @test result.M == 3
        @test sort(result.lambda[1:result.M]) ≈ [2.0, 3.0, 4.0] atol=1e-8
    end

    @testset "Matrix-Free Callback Allocation" begin
        n = 8
        A = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        A_mul!(y, op, x) = mul!(y, A, x)
        op = MatrixVecFunction{Float64}(A_mul!, (n, n), issymmetric=true)
        x = ones(n)
        y = zeros(n)

        function apply_repeated!(y, op, x)
            for _ in 1:1_000
                mul!(y, op, x)
            end
            return y
        end

        apply_repeated!(y, op, x)
        bytes = @allocated begin
            apply_repeated!(y, op, x)
        end

        @test bytes < 1024
    end

    @testset "Matrix-Free Shifted Operator Type Stability" begin
        n = 8
        A = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        A_mul!(y, x) = mul!(y, A, x)
        B_mul!(y, x) = copyto!(y, x)

        op = FeastKit.MatrixFreeShiftedOperator(n, 0.5 + 0.25im, A_mul!, B_mul!, Float64)

        @test fieldtype(typeof(op), :A_matvec!) === typeof(A_mul!)
        @test fieldtype(typeof(op), :B_matvec!) === typeof(B_mul!)

        x = randn(ComplexF64, n)
        y = zeros(ComplexF64, n)
        @inferred mul!(y, op, x)
    end

    @testset "Polynomial Companion Operator Allocation" begin
        n = 4
        C0 = ComplexF64.(-diagm(0 => collect(1.0:n)))
        C1 = Matrix{ComplexF64}(I, n, n)

        coeffs_ops = [
            LinearOperator{ComplexF64}(DenseTestMul(C0), (n, n)),
            LinearOperator{ComplexF64}(DenseTestMul(C1), (n, n)),
        ]

        A_comp, B_comp = FeastKit._matrix_free_polynomial_companion_operators(coeffs_ops)
        x = randn(ComplexF64, n)
        y = zeros(ComplexF64, n)

        apply_repeated_test!(y, A_comp, x, 1)
        bytes_a = @allocated begin
            apply_repeated_test!(y, A_comp, x, 1_000)
        end

        apply_repeated_test!(y, B_comp, x, 1)
        bytes_b = @allocated begin
            apply_repeated_test!(y, B_comp, x, 1_000)
        end

        @test bytes_a < 1024
        @test bytes_b < 1024

        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[2] = 4
        fpm[4] = 3

        result = feast_polynomial(coeffs_ops, 2.5 + 0im, 3.0;
                                  M0=n, fpm=fpm,
                                  solver=:gmres,
                                  solver_opts=(rtol=1e-10, maxiter=100, restart=10))

        @test result.info == 0
        @test result.lambda isa Vector{ComplexF64}
        @test sort(real.(result.lambda[1:result.M])) ≈ collect(1.0:n) atol=1e-8
    end

    @testset "Typed Custom Contour Lookup" begin
        fpm = zeros(Int, 64)
        feastinit!(fpm)

        Zne = ComplexF64[1 + 0im, 0 + 1im, -1 + 0im, 0 - 1im]
        Wne = ComplexF64[0.25 + 0im, 0 + 0.25im, -0.25 + 0im, 0 - 0.25im]
        contour = FeastKit.FeastContour{Float64}(Zne, Wne)
        FeastKit.feast_set_custom_contour!(fpm, contour)

        try
            return_type = only(Base.return_types(FeastKit.feast_get_custom_contour,
                                                 (Type{Float64}, Vector{Int})))
            @test return_type == Union{Nothing, FeastKit.FeastContour{Float64}}

            typed = FeastKit.feast_get_custom_contour(Float64, fpm)
            @test typed isa FeastKit.FeastContour{Float64}
            @test typed.Zne == Zne
            @test typed.Wne == Wne
        finally
            FeastKit.feast_clear_custom_contour!(fpm)
        end
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
