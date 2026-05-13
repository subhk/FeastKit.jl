using Test
using FeastKit
using LinearAlgebra
using SparseArrays

function _repeat_reorder!(lambda, vectors, lambda_src, vectors_src,
                          perm, lambda_tmp, vector_tmp, Emin, Emax, nrepeat)
    for _ in 1:nrepeat
        copyto!(lambda, lambda_src)
        copyto!(vectors, vectors_src)
        FeastKit._feast_reorder_by_interval!(lambda, vectors, perm,
                                              lambda_tmp, vector_tmp,
                                              Emin, Emax, length(lambda))
    end
    return lambda, vectors
end

function _repeat_copy_real!(dest, src, nrepeat)
    for _ in 1:nrepeat
        FeastKit._feast_copy_real!(dest, src)
    end
    return dest
end

function _repeat_dense_shifted_identity!(dest, z, A, nrepeat)
    for _ in 1:nrepeat
        FeastKit._feast_dense_shifted_identity_minus!(dest, z, A)
    end
    return dest
end

@testset "Allocation helper regressions" begin
    @testset "In-place interval reorder" begin
        lambda_src = [4.0, 2.0, 1.0, 3.0]
        vectors_src = ComplexF64[
            11 12 13 14
            21 22 23 24
            31 32 33 34
        ]
        lambda = copy(lambda_src)
        vectors = copy(vectors_src)
        perm = Vector{Int}(undef, length(lambda))
        lambda_tmp = similar(lambda)
        vector_tmp = similar(vectors)

        m = FeastKit._feast_reorder_by_interval!(lambda, vectors, perm,
                                                 lambda_tmp, vector_tmp,
                                                 1.5, 3.5, length(lambda))

        @test m == 2
        @test lambda == [2.0, 3.0, 4.0, 1.0]
        @test vectors == vectors_src[:, [2, 4, 1, 3]]

        _repeat_reorder!(lambda, vectors, lambda_src, vectors_src, perm,
                         lambda_tmp, vector_tmp, 1.5, 3.5, 1)
        bytes = @allocated _repeat_reorder!(lambda, vectors, lambda_src, vectors_src,
                                            perm, lambda_tmp, vector_tmp, 1.5, 3.5,
                                            1_000)
        @test bytes < 1024
    end

    @testset "In-place real copy" begin
        src = ComplexF64[
            1 + 2im 3 + 4im
            5 + 6im 7 + 8im
        ]
        dest = zeros(Float64, size(src))

        FeastKit._feast_copy_real!(dest, src)
        @test dest == real.(src)

        _repeat_copy_real!(dest, src, 1)
        _ = @allocated _repeat_copy_real!(dest, src, 1)
        bytes = @allocated _repeat_copy_real!(dest, src, 1_000)
        @test bytes < 1024
    end

    @testset "Complex-to-real result conversion avoids column temporaries" begin
        n = 1_000
        m = 100
        result = FeastResult{Float64, ComplexF64}(collect(1.0:m),
                                                  randn(ComplexF64, n, m),
                                                  m,
                                                  ones(m),
                                                  0,
                                                  0.0,
                                                  1)
        converted = FeastKit._complex_to_real_result(result)
        @test converted isa FeastResult{Float64, Float64}
        @test converted.q == real.(result.q)

        FeastKit._complex_to_real_result(result)
        bytes = @allocated FeastKit._complex_to_real_result(result)
        @test bytes < 1_300_000
    end

    @testset "Dense standard solver avoids generalized identity workspace" begin
        n = 40
        A = Matrix(Symmetric(diagm(0 => collect(range(1.0, 4.0; length=n)),
                                  1 => fill(0.01, n - 1),
                                  -1 => fill(0.01, n - 1))))
        B = Matrix{Float64}(I, n, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 4
        fpm[4] = 4

        feast_syev!(copy(A), 1.2, 3.8, 20, copy(fpm))
        feast_sygv!(copy(A), copy(B), 1.2, 3.8, 20, copy(fpm))
        feast(A, (1.2, 3.8); M0=20, fpm=copy(fpm), backend=:serial)
        feast(A, B, (1.2, 3.8); M0=20, fpm=copy(fpm), backend=:serial)
        standard_bytes = @allocated feast_syev!(copy(A), 1.2, 3.8, 20, copy(fpm))
        generalized_bytes = @allocated feast_sygv!(copy(A), copy(B), 1.2, 3.8, 20, copy(fpm))
        high_level_standard_bytes = @allocated feast(A, (1.2, 3.8);
                                                     M0=20, fpm=copy(fpm),
                                                     backend=:serial)
        high_level_generalized_bytes = @allocated feast(A, B, (1.2, 3.8);
                                                       M0=20, fpm=copy(fpm),
                                                       backend=:serial)
        @test standard_bytes < generalized_bytes
        @test high_level_standard_bytes < high_level_generalized_bytes
    end

    @testset "Sparse standard solver avoids generalized identity workspace" begin
        n = 40
        A = sparse(Matrix(Symmetric(diagm(0 => collect(range(1.0, 4.0; length=n)),
                                         1 => fill(0.01, n - 1),
                                         -1 => fill(0.01, n - 1)))))
        B = sparse(Matrix{Float64}(I, n, n))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 4
        fpm[4] = 4

        feast_scsrev!(copy(A), 1.2, 3.8, 20, copy(fpm))
        feast_scsrgv!(copy(A), copy(B), 1.2, 3.8, 20, copy(fpm))
        standard_bytes = @allocated feast_scsrev!(copy(A), 1.2, 3.8, 20, copy(fpm))
        generalized_bytes = @allocated feast_scsrgv!(copy(A), copy(B), 1.2, 3.8, 20, copy(fpm))
        @test standard_bytes < generalized_bytes
    end

    @testset "Shifted standard systems avoid explicit identity allocation" begin
        n = 120
        z = 1.5 + 0.25im
        A_dense = ComplexF64.(Matrix(Symmetric(diagm(0 => collect(range(1.0, 4.0; length=n)),
                                                1 => fill(0.01, n - 1),
                                                -1 => fill(0.01, n - 1)))))
        shifted_dense = similar(A_dense)

        FeastKit._feast_dense_shifted_identity_minus!(shifted_dense, z, A_dense)
        @test shifted_dense ≈ z * Matrix{ComplexF64}(I, n, n) - A_dense

        _repeat_dense_shifted_identity!(shifted_dense, z, A_dense, 1)
        _ = @allocated _repeat_dense_shifted_identity!(shifted_dense, z, A_dense, 1)
        bytes_dense = @allocated _repeat_dense_shifted_identity!(shifted_dense,
                                                                 z, A_dense,
                                                                 1_000)
        @test bytes_dense < 1024

        A_sparse = sparse(A_dense)
        shifted_sparse = FeastKit._feast_sparse_shifted_identity_minus(A_sparse, z)
        @test Matrix(shifted_sparse) ≈ z * Matrix{ComplexF64}(I, n, n) - A_dense
        @test nnz(shifted_sparse) == nnz(A_sparse)

        generic_bytes = @allocated z * spdiagm(0 => fill(one(ComplexF64), n)) - A_sparse
        helper_bytes = @allocated FeastKit._feast_sparse_shifted_identity_minus(A_sparse, z)
        @test helper_bytes < generic_bytes
    end

    @testset "Dense general standard avoids explicit identity matrix" begin
        n = 80
        M0 = 16
        eigenvalues = complex.(collect(1.0:n), 0.02 .* collect(1.0:n))
        A = Matrix(Diagonal(eigenvalues))
        B = Matrix{ComplexF64}(I, n, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 8
        fpm[4] = 1
        center = 20.0 + 0.4im
        radius = 12.0

        feast_geev!(copy(A), center, radius, M0, copy(fpm))
        feast_gegv!(copy(A), B, center, radius, M0, copy(fpm))
        standard_bytes = @allocated feast_geev!(copy(A), center, radius, M0, copy(fpm))
        generalized_bytes = @allocated feast_gegv!(copy(A), B, center, radius, M0, copy(fpm))
        _ = Matrix{ComplexF64}(I, n, n)
        identity_bytes = @allocated Matrix{ComplexF64}(I, n, n)

        # The standard and generalized paths are within allocator noise of each
        # other across Julia/OS versions. This regression guard is sized to
        # catch reintroducing a dense identity matrix, not byte-for-byte order.
        identity_slack = max(16_384, identity_bytes ÷ 4)
        @test standard_bytes <= generalized_bytes + identity_slack
    end

    @testset "Threaded moment helpers avoid avoidable temporaries" begin
        n = 48
        M0 = 8
        A = Matrix(Diagonal(collect(1.0:n)))
        B = Matrix{Float64}(I, n, n)
        A_sparse = sparse(A)
        B_sparse = sparse(B)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 6
        fpm[4] = 1
        contour = FeastKit.feast_contour(5.5, 18.5, fpm)
        work = randn(n, M0)

        FeastKit.pfeast_compute_moments_threaded(A, B, work, contour, M0)
        FeastKit.pfeast_compute_sparse_moments_threaded(A_sparse, B_sparse, work, contour, M0)
        dense_bytes = @allocated FeastKit.pfeast_compute_moments_threaded(A, B, work,
                                                                          contour, M0)
        sparse_bytes = @allocated FeastKit.pfeast_compute_sparse_moments_threaded(A_sparse,
                                                                                 B_sparse,
                                                                                 work,
                                                                                 contour,
                                                                                 M0)

        @test dense_bytes < 600_000
        @test sparse_bytes < 420_000
    end

    @testset "Matrix-free complex workspace has RHS scratch" begin
        workspace = allocate_matfree_workspace(ComplexF64, 8, 3)
        @test hasproperty(workspace, :rhs)
        @test size(workspace.rhs) == (8, 3)
        @test eltype(workspace.rhs) === ComplexF64
    end

    @testset "Matrix-free GMRES reuses shifted solve scratch" begin
        n = 32
        M0 = 10
        values = collect(range(1.0, 6.0; length=n))
        A = Matrix(Diagonal(values))
        A_op = LinearOperator{Float64}((y, x) -> mul!(y, A, x), (n, n);
                                       issymmetric=true)
        B_op = LinearOperator{Float64}((y, x) -> copyto!(y, x), (n, n);
                                       issymmetric=true, isposdef=true)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 6
        fpm[4] = 1

        feast(A_op, B_op, (1.5, 4.8); M0=M0, fpm=copy(fpm))
        GC.gc()
        bytes = @allocated feast(A_op, B_op, (1.5, 4.8); M0=M0, fpm=copy(fpm))

        @test bytes < 500_000
    end

    @testset "Matrix-free general RCI avoids projection temporaries" begin
        n = 32
        M0 = 10
        values = complex.(collect(range(1.0, 6.0; length=n)),
                          0.02 .* collect(1.0:n))
        A = Matrix(Diagonal(values))
        A_op = LinearOperator{ComplexF64}((y, x) -> mul!(y, A, x), (n, n))
        B_op = LinearOperator{ComplexF64}((y, x) -> copyto!(y, x), (n, n))
        workspace = allocate_matfree_workspace(ComplexF64, n, M0)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 6
        fpm[4] = 1
        center = 3.0 + 0.15im
        radius = 2.4

        exact_solver = function (Y, z, X)
            @inbounds for j in axes(X, 2), i in axes(X, 1)
                Y[i, j] = X[i, j] / (z - A[i, i])
            end
            return Y
        end

        result = FeastKit.feast_matfree_grci!(A_op, B_op, center, radius, M0;
                                              fpm=copy(fpm),
                                              linear_solver=exact_solver,
                                              workspace=workspace)
        @test result.M > 0
        GC.gc()
        bytes = @allocated FeastKit.feast_matfree_grci!(A_op, B_op, center,
                                                        radius, M0;
                                                        fpm=copy(fpm),
                                                        linear_solver=exact_solver,
                                                        workspace=workspace)

        @test bytes < 2_000_000
    end
end
