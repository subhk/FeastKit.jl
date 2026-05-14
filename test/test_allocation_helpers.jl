using Test
using FeastKit
using LinearAlgebra
using SparseArrays

# Repetition helpers isolate allocation measurements from setup allocations and
# keep the test bodies focused on the helper under scrutiny.
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

function _repeat_feast_sort!(lambda, q, res, lambda_src, q_src, res_src, nrepeat)
    for _ in 1:nrepeat
        copyto!(lambda, lambda_src)
        copyto!(q, q_src)
        copyto!(res, res_src)
        FeastKit.feast_sort!(lambda, q, res, length(lambda))
    end
    return lambda, q, res
end

@testset "Allocation helper regressions" begin
    @testset "In-place interval reorder" begin
        # Values inside the interval should be compacted to the front while
        # preserving the matching eigenvector columns.
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

    @testset "In-place real eigenpair sort" begin
        lambda_src = [4.0, 2.0, 1.0, 3.0]
        q_src = [
            11.0 12.0 13.0 14.0
            21.0 22.0 23.0 24.0
            31.0 32.0 33.0 34.0
        ]
        res_src = [0.4, 0.2, 0.1, 0.3]
        lambda = copy(lambda_src)
        q = copy(q_src)
        res = copy(res_src)

        FeastKit.feast_sort!(lambda, q, res, length(lambda))

        @test lambda == [1.0, 2.0, 3.0, 4.0]
        @test q == q_src[:, [3, 2, 4, 1]]
        @test res == [0.1, 0.2, 0.3, 0.4]

        _repeat_feast_sort!(lambda, q, res, lambda_src, q_src, res_src, 1)
        bytes = @allocated _repeat_feast_sort!(lambda, q, res, lambda_src,
                                               q_src, res_src, 1_000)
        @test bytes < 1024
    end

    @testset "QR compression preserves projected subspace rank" begin
        # Rank-deficient projected spaces arise when M0 is larger than the target
        # eigenspace. Compression must keep the represented subspace, not just
        # orthonormalize arbitrary columns.
        src = ComplexF64[
            1.0 + 0.0im 2.0 + 0.0im 0.0 + 0.0im 1.0e-15 + 0.0im
            0.0 + 1.0im 0.0 + 2.0im 1.0 + 0.0im 0.0 + 1.0e-15im
            0.0 + 0.0im 0.0 + 0.0im 0.0 + 1.0im 0.0 + 0.0im
            0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im
        ]
        basis = similar(src)

        rank = FeastKit._feast_qr_compress!(basis, src, 4; rank_tol=sqrt(eps(Float64)))

        Q = basis[:, 1:rank]
        @test rank == 2
        @test Q' * Q ≈ Matrix{ComplexF64}(I, rank, rank) atol=1.0e-12
        @test norm(src - Q * (Q' * src)) <= 1.0e-12
    end

    @testset "Dense Hermitian solver compresses oversized projected subspaces" begin
        n = 80
        A = Matrix(Diagonal(collect(1.0:n)))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 8
        fpm[3] = 7
        fpm[4] = 4

        result = feast_syev!(A, 10.5, 12.5, 32, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda) ≈ [11.0, 12.0] atol=1.0e-8
        @test maximum(result.res) < 1.0e-7
    end

    @testset "Sparse Hermitian solver compresses oversized projected subspaces" begin
        n = 80
        A = sparse(Matrix(Diagonal(collect(1.0:n))))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 8
        fpm[3] = 7
        fpm[4] = 4

        result = feast_scsrev!(A, 10.5, 12.5, 32, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda) ≈ [11.0, 12.0] atol=1.0e-8
        @test maximum(result.res) < 1.0e-7
    end

    @testset "Banded Hermitian solver compresses oversized projected subspaces" begin
        n = 80
        A = reshape(ComplexF64.(collect(1.0:n)), 1, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[2] = 8
        fpm[3] = 7
        fpm[4] = 4

        result = feast_hbev!(copy(A), 0, 10.5, 12.5, 32, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda) ≈ [11.0, 12.0] atol=1.0e-8
        @test maximum(result.res) < 1.0e-7
    end

    @testset "Dense complex-symmetric solver compresses oversized projected subspaces" begin
        n = 80
        values = complex.(collect(1.0:n), 0.02 .* collect(1.0:n))
        A = Matrix(Diagonal(values))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[8] = 8
        fpm[3] = 10
        fpm[4] = 4
        center = 11.5 + 0.23im
        radius = 0.65

        result = feast_geev_complex_sym!(A, center, radius, 64, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda, by=real) ≈ [11.0 + 0.22im, 12.0 + 0.24im] atol=1.0e-8
        @test maximum(result.res) < 1.0e-10
    end

    @testset "Sparse complex-symmetric solver compresses oversized projected subspaces" begin
        n = 80
        values = complex.(collect(1.0:n), 0.02 .* collect(1.0:n))
        A = sparse(Matrix(Diagonal(values)))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[8] = 8
        fpm[3] = 10
        fpm[4] = 4
        center = 11.5 + 0.23im
        radius = 0.65

        result = feast_scsrev_complex!(A, center, radius, 32, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda, by=real) ≈ [11.0 + 0.22im, 12.0 + 0.24im] atol=1.0e-8
        @test maximum(result.res) < 1.0e-10
    end

    @testset "Banded complex-symmetric solver compresses oversized projected subspaces" begin
        n = 80
        values = complex.(collect(1.0:n), 0.02 .* collect(1.0:n))
        A = reshape(values, 1, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[8] = 8
        fpm[3] = 10
        fpm[4] = 4
        center = 11.5 + 0.23im
        radius = 0.65

        result = feast_sbev_complex!(copy(A), 0, center, radius, 48, copy(fpm))

        @test result.info == 0
        @test result.M == 2
        @test sort(result.lambda, by=real) ≈ [11.0 + 0.22im, 12.0 + 0.24im] atol=1.0e-8
        @test maximum(result.res) < 1.0e-10
    end

    @testset "Complex-to-real result conversion avoids column temporaries" begin
        # This regression guards the summary/result conversion code path against
        # accidentally allocating one vector per eigenpair.
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

    @testset "High-level general wrappers avoid redundant dense materialization" begin
        n = 80
        M0 = 16
        eigenvalues = complex.(collect(1.0:n), 0.02 .* collect(1.0:n))
        A = Matrix(Diagonal(eigenvalues))
        B = Matrix(Diagonal(fill(1.1 + 0.02im, n)))
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        fpm[8] = 8
        fpm[4] = 1
        center = 20.0 + 0.4im
        radius = 12.0

        feast_general(A, center, radius; M0=M0, fpm=copy(fpm), backend=:serial)
        feast_geev!(A, center, radius, M0, copy(fpm))
        standard_high_level_bytes = @allocated feast_general(A, center, radius;
                                                             M0=M0,
                                                             fpm=copy(fpm),
                                                             backend=:serial)
        standard_low_level_bytes = @allocated feast_geev!(A, center, radius,
                                                          M0, copy(fpm))
        identity_bytes = @allocated Matrix{ComplexF64}(I, n, n)
        @test standard_high_level_bytes <= standard_low_level_bytes +
                                           max(16_384, identity_bytes)

        feast_general(A, B, center, radius; M0=M0, fpm=copy(fpm),
                      backend=:serial)
        feast_gegv!(A, B, center, radius, M0, copy(fpm))
        generalized_high_level_bytes = @allocated feast_general(A, B, center,
                                                                radius;
                                                                M0=M0,
                                                                fpm=copy(fpm),
                                                                backend=:serial)
        generalized_low_level_bytes = @allocated feast_gegv!(A, B, center,
                                                             radius, M0,
                                                             copy(fpm))
        dense_copy_bytes = @allocated copy(A)
        @test generalized_high_level_bytes <= generalized_low_level_bytes +
                                               max(16_384, dense_copy_bytes)

        A_real = Matrix(Symmetric(Diagonal(collect(range(1.0, 4.0; length=n)))))
        B_real = Matrix{Float64}(I, n, n)
        real_fpm = zeros(Int, 64)
        feastinit!(real_fpm)
        real_fpm[1] = 0
        real_fpm[2] = 4
        real_fpm[4] = 1
        interval = (1.2, 3.8)

        FeastKit.feast_serial(A_real, B_real, interval, M0, copy(real_fpm))
        feast_sygv!(A_real, B_real, interval[1], interval[2], M0,
                    copy(real_fpm))
        serial_bytes = @allocated FeastKit.feast_serial(A_real, B_real,
                                                        interval, M0,
                                                        copy(real_fpm))
        direct_bytes = @allocated feast_sygv!(A_real, B_real, interval[1],
                                              interval[2], M0,
                                              copy(real_fpm))
        real_copy_bytes = @allocated copy(A_real)
        @test serial_bytes <= direct_bytes + max(16_384, real_copy_bytes)
    end

    @testset "Direct solvers reuse contour factorizations across refinement loops" begin
        n = 120
        M0 = 30
        dense_A = Matrix(Symmetric(Diagonal(collect(range(1.0, 4.0; length=n)))))
        sparse_A = sparse(dense_A)

        fpm_one = zeros(Int, 64)
        feastinit!(fpm_one)
        fpm_one[1] = 0
        fpm_one[2] = 8
        fpm_one[3] = 16
        fpm_one[4] = 1

        fpm_three = copy(fpm_one)
        fpm_three[4] = 3

        feast_syev!(dense_A, 1.2, 3.8, M0, copy(fpm_one))
        feast_syev!(dense_A, 1.2, 3.8, M0, copy(fpm_three))
        dense_one_loop_bytes = @allocated feast_syev!(dense_A, 1.2, 3.8,
                                                       M0, copy(fpm_one))
        dense_three_loop_bytes = @allocated feast_syev!(dense_A, 1.2, 3.8,
                                                         M0, copy(fpm_three))
        @test dense_three_loop_bytes <= round(Int, 1.45 * dense_one_loop_bytes)

        feast_scsrev!(sparse_A, 1.2, 3.8, M0, copy(fpm_one))
        feast_scsrev!(sparse_A, 1.2, 3.8, M0, copy(fpm_three))
        sparse_one_loop_bytes = @allocated feast_scsrev!(sparse_A, 1.2, 3.8,
                                                         M0, copy(fpm_one))
        sparse_three_loop_bytes = @allocated feast_scsrev!(sparse_A, 1.2, 3.8,
                                                           M0, copy(fpm_three))
        @test sparse_three_loop_bytes <= round(Int, 1.45 * sparse_one_loop_bytes)

        banded_n = 200
        banded_M0 = 30
        banded_A = zeros(ComplexF64, 3, banded_n)
        banded_A[1, :] .= collect(range(1.0, 4.0; length=banded_n))
        for j in 1:(banded_n - 1)
            banded_A[2, j] = 0.005 + 0.001im
        end
        for j in 1:(banded_n - 2)
            banded_A[3, j] = 0.002 + 0.0005im
        end

        feast_hbev!(banded_A, 2, 1.2, 3.8, banded_M0, copy(fpm_one))
        feast_hbev!(banded_A, 2, 1.2, 3.8, banded_M0, copy(fpm_three))
        banded_one_loop_bytes = @allocated feast_hbev!(banded_A, 2, 1.2, 3.8,
                                                        banded_M0,
                                                        copy(fpm_one))
        banded_three_loop_bytes = @allocated feast_hbev!(banded_A, 2, 1.2, 3.8,
                                                          banded_M0,
                                                          copy(fpm_three))
        @test banded_three_loop_bytes <= round(Int, 1.45 * banded_one_loop_bytes)

        real_banded_n = 100
        real_banded_M0 = 24
        real_banded_k = 2
        real_banded_A = zeros(Float64, real_banded_k + 1, real_banded_n)
        real_banded_A[real_banded_k + 1, :] .=
            collect(range(1.0, 4.0; length=real_banded_n))
        for j in 1:(real_banded_n - 1)
            real_banded_A[real_banded_k, j + 1] = 0.005
        end
        for j in 1:(real_banded_n - 2)
            real_banded_A[real_banded_k - 1, j + 2] = 0.002
        end
        real_banded_B = zeros(Float64, 1, real_banded_n)
        real_banded_B[1, :] .= 1.0
        real_banded_fpm_five = copy(fpm_one)
        real_banded_fpm_five[4] = 5

        feast_sbgv!(real_banded_A, real_banded_B, real_banded_k, 0, 1.2,
                    3.8, real_banded_M0, copy(fpm_one))
        feast_sbgv!(real_banded_A, real_banded_B, real_banded_k, 0, 1.2,
                    3.8, real_banded_M0, copy(real_banded_fpm_five))
        real_banded_one_loop_bytes =
            @allocated feast_sbgv!(real_banded_A, real_banded_B,
                                   real_banded_k, 0, 1.2, 3.8,
                                   real_banded_M0, copy(fpm_one))
        real_banded_five_loop_bytes =
            @allocated feast_sbgv!(real_banded_A, real_banded_B,
                                   real_banded_k, 0, 1.2, 3.8,
                                   real_banded_M0,
                                   copy(real_banded_fpm_five))
        @test real_banded_five_loop_bytes <=
              round(Int, 1.44 * real_banded_one_loop_bytes)

        general_values = complex.(collect(range(1.0, 4.0; length=n)),
                                  0.01 .* collect(1:n))
        dense_general_A = Matrix(Diagonal(general_values))
        sparse_general_A = sparse(dense_general_A)
        general_center = 2.5 + 0.4im
        general_radius = 2.0
        general_fpm_one = zeros(Int, 64)
        feastinit!(general_fpm_one)
        general_fpm_one[1] = 0
        general_fpm_one[8] = 8
        general_fpm_one[3] = 16
        general_fpm_one[4] = 1
        general_fpm_three = copy(general_fpm_one)
        general_fpm_three[4] = 3
        general_fpm_five = copy(general_fpm_one)
        general_fpm_five[4] = 5

        feast_geev!(dense_general_A, general_center, general_radius, M0,
                    copy(general_fpm_one))
        feast_geev!(dense_general_A, general_center, general_radius, M0,
                    copy(general_fpm_three))
        dense_general_one_loop_bytes = @allocated feast_geev!(dense_general_A,
                                                              general_center,
                                                              general_radius,
                                                              M0,
                                                              copy(general_fpm_one))
        dense_general_three_loop_bytes = @allocated feast_geev!(dense_general_A,
                                                                general_center,
                                                                general_radius,
                                                                M0,
                                                                copy(general_fpm_three))
        @test dense_general_three_loop_bytes <=
              round(Int, 1.25 * dense_general_one_loop_bytes)

        feast_gcsrev!(sparse_general_A, general_center, general_radius, M0,
                      copy(general_fpm_one))
        feast_gcsrev!(sparse_general_A, general_center, general_radius, M0,
                      copy(general_fpm_three))
        sparse_general_one_loop_bytes = @allocated feast_gcsrev!(sparse_general_A,
                                                                 general_center,
                                                                 general_radius,
                                                                 M0,
                                                                 copy(general_fpm_one))
        sparse_general_three_loop_bytes = @allocated feast_gcsrev!(sparse_general_A,
                                                                   general_center,
                                                                   general_radius,
                                                                   M0,
                                                                   copy(general_fpm_three))
        @test sparse_general_three_loop_bytes <=
              round(Int, 1.45 * sparse_general_one_loop_bytes)

        banded_general_n = 100
        banded_general_M0 = 24
        banded_general_k = 2
        banded_general_A = zeros(ComplexF64, 2 * banded_general_k + 1,
                                 banded_general_n)
        banded_general_A[banded_general_k + 1, :] .=
            complex.(collect(range(1.0, 4.0; length=banded_general_n)),
                     0.01 .* collect(1:banded_general_n))
        for j in 1:(banded_general_n - 1)
            banded_general_A[banded_general_k, j + 1] = 0.004 - 0.001im
            banded_general_A[banded_general_k + 2, j] = 0.006 + 0.002im
        end
        for j in 1:(banded_general_n - 2)
            banded_general_A[banded_general_k - 1, j + 2] = 0.001 + 0.0005im
            banded_general_A[banded_general_k + 3, j] = 0.002 - 0.0005im
        end

        feast_gbev!(banded_general_A, banded_general_k, general_center,
                    general_radius, banded_general_M0, copy(general_fpm_one))
        feast_gbev!(banded_general_A, banded_general_k, general_center,
                    general_radius, banded_general_M0, copy(general_fpm_five))
        banded_general_one_loop_bytes =
            @allocated feast_gbev!(banded_general_A, banded_general_k,
                                   general_center, general_radius,
                                   banded_general_M0, copy(general_fpm_one))
        banded_general_five_loop_bytes =
            @allocated feast_gbev!(banded_general_A, banded_general_k,
                                   general_center, general_radius,
                                   banded_general_M0, copy(general_fpm_five))
        @test banded_general_five_loop_bytes <=
              round(Int, 1.65 * banded_general_one_loop_bytes)

        complex_sym_n = 100
        complex_sym_M0 = 24
        complex_sym_values = complex.(collect(range(1.0, 4.0; length=complex_sym_n)),
                                      0.02 .* collect(1:complex_sym_n))
        dense_complex_sym_A = Matrix(Diagonal(complex_sym_values))
        sparse_complex_sym_A = sparse(dense_complex_sym_A)
        complex_sym_center = 2.5 + 0.8im
        complex_sym_radius = 2.0
        complex_sym_fpm_one = zeros(Int, 64)
        feastinit!(complex_sym_fpm_one)
        complex_sym_fpm_one[1] = 0
        complex_sym_fpm_one[8] = 8
        complex_sym_fpm_one[3] = 16
        complex_sym_fpm_one[4] = 1
        complex_sym_fpm_three = copy(complex_sym_fpm_one)
        complex_sym_fpm_three[4] = 3

        feast_geev_complex_sym!(dense_complex_sym_A, complex_sym_center,
                                complex_sym_radius, complex_sym_M0,
                                copy(complex_sym_fpm_one))
        feast_geev_complex_sym!(dense_complex_sym_A, complex_sym_center,
                                complex_sym_radius, complex_sym_M0,
                                copy(complex_sym_fpm_three))
        dense_complex_sym_one_loop_bytes =
            @allocated feast_geev_complex_sym!(dense_complex_sym_A,
                                               complex_sym_center,
                                               complex_sym_radius,
                                               complex_sym_M0,
                                               copy(complex_sym_fpm_one))
        dense_complex_sym_three_loop_bytes =
            @allocated feast_geev_complex_sym!(dense_complex_sym_A,
                                               complex_sym_center,
                                               complex_sym_radius,
                                               complex_sym_M0,
                                               copy(complex_sym_fpm_three))
        @test dense_complex_sym_three_loop_bytes <=
              round(Int, 1.35 * dense_complex_sym_one_loop_bytes)

        feast_scsrev_complex!(sparse_complex_sym_A, complex_sym_center,
                              complex_sym_radius, complex_sym_M0,
                              copy(complex_sym_fpm_one))
        feast_scsrev_complex!(sparse_complex_sym_A, complex_sym_center,
                              complex_sym_radius, complex_sym_M0,
                              copy(complex_sym_fpm_three))
        sparse_complex_sym_one_loop_bytes =
            @allocated feast_scsrev_complex!(sparse_complex_sym_A,
                                             complex_sym_center,
                                             complex_sym_radius,
                                             complex_sym_M0,
                                             copy(complex_sym_fpm_one))
        sparse_complex_sym_three_loop_bytes =
            @allocated feast_scsrev_complex!(sparse_complex_sym_A,
                                             complex_sym_center,
                                             complex_sym_radius,
                                             complex_sym_M0,
                                             copy(complex_sym_fpm_three))
        @test sparse_complex_sym_three_loop_bytes <=
              round(Int, 1.45 * sparse_complex_sym_one_loop_bytes)

        banded_complex_sym_n = 200
        banded_complex_sym_M0 = 30
        banded_complex_sym_values =
            complex.(collect(range(1.0, 4.0; length=banded_complex_sym_n)),
                     0.02 .* collect(1:banded_complex_sym_n))
        banded_complex_sym_A = zeros(ComplexF64, 3, banded_complex_sym_n)
        banded_complex_sym_A[1, :] .= banded_complex_sym_values
        for j in 1:(banded_complex_sym_n - 1)
            banded_complex_sym_A[2, j] = 0.005 + 0.001im
        end
        for j in 1:(banded_complex_sym_n - 2)
            banded_complex_sym_A[3, j] = 0.002 + 0.0005im
        end

        feast_sbev_complex!(banded_complex_sym_A, 2, complex_sym_center,
                            complex_sym_radius, banded_complex_sym_M0,
                            copy(complex_sym_fpm_one))
        feast_sbev_complex!(banded_complex_sym_A, 2, complex_sym_center,
                            complex_sym_radius, banded_complex_sym_M0,
                            copy(complex_sym_fpm_three))
        banded_complex_sym_one_loop_bytes =
            @allocated feast_sbev_complex!(banded_complex_sym_A, 2,
                                           complex_sym_center,
                                           complex_sym_radius,
                                           banded_complex_sym_M0,
                                           copy(complex_sym_fpm_one))
        banded_complex_sym_three_loop_bytes =
            @allocated feast_sbev_complex!(banded_complex_sym_A, 2,
                                           complex_sym_center,
                                           complex_sym_radius,
                                           banded_complex_sym_M0,
                                           copy(complex_sym_fpm_three))
        @test banded_complex_sym_three_loop_bytes <=
              round(Int, 1.45 * banded_complex_sym_one_loop_bytes)
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
