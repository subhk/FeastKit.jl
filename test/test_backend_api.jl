using Test
using FeastKit
using LinearAlgebra
using SparseArrays

function _backend_test_problem(n)
    A = diagm(0 => 2.0 * ones(n), 1 => -ones(n - 1), -1 => -ones(n - 1))
    B = Matrix{Float64}(I, n, n)
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    fpm[2] = 8
    fpm[4] = 20
    return A, B, fpm
end

function _sorted_lambda(result)
    return sort(result.lambda[1:result.M])
end

@testset "Backend API" begin
    n = 10
    interval = (0.1, 3.9)
    A, B, fpm = _backend_test_problem(n)

    serial = feast(A, B, interval; M0=n, fpm=copy(fpm), backend=:serial)
    legacy_serial = feast(A, B, interval; M0=n, fpm=copy(fpm), parallel=:serial)

    @test serial.info == 0
    @test legacy_serial.info == serial.info
    @test legacy_serial.M == serial.M
    @test _sorted_lambda(legacy_serial) ≈ _sorted_lambda(serial) atol=1e-10

    @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                     backend=:serial, parallel=:threads)
    @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                     backend=:bogus)

    if Threads.nthreads() > 1
        dense_threaded_fallback = feast(A, B, interval; M0=n, fpm=copy(fpm),
                                        backend=:threads)
        @test dense_threaded_fallback.info == serial.info
        @test dense_threaded_fallback.M == serial.M
        @test _sorted_lambda(dense_threaded_fallback) ≈ _sorted_lambda(serial) atol=1e-10

        @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                         backend=:threads, strict_backend=true)

        sparse_serial = feast(sparse(A), sparse(B), interval; M0=n,
                              fpm=copy(fpm), backend=:serial)
        sparse_threaded = feast(sparse(A), sparse(B), interval; M0=n,
                                fpm=copy(fpm), backend=:threads)

        @test sparse_threaded.info == sparse_serial.info
        @test sparse_threaded.M == sparse_serial.M
        @test _sorted_lambda(sparse_threaded) ≈ _sorted_lambda(sparse_serial) atol=1e-8
    end
end
