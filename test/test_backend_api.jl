using Test
using FeastKit
using Distributed
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

    auto = feast(A, B, interval; M0=n, fpm=copy(fpm), backend=:auto)
    @test auto.info == serial.info
    @test auto.M == serial.M
    @test _sorted_lambda(auto) ≈ _sorted_lambda(serial) atol=1e-10

    @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                     backend=:threads)
    @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                     parallel=:threads)
    @test_throws ArgumentError feast_general(ComplexF64.(A), 0.0 + 0.0im, 3.0;
                                             M0=n, fpm=copy(fpm), backend=:threads)

    if !mpi_available()
        @test_throws ArgumentError feast(A, B, interval; M0=n, fpm=copy(fpm),
                                         backend=:mpi)
    end

    if nworkers() == 1
        @test_throws ArgumentError feast(sparse(A), sparse(B), interval; M0=n,
                                         fpm=copy(fpm), backend=:distributed)
    end

    if Threads.nthreads() > 1
        sparse_serial = feast(sparse(A), sparse(B), interval; M0=n,
                              fpm=copy(fpm), backend=:serial)
        sparse_threaded = feast(sparse(A), sparse(B), interval; M0=n,
                                fpm=copy(fpm), backend=:threads)

        @test sparse_threaded.info == sparse_serial.info
        @test sparse_threaded.M == sparse_serial.M
        @test _sorted_lambda(sparse_threaded) ≈ _sorted_lambda(sparse_serial) atol=1e-8
    end
end
