using Test
using FeastKit
using Distributed
using LinearAlgebra
using SparseArrays

function _parallel_backend_problem(n)
    A = sparse(diagm(0 => 2.0 * ones(n), 1 => -ones(n - 1), -1 => -ones(n - 1)))
    B = sparse(Matrix{Float64}(I, n, n))
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    fpm[2] = 8
    fpm[4] = 20
    return A, B, fpm
end

@testset "Distributed backend" begin
    if get(ENV, "FEASTKIT_TEST_DISTRIBUTED", "false") == "true"
        @test nworkers() > 1

        n = 10
        interval = (0.1, 3.9)
        A, B, fpm = _parallel_backend_problem(n)

        serial = feast(A, B, interval; M0=n, fpm=copy(fpm), backend=:serial)
        distributed = feast(A, B, interval; M0=n, fpm=copy(fpm),
                            backend=:distributed, strict_backend=true)
        distributed_alias = pdfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2],
                                            n, copy(fpm); use_threads=false)
        distributed_standard_alias = pdfeast_scsrev!(copy(A), interval[1], interval[2],
                                                     n, copy(fpm); use_threads=false)

        @test distributed.info == serial.info
        @test distributed.M == serial.M
        @test sort(distributed.lambda[1:distributed.M]) ≈ sort(serial.lambda[1:serial.M]) atol=1e-8
        @test distributed_alias.info == serial.info
        @test distributed_alias.M == serial.M
        @test sort(distributed_alias.lambda[1:distributed_alias.M]) ≈ sort(serial.lambda[1:serial.M]) atol=1e-8
        @test distributed_standard_alias.info == serial.info
        @test distributed_standard_alias.M == serial.M
        @test sort(distributed_standard_alias.lambda[1:distributed_standard_alias.M]) ≈ sort(serial.lambda[1:serial.M]) atol=1e-8
    else
        @info "Skipping distributed backend execution test (set FEASTKIT_TEST_DISTRIBUTED=true and add workers)"
    end
end

@testset "MPI backend" begin
    if get(ENV, "FEASTKIT_TEST_MPI", "false") == "true"
        @eval using MPI
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nranks = MPI.Comm_size(comm)
        @test nranks > 1

        n = 10
        interval = (0.1, 3.9)
        A, B, fpm = _parallel_backend_problem(n)

        @test determine_parallel_backend(:mpi, comm) == :mpi

        result = mpi_feast(A, B, interval; M0=n, fpm=copy(fpm), comm=comm)
        highlevel = feast(A, B, interval; M0=n, fpm=copy(fpm),
                          backend=:mpi, strict_backend=true, comm=comm)
        mpi_alias = pdfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2],
                                    n, copy(fpm); comm=comm)
        mpi_standard_alias = pdfeast_scsrev!(copy(A), interval[1], interval[2],
                                             n, copy(fpm); comm=comm)
        expected = eigvals(Matrix(A))
        expected_inside = sort(expected[interval[1] .<= expected .<= interval[2]])

        @test result.info == 0
        @test result.M == length(expected_inside)
        @test sort(result.lambda[1:result.M]) ≈ expected_inside atol=1e-8
        @test highlevel.info == 0
        @test highlevel.M == length(expected_inside)
        @test sort(highlevel.lambda[1:highlevel.M]) ≈ expected_inside atol=1e-8
        @test mpi_alias.info == 0
        @test mpi_alias.M == length(expected_inside)
        @test sort(mpi_alias.lambda[1:mpi_alias.M]) ≈ expected_inside atol=1e-8
        @test mpi_standard_alias.info == 0
        @test mpi_standard_alias.M == length(expected_inside)
        @test sort(mpi_standard_alias.lambda[1:mpi_standard_alias.M]) ≈ expected_inside atol=1e-8

        rank == 0 && println("MPI backend verified on $nranks ranks")
        MPI.Finalize()
    else
        @info "Skipping MPI backend execution test (run under mpiexec with FEASTKIT_TEST_MPI=true)"
    end
end
