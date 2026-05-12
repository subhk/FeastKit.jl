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

        n_complex = 4
        complex_interval = (0.4, 1.6)
        A_h = sparse(Diagonal(ComplexF64[0.5, 1.0, 1.5, 3.0]))
        B_h = spdiagm(0 => ones(ComplexF64, n_complex))
        fpm_h = zeros(Int, 64)
        feastinit!(fpm_h)
        fpm_h[1] = 0
        fpm_h[2] = 8
        fpm_h[4] = 12
        fpm_h_iter = copy(fpm_h)
        fpm_h_iter[3] = 8
        expected_h = [0.5, 1.0, 1.5]

        herm_mpi = mpi_feast_hcsrgv!(copy(A_h), copy(B_h), complex_interval[1], complex_interval[2],
                                     n_complex, copy(fpm_h); comm=comm)
        herm_highlevel = feast(copy(A_h), copy(B_h), complex_interval; M0=n_complex,
                               fpm=copy(fpm_h), backend=:mpi, strict_backend=true, comm=comm)
        herm_alias = pzfeast_hcsrgv!(copy(A_h), copy(B_h), complex_interval[1], complex_interval[2],
                                     n_complex, copy(fpm_h); comm=comm)
        herm_iter_alias = pzifeast_hcsrgv!(copy(A_h), copy(B_h), complex_interval[1], complex_interval[2],
                                           length(expected_h), copy(fpm_h_iter); comm=comm,
                                           solver_tol=1e-10, solver_maxiter=100)

        @test herm_mpi.info == 0
        @test herm_mpi.M == length(expected_h)
        @test sort(herm_mpi.lambda[1:herm_mpi.M]) ≈ expected_h atol=1e-8
        @test herm_highlevel.info == 0
        @test herm_highlevel.M == length(expected_h)
        @test sort(herm_highlevel.lambda[1:herm_highlevel.M]) ≈ expected_h atol=1e-8
        @test herm_alias.info == 0
        @test herm_alias.M == length(expected_h)
        @test sort(herm_alias.lambda[1:herm_alias.M]) ≈ expected_h atol=1e-8
        @test herm_iter_alias.info == 0
        @test herm_iter_alias.M == length(expected_h)
        @test sort(herm_iter_alias.lambda[1:herm_iter_alias.M]) ≈ expected_h atol=1e-8

        center = 1.0 + 0.1im
        radius = 1.3
        A_g = sparse(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 + 0.2im, 2.0 - 0.1im, 4.0 + 0.0im]))
        B_g = spdiagm(0 => ones(ComplexF64, n_complex))
        fpm_g = zeros(Int, 64)
        feastinit!(fpm_g)
        fpm_g[1] = 0
        fpm_g[3] = 11
        fpm_g[4] = 12
        fpm_g[8] = 12
        expected_g = ComplexF64[0.5 + 0.1im, 1.0 + 0.2im, 2.0 - 0.1im]
        sort_complex(vals) = sort(collect(vals), by=x -> (round(real(x), digits=10),
                                                          round(imag(x), digits=10)))

        general_mpi = mpi_feast_gcsrgv!(copy(A_g), copy(B_g), center, radius,
                                        n_complex, copy(fpm_g); comm=comm)
        general_highlevel = feast_general(copy(A_g), copy(B_g), center, radius;
                                          M0=n_complex, fpm=copy(fpm_g),
                                          backend=:mpi, strict_backend=true, comm=comm)
        general_alias = pzfeast_gcsrgv!(copy(A_g), copy(B_g), center, radius,
                                        n_complex, copy(fpm_g); comm=comm)
        general_iter_alias = pzifeast_gcsrgv!(copy(A_g), copy(B_g), center, radius,
                                              n_complex, copy(fpm_g); comm=comm,
                                              solver_tol=1e-10, solver_maxiter=100)

        @test general_mpi.info == 0
        @test general_mpi.M == length(expected_g)
        @test isapprox(sort_complex(general_mpi.lambda[1:general_mpi.M]), sort_complex(expected_g); atol=1e-8)
        @test general_highlevel.info == 0
        @test general_highlevel.M == length(expected_g)
        @test isapprox(sort_complex(general_highlevel.lambda[1:general_highlevel.M]), sort_complex(expected_g); atol=1e-8)
        @test general_alias.info == 0
        @test general_alias.M == length(expected_g)
        @test isapprox(sort_complex(general_alias.lambda[1:general_alias.M]), sort_complex(expected_g); atol=1e-8)
        @test general_iter_alias.info == 0
        @test general_iter_alias.M == length(expected_g)
        @test isapprox(sort_complex(general_iter_alias.lambda[1:general_iter_alias.M]), sort_complex(expected_g); atol=1e-8)

        A_hd = Matrix(A_h)
        B_hd = Matrix(B_h)
        dense_herm_mpi = mpi_feast_hegv!(copy(A_hd), copy(B_hd), complex_interval[1], complex_interval[2],
                                         n_complex, copy(fpm_h); comm=comm)
        dense_herm_highlevel = feast(copy(A_hd), copy(B_hd), complex_interval;
                                     M0=n_complex, fpm=copy(fpm_h),
                                     backend=:mpi, strict_backend=true, comm=comm)
        dense_herm_alias = pzfeast_hegv!(copy(A_hd), copy(B_hd), complex_interval[1], complex_interval[2],
                                         n_complex, copy(fpm_h); comm=comm)
        dense_herm_iter_alias = pzifeast_hegv!(copy(A_hd), copy(B_hd),
                                               complex_interval[1], complex_interval[2],
                                               length(expected_h), copy(fpm_h_iter); comm=comm,
                                               solver_tol=1e-10, solver_maxiter=100)
        dense_herm_standard_mpi = mpi_feast_heev!(copy(A_hd), complex_interval[1], complex_interval[2],
                                                  n_complex, copy(fpm_h); comm=comm)
        dense_herm_standard_highlevel = feast(copy(A_hd), complex_interval;
                                              M0=n_complex, fpm=copy(fpm_h),
                                              backend=:mpi, strict_backend=true, comm=comm)
        dense_herm_standard_alias = pzfeast_heev!(copy(A_hd), complex_interval[1], complex_interval[2],
                                                  n_complex, copy(fpm_h); comm=comm)
        dense_herm_standard_iter_alias = pzifeast_heev!(copy(A_hd), complex_interval[1], complex_interval[2],
                                                        length(expected_h), copy(fpm_h_iter); comm=comm,
                                                        solver_tol=1e-10, solver_maxiter=100)

        @test dense_herm_mpi.info == 0
        @test dense_herm_mpi.M == length(expected_h)
        @test sort(dense_herm_mpi.lambda[1:dense_herm_mpi.M]) ≈ expected_h atol=1e-8
        @test dense_herm_highlevel.info == 0
        @test dense_herm_highlevel.M == length(expected_h)
        @test sort(dense_herm_highlevel.lambda[1:dense_herm_highlevel.M]) ≈ expected_h atol=1e-8
        @test dense_herm_alias.info == 0
        @test dense_herm_alias.M == length(expected_h)
        @test sort(dense_herm_alias.lambda[1:dense_herm_alias.M]) ≈ expected_h atol=1e-8
        @test dense_herm_iter_alias.info == 0
        @test dense_herm_iter_alias.M == length(expected_h)
        @test sort(dense_herm_iter_alias.lambda[1:dense_herm_iter_alias.M]) ≈ expected_h atol=1e-8
        @test dense_herm_standard_mpi.info == 0
        @test dense_herm_standard_mpi.M == length(expected_h)
        @test sort(dense_herm_standard_mpi.lambda[1:dense_herm_standard_mpi.M]) ≈ expected_h atol=1e-8
        @test dense_herm_standard_highlevel.info == 0
        @test dense_herm_standard_highlevel.M == length(expected_h)
        @test sort(dense_herm_standard_highlevel.lambda[1:dense_herm_standard_highlevel.M]) ≈ expected_h atol=1e-8
        @test dense_herm_standard_alias.info == 0
        @test dense_herm_standard_alias.M == length(expected_h)
        @test sort(dense_herm_standard_alias.lambda[1:dense_herm_standard_alias.M]) ≈ expected_h atol=1e-8
        @test dense_herm_standard_iter_alias.info == 0
        @test dense_herm_standard_iter_alias.M == length(expected_h)
        @test sort(dense_herm_standard_iter_alias.lambda[1:dense_herm_standard_iter_alias.M]) ≈ expected_h atol=1e-8

        A_gd = Matrix(A_g)
        B_gd = Matrix(B_g)
        dense_general_mpi = mpi_feast_gegv!(copy(A_gd), copy(B_gd), center, radius,
                                            n_complex, copy(fpm_g); comm=comm)
        dense_general_highlevel = feast_general(copy(A_gd), copy(B_gd), center, radius;
                                                M0=n_complex, fpm=copy(fpm_g),
                                                backend=:mpi, strict_backend=true, comm=comm)
        dense_general_alias = pzfeast_gegv!(copy(A_gd), copy(B_gd), center, radius,
                                            n_complex, copy(fpm_g); comm=comm)
        dense_general_iter_alias = pzifeast_gegv!(copy(A_gd), copy(B_gd), center, radius,
                                                  n_complex, copy(fpm_g); comm=comm,
                                                  solver_tol=1e-10, solver_maxiter=100)
        dense_general_standard_mpi = mpi_feast_geev!(copy(A_gd), center, radius,
                                                     n_complex, copy(fpm_g); comm=comm)
        dense_general_standard_highlevel = feast_general(copy(A_gd), center, radius;
                                                         M0=n_complex, fpm=copy(fpm_g),
                                                         backend=:mpi, strict_backend=true, comm=comm)
        dense_general_standard_alias = pzfeast_geev!(copy(A_gd), center, radius,
                                                     n_complex, copy(fpm_g); comm=comm)
        dense_general_standard_iter_alias = pzifeast_geev!(copy(A_gd), center, radius,
                                                           n_complex, copy(fpm_g); comm=comm,
                                                           solver_tol=1e-10, solver_maxiter=100)

        @test dense_general_mpi.info == 0
        @test dense_general_mpi.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_mpi.lambda[1:dense_general_mpi.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_highlevel.info == 0
        @test dense_general_highlevel.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_highlevel.lambda[1:dense_general_highlevel.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_alias.info == 0
        @test dense_general_alias.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_alias.lambda[1:dense_general_alias.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_iter_alias.info == 0
        @test dense_general_iter_alias.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_iter_alias.lambda[1:dense_general_iter_alias.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_standard_mpi.info == 0
        @test dense_general_standard_mpi.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_standard_mpi.lambda[1:dense_general_standard_mpi.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_standard_highlevel.info == 0
        @test dense_general_standard_highlevel.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_standard_highlevel.lambda[1:dense_general_standard_highlevel.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_standard_alias.info == 0
        @test dense_general_standard_alias.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_standard_alias.lambda[1:dense_general_standard_alias.M]), sort_complex(expected_g); atol=1e-8)
        @test dense_general_standard_iter_alias.info == 0
        @test dense_general_standard_iter_alias.M == length(expected_g)
        @test isapprox(sort_complex(dense_general_standard_iter_alias.lambda[1:dense_general_standard_iter_alias.M]), sort_complex(expected_g); atol=1e-8)

        rank == 0 && println("MPI backend verified on $nranks ranks")
        MPI.Finalize()
    else
        @info "Skipping MPI backend execution test (run under mpiexec with FEASTKIT_TEST_MPI=true)"
    end
end
