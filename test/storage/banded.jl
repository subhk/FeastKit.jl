include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Banded matrix utilities" begin
    # Test banded matrix conversion utilities
    n = 5
    k = 1  # One super-diagonal

    # Create a simple banded matrix in full format
    A_full = diagm(0 => 2*ones(n), 1 => -ones(n-1))

    # Convert to banded format
    A_banded = full_to_banded(A_full, k)
    @test size(A_banded, 1) == k + 1
    @test size(A_banded, 2) == n

    # Convert back to full format
    A_recovered = banded_to_full(A_banded, k, n)
    @test A_recovered ≈ A_full

    # Test banded matrix info
    info = feast_banded_info(A_banded, k, n)
    @test info[1] == n
    @test info[2] == 2*k + 1  # Bandwidth
end

@testset "Banded iterative FEAST" begin
    n = 8
    ka = 1
    kb = 0
    A_full = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
    B_full = Matrix{Float64}(I, n, n)
    A_band = full_to_banded(A_full, ka)
    B_band = full_to_banded(B_full, kb)

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    fpm_gmres = copy(fpm)
    fpm_gmres[3] = 8

    Emax_real = 3.1  # Avoid the exact λ = 3.0 endpoint of this tridiagonal spectrum.
    direct = feast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, Emax_real, n, copy(fpm))
    @test direct.info == 0
    @test direct.M > 0

    gmres_result = feast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, Emax_real, n, copy(fpm_gmres);
                               solver=:gmres, solver_tol=1e-8,
                               solver_maxiter=400, solver_restart=30)
    @test gmres_result.info == 0
    @test gmres_result.M == direct.M
    @test isapprox(sort(gmres_result.lambda[1:gmres_result.M]), sort(direct.lambda[1:direct.M]); atol=1e-8)

    wrapper = difeast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, Emax_real, n, copy(fpm_gmres);
                            solver_tol=1e-8, solver_maxiter=400, solver_restart=30)
    @test wrapper.info == 0
    @test wrapper.M == direct.M
    @test isapprox(sort(wrapper.lambda[1:wrapper.M]), sort(direct.lambda[1:direct.M]); atol=1e-8)

    A_h_full = Matrix(Diagonal(ComplexF64[-1.5, -0.5, 0.2, 0.8, 1.4, 2.2, 3.0, 3.8]))
    B_h_full = Matrix(Diagonal(ComplexF64[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]))
    A_h_band = full_to_banded(A_h_full, ka)
    B_h_band = full_to_banded(B_h_full, ka)

    direct_h = feast_hbgv!(copy(A_h_band), copy(B_h_band), ka, ka, -2.0, 2.0, n, copy(fpm))
    @test direct_h.info == 0
    @test direct_h.M > 0
    fpm_h_gmres = copy(fpm)
    # This platform-sensitive GMRES check is an iterative smoke test, not a 1e-12 solve.
    fpm_h_gmres[3] = 5
    gmres_h = feast_hbgv!(copy(A_h_band), copy(B_h_band), ka, ka, -2.0, 2.0, n, copy(fpm_h_gmres);
                          solver=:gmres, solver_tol=1e-7,
                          solver_maxiter=400, solver_restart=30)
    @test gmres_h.info == 0
    @test isapprox(sort(real.(gmres_h.lambda[1:gmres_h.M])), sort(real.(direct_h.lambda[1:direct_h.M])); atol=1e-5)

    wrapper_h = zifeast_hbgv!(copy(A_h_band), copy(B_h_band), ka, ka, -2.0, 2.0, n, copy(fpm_h_gmres);
                              solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test wrapper_h.info == 0
    @test isapprox(sort(real.(wrapper_h.lambda[1:wrapper_h.M])), sort(real.(direct_h.lambda[1:direct_h.M])); atol=1e-5)

    A_g_full = Matrix(Diagonal(ComplexF64[1.0 + 0.1im, 1.5 - 0.2im, 2.0 + 0.3im,
                                           2.8 - 0.1im, 3.5 + 0.2im, 4.5,
                                           5.2 - 0.1im, 6.0]))
    B_g_full = Matrix(Diagonal(fill(ComplexF64(1), n)))
    A_g_band = full_to_banded(A_g_full, ka)
    B_g_band = full_to_banded(B_g_full, ka)

    Emid = complex(2.5, 0.0)
    r = 4.0
    direct_g = feast_gbgv!(copy(A_g_band), copy(B_g_band), ka, ka, Emid, r, n, copy(fpm))
    @test direct_g.info == 0
    @test direct_g.M > 0
    gmres_g = feast_gbgv!(copy(A_g_band), copy(B_g_band), ka, ka, Emid, r, n, copy(fpm);
                          solver=:gmres, solver_tol=1e-7,
                          solver_maxiter=400, solver_restart=30)
    @test gmres_g.info == 0
    @test isapprox(sort(real.(gmres_g.lambda[1:gmres_g.M])), sort(real.(direct_g.lambda[1:direct_g.M])); atol=1e-6)

    wrapper_g = zifeast_gbgv!(copy(A_g_band), copy(B_g_band), ka, ka, Emid, r, n, copy(fpm);
                              solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test wrapper_g.info == 0
    @test isapprox(sort(real.(wrapper_g.lambda[1:wrapper_g.M])), sort(real.(direct_g.lambda[1:direct_g.M])); atol=1e-6)

    @testset "Iterative banded storage avoids dense conversion" begin
        @test isdefined(FeastKit, :_solve_banded_shifted!)

        n_iter = 220
        fpm_iter_alloc = zeros(Int, 64)
        feastinit!(fpm_iter_alloc)
        fpm_iter_alloc[1] = 0
        fpm_iter_alloc[2] = 4
        fpm_iter_alloc[4] = 6
        fpm_iter_alloc[8] = 4
        iter_storage_tol = 1e-3

        A_real_large = Matrix(Diagonal(collect(range(-2.0, 3.0, length=n_iter))))
        B_real_large = Matrix{Float64}(I, n_iter, n_iter)
        A_real_band = full_to_banded(A_real_large, 0)
        B_real_band = full_to_banded(B_real_large, 0)
        feast_sbgv!(copy(A_real_band), copy(B_real_band), 0, 0,
                    -1.0, 1.0, 8, copy(fpm_iter_alloc);
                    solver=:gmres, solver_tol=iter_storage_tol,
                    solver_maxiter=80, solver_restart=10)
        real_iter_alloc = @allocated feast_sbgv!(copy(A_real_band), copy(B_real_band), 0, 0,
                                                 -1.0, 1.0, 8, copy(fpm_iter_alloc);
                                                 solver=:gmres, solver_tol=iter_storage_tol,
                                                 solver_maxiter=80, solver_restart=10)
        @test real_iter_alloc < 20_000_000

        A_herm_large = Matrix(Diagonal(ComplexF64.(range(-2.0, 3.0, length=n_iter))))
        B_herm_large = Matrix(Diagonal(fill(ComplexF64(1), n_iter)))
        A_herm_band = full_to_banded(A_herm_large, 0)
        B_herm_band = full_to_banded(B_herm_large, 0)
        feast_hbgv!(copy(A_herm_band), copy(B_herm_band), 0, 0,
                    -1.0, 1.0, 8, copy(fpm_iter_alloc);
                    solver=:gmres, solver_tol=iter_storage_tol,
                    solver_maxiter=80, solver_restart=10)
        herm_iter_alloc = @allocated feast_hbgv!(copy(A_herm_band), copy(B_herm_band), 0, 0,
                                                 -1.0, 1.0, 8, copy(fpm_iter_alloc);
                                                 solver=:gmres, solver_tol=iter_storage_tol,
                                                 solver_maxiter=80, solver_restart=10)
        @test herm_iter_alloc < 20_000_000

        A_sym_large = Matrix(Diagonal(ComplexF64.(range(-2.0, 3.0, length=n_iter)) .+ 0.02im))
        B_sym_large = Matrix(Diagonal(fill(ComplexF64(1), n_iter)))
        A_sym_band = full_to_banded(A_sym_large, 0)
        B_sym_band = full_to_banded(B_sym_large, 0)
        feast_sbgv_complex!(copy(A_sym_band), copy(B_sym_band), 0, 0,
                            0.0 + 0.02im, 1.0, 8, copy(fpm_iter_alloc);
                            solver=:gmres, solver_tol=iter_storage_tol,
                            solver_maxiter=80, solver_restart=10)
        sym_iter_alloc = @allocated feast_sbgv_complex!(copy(A_sym_band), copy(B_sym_band), 0, 0,
                                                        0.0 + 0.02im, 1.0, 8, copy(fpm_iter_alloc);
                                                        solver=:gmres, solver_tol=iter_storage_tol,
                                                        solver_maxiter=80, solver_restart=10)
        @test sym_iter_alloc < 20_000_000

        A_general_large = Matrix(Diagonal(ComplexF64.(range(-2.0, 3.0, length=n_iter)) .+ 0.03im))
        B_general_large = Matrix(Diagonal(fill(1.1 + 0.02im, n_iter)))
        A_general_band = full_to_general_banded(A_general_large, 0)
        B_general_band = full_to_general_banded(B_general_large, 0)
        feast_gbgv!(copy(A_general_band), copy(B_general_band), 0, 0,
                    0.0 + 0.03im, 1.0, 8, copy(fpm_iter_alloc);
                    solver=:gmres, solver_tol=iter_storage_tol,
                    solver_maxiter=80, solver_restart=10)
        general_iter_alloc = @allocated feast_gbgv!(copy(A_general_band), copy(B_general_band), 0, 0,
                                                    0.0 + 0.03im, 1.0, 8, copy(fpm_iter_alloc);
                                                    solver=:gmres, solver_tol=iter_storage_tol,
                                                    solver_maxiter=80, solver_restart=10)
        @test general_iter_alloc < 20_000_000
    end

    @testset "General banded direct storage avoids dense conversion" begin
        @test isdefined(FeastKit, :_feast_banded_general)
        @test isdefined(FeastKit, :full_to_general_banded)

        A_general_full = ComplexF64[
            0.5 + 0.2im   0.3 - 0.1im   0.0 + 0.0im  0.0 + 0.0im  0.0 + 0.0im
           -0.2 + 0.4im   1.0 + 0.1im   0.4 + 0.2im  0.0 + 0.0im  0.0 + 0.0im
            0.0 + 0.0im   0.1 - 0.3im   1.6 - 0.2im  0.2 + 0.5im  0.0 + 0.0im
            0.0 + 0.0im   0.0 + 0.0im  -0.3 + 0.2im  2.2 + 0.3im  0.5 - 0.4im
            0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im  0.2 + 0.1im  2.9 - 0.2im
        ]
        B_general_full = ComplexF64[
            1.3 + 0.0im   0.1 + 0.1im   0.0 + 0.0im  0.0 + 0.0im  0.0 + 0.0im
            0.0 + 0.2im   1.2 + 0.1im   0.1 - 0.1im  0.0 + 0.0im  0.0 + 0.0im
            0.0 + 0.0im  -0.1 + 0.0im   1.1 - 0.1im  0.0 + 0.2im  0.0 + 0.0im
            0.0 + 0.0im   0.0 + 0.0im   0.2 - 0.1im  1.4 + 0.0im  0.1 + 0.0im
            0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im -0.1 + 0.1im  1.5 + 0.1im
        ]
        A_general_band = full_to_general_banded(A_general_full, 1)
        B_general_band = full_to_general_banded(B_general_full, 1)
        Emid_general = 1.35 + 0.05im
        r_general = 1.6
        sort_complex(vals) = sort(collect(vals), by=x -> (round(real(x), digits=10),
                                                          round(imag(x), digits=10)))

        expected_general = eigvals(A_general_full, B_general_full)
        expected_general = expected_general[[feast_inside_gcontour(λ, Emid_general, r_general; fpm=fpm) for λ in expected_general]]
        result_general = feast_gbgv!(copy(A_general_band), copy(B_general_band),
                                     1, 1, Emid_general, r_general, 5, copy(fpm))
        @test result_general.info == 0
        @test result_general.M == length(expected_general)
        @test isapprox(sort_complex(result_general.lambda), sort_complex(expected_general); atol=1e-7)

        expected_standard_general = eigvals(A_general_full)
        expected_standard_general = expected_standard_general[[feast_inside_gcontour(λ, Emid_general, r_general; fpm=fpm) for λ in expected_standard_general]]
        result_standard_general = feast_gbev!(copy(A_general_band), 1, Emid_general, r_general, 5, copy(fpm))
        @test result_standard_general.info == 0
        @test result_standard_general.M == length(expected_standard_general)
        @test isapprox(sort_complex(result_standard_general.lambda), sort_complex(expected_standard_general); atol=1e-7)

        n_large = 260
        diag_vals = ComplexF64.(range(-2.0, 3.0, length=n_large)) .+ 0.05im
        A_large = Matrix(Diagonal(diag_vals))
        A_large += diagm(1 => fill(0.15 - 0.05im, n_large - 1),
                         -1 => fill(-0.08 + 0.03im, n_large - 1))
        B_large = Matrix(Diagonal(fill(1.2 + 0.1im, n_large)))
        B_large += diagm(1 => fill(0.02 + 0.01im, n_large - 1),
                         -1 => fill(-0.01 + 0.02im, n_large - 1))
        A_large_band = full_to_general_banded(A_large, 1)
        B_large_band = full_to_general_banded(B_large, 1)
        fpm_large = zeros(Int, 64)
        feastinit!(fpm_large)
        fpm_large[1] = 0
        fpm_large[8] = 6
        fpm_large[4] = 8

        feast_gbgv!(copy(A_large_band), copy(B_large_band),
                    1, 1, 0.0 + 0.0im, 1.0, 8, copy(fpm_large))
        direct_alloc = @allocated feast_gbgv!(copy(A_large_band), copy(B_large_band),
                                              1, 1, 0.0 + 0.0im, 1.0, 8, copy(fpm_large))
        @test direct_alloc < 30_000_000
    end

    @testset "Complex-symmetric banded direct storage" begin
        @test isdefined(FeastKit, :_feast_banded_complex_symmetric)
        @test isdefined(FeastKit, :feast_sbgv_complex!)
        @test isdefined(FeastKit, :feast_sbev_complex!)

        A_cs_full = ComplexF64[
            0.3 + 0.2im  0.1 + 0.4im  0.0 + 0.0im  0.0 + 0.0im
            0.1 + 0.4im  0.9 - 0.1im  0.0 + 0.2im  0.0 + 0.0im
            0.0 + 0.0im  0.0 + 0.2im  1.4 + 0.3im  0.15 - 0.1im
            0.0 + 0.0im  0.0 + 0.0im  0.15 - 0.1im  2.2 + 0.1im
        ]
        B_cs_full = Matrix(Diagonal(ComplexF64[1.0, 1.1, 1.2, 1.3]))
        A_cs_band = full_to_banded(A_cs_full, 1)
        B_cs_band = full_to_banded(B_cs_full, 0)
        Emid_cs = 1.0 + 0.1im
        r_cs = 1.5
        sort_complex(vals) = sort(collect(vals), by=x -> (round(real(x), digits=10),
                                                          round(imag(x), digits=10)))

        expected_banded = eigvals(A_cs_full, B_cs_full)
        expected_banded = expected_banded[[feast_inside_gcontour(λ, Emid_cs, r_cs; fpm=fpm) for λ in expected_banded]]
        result_banded = feast_sbgv_complex!(copy(A_cs_band), copy(B_cs_band),
                                            1, 0, Emid_cs, r_cs, 4, copy(fpm))
        @test result_banded.info == 0
        @test result_banded.M == length(expected_banded)
        @test isapprox(sort_complex(result_banded.lambda), sort_complex(expected_banded); atol=1e-7)

        expected_standard = eigvals(A_cs_full)
        expected_standard = expected_standard[[feast_inside_gcontour(λ, Emid_cs, r_cs; fpm=fpm) for λ in expected_standard]]
        result_standard = feast_sbev_complex!(copy(A_cs_band), 1, Emid_cs, r_cs, 4, copy(fpm))
        @test result_standard.info == 0
        @test result_standard.M == length(expected_standard)
        @test isapprox(sort_complex(result_standard.lambda), sort_complex(expected_standard); atol=1e-7)
    end

    @testset "Hermitian banded direct storage avoids dense conversion" begin
        n_large = 260
        vals = collect(range(-2.0, 3.0, length=n_large))
        A_large = Matrix(Diagonal(ComplexF64.(vals)))
        B_large = Matrix{ComplexF64}(I, n_large, n_large)
        A_large_band = full_to_banded(A_large, 0)
        B_large_band = full_to_banded(B_large, 0)
        fpm_large = zeros(Int, 64)
        feastinit!(fpm_large)
        fpm_large[1] = 0
        fpm_large[2] = 4
        fpm_large[4] = 8

        feast_hbev!(copy(A_large_band), 0, -1.0, 1.0, 10, copy(fpm_large))

        standard_alloc = @allocated feast_hbev!(copy(A_large_band), 0, -1.0, 1.0, 10, copy(fpm_large))
        generalized_alloc = @allocated feast_hbgv!(copy(A_large_band), copy(B_large_band),
                                                   0, 0, -1.0, 1.0, 10, copy(fpm_large))
        @test standard_alloc < 30_000_000
        @test generalized_alloc < 30_000_000
    end
end
