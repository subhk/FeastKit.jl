include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Sparse matrix support" begin
    # Test sparse matrix creation and info
    n = 10
    A_sparse = spdiagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))

    info = feast_sparse_info(A_sparse)
    @test info[1] == n  # Size
    @test info[2] > 0   # Non-zeros
    @test info[3] > 0   # Density

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    result = feast_scsrev!(copy(A_sparse), 0.0, 4.0, n, fpm)
    @test result.info == 0
    @test result.M == n
end

@testset "Sparse Hermitian generalized" begin
    n = 6
    A = spdiagm(0 => ComplexF64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    B = spdiagm(0 => ComplexF64[1.0, 1.2, 1.5, 2.5, 4.0, 5.0])

    dense_vals = sort(real.(eigvals(Matrix(A), Matrix(B))))
    Emin = 0.5
    Emax = 3.1
    expected = [λ for λ in dense_vals if Emin <= λ <= Emax]

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    result = feast_hcsrgv!(copy(A), copy(B), Emin, Emax, n, copy(fpm))

    @test result.info == 0
    @test result.M == length(expected)
    @test isapprox(sort(result.lambda), sort(expected); atol=1e-8)

    contour = feast_contour(Emin, Emax, copy(fpm))
    fpm_custom = copy(fpm)
    result_x = feast_hcsrgvx!(copy(A), copy(B), Emin, Emax, n, fpm_custom,
                              contour.Zne, contour.Wne)
    @test result_x.info == 0
    @test isapprox(sort(result_x.lambda), sort(result.lambda); atol=1e-8)
end

@testset "Sparse complex iterative" begin
    n = 6
    main_diag = ComplexF64[1.0, 1.5, 2.0, 2.8, 3.5, 4.2]
    A = spdiagm(0 => main_diag)

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    Emin, Emax = 0.5, 4.0

    direct = feast_hcsrev!(copy(A), Emin, Emax, n, copy(fpm))
    gmres = feast_hcsrev!(copy(A), Emin, Emax, n, copy(fpm);
                          solver=:gmres, solver_tol=1e-8,
                          solver_maxiter=400, solver_restart=30)
    @test gmres.info == 0
    @test gmres.M == direct.M
    @test isapprox(sort(gmres.lambda), sort(direct.lambda); atol=1e-8)

    wrapper = zifeast_hcsrev!(copy(A), Emin, Emax, n, copy(fpm);
                              solver_tol=1e-8, solver_maxiter=400, solver_restart=30)
    @test wrapper.info == 0
    @test wrapper.M == direct.M
    @test isapprox(sort(wrapper.lambda), sort(direct.lambda); atol=1e-8)

    Bdiag = ComplexF64[1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    B = spdiagm(0 => Bdiag)
    direct_g = feast_hcsrgv!(copy(A), copy(B), Emin, Emax, n, copy(fpm))
    gmres_g = feast_hcsrgv!(copy(A), copy(B), Emin, Emax, n, copy(fpm);
                            solver=:gmres, solver_tol=1e-8,
                            solver_maxiter=400, solver_restart=30)
    @test gmres_g.info == 0
    @test gmres_g.M == direct_g.M
    @test isapprox(sort(gmres_g.lambda), sort(direct_g.lambda); atol=1e-8)

    wrapper_g = zifeast_hcsrgv!(copy(A), copy(B), Emin, Emax, n, copy(fpm);
                                solver_tol=1e-8, solver_maxiter=400, solver_restart=30)
    @test wrapper_g.info == 0
    @test wrapper_g.M == direct_g.M
    @test isapprox(sort(wrapper_g.lambda), sort(direct_g.lambda); atol=1e-8)

    A_general = spdiagm(0 => ComplexF64[1.0 + 0.1im, 1.5 - 0.2im, 2.0 + 0.3im,
                                        2.8 - 0.1im, 3.5 + 0.2im, 4.5])
    B_general = spdiagm(0 => fill(ComplexF64(1), n))
    Emid = complex(2.0, 0.0)
    radius = 3.0
    direct_gen = feast_gcsrgv!(copy(A_general), copy(B_general), Emid, radius, n, copy(fpm))
    gmres_gen = feast_gcsrgv!(copy(A_general), copy(B_general), Emid, radius, n, copy(fpm);
                              solver=:gmres, solver_tol=1e-7,
                              solver_maxiter=400, solver_restart=30)
    @test gmres_gen.info == 0
    @test isapprox(sort(real.(gmres_gen.lambda)), sort(real.(direct_gen.lambda)); atol=1e-7)
    wrapper_gen = zifeast_gcsrgv!(copy(A_general), copy(B_general), Emid, radius, n, copy(fpm);
                                  solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test isapprox(sort(real.(wrapper_gen.lambda)), sort(real.(direct_gen.lambda)); atol=1e-7)

    direct_std = feast_gcsrev!(copy(A_general), Emid, radius, n, copy(fpm))
    gmres_std = feast_gcsrev!(copy(A_general), Emid, radius, n, copy(fpm);
                              solver=:gmres, solver_tol=1e-7,
                              solver_maxiter=400, solver_restart=30)
    @test gmres_std.info == 0
    @test gmres_std.M == direct_std.M
    @test isapprox(sort(real.(gmres_std.lambda)), sort(real.(direct_std.lambda)); atol=1e-6)
    wrapper_std = zifeast_gcsrev!(copy(A_general), Emid, radius, n, copy(fpm);
                                  solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test wrapper_std.info == 0
    @test wrapper_std.M == direct_std.M
    @test isapprox(sort(real.(wrapper_std.lambda)), sort(real.(direct_std.lambda)); atol=1e-6)

    # Complex-symmetric wrappers use a transpose-bilinear Ritz projection.
    @test isdefined(FeastKit, :_feast_sparse_complex_symmetric)
    A_sym = ComplexF64[
        0.3 + 0.2im  0.1 + 0.4im  0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im
        0.1 + 0.4im  0.9 - 0.1im  0.0 + 0.2im   0.0 + 0.0im   0.0 + 0.0im   0.0 + 0.0im
        0.0 + 0.0im  0.0 + 0.2im  1.4 + 0.3im   0.15 - 0.1im  0.0 + 0.0im   0.0 + 0.0im
        0.0 + 0.0im  0.0 + 0.0im  0.15 - 0.1im  2.2 + 0.1im   0.2 + 0.0im   0.0 + 0.0im
        0.0 + 0.0im  0.0 + 0.0im  0.0 + 0.0im   0.2 + 0.0im   3.0 - 0.2im   0.1 + 0.1im
        0.0 + 0.0im  0.0 + 0.0im  0.0 + 0.0im   0.0 + 0.0im   0.1 + 0.1im   4.2 + 0.1im
    ]
    B_sym = spdiagm(0 => ComplexF64[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    Emid_cs = complex(1.2, 0.1)
    r_cs = 1.7
    expected_cs = eigvals(A_sym, Matrix(B_sym))
    expected_cs = expected_cs[[feast_inside_gcontour(λ, Emid_cs, r_cs; fpm=fpm) for λ in expected_cs]]
    sort_complex(vals) = sort(collect(vals), by=x -> (round(real(x), digits=10),
                                                      round(imag(x), digits=10)))
    cs_wrap = feast_scsrgv_complex!(sparse(A_sym), sparse(B_sym), Emid_cs, r_cs, n, copy(fpm))
    @test cs_wrap.info == 0
    @test cs_wrap.M == length(expected_cs)
    @test isapprox(sort_complex(cs_wrap.lambda), sort_complex(expected_cs); atol=1e-7)

    cs_iter = zifeast_scsrgv_complex!(sparse(A_sym), sparse(B_sym), Emid_cs, r_cs, n, copy(fpm);
                                      solver_tol=1e-8, solver_maxiter=400, solver_restart=30)
    @test cs_iter.info == 0
    @test cs_iter.M == length(expected_cs)
    @test isapprox(sort_complex(cs_iter.lambda), sort_complex(expected_cs); atol=1e-7)

    cs_std = feast_scsrev_complex!(sparse(A_sym), Emid_cs, r_cs, n, copy(fpm))
    expected_std = eigvals(A_sym)
    expected_std = expected_std[[feast_inside_gcontour(λ, Emid_cs, r_cs; fpm=fpm) for λ in expected_std]]
    @test cs_std.info == 0
    @test cs_std.M == length(expected_std)
    @test isapprox(sort_complex(cs_std.lambda), sort_complex(expected_std); atol=1e-7)

    A_non_symmetric = sparse([1.0 + 0im 2.0 + 1im; 0.0 + 0im 3.0 + 0im])
    B_non_symmetric = sparse(Matrix{ComplexF64}(I, 2, 2))
    @test_throws ArgumentError feast_scsrgv_complex!(A_non_symmetric, B_non_symmetric,
                                                     Emid_cs, r_cs, 2, copy(fpm))
end

@testset "Sparse iterative FEAST (GMRES)" begin
    n = 12
    A = spdiagm(-1 => -ones(n-1), 0 => 2.0 .* ones(n), 1 => -ones(n-1))
    B = spdiagm(0 => ones(n))
    Emin, Emax = 0.1, 3.9
    M0 = n

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0

    direct = feast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, copy(fpm))
    gmres_result = feast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, copy(fpm);
                                 solver=:gmres, solver_tol=1e-6,
                                 solver_maxiter=400, solver_restart=20)
    @test gmres_result.info == 0
    @test gmres_result.M == direct.M
    @test isapprox(sort(gmres_result.lambda), sort(direct.lambda); atol=1e-6)

    fpm_iter = zeros(Int, 64)
    feastinit!(fpm_iter)
    fpm_iter[1] = 0
    iter_result = difeast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, fpm_iter;
                                  solver_tol=1e-6, solver_maxiter=400,
                                  solver_restart=20)
    @test iter_result.info == 0
    @test iter_result.M == direct.M
    @test isapprox(sort(iter_result.lambda), sort(direct.lambda); atol=1e-5)
end
