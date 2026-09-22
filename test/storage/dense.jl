include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Simple eigenvalue problems" begin
    # Test with small matrix that has known eigenvalues
    n = 4

    # Create a simple tridiagonal matrix
    A = diagm(0 => [2.0, 2.0, 2.0, 2.0],
             1 => [-1.0, -1.0, -1.0],
            -1 => [-1.0, -1.0, -1.0])

    # Eigenvalues should be approximately [0.17, 1.0, 2.0, 3.83]
    # Let's search for eigenvalues in [0.5, 2.5] (should find λ ≈ 1.0, 2.0)

    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0  # No output for testing

    if get(ENV, "FEAST_RUN_LONG_TESTS", "false") == "true"
        # The high-level driver has to produce the right answer, not merely
        # avoid crashing. Swallowing every exception here is what let the
        # entry point go unexercised.
        reference = eigvals(Symmetric(A))
        inside = filter(lambda -> 0.5 <= lambda <= 2.5, reference)
        result = feast(A, (0.5, 2.5), M0=4, fpm=fpm, parallel=:serial)
        @test result.info == 0
        @test result.M == length(inside)
        @test isapprox(sort(result.lambda[1:result.M]), sort(inside); atol=1e-8)
        for j in 1:result.M
            q = result.q[:, j]
            @test isapprox(norm(q), 1.0; atol=1e-8)
            @test norm(A * q - result.lambda[j] * q) / norm(q) < 1e-8
        end
    else
        @info "Skipping high-level feast() smoke run (set FEAST_RUN_LONG_TESTS=true to enable)"
    end
end

@testset "General eigenvalue problems" begin
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0

    # Dense standard problem (B = I)
    @info "General: dense standard"
    A_dense = ComplexF64[1  2+1im;
                         0  3]
    center = 2 + 0im
    radius = 2.5
    result_standard = feast_general(A_dense, center, radius; M0=size(A_dense, 1), fpm=copy(fpm), parallel=:serial)
    @test result_standard.info == 0
    @test result_standard.M == 2
    expected_dense = sort(real.(eigvals(Matrix(A_dense))))
    @test isapprox(sort(real.(result_standard.lambda)), expected_dense; atol=1e-9)

    # Dense generalized problem with diagonal B
    @info "General: dense generalized"
    B_dense = ComplexF64[1 0;
                         0 2]
    result_general = feast_general(A_dense, B_dense, center, radius; M0=size(A_dense, 1), fpm=copy(fpm), parallel=:serial)
    @test result_general.info == 0
    @test result_general.M == 2
    expected_general = sort(real.(eigvals(Matrix(A_dense), Matrix(B_dense))))
    @test isapprox(sort(real.(result_general.lambda)), expected_general; atol=1e-9)

    # Sparse standard problem (automatic type promotion)
    @info "General: sparse standard"
    A_sparse = sparse(A_dense)
    result_sparse = feast_general(A_sparse, center, radius; M0=size(A_sparse, 1), fpm=copy(fpm), parallel=:serial)
    @test result_sparse.info == 0
    @test result_sparse.M == 2
    @test isapprox(sort(real.(result_sparse.lambda)), expected_dense; atol=1e-9)

    # Real input should be promoted to complex
    @info "General: real input promotion"
    A_real = [1.0 2.0; 0.0 3.0]
    result_real = feast_general(A_real, center, radius; M0=size(A_real, 1), fpm=copy(fpm), parallel=:serial)
    @test result_real.info == 0
    @test result_real.M == 2
    @test isapprox(sort(real.(result_real.lambda)), expected_dense; atol=1e-9)

    # Dense complex-symmetric wrappers use a transpose-bilinear projection.
    @test isdefined(FeastKit, :_feast_dense_complex_symmetric)
    A_cs_dense = ComplexF64[
        0.3 + 0.2im  0.1 + 0.4im  0.0 + 0.0im  0.0 + 0.0im
        0.1 + 0.4im  0.9 - 0.1im  0.0 + 0.2im  0.0 + 0.0im
        0.0 + 0.0im  0.0 + 0.2im  1.4 + 0.3im  0.15 - 0.1im
        0.0 + 0.0im  0.0 + 0.0im  0.15 - 0.1im  2.2 + 0.1im
    ]
    B_cs_dense = Matrix(Diagonal(ComplexF64[1.0, 1.1, 1.2, 1.3]))
    Emid_cs_dense = 1.0 + 0.1im
    r_cs_dense = 1.5
    expected_cs_dense = eigvals(A_cs_dense, B_cs_dense)
    expected_cs_dense = expected_cs_dense[[feast_inside_gcontour(λ, Emid_cs_dense, r_cs_dense; fpm=fpm) for λ in expected_cs_dense]]
    sort_complex(vals) = sort(collect(vals), by=x -> (round(real(x), digits=10),
                                                      round(imag(x), digits=10)))

    cs_dense = feast_gegv_complex_sym!(copy(A_cs_dense), copy(B_cs_dense),
                                       Emid_cs_dense, r_cs_dense, 4, copy(fpm))
    @test cs_dense.info == 0
    @test cs_dense.M == length(expected_cs_dense)
    @test isapprox(sort_complex(cs_dense.lambda), sort_complex(expected_cs_dense); atol=1e-7)

    cs_dense_standard = feast_geev_complex_sym!(copy(A_cs_dense), Emid_cs_dense,
                                                r_cs_dense, 4, copy(fpm))
    expected_cs_standard = eigvals(A_cs_dense)
    expected_cs_standard = expected_cs_standard[[feast_inside_gcontour(λ, Emid_cs_dense, r_cs_dense; fpm=fpm) for λ in expected_cs_standard]]
    @test cs_dense_standard.info == 0
    @test cs_dense_standard.M == length(expected_cs_standard)
    @test isapprox(sort_complex(cs_dense_standard.lambda), sort_complex(expected_cs_standard); atol=1e-7)

    A_dense_non_symmetric = ComplexF64[1 2; 0 3]
    @test_throws ArgumentError feast_geev_complex_sym!(A_dense_non_symmetric,
                                                       Emid_cs_dense, r_cs_dense,
                                                       2, copy(fpm))

    # Invalid configurations
    @test_throws ArgumentError feast_general(A_dense, B_dense, center, 0.0; M0=4, fpm=copy(fpm))
    B_bad = rand(ComplexF64, 3, 3)
    @test_throws ArgumentError feast_general(A_dense, B_bad, center, radius; M0=4, fpm=copy(fpm))
end

@testset "Single precision support" begin
    n = 4
    A = Matrix{Float32}(SymTridiagonal(fill(2.0f0, n), fill(-1.0f0, n-1)))
    B = Matrix{Float32}(I, n, n)
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    result = feast_sygv!(copy(A), copy(B), 0.0f0, 4.0f0, n, fpm)
    @test result.info == 0
    @test result.M >= 1
    @test eltype(result.lambda) === Float32
    @test eltype(result.q) === Float32

    high_level = feast(A, B, (0.0f0, 4.0f0); M0=n, fpm=copy(fpm), backend=:serial)
    @test high_level.info == 0
    @test high_level.M >= 1
    @test eltype(high_level.lambda) === Float32
    @test eltype(high_level.q) === Float32

    A_complex = Matrix(Diagonal(ComplexF32[0.25f0, 1.25f0, 2.25f0, 3.25f0]))
    result_complex = feast_heev!(copy(A_complex), -2.0f0, 2.0f0, n, fpm)
    @test result_complex.info == 0
    @test result_complex.M >= 1
end

@testset "Dense iterative FEAST" begin
    Random.seed!(1)

    n = 6
    A = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
    B = Matrix{Float64}(I, n, n)
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    fpm_gmres8 = copy(fpm)
    # Match the FEAST outer residual target to the requested inner GMRES accuracy.
    fpm_gmres8[3] = 8

    direct = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm))
    gmres_result = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm_gmres8);
                               solver=:gmres, solver_tol=1e-8,
                               solver_maxiter=400, solver_restart=25)
    @test gmres_result.info == 0
    @test gmres_result.M == direct.M
    @test isapprox(sort(gmres_result.lambda), sort(direct.lambda); atol=1e-8)

    alias_result = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm_gmres8);
                               solver=:iterative, solver_tol=1e-8,
                               solver_maxiter=400, solver_restart=25)
    @test alias_result.info == 0
    @test alias_result.M == direct.M
    @test isapprox(sort(alias_result.lambda), sort(direct.lambda); atol=1e-8)

    wrapper = difeast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm_gmres8);
                            solver_tol=1e-8, solver_maxiter=400, solver_restart=25)
    @test wrapper.info == 0
    @test wrapper.M == direct.M
    @test isapprox(sort(wrapper.lambda), sort(direct.lambda); atol=1e-8)

    A_complex = Matrix(Diagonal(ComplexF64[-1.5, -0.5, 0.2, 0.8, 1.4, 2.2]))
    direct_h = feast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm))
    gmres_h = feast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm_gmres8);
                           solver=:gmres, solver_tol=1e-8,
                           solver_maxiter=400, solver_restart=25)
    @test gmres_h.info == 0
    @test gmres_h.M == direct_h.M
    @test isapprox(sort(gmres_h.lambda), sort(direct_h.lambda); atol=1e-8)

    wrapper_h = zifeast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm_gmres8);
                              solver_tol=1e-8, solver_maxiter=400, solver_restart=25)
    @test wrapper_h.info == 0
    @test wrapper_h.M == direct_h.M
    @test isapprox(sort(wrapper_h.lambda), sort(direct_h.lambda); atol=1e-8)

    B_complex = Matrix(Diagonal(ComplexF64[1.2, 1.1, 1.3, 1.4, 1.5, 1.6]))
    direct_hg = feast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm))
    gmres_hg = feast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm_gmres8);
                            solver=:gmres, solver_tol=1e-8,
                            solver_maxiter=400, solver_restart=25)
    @test gmres_hg.info == 0
    @test gmres_hg.M == direct_hg.M
    @test isapprox(sort(gmres_hg.lambda), sort(direct_hg.lambda); atol=1e-8)

    wrapper_hg = zifeast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm_gmres8);
                               solver_tol=1e-8, solver_maxiter=400, solver_restart=25)
    @test wrapper_hg.info == 0
    @test wrapper_hg.M == direct_hg.M
    @test isapprox(sort(wrapper_hg.lambda), sort(direct_hg.lambda); atol=1e-8)

    A_general = Matrix{ComplexF64}(rand(ComplexF64, n, n))
    B_general = A_general' + I
    Emid = complex(0.0, 0.0)
    r = 3.0
    direct_gen = feast_gegv!(copy(A_general), copy(B_general), Emid, r, n, copy(fpm))
    gmres_gen = feast_gegv!(copy(A_general), copy(B_general), Emid, r, n, copy(fpm);
                            solver=:gmres, solver_tol=1e-7,
                            solver_maxiter=400, solver_restart=30)
    @test gmres_gen.info == 0
    @test gmres_gen.M == direct_gen.M
    @test isapprox(sort(real.(gmres_gen.lambda)), sort(real.(direct_gen.lambda)); atol=1e-7)

    wrapper_gen = zifeast_gegv!(copy(A_general), copy(B_general), Emid, r, n, copy(fpm);
                                solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test isapprox(sort(real.(wrapper_gen.lambda)), sort(real.(direct_gen.lambda)); atol=1e-7)

    direct_geev = feast_geev!(copy(A_general), Emid, r, n, copy(fpm))
    gmres_geev = feast_geev!(copy(A_general), Emid, r, n, copy(fpm);
                             solver=:gmres, solver_tol=1e-7,
                             solver_maxiter=400, solver_restart=30)
    @test isapprox(sort(real.(gmres_geev.lambda)), sort(real.(direct_geev.lambda)); atol=1e-7)

    wrapper_geev = zifeast_geev!(copy(A_general), Emid, r, n, copy(fpm);
                                 solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
    @test isapprox(sort(real.(wrapper_geev.lambda)), sort(real.(direct_geev.lambda)); atol=1e-7)
end
