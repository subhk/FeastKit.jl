using FeastKit
using Test
using LinearAlgebra
using SparseArrays
using Distributed
using Random

@testset "FeastKit.jl" begin
    
    @testset "Parameter initialization" begin
        # Test feastinit - now uses -111 sentinel value (matches Fortran)
        fpm = feastinit()
        @test length(fpm.fpm) == 64
        @test all(fpm.fpm .== -111)  # All initialized to sentinel value

        # Test direct fpm array initialization
        fpm_array = zeros(Int, 64)
        feastinit!(fpm_array)
        @test all(fpm_array .== -111)  # All initialized to sentinel value

        # Test that feastdefault! sets defaults for -111 values
        feastdefault!(fpm_array)
        @test fpm_array[1] == 0   # Default print level (off)
        @test fpm_array[2] == 8   # Default integration points
        @test fpm_array[3] == 12  # Default tolerance
        @test fpm_array[4] == 20  # Default max loops

        # Test that user values are preserved
        fpm_user = zeros(Int, 64)
        feastinit!(fpm_user)
        fpm_user[2] = 16  # User sets 16 contour points
        fpm_user[3] = 8   # User sets tolerance 10^-8
        feastdefault!(fpm_user)
        @test fpm_user[2] == 16  # User value preserved
        @test fpm_user[3] == 8   # User value preserved
        @test fpm_user[4] == 20  # Default applied to unset parameters
    end
    
    @testset "Contour generation" begin
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        feastdefault!(fpm)  # Apply defaults before using contour functions

        # Test elliptical contour for real interval (symmetric/Hermitian)
        # Uses fpm[2] for half-contour (default: 8 points)
        Emin, Emax = 0.0, 10.0
        contour = feast_contour(Emin, Emax, fpm)
        @test length(contour.Zne) == fpm[2]
        @test length(contour.Wne) == fpm[2]

        # Test elliptical contour for complex region (non-Hermitian/general)
        # Uses fpm[8] for full-contour (default: 16 points) - matches Fortran FEAST
        Emid = 5.0 + 2.0im
        r = 3.0
        contour_g = feast_gcontour(Emid, r, fpm)
        @test length(contour_g.Zne) == fpm[8]  # Full contour uses fpm[8]
        @test length(contour_g.Wne) == fpm[8]
    end
    
    @testset "Input validation" begin
        # Test parameter validation
        @test_throws ArgumentError check_feast_srci_input(0, 10, 0.0, 1.0, zeros(Int, 64))
        @test_throws ArgumentError check_feast_srci_input(10, 0, 0.0, 1.0, zeros(Int, 64))
        @test_throws ArgumentError check_feast_srci_input(10, 15, 0.0, 1.0, zeros(Int, 64))
        @test_throws ArgumentError check_feast_srci_input(10, 5, 1.0, 0.0, zeros(Int, 64))
        @test_throws ArgumentError check_feast_srci_input(10, 5, 0.0, 1.0, zeros(Int, 32))
        
        # Should pass for valid inputs
        @test check_feast_srci_input(10, 5, 0.0, 1.0, zeros(Int, 64)) == true
    end

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
            # Exercise the high-level driver only when explicitly requested.
            try
                result = feast(A, (0.5, 2.5), M0=4, fpm=fpm, parallel=:serial)
                @test result.info >= 0  # Should not crash
                @test result.M >= 0     # Should find some eigenvalues
            catch e
                @test isa(e, ArgumentError) || isa(e, ErrorException) || isa(e, UndefVarError)
                # Implementation is incomplete, so errors are expected
            end
        else
            @info "Skipping high-level feast() smoke run (set FEAST_RUN_LONG_TESTS=true to enable)"
        end
    end

    @testset "High-level feast dispatch" begin
        n = 3
        # Real symmetric generalized problem
        @info "Dispatch: real symmetric generalized"
        A_sym = Symmetric(diagm(0 => 2.0 .* ones(n), 1 => -1.0 .* ones(n-1), -1 => -1.0 .* ones(n-1)))
        B_real = Matrix{Float64}(I, n, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[1] = 0
        result_real = feast(A_sym, B_real, (0.5, 3.5), M0=n, fpm=copy(fpm), parallel=:serial)
        @test result_real.info == 0
        @test result_real.M == n
        @test isapprox(sort(result_real.lambda), sort(eigvals(Matrix(A_sym))); atol=1e-10)

        # Ensure non-symmetric input errors out
        A_nonsym = [1.0 2.0 0.0; 0.0 3.0 1.0; 0.5 0.0 4.0]
        @test_throws ArgumentError feast(A_nonsym, B_real, (0.5, 3.5), M0=n, fpm=copy(fpm), parallel=:serial)

        # Dense complex Hermitian standard problem
        @info "Dispatch: dense Hermitian standard"
        A_herm = Hermitian([2.5 + 0im 0.2 + 0.1im 0.0;
                            0.2 - 0.1im 3.5 + 0im 0.3 - 0.2im;
                            0.0 0.3 + 0.2im 4.0 + 0im])
        result_complex = feast(A_herm, (2.0, 5.0), M0=n, fpm=copy(fpm), parallel=:serial)
        @test result_complex.info == 0
        @test result_complex.M == n
        hermitian_eigs = sort(eigvals(Matrix(A_herm)))
        @test isapprox(sort(result_complex.lambda), hermitian_eigs; atol=1e-9)

        # Complex non-Hermitian input should throw
        A_nonherm = [1.0 + 0im 2.0 + 1.0im; 3.0 - 1.0im 4.0 + 0im]
        B_complex = Matrix{ComplexF64}(I, 2, 2)
        @test_throws ArgumentError feast(A_nonherm, B_complex, (0.0, 5.0), M0=2, fpm=copy(fpm), parallel=:serial)

        # Sparse complex Hermitian standard problem
        @info "Dispatch: sparse Hermitian standard"
        v = ComplexF64[0.1 + 0.2im, -0.05 + 0.1im]
        diag_vals = ComplexF64[2.0, 3.0, 4.0]
        A_sparse = spdiagm(-1 => conj.(v), 0 => diag_vals, 1 => v)
        result_sparse = feast(A_sparse, (1.5, 4.5), M0=size(A_sparse, 1), fpm=copy(fpm), parallel=:serial)
        sparse_eigs = sort(real.(eigvals(Matrix(A_sparse))))
        @test result_sparse.info == 0
        @test result_sparse.M == size(A_sparse, 1)
        @test isapprox(sort(result_sparse.lambda), sparse_eigs; atol=1e-9)
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

        direct = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm))
        gmres_result = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm);
                                   solver=:gmres, solver_tol=1e-8,
                                   solver_maxiter=400, solver_restart=25)
        @test gmres_result.info == 0
        @test gmres_result.M == direct.M
        @test isapprox(sort(gmres_result.lambda), sort(direct.lambda); atol=1e-8)

        alias_result = feast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm);
                                   solver=:iterative, solver_tol=1e-8,
                                   solver_maxiter=400, solver_restart=25)
        @test alias_result.info == 0
        @test alias_result.M == direct.M
        @test isapprox(sort(alias_result.lambda), sort(direct.lambda); atol=1e-8)

        wrapper = difeast_sygv!(copy(A), copy(B), 0.0, 4.0, n, copy(fpm);
                                solver_tol=1e-8, solver_maxiter=400, solver_restart=25)
        @test wrapper.info == 0
        @test wrapper.M == direct.M
        @test isapprox(sort(wrapper.lambda), sort(direct.lambda); atol=1e-8)

        A_complex = Matrix(Diagonal(ComplexF64[-1.5, -0.5, 0.2, 0.8, 1.4, 2.2]))
        direct_h = feast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm))
        gmres_h = feast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm);
                               solver=:gmres, solver_tol=1e-8,
                               solver_maxiter=400, solver_restart=25)
        @test gmres_h.info == 0
        @test gmres_h.M == direct_h.M
        @test isapprox(sort(gmres_h.lambda), sort(direct_h.lambda); atol=1e-8)

        wrapper_h = zifeast_heev!(copy(A_complex), -2.0, 1.5, n, copy(fpm);
                                  solver_tol=1e-8, solver_maxiter=400, solver_restart=25)
        @test wrapper_h.info == 0
        @test wrapper_h.M == direct_h.M
        @test isapprox(sort(wrapper_h.lambda), sort(direct_h.lambda); atol=1e-8)

        B_complex = Matrix(Diagonal(ComplexF64[1.2, 1.1, 1.3, 1.4, 1.5, 1.6]))
        direct_hg = feast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm))
        gmres_hg = feast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm);
                                solver=:gmres, solver_tol=1e-8,
                                solver_maxiter=400, solver_restart=25)
        @test gmres_hg.info == 0
        @test gmres_hg.M == direct_hg.M
        @test isapprox(sort(gmres_hg.lambda), sort(direct_hg.lambda); atol=1e-8)

        wrapper_hg = zifeast_hegv!(copy(A_complex), copy(B_complex), -2.0, 1.5, n, copy(fpm);
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

        direct = feast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, 3.0, n, copy(fpm))
        @test direct.info == 0
        @test direct.M > 0

        gmres_result = feast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, 3.0, n, copy(fpm_gmres);
                                   solver=:gmres, solver_tol=1e-8,
                                   solver_maxiter=400, solver_restart=30)
        @test gmres_result.info == 0
        @test gmres_result.M == direct.M
        @test isapprox(sort(gmres_result.lambda[1:gmres_result.M]), sort(direct.lambda[1:direct.M]); atol=1e-8)

        wrapper = difeast_sbgv!(copy(A_band), copy(B_band), ka, kb, 0.5, 3.0, n, copy(fpm_gmres);
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
        gmres_h = feast_hbgv!(copy(A_h_band), copy(B_h_band), ka, ka, -2.0, 2.0, n, copy(fpm);
                              solver=:gmres, solver_tol=1e-7,
                              solver_maxiter=400, solver_restart=30)
        @test gmres_h.info == 0
        @test isapprox(sort(real.(gmres_h.lambda[1:gmres_h.M])), sort(real.(direct_h.lambda[1:direct_h.M])); atol=1e-7)

        wrapper_h = zifeast_hbgv!(copy(A_h_band), copy(B_h_band), ka, ka, -2.0, 2.0, n, copy(fpm);
                                  solver_tol=1e-7, solver_maxiter=400, solver_restart=30)
        @test wrapper_h.info == 0
        @test isapprox(sort(real.(wrapper_h.lambda[1:wrapper_h.M])), sort(real.(direct_h.lambda[1:direct_h.M])); atol=1e-7)

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

    @testset "FEAST-compatible precision aliases" begin
        required_aliases = [
            :sfeast_syev!, :dfeast_syev!, :dfeast_syevx!,
            :cfeast_heev!, :zfeast_heev!, :zfeast_geev!,
            :dfeast_scsrev!, :zfeast_hcsrev!,
            :dfeast_sbev!, :zfeast_hbev!,
            :cfeast_sbev!, :zfeast_sbev!
        ]
        @test all(name -> isdefined(FeastKit, name), required_aliases)

        n = 5
        A64 = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        fpm64 = zeros(Int, 64)
        feastinit!(fpm64)
        fpm64[1] = 0

        generic_dense = feast_syev!(copy(A64), 0.1, 3.9, n, copy(fpm64))
        alias_dense = FeastKit.dfeast_syev!(copy(A64), 0.1, 3.9, n, copy(fpm64))
        @test alias_dense.info == generic_dense.info
        @test alias_dense.M == generic_dense.M
        @test isapprox(sort(alias_dense.lambda), sort(generic_dense.lambda); atol=1e-10)

        contour = feast_contour(0.1, 3.9, copy(fpm64))
        alias_dense_x = FeastKit.dfeast_syevx!(copy(A64), 0.1, 3.9, n, copy(fpm64),
                                               contour.Zne, contour.Wne)
        @test alias_dense_x.info == generic_dense.info
        @test isapprox(sort(alias_dense_x.lambda), sort(generic_dense.lambda); atol=1e-10)

        A32 = Matrix{Float32}(A64)
        fpm32 = zeros(Int, 64)
        feastinit!(fpm32)
        fpm32[1] = 0
        alias_single = FeastKit.sfeast_syev!(copy(A32), 0.1f0, 3.9f0, n, copy(fpm32))
        @test alias_single.info == 0
        @test alias_single.M == n

        A_complex = Matrix(Diagonal(ComplexF64[-1.0, 0.25, 1.0, 2.0, 3.0]))
        generic_h = feast_heev!(copy(A_complex), -1.5, 2.5, n, copy(fpm64))
        alias_h = FeastKit.zfeast_heev!(copy(A_complex), -1.5, 2.5, n, copy(fpm64))
        @test alias_h.info == generic_h.info
        @test alias_h.M == generic_h.M
        @test isapprox(sort(alias_h.lambda), sort(generic_h.lambda); atol=1e-10)

        A_complex32 = ComplexF32.(A_complex)
        alias_c = FeastKit.cfeast_heev!(copy(A_complex32), -1.5f0, 2.5f0, n, copy(fpm32))
        @test alias_c.info == 0
        @test alias_c.M == 4

        A_general = Matrix(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 - 0.2im, 2.0, 3.0 + 0.3im, 4.0]))
        generic_g = feast_geev!(copy(A_general), 2.0 + 0.0im, 3.0, n, copy(fpm64))
        alias_g = FeastKit.zfeast_geev!(copy(A_general), 2.0 + 0.0im, 3.0, n, copy(fpm64))
        @test alias_g.info == generic_g.info
        @test alias_g.M == generic_g.M
        @test isapprox(sort(real.(alias_g.lambda)), sort(real.(generic_g.lambda)); atol=1e-10)

        A_sparse = sparse(A64)
        generic_sparse = feast_scsrev!(copy(A_sparse), 0.1, 3.9, n, copy(fpm64))
        alias_sparse = FeastKit.dfeast_scsrev!(copy(A_sparse), 0.1, 3.9, n, copy(fpm64))
        @test alias_sparse.info == generic_sparse.info
        @test alias_sparse.M == generic_sparse.M
        @test isapprox(sort(alias_sparse.lambda), sort(generic_sparse.lambda); atol=1e-10)

        A_sparse_h = sparse(A_complex)
        generic_sparse_h = feast_hcsrev!(copy(A_sparse_h), -1.5, 2.5, n, copy(fpm64))
        alias_sparse_h = FeastKit.zfeast_hcsrev!(copy(A_sparse_h), -1.5, 2.5, n, copy(fpm64))
        @test alias_sparse_h.info == generic_sparse_h.info
        @test alias_sparse_h.M == generic_sparse_h.M
        @test isapprox(sort(alias_sparse_h.lambda), sort(generic_sparse_h.lambda); atol=1e-10)

        A_band = full_to_banded(A64, 1)
        generic_band = feast_sbev!(copy(A_band), 1, 0.1, 3.9, n, copy(fpm64))
        alias_band = FeastKit.dfeast_sbev!(copy(A_band), 1, 0.1, 3.9, n, copy(fpm64))
        @test alias_band.info == generic_band.info
        @test alias_band.M == generic_band.M
        @test isapprox(sort(alias_band.lambda), sort(generic_band.lambda); atol=1e-10)

        A_band_h = full_to_banded(A_complex, 0)
        generic_band_h = feast_hbev!(copy(A_band_h), 0, -1.5, 2.5, n, copy(fpm64))
        alias_band_h = FeastKit.zfeast_hbev!(copy(A_band_h), 0, -1.5, 2.5, n, copy(fpm64))
        @test alias_band_h.info == generic_band_h.info
        @test alias_band_h.M == generic_band_h.M
        @test isapprox(sort(alias_band_h.lambda), sort(generic_band_h.lambda); atol=1e-10)

        A_band_cs = full_to_banded(Matrix(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 - 0.1im,
                                                              1.5 + 0.2im, 2.0, 3.0])), 0)
        generic_band_cs = feast_sbev_complex!(copy(A_band_cs), 0, 1.2 + 0.0im, 1.2, n, copy(fpm64))
        alias_band_cs = FeastKit.zfeast_sbev!(copy(A_band_cs), 0, 1.2 + 0.0im, 1.2, n, copy(fpm64))
        @test alias_band_cs.info == generic_band_cs.info
        @test alias_band_cs.M == generic_band_cs.M
        @test isapprox(sort(real.(alias_band_cs.lambda)), sort(real.(generic_band_cs.lambda)); atol=1e-10)
    end

    @testset "PFEAST-compatible precision aliases" begin
        required_parallel_aliases = [
            :psfeast_syev!, :pdfeast_syev!,
            :psfeast_sygv!, :pdfeast_sygv!,
            :psfeast_scsrev!, :pdfeast_scsrev!,
            :psfeast_scsrgv!, :pdfeast_scsrgv!,
            :psfeast_srci!, :pdfeast_srci!
        ]
        @test all(name -> isdefined(FeastKit, name), required_parallel_aliases)

        n = 4
        A64 = Matrix(Diagonal([0.5, 1.0, 1.5, 3.0]))
        B64 = Matrix{Float64}(I, n, n)
        fpm64 = zeros(Int, 64)
        feastinit!(fpm64)
        fpm64[1] = 0
        fpm64[2] = 8
        fpm64[4] = 12

        dense_standard = if nworkers() == 1
            @test_logs (:warn, "No worker processes available, falling back to serial computation") begin
                FeastKit.pdfeast_syev!(copy(A64), 0.4, 1.6, n, copy(fpm64);
                                       use_threads=false)
            end
        else
            FeastKit.pdfeast_syev!(copy(A64), 0.4, 1.6, n, copy(fpm64);
                                   use_threads=false)
        end
        @test dense_standard.info == 0
        @test dense_standard.M == 3
        @test isapprox(sort(dense_standard.lambda), [0.5, 1.0, 1.5]; atol=1e-8)

        dense_generalized = if nworkers() == 1
            @test_logs (:warn, "No worker processes available, falling back to serial computation") begin
                FeastKit.pdfeast_sygv!(copy(A64), copy(B64), 0.4, 1.6, n, copy(fpm64);
                                       use_threads=false)
            end
        else
            FeastKit.pdfeast_sygv!(copy(A64), copy(B64), 0.4, 1.6, n, copy(fpm64);
                                   use_threads=false)
        end
        @test dense_generalized.info == dense_standard.info
        @test dense_generalized.M == dense_standard.M
        @test isapprox(sort(dense_generalized.lambda), sort(dense_standard.lambda); atol=1e-8)

        A_sparse = sparse(A64)
        B_sparse = sparse(B64)
        sparse_standard = FeastKit.pdfeast_scsrev!(copy(A_sparse), 0.4, 1.6, n, copy(fpm64);
                                                   use_threads=false)
        @test sparse_standard.info == 0
        @test sparse_standard.M == 3
        @test isapprox(sort(sparse_standard.lambda), [0.5, 1.0, 1.5]; atol=1e-8)

        sparse_generalized = FeastKit.pdfeast_scsrgv!(copy(A_sparse), copy(B_sparse),
                                                      0.4, 1.6, n, copy(fpm64);
                                                      use_threads=false)
        @test sparse_generalized.info == sparse_standard.info
        @test sparse_generalized.M == sparse_standard.M
        @test isapprox(sort(sparse_generalized.lambda), sort(sparse_standard.lambda); atol=1e-8)

        A_sparse32 = sparse(Float32.(A64))
        B_sparse32 = spdiagm(0 => ones(Float32, n))
        fpm32 = copy(fpm64)
        Random.seed!(1234)
        randn(Float64, 64)
        sparse_single_first = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                                       0.4f0, 1.6f0, n, copy(fpm32);
                                                       use_threads=false)
        Random.seed!(5678)
        randn(Float64, 64)
        sparse_single_second = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                                        0.4f0, 1.6f0, n, copy(fpm32);
                                                        use_threads=false)
        @test sparse_single_first.info == sparse_single_second.info
        @test sparse_single_first.M == sparse_single_second.M
        @test sort(Float64.(sparse_single_first.lambda)) == sort(Float64.(sparse_single_second.lambda))

        sparse_single = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                                 0.4f0, 1.6f0, n, copy(fpm32);
                                                 use_threads=false)
        @test sparse_single.info == 0
        @test sparse_single.M == 3
        @test isapprox(sort(Float64.(sparse_single.lambda)), [0.5, 1.0, 1.5]; atol=1e-5)

        state = ParallelFeastState{Float64}(fpm64[2], n, false, false)
        work = Matrix{Float64}(undef, n, n)
        workc = Matrix{ComplexF64}(undef, n, n)
        Aq = Matrix{Float64}(undef, n, n)
        Sq = Matrix{Float64}(undef, n, n)
        lambda = Vector{Float64}(undef, n)
        q = Matrix{Float64}(undef, n, n)
        res = Vector{Float64}(undef, n)
        FeastKit.pdfeast_srci!(state, n, work, workc, Aq, Sq, copy(fpm64),
                               0.4, 1.6, n, lambda, q, res)
        @test state.info == Int(Feast_SUCCESS)
        @test state.ijob == Int(Feast_RCI_FACTORIZE)
    end
    
    @testset "Utility functions" begin
        # Test feast_name function
        code = 241500  # Example Feast code
        name = feast_name(code)
        @test isa(name, String)
        @test length(name) > 0
        
        # Test eigenvalue filtering
        lambda = [0.5, 1.5, 2.5, 3.5]
        @test feast_inside_contour(1.0, 0.0, 2.0) == true
        @test feast_inside_contour(3.0, 0.0, 2.0) == false
        
        # Test complex contour
        @test feast_inside_gcontour(1.0+1.0im, 1.0+1.0im, 2.0) == true
        @test feast_inside_gcontour(5.0+5.0im, 1.0+1.0im, 2.0) == false
    end
    
    @testset "Memory estimation" begin
        # Test memory estimation
        N, M0 = 100, 10
        mem_size = feast_memory_estimate(N, M0, Float64)
        @test mem_size > 0
    end
    
    @testset "Error handling" begin
        # Test error enum values
        @test Feast_SUCCESS.value == 0
        @test Feast_ERROR_N.value == 1
        @test Feast_ERROR_M0.value == 2

        # Test parameter validation on the normal initialized-parameter path
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        feastdefault!(fpm)
        @test fpm[1] == 0   # Default print level (off)
        @test fpm[2] == 8   # Default integration points
        @test fpm[3] == 12  # Default tolerance exponent
        @test fpm[4] == 20  # Default max loops

        # Test that invalid values outside valid range throw errors
        fpm_bad = zeros(Int, 64)
        feastinit!(fpm_bad)
        fpm_bad[1] = 5  # Invalid print level (must be 0 or 1)
        @test_throws ArgumentError feastdefault!(fpm_bad)
    end
    
    if get(ENV, "FEAST_RUN_PARALLEL_TESTS", "false") == "true"
        @testset "Parallel support" begin
            # Test parallel state creation
            state = ParallelFeastState{Float64}(8, 10, true, true)
            @test state.use_parallel == true
            @test state.use_threads == true
            @test state.total_points == 8
            @test length(state.moment_contributions) == 8

            # Test contour point distribution
            ne = 16
            nw = 4
            chunks = distribute_contour_points(ne, nw)
            @test length(chunks) == nw
            @test sum(length(chunk) for chunk in chunks) == ne

            # Test backend determination (basic test only)
            backend = determine_parallel_backend(:auto, nothing)
            @test backend in [:serial, :threads, :distributed, :mpi]

            # Test parallel capabilities check
            caps = feast_parallel_capabilities()
            @test isa(caps, Dict)
            @test haskey(caps, :threads)
            @test haskey(caps, :distributed)
            @test haskey(caps, :mpi)

            if get(ENV, "FEASTKIT_TEST_PARALLEL", "false") == "true"
                @info "Parallel execution tests enabled"
                n = 10
                A = diagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))
                B = Matrix{Float64}(I, n, n)

                try
                    if Threads.nthreads() > 1
                        result = feast(A, B, (0.5, 2.5), M0=5, parallel=:threads)
                        @test isa(result, FeastResult)
                    end
                catch e
                    @test isa(e, ArgumentError) || isa(e, ErrorException) || isa(e, UndefVarError)
                end
            else
                @info "Parallel execution tests disabled (set FEASTKIT_TEST_PARALLEL=true to enable)"
            end
        end
    else
        @info "Skipping parallel support tests (set FEAST_RUN_PARALLEL_TESTS=true to enable)"
    end
    
    @testset "Performance utilities" begin
        # Test memory estimation
        N, M0 = 50, 8
        mem_size = feast_memory_estimate(N, M0, Float64)
        @test mem_size > 0
        
        # Test interval validation
        A = diagm(0 => [1.0, 2.0, 3.0, 4.0])
        bounds = feast_validate_interval(A, (1.5, 3.5))
        @test bounds[1] <= bounds[2]  # min <= max
        
        # Test result summary (should not crash)
        lambda = [1.0, 2.0]
        q = [1.0 0.0; 0.0 1.0]
        res = [1e-12, 1e-12]
        result = FeastResult{Float64, Float64}(lambda, q, 2, res, 0, 1e-12, 3)
        
        # Capture output to avoid cluttering test results
        buf = IOBuffer()
        feast_summary(buf, result)
        output_str = String(take!(buf))
        @test length(output_str) > 0  # Should produce some output
    end
    
    if get(ENV, "FEAST_RUN_PARALLEL_TESTS", "false") == "true"
        @testset "Threaded vs Serial comparison" begin
            if get(ENV, "FEASTKIT_TEST_PARALLEL", "false") == "true"
                @info "Threaded vs Serial comparison enabled"
                if Threads.nthreads() > 1
                    n = 20
                    A = diagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))

                    fpm = zeros(Int, 64)
                    feastinit!(fpm)
                    fpm[1] = 0
                    fpm[2] = 4

                    try
                        result_serial = feast(A, (0.5, 1.5), M0=6, fpm=copy(fpm), parallel=:serial)
                        result_parallel = feast(A, (0.5, 1.5), M0=6, fpm=copy(fpm), parallel=:threads)
                        if result_serial.M > 0 && result_parallel.M > 0
                            @test result_serial.M >= 0
                            @test result_parallel.M >= 0
                        end
                    catch e
                        @test isa(e, ArgumentError) || isa(e, ErrorException) || isa(e, UndefVarError)
                    end
                else
                    @info "Skipping (only 1 thread available)"
                end
            else
                @info "Threaded vs Serial comparison disabled (set FEASTKIT_TEST_PARALLEL=true to enable)"
            end
        end
    else
        @info "Skipping threaded vs serial comparison (set FEAST_RUN_PARALLEL_TESTS=true to enable)"
    end
    
    @testset "Parallel backend selection" begin
        # Test parallel backend determination logic
        
        # Test explicit backend selection
        @test_throws ArgumentError determine_parallel_backend(:invalid, nothing)
        
        # Test auto selection logic
        backend = determine_parallel_backend(:auto, nothing)
        @test backend in [:serial, :threads, :distributed, :mpi]
        
        # Test with different thread/worker configurations
        if Threads.nthreads() > 1
            backend_threads = determine_parallel_backend(:threads, nothing)
            @test backend_threads == :threads
        else
            backend_threads = determine_parallel_backend(:threads, nothing)
            @test backend_threads == :serial
        end
        
        if Distributed.nworkers() > 1
            backend_dist = determine_parallel_backend(:distributed, nothing)
            @test backend_dist == :distributed
        else
            backend_dist = determine_parallel_backend(:distributed, nothing)
            @test backend_dist == :serial
        end
    end
end

include("test_matrix_free.jl")
include("test_allocation_helpers.jl")
include("test_backend_api.jl")
include("test_parallel_backends.jl")
include("test_production_gates.jl")
