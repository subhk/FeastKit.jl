using Test, FeastKit, LinearAlgebra, SparseArrays, Random, Logging

# Regressions for convergence decisions that used to depend on the units of the
# problem, spurious Ritz pairs that were counted as eigenvalues, and empty search
# regions that were reported as failures.

quietly(f) = with_logger(f, NullLogger())

@testset "Solver robustness" begin
    @testset "Spurious Ritz pairs are not counted" begin
        # Eigenvalues 1 and 7 (and 2 and 6) sit symmetrically about the search
        # interval, so the contour filter damps them equally. A trial direction
        # mixing their eigenvectors never separates under filtering, and its
        # Rayleigh quotient (~5.21) lies inside (2.5, 5.5). It used to be counted
        # as an eigenvalue: 20 loops then info=5 (M0=6), or saturation (M0=4).
        D = collect(1.0:10.0)
        expected = [3.0, 4.0, 5.0]
        for M0 in (4, 5, 6), T in (Float64, ComplexF64), storage in (Matrix, sparse)
            A = storage(Matrix(Diagonal(T.(D))))
            r = quietly(() -> feast(A, (2.5, 5.5); M0=M0))
            @test r.info == 0
            @test r.M == 3
            @test sort(real.(r.values)) ≈ expected atol=1e-10
        end
        # The convenience wrapper used to return the spurious value silently.
        values = quietly(() -> eigvals_feast(Matrix(Diagonal(D)), (2.5, 5.5); M0=6))
        @test sort(values) ≈ expected atol=1e-10

        # Band storage and the full-contour (general) kernel.
        r = quietly(() -> feast_banded(full_to_banded(Matrix(Diagonal(D)), 0), 0, (2.5, 5.5); M0=6))
        @test r.info == 0 && r.M == 3
        for storage in (Matrix, sparse)
            G = storage(Matrix(Diagonal(ComplexF64.(D))))
            r = quietly(() -> feast_general(G, 4.0 + 0im, 1.5; M0=6))
            @test r.info == 0
            @test r.M == 3
            @test sort(real.(r.values)) ≈ expected atol=1e-10
        end
        if Threads.nthreads() > 1
            for storage in (Matrix, sparse)
                r = quietly(() -> feast(storage(Matrix(Diagonal(D))), (2.5, 5.5);
                                        M0=6, backend=:threads))
                @test r.info == 0 && r.M == 3
            end
        end
    end

    @testset "Oblique spurious pairs in non-normal problems" begin
        # Two eigenvalues inside the circle, two real ones just outside, then
        # a conjugate pair 0.2 ± 2i whose equal filter response makes the
        # M0 = 5 subspace hold only half of it. The one-sided Rayleigh-Ritz of
        # this non-normal matrix then mixes an in-region eigenvector into the
        # leftover direction, so the spurious pair's own filter response is
        # high; counting the in-region directions of the whole subspace is
        # what exposes it. It used to end in info=5 after the full loop budget.
        n = 8
        D = zeros(n, n)
        for (i, v) in enumerate((0.0, 0.4, 1.3, -0.95))
            D[i, i] = v
        end
        D[5, 5] = D[6, 6] = 0.2
        D[5, 6], D[6, 5] = 2.0, -2.0
        for i in 7:n
            D[i, i] = 3.0 + i
        end
        V = I + 0.4 .* [sin(i * j + 1.0) for i in 1:n, j in 1:n] ./ sqrt(n)
        A = V * D / V
        for storage in (Matrix, sparse)
            r = quietly(() -> feast_general(storage(A), 0.2 + 0im, 0.5; M0=5))
            @test r.info == 0
            @test r.M == 2
            @test sort(real.(r.values)) ≈ [0.0, 0.4] atol=1e-10
        end

        # The restricted filter's eigenvalues count the kept directions: a
        # projector onto two coordinates keeps two of four trial directions,
        # however the trial basis mixes them.
        X = [1.0 0.2 0.3 0.1; 0.5 1.0 0.1 0.2; 0.3 0.4 1.0 0.6; 0.2 0.1 0.5 1.0;
             0.7 0.3 0.2 0.9; 0.1 0.8 0.6 0.3]
        P = Diagonal([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        @test FeastKit._feast_filter_rank(P * X, X) == 2
        @test FeastKit._feast_filter_rank(0.9 .* X, X) == 4
    end

    @testset "Polynomial solvers do not depend on units" begin
        # P(λ) = λ²I - diag(1², …, 40²): roots ±1 … ±40. The disc holds 4, 5, 6.
        nn = 40
        K = Matrix(Diagonal([-(float(j))^2 for j in 1:nn]))
        C = zeros(nn, nn)
        Mm = Matrix{Float64}(I, nn, nn)
        expected = [4.0, 5.0, 6.0]
        center, radius = 5.0 + 0.0im, 1.2
        fpm_moments() = (f = feastinit().fpm; f[8] = 32; f[16] = 1; f)
        paths = [
            ("moments", (c, z, r) -> feast_srcipev!(c, 2, z, r, 8, fpm_moments())),
            ("companion", (c, z, r) -> feast_polynomial([ComplexF64.(A) for A in c], z, r; M0=8)),
            ("sparse companion", (c, z, r) -> feast_scsrpev!(sparse.(c), 2, z, r, 8, feastinit().fpm)),
        ]
        for (label, solve) in paths
            residuals = Float64[]
            for (coeffs, z, r, back) in (([K, C, Mm], center, radius, 1.0),
                                         ([1e-8 .* K, 1e-8 .* C, 1e-8 .* Mm], center, radius, 1.0),
                                         ([1e8 .* K, 1e8 .* C, 1e8 .* Mm], center, radius, 1.0),
                                         ([K, C ./ 1e4, Mm ./ 1e8], 1e4 * center, 1e4 * radius, 1e4),
                                         ([K, C .* 1e4, Mm .* 1e8], 1e-4 * center, 1e-4 * radius, 1e-4))
                result = quietly(() -> solve(coeffs, z, r))
                @test result.info == 0
                @test result.M == 3
                @test sort(real.(result.values)) ./ back ≈ expected rtol=1e-10
                push!(residuals, result.epsout)
            end
            # One problem in five sets of units: one residual, up to roundoff.
            @test maximum(residuals) <= 10 * minimum(residuals) + 1e-15
        end
        # Matrix-free operators go through GMRES on the companion system, so a
        # smaller instance (roots ±1 … ±12) keeps the solves quick.
        ops = c -> [LinearOperator{ComplexF64}((y, x) -> mul!(y, ComplexF64.(A), x), size(A)) for A in c]
        Ks = Matrix(Diagonal([-(float(j))^2 for j in 1:12]))
        Cs, Ms = zeros(12, 12), Matrix{Float64}(I, 12, 12)
        for (coeffs, z, r, back) in (([Ks, Cs, Ms], center, radius, 1.0),
                                     ([1e-8 .* Ks, 1e-8 .* Cs, 1e-8 .* Ms], center, radius, 1.0),
                                     ([Ks, Cs ./ 1e4, Ms ./ 1e8], 1e4 * center, 1e4 * radius, 1e4))
            result = quietly(() -> feast_polynomial(ops(coeffs), z, r; M0=8,
                                                    solver_opts=(rtol=1e-13, maxiter=500, restart=30)))
            @test result.info == 0 && result.M == 3
            @test sort(real.(result.values)) ./ back ≈ expected rtol=1e-10
        end

        # Validation verdicts are relative to the size of the terms.
        x = zeros(ComplexF64, 12); x[5] = 1
        for s in (1e-12, 1.0, 1e12)
            coeff_ops = ops([s .* Ks, s .* Cs, s .* Ms])
            A_comp, B_comp = FeastKit._matrix_free_polynomial_companion_operators(coeff_ops)
            v = FeastKit.validate_companion_matrices(A_comp.A_mul!, B_comp.A_mul!, coeff_ops,
                                                     5.0 + 0im, x)
            @test v.polynomial_valid && v.companion_valid
        end
    end

    @testset "Screening keeps genuine pairs" begin
        rng = MersenneTwister(1)
        keep = Vector{Bool}(undef, 3)
        X = Matrix(qr(randn(rng, 20, 3)).Q)
        # Two converged pairs and one unconverged pair with a strong response:
        # still converging, so nothing is dropped yet.
        @test FeastKit._feast_screen_spurious!(keep, 0.9 .* X, X, [1e-14, 1e-14, 1e-3], 3, 1e-12) === nothing
        # The same pair with a weak response is spurious.
        Y = copy(X); Y[:, 3] .*= 1e-3
        @test FeastKit._feast_screen_spurious!(keep, Y, X, [1e-14, 1e-14, 1e-3], 3, 1e-12) == 2
        @test keep == [true, true, false]
        # Nothing to screen when every pair converged.
        @test FeastKit._feast_screen_spurious!(keep, Y, X, zeros(3), 3, 1e-12) === nothing
    end

    @testset "Convergence does not depend on units" begin
        rng = MersenneTwister(11)
        n = 60
        Q = Matrix(qr(randn(rng, n, n)).Q)
        U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
        base = Dict(:real => Matrix(Symmetric(Q * Diagonal(1.0:n) * Q')),
                    :hermitian => Matrix(Hermitian(U * Diagonal(1.0:n) * U')))
        for (kind, A1) in base
            loops = Int[]
            for s in (1e-10, 1.0, 1e10)
                A = A1 .* s
                r = quietly(() -> feast(A, (2.5s, 8.5s); M0=7))
                @test r.info == 0
                @test r.M == 6
                @test sort(real.(r.values)) ./ s ≈ collect(3.0:8.0) rtol=1e-12
                # A true relative residual, independent of the solver's own measure.
                @test maximum(norm(A * r.vectors[:, j] - r.values[j] * r.vectors[:, j]) /
                              (s * n) for j in 1:r.M) < 1e-12
                push!(loops, r.loop)
            end
            @test allequal(loops)
        end
        # Near-zero eigenvalues of a large matrix: small relative to the pencil,
        # so they are judged by backward error and still converge.
        L = Matrix(SymTridiagonal(fill(2.0, 100), fill(-1.0, 99)))
        ref = eigvals(Symmetric(L))
        for s in (1e-8, 1.0, 1e8)
            r = quietly(() -> feast(L .* s, ((ref[1] - 1e-9) * s, (ref[8] + 1e-9) * s); M0=16))
            @test r.info == 0 && r.M == 8
            @test sort(r.values) ./ s ≈ ref[1:8] rtol=1e-10
        end
    end

    @testset "An empty region is a complete answer" begin
        A = Matrix(Diagonal(1.0:10.0))
        r = quietly(() -> feast(A, (20.0, 30.0); M0=4))
        @test r.info == 0 && r.M == 0 && r.converged
        @test occursin("no eigenvalues", r.message)
        @test isempty(quietly(() -> eigvals_feast(A, (20.0, 30.0); M0=4, check=true)))
        r = quietly(() -> feast_general(ComplexF64.(A), 20.0 + 0im, 1.0; M0=4))
        @test r.info == 0 && r.M == 0
        # A caller's seed that misses the enclosed eigenvector (here e1, whose
        # eigenvalue is outside) is checked with independent probes first.
        r = quietly(() -> feast(A, (2.5, 3.5); M0=2, initial_subspace=Matrix(1.0I, 10, 2)[:, 1:1]))
        @test r.info == 0 && r.M == 1
        @test r.values ≈ [3.0]
    end

    @testset "Unconverged convenience results warn" begin
        A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
        @test_logs (:warn, r"did not converge") match_mode=:any eigvals_feast(A, (0.5, 4.5); subspace_size=2)
        @test_logs (:warn, r"did not converge") match_mode=:any eigen_feast(A, (0.5, 4.5); subspace_size=2)
        @test_logs min_level=Logging.Warn eigvals_feast(A, (0.5, 2.5); subspace_size=4)
    end

    @testset "Generalized interval check bounds the pencil" begin
        A = Matrix(2.0I, 4, 4)
        B = Matrix(2.0I, 4, 4)
        # Every eigenvalue of (2I, 2I) is 1; bounding A alone warned "[2, 2]".
        @test_logs min_level=Logging.Warn feast_validate_interval(A, B, (0.5, 1.5))
        @test feast_validate_interval(A, B, (0.5, 1.5)) == (1.0, 1.0)
        @test_logs min_level=Logging.Warn feast(A, B, (0.5, 1.5); M0=4)
        @test_logs (:warn, r"may not contain eigenvalues") feast_validate_interval(A, B, (5.0, 6.0))
        # Gershgorin cannot certify this B positive definite: no bound, no warning.
        Bweak = [1.0 2.0; 2.0 1.0]
        @test_logs min_level=Logging.Warn feast_validate_interval([1.0 0.0; 0.0 2.0], Bweak, (5.0, 6.0))
        @test feast_validate_interval([1.0 0.0; 0.0 2.0], Bweak, (5.0, 6.0)) == (-Inf, Inf)
    end

    @testset "API regressions" begin
        @test FeastLinearOperator === LinearOperator
        # feastinit() leaves the contour size unset until defaults are applied.
        @test ParallelFeastState{Float64}(feastinit().fpm[2], 4, false, false) isa ParallelFeastState{Float64}
        if Base.get_extension(FeastKit, :FeastKitMPIExt) === nothing
            A = Matrix(Diagonal(1.0:4.0))
            @test_throws "using MPI" mpi_feast(A, (0.5, 1.5); M0=2)
            @test_throws "using MPI" feast_hybrid(A, A, (0.5, 1.5))
            withenv("FEASTKIT_ENABLE_MPI" => "true") do
                @test !mpi_available()   # opted in, but MPI is not loaded
            end
        end
        withenv("FEASTKIT_ENABLE_MPI" => nothing) do
            @test !mpi_available()
        end
    end

    @testset "Matrix-free GMRES reaches the default tolerance" begin
        n = 100
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        r = quietly(() -> feast_matvec((y, x) -> mul!(y, A, x), (y, x) -> copyto!(y, x),
                                       n, (0.5, 1.5); M0=30))
        @test r.info == 0
        @test r.M == 19
        @test sort(r.values) ≈ filter(x -> 0.5 < x < 1.5, eigvals(Symmetric(A))) atol=1e-10
    end

    @testset "Sparse complex-symmetric solves keep a caller's seed" begin
        d = ComplexF64[1 + 0.1im, 2 - 0.1im, 3 + 0.05im, 7, 8, 9]
        A = sparse(Diagonal(d))
        exact = Matrix{ComplexF64}(I, 6, 6)[:, 1:3]
        seeded = quietly(() -> feast_scsrev_complex!(copy(A), 2.0 + 0im, 1.6, 4, feastinit().fpm;
                                                     initial_subspace=exact))
        @test seeded.info == 0 && seeded.M == 3
        @test sort(seeded.lambda; by=real) ≈ d[1:3] atol=1e-10
    end
end
