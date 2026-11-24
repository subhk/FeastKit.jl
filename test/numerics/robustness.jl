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
        # A closed 16-node circle around the Ritz values: filter value 1 there.
        Zne = [cis(2pi * (k - 0.5) / 16) for k in 1:16]
        Wne = Zne ./ 16
        ritz = [0.0, 0.1, -0.1]
        screen(Y, res, W) = FeastKit._feast_screen_spurious!(keep, Y, X, res, 3, 1e-12,
                                                             ritz, Zne, W)
        @test FeastKit._feast_filter_response(0.1, Zne, Wne, false) ≈ 1 atol=1e-8
        # Two converged pairs and one unconverged pair with a strong response:
        # still converging, so nothing is dropped yet.
        @test screen(0.9 .* X, [1e-14, 1e-14, 1e-3], Wne) === nothing
        # The same pair with a weak response is spurious.
        Y = copy(X); Y[:, 3] .*= 1e-3
        @test screen(Y, [1e-14, 1e-14, 1e-3], Wne) == 2
        @test keep == [true, true, false]
        # Nothing to screen when every pair converged.
        @test screen(Y, zeros(3), Wne) === nothing
        # Responses are measured against the filter's own value, so rescaling
        # the weights (and with them every filtered vector) changes nothing.
        for s in (0.05, 0.2, 7.0)
            @test screen(0.9s .* X, [1e-14, 1e-14, 1e-3], s .* Wne) === nothing
            @test screen(s .* Y, [1e-14, 1e-14, 1e-3], s .* Wne) == 2
        end
        # Protect a weak pair one representable step inside the boundary
        # when its absolute residual is on the scale of spectral roundoff.
        center = 1e12
        nodes = center .+ Zne
        near_edge = prevfloat(maximum(real, nodes))
        translated_ritz = [center, center + 0.1, near_edge]
        @test FeastKit._feast_screen_spurious!(keep, Y, X, [0.0, 0.0, 5e-16], 3,
                                                1e-16, translated_ritz, nodes, Wne;
                                                residual_scale=center) === nothing
    end

    @testset "Filters whose in-region value is not 1 lose no eigenpairs" begin
        N = 100
        A = Matrix(SymTridiagonal(fill(2.0, N), fill(-1.0, N - 1)))
        lam = eigvals(Symmetric(A))
        a, b = (lam[20] + lam[21]) / 2, (lam[28] + lam[29]) / 2
        expected = lam[21:28]
        # Flat built-in ellipses: the filter is far below 1 on the interval.
        for (ne, aspect) in ((6, 5), (8, 1))
            fpm = feastinit().fpm; fpm[2] = ne; fpm[18] = aspect
            r = quietly(() -> feast_heev!(ComplexF64.(A), a, b, 12, fpm))
            @test r.info == 0 && r.M == 8
            @test sort(r.lambda[1:r.M]) ≈ expected atol=1e-9
        end
        # A user contour with uniformly rescaled weights is the same method.
        f0 = feastinit().fpm; f0[2] = 4; feastdefault!(f0)
        contour = feast_contour(a, b, f0)
        for s in (0.1, 0.2, 1.0, 3.0)
            fpm = feastinit().fpm; fpm[2] = 4
            r = quietly(() -> feast_syevx!(A, a, b, 12, fpm, contour.Zne, s .* contour.Wne))
            @test r.info == 0 && r.M == 8
            @test sort(r.lambda[1:r.M]) ≈ expected atol=1e-9
        end
    end

    @testset "An empty Ritz set is not taken for an empty interval" begin
        # One eigenvalue just inside the interval and 299 crowded just outside
        # it: a two-vector subspace starts out dominated by the crowd.
        rng = MersenneTwister(3)
        N = 300
        d = vcat([1.00095], 1.00105 .+ 0.0005 .* rand(rng, N - 1))
        Q = Matrix(qr(randn(rng, N, N)).Q)
        A = Q * Diagonal(d) * Q'; A = (A + A') / 2
        for solve in (() -> feast_syev!(A, 0.999, 1.001, 2, feastinit().fpm),
                      () -> feast_heev!(ComplexF64.(A), 0.999, 1.001, 2, feastinit().fpm),
                      () -> feast_geev!(ComplexF64.(A), 1.0 + 0im, 0.001, 2, feastinit().fpm))
            r = quietly(solve)
            # Either the eigenvalue is found or the run says it did not finish;
            # it must never report a converged empty interval.
            @test r.M == 1 || r.info != 0
            r.info == 0 && @test real(r.lambda[1]) ≈ 1.00095 atol=1e-9
        end
        # Genuinely empty intervals are still recognized, next to the spectrum
        # and far from it.
        T2 = Matrix(SymTridiagonal(fill(2.0, 50), fill(-1.0, 49)))
        ev = eigvals(Symmetric(T2))
        for (lo, hi) in ((ev[10] + 0.2 * (ev[11] - ev[10]), ev[10] + 0.8 * (ev[11] - ev[10])),
                         (10.0, 11.0))
            for r in (quietly(() -> feast_syev!(T2, lo, hi, 6, feastinit().fpm)),
                      quietly(() -> feast_heev!(ComplexF64.(T2), lo, hi, 6, feastinit().fpm)),
                      quietly(() -> feast_geev!(ComplexF64.(T2), complex((lo + hi) / 2),
                                                (hi - lo) / 2, 6, feastinit().fpm)))
                @test r.info == 0 && r.M == 0
            end
        end
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

    @testset "Shape contours select by the shape, not the node polygon" begin
        rng = MersenneTwister(1)
        Q = Matrix(qr(randn(rng, 30, 30)).Q)
        inside = [-0.8, -0.3, 0.1, 0.5, 0.95]
        A = Matrix(Symmetric(Q * Diagonal(vcat(inside, range(3, 6, length=25))) * Q'))
        # 0.95 lies inside the unit circle but outside the 8-node polygon.
        for n in (8, 16, 64)
            r = quietly(() -> feast(A, feast_circle(0.0, 1.0; n=n); M0=10))
            @test r.info == 0 && r.M == 5
            @test sort(real.(r.lambda)) ≈ inside atol=1e-8
        end
        @test sort(real.(quietly(() -> eigvals_feast(A, feast_circle(0.0, 1.0); M0=10)))) ≈ inside atol=1e-8
        @test quietly(() -> eigen_feast(A, feast_circle(0.0, 1.0); M0=10)).values isa AbstractVector
    end

    @testset "Closed contours are not completed twice" begin
        for T in (Float64, Float32), n in (8, 12, 16, 32, 64, 65)
            c = feast_circle(T(0), T(1); n=n)
            d = FeastKit._feast_complete_hermitian_contour(c)
            @test length(d.Zne) == n
            @test real(sum(d.Wne ./ (d.Zne .- T(0.3)))) ≈ 1 atol=1e-3
        end
        half = feast_contour_expert(-1.0, 1.0, 8)
        @test length(FeastKit._feast_complete_hermitian_contour(half).Zne) == 16
    end

    @testset "Parameter defaults and validation" begin
        L = Matrix(SymTridiagonal(fill(2.0, 60), fill(-1.0, 59)))
        # FeastParameters() means "all defaults", not a tolerance of 10^0.
        r = quietly(() -> feast(L, (0.0, 1.0); M0=30, fpm=FeastParameters()))
        @test r.info == 0 && r.epsout <= 1e-10
        for N in (1, 64, 401, 500, 529, 1025, 4096, 10^6)
            @test feastdefault!(feastinit_driver(N)) isa Vector{Int}
        end
        # Zolotarev is only available where it is tabulated.
        for n in (9, 20, 24)
            f = feastinit().fpm; f[2] = n; f[16] = 2
            @test_throws ArgumentError feastdefault!(f)
        end
        f = feastinit().fpm; f[2] = 16; f[16] = 2
        @test feastdefault!(f)[2] == 16
        # An explicit Float32-representable tolerance is honored.
        f = feastinit().fpm; f[3] = 6
        @test FeastKit.feast_tolerance(f, Float32) == 1f-6
        @test FeastKit.feast_tolerance(feastdefault!(feastinit().fpm), Float32) == sqrt(eps(Float32))
        # Non-finite intervals are argument errors, not factorization failures.
        @test_throws ArgumentError feast(L, (NaN, 1.0))
        @test_throws ArgumentError feast(L, (-Inf, 1.0))
        @test_throws ArgumentError feast_syev!(L, NaN, 1.0, 10, feastinit().fpm)
        # Custom contours are not limited to the Gauss node counts.
        ne = 30
        θ = [2pi * (k - 0.5) / ne for k in 1:ne]
        Z = ComplexF64[0.5 + 0.6 * cis(t) for t in θ]
        W = ComplexF64[0.6 * cis(t) / ne for t in θ]
        ev = eigvals(Symmetric(L))
        r = quietly(() -> feast_geevx!(ComplexF64.(L), 0.5 + 0im, 0.6, 30, feastinit().fpm, Z, W))
        @test r.info == 0 && r.M == count(x -> abs(x - 0.5) < 0.6, ev)
    end

    @testset "Storage and backend checks" begin
        G = Matrix{ComplexF64}(Tridiagonal(fill(-1.0, 19), collect(1.0:20.0), fill(-3.0, 19)))
        # Symmetric (upper-only) band storage is not general band storage.
        @test_throws ArgumentError feast_gbev!(full_to_banded(G, 1), 1, 5.0 + 0im, 1.6, 10, feastinit().fpm)
        r = quietly(() -> feast_gbev!(full_to_general_banded(G, 1), 1, 5.0 + 0im, 1.6, 10, feastinit().fpm))
        expected = sort(filter(x -> abs(x - 5) < 1.6, real.(eigvals(G))))
        @test r.info == 0
        @test sort(real.(r.lambda[1:r.M])) ≈ expected atol=1e-8
        # Too small a subspace is reported as such by the Hermitian band driver.
        H = ComplexF64.(Matrix(SymTridiagonal(fill(2.0, 40), fill(-1.0, 39))))
        ev = eigvals(Hermitian(H))
        r = quietly(() -> feast_hbev!(full_to_banded(H, 1), 1, ev[5] - 0.01, ev[14] + 0.01, 4, feastinit().fpm))
        @test r.info == Int(Feast_ERROR_M0)
        # The threaded drivers use a registered custom contour: three nodes
        # need visibly more refinement loops than the default eight.
        T2 = Matrix(SymTridiagonal(fill(2.0, 100), fill(-1.0, 99)))
        B2 = Matrix{Float64}(I, 100, 100)
        lam = eigvals(Symmetric(T2))
        a, b = (lam[20] + lam[21]) / 2, (lam[28] + lam[29]) / 2
        for solve in ((f) -> pfeast_sygv!(copy(T2), copy(B2), a, b, 12, f),
                      (f) -> pfeast_scsrgv!(sparse(T2), sparse(B2), a, b, 12, f))
            default = quietly(() -> solve(feastinit().fpm))
            f = feastinit().fpm; feastdefault!(f)
            FeastKit.feast_set_custom_contour!(f, feast_contour_expert(a, b, 3))
            custom = try
                quietly(() -> solve(f))
            finally
                FeastKit.feast_clear_custom_contour!(f)
            end
            @test default.info == 0 && custom.info == 0 && custom.M == default.M == 8
            @test custom.loop >= default.loop + 2
        end
        # Overlapping threaded solves leave the BLAS thread count as it was.
        before = BLAS.get_num_threads()
        @sync for _ in 1:4
            Threads.@spawn quietly(() -> pfeast_sygv!(copy(T2), copy(B2), a, b, 12, feastinit().fpm))
        end
        @test BLAS.get_num_threads() == before
        if !FeastKit.mpi_available()
            @test_throws ArgumentError feast_with_backend(T2, B2, (a, b), :mpi, 12, feastinit().fpm,
                                                          nothing, true; strict_backend=true)
        end
        # A failing solver callback is reported, not silently turned into info=8.
        Aop = LinearOperator{Float64}((y, x) -> mul!(y, T2, x), (100, 100); issymmetric=true)
        r = @test_logs (:warn, r"callback failed") match_mode=:any feast(
            Aop, (a, b); M0=12, solver=(Y, z, X) -> error("boom"))
        @test r.info == Int(Feast_ERROR_LAPACK)
    end

    @testset "Input types the high-level API accepts" begin
        L = Matrix(SymTridiagonal(fill(2.0, 60), fill(-1.0, 59)))
        Ls = sparse(L)
        expected = filter(x -> 0 <= x <= 1, eigvals(Symmetric(L)))
        check(r) = (@test r.info == 0; @test sort(real.(r.lambda)) ≈ expected atol=1e-8)
        # Sparse matrices with non-Int indices.
        check(quietly(() -> feast(SparseMatrixCSC{Float64,Int32}(Ls), (0.0, 1.0); M0=30)))
        check(quietly(() -> feast(Symmetric(SparseMatrixCSC{Float64,Int32}(Ls)), (0.0, 1.0); M0=30)))
        check(quietly(() -> feast_general(SparseMatrixCSC{Float64,Int32}(Ls), 0.5 + 0im, 0.5; M0=30)))
        # Pencils whose two matrices use different storage.
        check(quietly(() -> feast(Ls, Diagonal(ones(60)), (0.0, 1.0); M0=30)))
        check(quietly(() -> feast(L, sparse(1.0I, 60, 60), (0.0, 1.0); M0=30)))
        check(quietly(() -> feast(Ls, Matrix(1.0I, 60, 60), (0.0, 1.0); M0=30)))
        check(quietly(() -> feast_general(L, sparse(1.0I, 60, 60), 0.5 + 0im, 0.5; M0=30)))
        # Exact element types are promoted, and a real center is accepted.
        check(quietly(() -> feast(Rational{Int}.(round.(Int, L)), (0, 1); M0=30)))
        check(quietly(() -> feast_general(L, 0.5, 0.5; M0=30)))
        r = quietly(() -> feast(Complex{Int}[2 im; -im 2], (0, 2)))
        @test r.info == 0
        @test r.lambda ≈ [1.0] atol=1e-10
        @test feast_validate_interval(L, (0, 1)) == (0.0, 4.0)
        @test feast_validate_interval(Float32.(L), (0.0, 1.0)) == (0.0f0, 4.0f0)
        # Banded interface: integer storage and a mismatched B.
        Ib = round.(Int, full_to_banded(L, 1))
        check(quietly(() -> feast_banded(Ib, 1, (0, 1); M0=30)))
        check(quietly(() -> feast_banded(Ib, 1, (0.0, 1.0); B=ones(Int, 1, 60), M0=30)))
        @test_throws ArgumentError feast_banded(full_to_banded(L, 1), 1, (0.0, 1.0); B=ones(1, 10), M0=30)
        # The polynomial default subspace fits small problems.
        K = ComplexF64.(Matrix(Diagonal(-collect(1.0:6.0) .^ 2)))
        r = quietly(() -> feast_polynomial([K, zeros(ComplexF64, 6, 6), Matrix{ComplexF64}(I, 6, 6)], 2.0 + 0im, 0.5))
        @test r.info == 0
        @test real.(r.lambda) ≈ [2.0] atol=1e-8
        # A seeded run converges with enough time for verification.
        N = 100
        A = Matrix(SymTridiagonal(fill(2.0, N), fill(-1.0, N - 1)))
        lam = eigvals(Symmetric(A))
        a, b = (lam[20] + lam[21]) / 2, (lam[28] + lam[29]) / 2
        seed = randn(MersenneTwister(1), N, 12)
        full = quietly(() -> feast(A, (a, b); M0=12, initial_subspace=seed))
        @test full.info == 0 && full.M == 8
    end

    @testset "Iterative solves converge at the default inner tolerance" begin
        # Each shifted solve is warm-started from its trial vector, so GMRES's
        # fixed 1e-3 applies to a correction that shrinks with the residual.
        # Without that, these stalled near 4e-7 or kept spurious pairs.
        rng = MersenneTwister(4)
        n = 30
        X = randn(rng, n, n); A = (X + X') / 2
        Y = randn(rng, n, n); B = Y * Y' / n + I
        expected = filter(x -> -1 < x < 1, eigvals(Symmetric(A), Symmetric(B)))
        r = quietly(() -> difeast_sygv!(A, B, -1.0, 1.0, 20, feastinit().fpm))
        @test r.info == 0 && r.M == length(expected)
        @test sort(r.lambda) ≈ expected atol=1e-9
        r = quietly(() -> zifeast_hegv!(ComplexF64.(A), ComplexF64.(B), -1.0, 1.0, 20,
                                        feastinit().fpm))
        @test r.info == 0 && r.M == length(expected)
        G = randn(rng, ComplexF64, n, n)
        inside = filter(x -> abs(x - (0.3 + 0.2im)) < 2.5, eigvals(G))
        for st in (0.0, 1e-10)
            r = quietly(() -> zifeast_geev!(G, 0.3 + 0.2im, 2.5, 20, feastinit().fpm;
                                            solver_tol=st))
            @test r.info == 0 && r.M == length(inside)
            @test sort(r.lambda; by=real) ≈ sort(inside; by=real) atol=1e-8
        end
        # Built-in matrix-free solvers are marked for the warm start.
        Aop = LinearOperator{Float64}((y, x) -> mul!(y, A, x), (n, n); issymmetric=true)
        Bop = LinearOperator{Float64}((y, x) -> mul!(y, B, x), (n, n); issymmetric=true)
        @test create_iterative_solver(Aop, Bop) isa FeastKit.FeastIterativeSolver
        r = quietly(() -> feast(Aop, Bop, (-1.0, 1.0); M0=20))
        @test r.info == 0 && r.M == length(expected)
    end

    @testset "Polynomial RCI refinement converges" begin
        function companion_eigvals(C)
            d = length(C) - 1
            N = size(C[1], 1)
            AL = zeros(ComplexF64, d * N, d * N)
            BL = zeros(ComplexF64, d * N, d * N)
            for i in 1:(d - 1)
                AL[(i - 1) * N + 1:i * N, i * N + 1:(i + 1) * N] = I(N)
                BL[(i - 1) * N + 1:i * N, (i - 1) * N + 1:i * N] = I(N)
            end
            for j in 1:d
                AL[(d - 1) * N + 1:d * N, (j - 1) * N + 1:j * N] = -C[j]
            end
            BL[(d - 1) * N + 1:d * N, (d - 1) * N + 1:d * N] = C[d + 1]
            return filter(isfinite, eigvals(AL, BL))
        end
        rng = MersenneTwister(5)
        N = 12
        K = randn(rng, ComplexF64, N, N); C = randn(rng, ComplexF64, N, N)
        Mm = Matrix{ComplexF64}(I, N, N)
        ref = companion_eigvals([K, C, Mm])
        c, r = ref[5] + 0.15, 0.4
        inside = filter(x -> abs(x - c) < r, ref)
        # Re-running Beyn's extraction from the Ritz vectors stayed at the
        # quadrature error (2e-5 here) for every loop; the residual correction
        # converges in two.
        f = feastinit().fpm; f[8] = 16
        res = quietly(() -> feast_grcipev!([K, C, Mm], 2, c, r, 4, f))
        @test res.info == 0 && res.M == length(inside)
        @test all(l -> minimum(abs.(l .- inside)) < 1e-10, res.lambda)
        res = quietly(() -> feast_grcipev!([K, C, Mm], 2, c + 50, 0.4, 4, feastinit().fpm))
        @test res.info == 0 && res.M == 0
        rng = MersenneTwister(3)
        for d in (2, 3)
            P = [randn(rng, ComplexF64, N, N) for _ in 1:(d + 1)]
            ref = companion_eigvals(P)
            inside = filter(x -> abs(x - (0.2 - 0.1im)) < 0.7, ref)
            f = feastinit().fpm; f[8] = 64
            res = quietly(() -> feast_grcipev!(P, d, 0.2 - 0.1im, 0.7, 12, f))
            @test res.info == 0 && res.M == length(inside)
            @test all(l -> minimum(abs.(l .- inside)) < 1e-9, res.lambda)
        end
    end

    @testset "Threaded drivers count loops like the serial ones" begin
        T2 = Matrix(SymTridiagonal(fill(2.0, 60), fill(-1.0, 59)))
        B2 = Matrix{Float64}(I, 60, 60)
        # An interval far from the spectrum is settled by the first sweep,
        # which is loop 0, not 1.
        for solve in ((f) -> pfeast_sygv!(copy(T2), copy(B2), 10.0, 11.0, 8, f),
                      (f) -> pfeast_scsrgv!(sparse(T2), sparse(B2), 10.0, 11.0, 8, f))
            r = quietly(() -> solve(feastinit().fpm))
            @test r.info == 0 && r.M == 0 && r.loop == 0
        end
        # fpm[4] refinements follow the first sweep, as in serial.
        f = feastinit().fpm; f[4] = 1; f[2] = 3
        serial = quietly(() -> feast_sygv!(copy(T2), copy(B2), 3.9, 5.0, 8, copy(f)))
        threaded = quietly(() -> pfeast_sygv!(copy(T2), copy(B2), 3.9, 5.0, 8, copy(f)))
        @test serial.loop == threaded.loop == 1
        # They and feast_parallel start from the serial solver's subspace and
        # converge in as many loops. feast_parallel used to check that subspace
        # with independent probes, as if a caller had supplied it, which took
        # one or two extra loops. (Here the residual is 5.7e-11 after loop 1
        # and about 1e-15 after loop 2, far from the 1e-12 tolerance.)
        lam = eigvals(Symmetric(T2))
        a, b = (lam[10] + lam[11]) / 2, (lam[15] + lam[16]) / 2
        reference = quietly(() -> feast_sygv!(copy(T2), copy(B2), a, b, 10, feastinit().fpm))
        @test reference.info == 0 && reference.M == 5
        for solve in ((f) -> pfeast_sygv!(copy(T2), copy(B2), a, b, 10, f),
                      (f) -> pfeast_scsrgv!(sparse(T2), sparse(B2), a, b, 10, f),
                      (f) -> feast_parallel(T2, B2, (a, b); M0=10, fpm=f))
            r = quietly(() -> solve(feastinit().fpm))
            @test r.info == 0 && r.M == 5
            @test r.loop == reference.loop
        end
    end

    @testset "A seed check that runs out of loops keeps the converged pairs" begin
        N = 100
        A = Matrix(SymTridiagonal(fill(2.0, N), fill(-1.0, N - 1)))
        lam = eigvals(Symmetric(A))
        a, b = (lam[20] + lam[21]) / 2, (lam[28] + lam[29]) / 2
        seed = randn(MersenneTwister(1), N, 12)
        for maxloop in (3, 4)
            f = feastinit().fpm; f[4] = maxloop
            r = quietly(() -> feast_heev!(ComplexF64.(A), a, b, 12, f; initial_subspace=seed))
            converged = r.res .<= 1e-12
            @test r.info == 5 || (r.info == 0 && all(converged))
            @test sort(r.lambda[converged]) ≈ lam[21:28] atol=1e-10
            f = feastinit().fpm; f[4] = maxloop
            r = quietly(() -> feast_syev!(A, a, b, 12, f; initial_subspace=seed))
            converged = r.res .<= 1e-12
            @test r.info == 5 || (r.info == 0 && all(converged))
            @test sort(r.lambda[converged]) ≈ lam[21:28] atol=1e-10
        end
    end

    @testset "Seed verification preserves an unresolved boundary candidate" begin
        A = Matrix(Diagonal([0.0, 0.9999, 1.00001, 1.00011]))
        seed = repeat([1.0, 0.0, 0.0, 0.0], 1, 3)
        f = feastinit().fpm
        f[2] = 4
        f[4] = 1
        f[16] = 1
        r = quietly(() -> feast_syev!(A, -1.0, 1.0, 3, f; initial_subspace=seed))
        @test r.info == 5 && r.M == 2
        @test r.lambda[1] ≈ 0.0 atol=1e-12
        @test 0.9999 < r.lambda[2] < 1.0
        @test r.res[2] > 1e-12
    end

    @testset "A converged seed needs time for verification" begin
        A = Matrix(Diagonal([0.0, 0.5, 3.0, 4.0]))
        partial_seed = repeat([1.0, 0.0, 1.0, 0.0], 1, 3)
        full_seed = Matrix(1.0I, 4, 4)
        f = feastinit().fpm
        f[2] = 8
        f[4] = 1
        f[16] = 1
        Ac = ComplexF64.(A)
        for solve in ((m, seed) -> feast_syev!(A, -1.0, 1.0, m, copy(f); initial_subspace=seed),
                      (m, seed) -> feast_heev!(Ac, -1.0, 1.0, m, copy(f); initial_subspace=seed),
                      (m, seed) -> feast_geev!(Ac, 0.0 + 0im, 1.0, m, copy(f); initial_subspace=seed),
                      (m, seed) -> feast_scsrev_complex!(sparse(Ac), 0.0 + 0im, 1.0, m, copy(f);
                                                        initial_subspace=seed))
            r = quietly(() -> solve(3, partial_seed))
            @test r.info == 5 && r.M == 1 && r.loop == 1
            @test only(r.res) <= 1e-12
            r = quietly(() -> solve(4, full_seed))
            @test r.info == 0 && r.M == 2
            @test r.lambda ≈ [0.0, 0.5] atol=1e-12
        end
    end

    @testset "Non-normal problems with more eigenvalues than M0 are not called empty" begin
        # Triangular with large off-diagonals: far from normal. With fewer
        # columns than eigenvalues inside, the subspace stalls on part of the
        # in-region invariant subspace, or settles on an eigenvalue just
        # outside the circle, near a node, that the filter amplifies more than
        # the region itself. Its oblique Ritz pairs used to be screened out
        # against that node's filter value, or the settled pair taken as proof
        # of an empty region, ending in info = 0 with M = 0. Julia's random
        # streams differ between releases, so the matrices come from SplitMix64
        # with Irwin-Hall normals: every case failed on every supported release.
        # Only "not empty" is asserted: these matrices are so far from normal
        # that σmin(zI - A) on the circle is below eps * ‖A‖ (about 1e-20 of
        # it), so rounding alone can move an eigenvalue across the circle and
        # the number of eigenvalues inside is not determined in floating point.
        splitmix(s) = (s += 0x9e3779b97f4a7c15; z = s;
                       z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9;
                       z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb; (s, z ⊻ (z >> 31)))
        function nonnormal(seed)
            state = Ref(UInt64(seed))
            uniform() = ((state[], z) = splitmix(state[]); (z >> 11) * 2.0^-53)
            normal() = sum(uniform() for _ in 1:12) - 6
            cnormal() = complex(normal(), normal()) / sqrt(2)
            N = uniform() < 0.5 ? 40 : 100
            A = [cnormal() for _ in 1:N, _ in 1:N]
            A = triu(A) + 5 * triu([cnormal() for _ in 1:N, _ in 1:N], 1)
            lam = diag(A)
            c = lam[1 + floor(Int, uniform() * N)] + 0.1 * cnormal()
            r = (0.1 + uniform()) * max(0.3, 0.3 * maximum(abs, lam))
            return A, c, r, count(x -> abs(x - c) < r, lam)
        end
        for (seed, M0, ne) in ((1, 1, 16), (1, 2, 8), (19, 1, 8), (52, 1, 16), (80, 1, 8))
            A, c, r, nt = nonnormal(seed)
            @test nt > M0
            f = feastinit().fpm; f[8] = ne
            res = quietly(() -> feast_gegv!(A, nothing, c, r, M0, f))
            @test !(res.info == 0 && res.M == 0)
        end
        # A mixture of eigenvectors from either side is still screened out.
        G = Matrix(Diagonal(ComplexF64.(collect(1.0:10.0))))
        r = quietly(() -> feast_general(G, 4.5 + 0im, 0.4; M0=4))
        @test r.info == 0 && r.M == 0
        # An empty disc beside an eigenvalue just outside it is still empty:
        # M0 = 1 settles on that eigenvalue, whose filter value (0.36, then
        # 0.29) is below the filter's floor inside the disc (about 0.45).
        for d1 in (1.01, 1.015)
            E = Matrix(Diagonal(ComplexF64[d1, 3, 5, 7, -4, 2im]))
            r = quietly(() -> feast_general(E, 0.0 + 0im, 1.0; M0=1))
            @test r.info == 0 && r.M == 0
        end
    end

    @testset "A settled subspace beside an empty disc is read pair by pair" begin
        # Unit circle, 8 Gauss nodes: the filter's floor inside is 0.45 of its
        # central value; ρ(1.3) = 0.018, ρ(1.02i) = 0.47 and ρ(3) = 5e-4.
        fpm = feastinit().fpm; fpm[8] = 8; feastdefault!(fpm)
        c = 0.0 + 0im
        ct = feast_gcontour(c, 1.0, fpm)
        boundary = FeastKit._feast_region_boundary(Float64, c, 1.0, fpm)
        ρc = FeastKit._feast_filter_response(c, ct.Zne, ct.Wne, false)
        looks_empty(ritz, res, ratio) = FeastKit._feast_region_looks_empty(
            ritz, c, 1.0, ct.Zne, ct.Wne, false, minimum(abs, ritz);
            real_spectrum=false, nearest_residual=i -> res[i], tol=1e-12,
            boundary=boundary, filtered_ratio=ratio)[1]
        # A converged pair below the floor shows the disc empty. Checking an
        # unconverged pair first used to overwrite the floor with that pair's
        # accuracy bound (a variable shared with a nested function), hiding it.
        @test looks_empty(ComplexF64[3.0, 1.3], [1e-3, 1e-15], nothing)
        # So does a subspace the last sweep kept less of than the floor ...
        @test looks_empty(ComplexF64[3.0, 1.02im], [1e-3, 1e-15], 0.3ρc)
        # ... but not one kept more strongly, or an unconverged nearest pair.
        @test !looks_empty(ComplexF64[3.0, 1.02im], [1e-3, 1e-15], 0.6ρc)
        @test !looks_empty(ComplexF64[3.0, 1.02im], [1e-3, 1e-6], 0.3ρc)
    end

    @testset "Defective eigenvalues keep their algebraic multiplicity" begin
        # The Ritz vectors of a Jordan block of size k converge to within about
        # eps^(1/k) of each other. Swept as they were, the next rank compression
        # kept only one direction of the block: the general solver reported
        # fewer eigenvalues than the disc holds, as a success, and the
        # complex-symmetric ones -- whose Rayleigh quotient xᵀAx / xᵀx is 0/0 at
        # such an eigenvector -- stopped converging. The matrices are built
        # without random numbers, so they are the same on every Julia release.
        λ0 = 0.3 + 0.1im
        n = 12
        others(k) = [2.0 + 0.25 * j * im for j in 1:(n - k)]
        jordan(k) = (J = Matrix(Diagonal(vcat(fill(λ0, k), others(k))));
                     for i in 1:k-1; J[i, i+1] = 1; end; J)
        U = Matrix(qr([cos(i * j) + im * sin(i + 2j) for i in 1:n, j in 1:n]).Q)
        for k in (2, 3)
            A = U * jordan(k) * U'
            for M0 in (k + 1, 2k + 2)
                r = quietly(() -> feast_geev!(copy(A), λ0, 0.5, M0, feastinit().fpm))
                @test r.info == 0 && r.M == k
                @test all(x -> abs(x - λ0) < 1e-3, r.lambda)
            end
            # With M0 equal to the multiplicity the count cannot be certified.
            r = quietly(() -> feast_geev!(copy(A), λ0, 0.5, k, feastinit().fpm))
            @test r.info == Int(FeastKit.Feast_ERROR_M0)
        end

        # Complex-symmetric Jordan blocks: S = T J T⁻¹ with T = (I + iK)/√2, K
        # the exchange matrix, is complex symmetric, and a real orthogonal
        # similarity keeps it so.
        function cs_jordan(k)
            Nk = diagm(1 => ones(k - 1)); K = reverse(Matrix(1.0I, k, k), dims=1)
            D = Matrix{ComplexF64}(Diagonal(vcat(fill(λ0, k), others(k))))
            D[1:k, 1:k] .+= (Nk + K * Nk * K) / 2 + im * (K * Nk - Nk * K) / 2
            return D
        end
        Qr = Matrix(qr([cos(i * j) + sin(i + 2j) for i in 1:n, j in 1:n]).Q)
        for k in (2, 3), M0 in (k + 1, 2k + 2)
            A = transpose(Qr) * cs_jordan(k) * Qr
            A = (A + transpose(A)) / 2
            rd = quietly(() -> feast_geev_complex_sym!(copy(A), λ0, 0.5, M0, feastinit().fpm))
            rs = quietly(() -> feast_scsrgv_complex!(sparse(A), sparse(ComplexF64(1) * I, n, n),
                                                    λ0, 0.5, M0, feastinit().fpm))
            for r in (rd, rs)
                @test r.info == 0 && r.M == k
                @test all(x -> abs(x - λ0) < 1e-3, r.lambda)
            end
        end
        # Banded: plane rotations on neighbouring indices keep the bandwidth small.
        function plane_rotations(θ)
            R = Matrix(1.0I, n, n)
            for off in (0, 1), i in (1 + off):2:(n - 1)
                G = Matrix(1.0I, n, n)
                G[i, i] = G[i+1, i+1] = cos(θ * i)
                G[i, i+1] = sin(θ * i); G[i+1, i] = -sin(θ * i)
                R = R * G
            end
            return R
        end
        R = plane_rotations(0.7)
        for k in (2, 3), M0 in (k + 1, 2k + 2)
            A = transpose(R) * cs_jordan(k) * R
            A = (A + transpose(A)) / 2
            bw = maximum(abs(i - j) for i in 1:n, j in 1:n if abs(A[i, j]) > 1e-14)
            r = quietly(() -> feast_sbev_complex!(FeastKit.full_to_banded(A, bw), bw,
                                                  λ0, 0.5, M0, feastinit().fpm))
            @test r.info == 0 && r.M == k
        end
    end

    @testset "Structured matrices keep sparse storage" begin
        n = 200
        T3 = SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1))
        S3 = sparse(T3)
        for A in (T3, Tridiagonal(T3), Diagonal(ones(4)), S3', Symmetric(S3))
            @test FeastKit._materialize_matrix(A) isa SparseMatrixCSC{Float64,Int}
        end
        @test FeastKit._materialize_matrix(T3; dense=true) isa Matrix{Float64}
        ev = eigvals(T3)
        iv = ((ev[5] + ev[6]) / 2, (ev[12] + ev[13]) / 2)
        for A in (T3, S3', Symmetric(S3), Tridiagonal(T3))
            r = quietly(() -> feast(A, iv; M0=12))
            @test r.info == 0
            @test sort(r.lambda) ≈ ev[6:12] atol=1e-10
        end
        @test FeastKit._gershgorin_bounds(Symmetric(S3)) == FeastKit._gershgorin_bounds(S3)
        # Mixed precision only exists for dense storage and still gets it.
        r = quietly(() -> feast(T3, iv; M0=12, mixed_precision=true))
        @test r.info == 0 && r.M == 7
    end

    @testset "Element types, small contours and contour helpers" begin
        L = Matrix(SymTridiagonal(fill(2.0, 30), fill(-1.0, 29)))
        count_inside = count(x -> 0 < x < 1, eigvals(Symmetric(L)))
        @test_throws ArgumentError feast(BigFloat.(L), (big(0.0), big(1.0)); M0=20)
        @test_throws ArgumentError feast_general(BigFloat.(L), big(0.5) + 0im, big(0.5); M0=20)
        @test_throws ArgumentError feast_syev!(BigFloat.(L), big(0.0), big(1.0), 20, feastinit().fpm)
        r = quietly(() -> feast(Float16.(L), (Float16(0), Float16(1)); M0=20))
        @test r.info == 0 && r.M == count_inside
        # FEAST accepts one- and two-point half contours.
        for npts in (1, 2)
            f = feastinit().fpm; f[2] = npts
            r = quietly(() -> feast_syev!(L, 0.0, 1.0, 20, f))
            @test r.info == 0 && r.M == count_inside
        end
        # Custom contour weights are spectrally accurate on smooth curves.
        for npts in (8, 16)
            nodes = [complex(5 + 3cos(2π * (k - 1) / npts), 3sin(2π * (k - 1) / npts)) for k in 1:npts]
            c = feast_customcontour(nodes, feastinit().fpm)
            @test abs(feast_grationalx(c.Zne, c.Wne, [complex(5.0, 0.0)])[1] - 1) < 1e-12
        end
        center = 1e12
        unit_nodes = [cispi(2 * (j + 0.5) / 32) for j in 0:31]
        nodes = center .+ unit_nodes
        contour = feast_customcontour(nodes, feastinit().fpm)
        centered = feast_customcontour(nodes .- center, feastinit().fpm)
        @test contour.Wne ≈ centered.Wne rtol=1e-12
        # The equally damped exterior pair can persist in the trial space.
        # At 1e13 its residual even meets the unadjusted relative tolerance.
        for shift in (1e12, 1e13), storage in (Matrix, sparse)
            shifted = feast_customcontour(shift .+ unit_nodes, feastinit().fpm)
            A = storage(Diagonal(complex.(shift .+ [0.25, 1.125, 1.25, -1.25])))
            r = quietly(() -> feast(A, shifted; M0=3))
            @test r.converged
            @test r.lambda .- shift ≈ [0.25] atol=5eps(shift)
        end
        A = Matrix(Diagonal(complex.(center .+ [0.25, 1.125, 1.25, -1.5])))
        r = quietly(() -> feast(A, contour; M0=3))
        @test r.converged
        @test r.lambda .- center ≈ [0.25] atol=5eps(center)
        nodes = [cispi(2 * (j + 0.5) / 56) for j in 0:55]
        A = Matrix(Diagonal([0.0, 2.0]))
        r = quietly(() -> FeastKit.feast_custom_contour(nodes, A, (-1.0, 1.0); M0=2))
        @test r.info == 0 && r.M == 1
        @test only(r.lambda) ≈ 0.0 atol=1e-12
        @test_throws ArgumentError feast_customcontour(ComplexF64[1, -1], feastinit().fpm)
        # A zero aspect ratio no longer flattens the contour onto the real axis.
        f = zeros(Int, 64); f[2] = 8
        @test all(z -> imag(z) > 0, feast_contour(-1.0, 1.0, f).Zne)
        f = feastinit().fpm; f[16] = 2
        @test_throws ArgumentError feast_gcontour(0.0 + 0im, 1.0, f)
        # The polygon test handles concave contours.
        P = ComplexF64[0, 2, 2 + 2im, 1 + 0.2im, 2im]
        @test !FeastKit.feast_inside_contourx(1.0 + 1.5im, P, P)
        @test FeastKit.feast_inside_contourx(0.5 + 0.5im, P, P)
        @test_throws ArgumentError FeastKit.feast_inside_contourx(0.0im, ComplexF64[], ComplexF64[])
        estimate = redirect_stdout(devnull) do
            FeastKit.feast_memory_estimate(3_000_000_000, 1_000_000_000, Float64)
        end
        @test estimate > 0
    end

    @testset "Zolotarev filters include their constant term" begin
        f = feastinit().fpm; f[2] = 4; f[16] = 2; feastdefault!(f)
        @test FeastKit._feast_filter_offset(Float64, f, false) ≈ real(FeastKit.zolotarev_point(4, 0)[2])
        @test FeastKit._feast_filter_offset(Float64, f, true) == 0
        g = feastinit().fpm; feastdefault!(g)
        @test FeastKit._feast_filter_offset(Float64, g, false) == 0
        L = Matrix(SymTridiagonal(fill(2.0, 60), fill(-1.0, 59)))
        expected = filter(x -> 0.5 < x < 1.5, eigvals(Symmetric(L)))
        for solve in ((p) -> feast_syev!(L, 0.5, 1.5, 30, p),
                      (p) -> feast_heev!(ComplexF64.(L), 0.5, 1.5, 30, p),
                      (p) -> pfeast_sygv!(copy(L), Matrix(1.0I, 60, 60), 0.5, 1.5, 30, p))
            p = feastinit().fpm; p[2] = 8; p[16] = 2
            r = quietly(() -> solve(p))
            @test r.info == 0
            @test sort(r.lambda) ≈ expected atol=1e-10
        end
        # The constant term keeps a small copy of every eigenvector in the first
        # filtered block. When the spectrum spans eleven orders of magnitude,
        # that is enough to drag the first Ritz values far out of a tiny
        # interval; it must not be read as an empty interval.
        rng = MersenneTwister(7)
        d = filter(x -> !(1.2e-5 < x < 2.5e-5), exp.(range(log(1e-6), log(1e5), length=199)))
        d = vcat(1.5e-5, d)
        Q = Matrix(qr(randn(rng, length(d), length(d))).Q)
        W = Matrix(Symmetric(Q * Diagonal(d) * Q'))
        for M0 in (2, 6)
            p = feastinit().fpm; p[2] = 16; p[16] = 2
            r = quietly(() -> feast_syev!(W, 1.4e-5, 2.2e-5, M0, p))
            @test r.info == 0 && r.M == 1
            @test r.lambda[1] ≈ 1.5e-5 rtol=1e-6
        end
    end

    @testset "Backend routing" begin
        T2 = Matrix(SymTridiagonal(fill(2.0, 40), fill(-1.0, 39)))
        B2 = Matrix{Float64}(I, 40, 40)
        @test_throws ArgumentError feast_with_backend(T2, B2, (0.1, 0.5), :threads, 12,
                                                      feastinit().fpm, nothing, false;
                                                      strict_backend=true)
        if Threads.nthreads() > 1
            # The comparison reports backends that cannot run a problem.
            results = redirect_stdout(devnull) do
                quietly(() -> feast_parallel_comparison(ComplexF64.(T2), ComplexF64.(B2),
                                                        (0.1, 0.5), 12))
            end
            @test haskey(results, :serial) && !haskey(results, :threads)
        end
        # feast_parallel's adapter factorizes once per solve.
        r = quietly(() -> feast_parallel(T2, B2, (0.1, 0.5); M0=12))
        @test r.info == 0
        @test sort(r.lambda) ≈ filter(x -> 0.1 < x < 0.5, eigvals(Symmetric(T2))) atol=1e-10
    end
end
