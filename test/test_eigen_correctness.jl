using FeastKit
using Test
using LinearAlgebra
using SparseArrays
using Random

# Ground-truth checks for every storage path.
#
# The rest of the suite mostly asserts `info == 0`, `M > 0`, and that two
# solvers agree with each other. That is not enough: it passed while the banded
# and matrix-free paths returned eigenvector columns of norm 1e-12, whose
# residual ||Aq - λq|| is trivially tiny, so a broken solve looked converged.
# Every case here compares against a dense `eigen`/`eigvals` reference and
# checks the eigenvectors themselves.

"""
    check_symmetric_result(result, A, B, interval; tol, expected)

Assert a real symmetric FEAST result against a dense reference:
the count inside the interval, each eigenvalue, unit-norm eigenvectors, and the
true generalized residual `||A q - λ B q|| / ||q||` recomputed from `A` and `B`
rather than trusting `result.res`.
"""
function check_symmetric_result(result, A, B, interval; tol = 1.0e-8,
                                expected = nothing)
    Emin, Emax = interval
    reference = B === nothing ? eigvals(Symmetric(Matrix(A))) :
                                eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(B)))
    inside = filter(λ -> Emin <= λ <= Emax, reference)
    expected === nothing || @test length(inside) == expected

    @test result.info == 0
    @test result.M == length(inside)
    @test isapprox(sort(result.lambda[1:result.M]), sort(inside); atol = tol)

    for j in 1:result.M
        q = result.q[:, j]
        qnorm = norm(q)
        # A near-zero column makes any absolute residual look converged.
        @test isapprox(qnorm, 1.0; atol = 1.0e-8)
        Bq = B === nothing ? q : Matrix(B) * q
        @test norm(Matrix(A) * q - result.lambda[j] * Bq) / qnorm < tol
        # The reported residual must agree with the recomputed one.
        @test result.res[j] < tol
    end
    @test result.epsout < tol
    return nothing
end

fresh_fpm() = (v = zeros(Int, 64); feastinit!(v); v)

@testset "Eigenpair correctness against dense reference" begin

    @testset "Real symmetric, all storage paths agree with eigen" begin
        n = 60
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        Bgen = Matrix(SymTridiagonal(fill(4.0, n), fill(1.0, n - 1)))
        Bid = Matrix{Float64}(I, n, n)

        std_ref = eigvals(Symmetric(A))
        std_interval = (std_ref[1] - 1.0e-9, std_ref[4] + 1.0e-9)
        gen_ref = eigvals(Symmetric(A), Symmetric(Bgen))
        gen_interval = (gen_ref[1] - 1.0e-9, gen_ref[4] + 1.0e-9)

        # M0 deliberately exceeds the number of eigenvalues in the interval:
        # that is the case that used to produce spurious Ritz pairs.
        M0 = 10

        @testset "dense standard" begin
            check_symmetric_result(
                feast_syev!(copy(A), std_interval..., M0, fresh_fpm()),
                A, nothing, std_interval; expected = 4)
        end

        @testset "dense generalized" begin
            check_symmetric_result(
                feast_sygv!(copy(A), copy(Bgen), gen_interval..., M0, fresh_fpm()),
                A, Bgen, gen_interval; expected = 4)
        end

        @testset "sparse standard" begin
            check_symmetric_result(
                feast_scsrev!(sparse(A), std_interval..., M0, fresh_fpm()),
                A, nothing, std_interval; expected = 4)
        end

        @testset "sparse generalized" begin
            check_symmetric_result(
                feast_scsrgv!(sparse(A), sparse(Bgen), gen_interval..., M0, fresh_fpm()),
                A, Bgen, gen_interval; expected = 4)
        end

        @testset "banded standard" begin
            check_symmetric_result(
                feast_sbev!(full_to_banded(A, 1), 1, std_interval..., M0, fresh_fpm()),
                A, nothing, std_interval; expected = 4)
        end

        @testset "banded generalized" begin
            # The banded path drives feast_srci!, whose residual used to ignore B
            # entirely and never normalized the Ritz vectors.
            check_symmetric_result(
                feast_sbgv!(full_to_banded(A, 1), full_to_banded(Bgen, 1), 1, 1,
                            gen_interval..., M0, fresh_fpm()),
                A, Bgen, gen_interval; expected = 4)
        end

        @testset "high-level feast" begin
            check_symmetric_result(feast(A, std_interval; M0 = M0),
                                   A, nothing, std_interval; expected = 4)
            check_symmetric_result(feast(A, Bgen, gen_interval; M0 = M0),
                                   A, Bgen, gen_interval; expected = 4)
            check_symmetric_result(feast(sparse(A), std_interval; M0 = M0),
                                   A, nothing, std_interval; expected = 4)
        end

        @testset "dense standard with B = I passed explicitly" begin
            check_symmetric_result(
                feast_sygv!(copy(A), copy(Bid), std_interval..., M0, fresh_fpm()),
                A, Bid, std_interval; expected = 4)
        end
    end

    @testset "Random symmetric matrix, interior interval" begin
        # A random dense matrix has no structure to hide a bad projection behind.
        rng = MersenneTwister(20240607)
        n = 60
        A = Matrix(Symmetric(randn(rng, n, n)))
        reference = eigvals(Symmetric(A))
        interval = (reference[10] - 1.0e-6, reference[15] + 1.0e-6)
        check_symmetric_result(feast(A, interval; M0 = 12), A, nothing, interval;
                               tol = 1.0e-7, expected = 6)
    end

    @testset "Matrix-free path matches the dense reference" begin
        n = 60
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        Bgen = Matrix(SymTridiagonal(fill(4.0, n), fill(1.0, n - 1)))

        A_op = LinearOperator{Float64}((y, x) -> mul!(y, A, x), (n, n); issymmetric = true)
        id_op = LinearOperator{Float64}((y, x) -> copyto!(y, x), (n, n); issymmetric = true)
        B_op = LinearOperator{Float64}((y, x) -> mul!(y, Bgen, x), (n, n); issymmetric = true)

        std_ref = eigvals(Symmetric(A))
        std_interval = (std_ref[1] - 1.0e-9, std_ref[4] + 1.0e-9)
        std_solve = (dest, z, rhs) -> (dest .= (z * I - A) \ ComplexF64.(rhs); dest)
        check_symmetric_result(
            feast_matfree_srci!(A_op, id_op, std_interval, 10; linear_solver = std_solve),
            A, nothing, std_interval; expected = 4)

        gen_ref = eigvals(Symmetric(A), Symmetric(Bgen))
        gen_interval = (gen_ref[1] - 1.0e-9, gen_ref[4] + 1.0e-9)
        # The callback owns the B multiply on the right-hand side.
        gen_solve = (dest, z, rhs) -> (dest .= (z * Bgen - A) \ (Bgen * ComplexF64.(rhs)); dest)
        check_symmetric_result(
            feast_matfree_srci!(A_op, B_op, gen_interval, 10; linear_solver = gen_solve),
            A, Bgen, gen_interval; expected = 4)
    end

    @testset "Complex Hermitian matches the dense reference" begin
        rng = MersenneTwister(11)
        n = 50
        A = randn(rng, ComplexF64, n, n)
        A = (A + A') / 2
        reference = eigvals(Hermitian(A))
        interval = (reference[5] - 1.0e-9, reference[8] + 1.0e-9)

        result = feast(Hermitian(A), interval; M0 = 10)
        @test result.info == 0
        @test result.M == 4
        @test isapprox(sort(result.lambda[1:result.M]),
                       sort(reference[5:8]); atol = 1.0e-8)
        for j in 1:result.M
            q = result.q[:, j]
            @test isapprox(norm(q), 1.0; atol = 1.0e-8)
            @test norm(A * q - result.lambda[j] * q) / norm(q) < 1.0e-8
        end
    end

    @testset "General non-Hermitian matches the dense reference" begin
        rng = MersenneTwister(42)
        n = 50
        A = randn(rng, ComplexF64, n, n)
        reference = eigvals(A)
        center, radius = 0.0 + 0.0im, 2.0
        inside = filter(λ -> abs(λ - center) <= radius, reference)

        result = feast_general(A, Matrix{ComplexF64}(I, n, n), center, radius;
                               M0 = 2 * length(inside) + 4)
        @test result.info == 0
        @test result.M == length(inside)
        for j in 1:result.M
            q = result.q[:, j]
            @test isapprox(norm(q), 1.0; atol = 1.0e-8)
            @test minimum(abs.(reference .- result.lambda[j])) < 1.0e-8
            @test norm(A * q - result.lambda[j] * q) / norm(q) < 1.0e-8
        end
    end

    @testset "A saturated subspace is reported as M0 too small" begin
        # FEAST's contract is that M0 exceeds the number of eigenvalues in the
        # interval. When it does not, the trial subspace is saturated: every
        # Ritz pair is inside the interval and there is no way to tell whether
        # eigenvalues were missed. Returning that as success or as plain
        # non-convergence hides a wrong answer -- the interval below holds
        # roughly 480 eigenvalues and M0 is 20.
        n = 4000
        A = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
        M0 = 20
        result = feast_scsrev!(A, 0.05, 0.35, M0, fresh_fpm())

        @test result.M == M0
        @test result.info == Int(Feast_ERROR_M0)

        # A correctly sized subspace on the same matrix is unaffected.
        lam(k) = 4 * sin(k * pi / (2 * (n + 1)))^2
        ok = feast_scsrev!(A, lam(100) - 1.0e-9, lam(108) + 1.0e-9, 20, fresh_fpm())
        @test ok.info == 0
        @test ok.M == 9
        @test ok.M < 20
    end

    @testset "feast_estimate_count sizes M0 before the solve" begin
        n = 4000
        A = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
        exact(lo, hi) = count(k -> lo <= 4 * sin(k * pi / (2 * (n + 1)))^2 <= hi, 1:n)

        # The estimator is a statistical trace estimate, so its relative error
        # shrinks as the count grows -- which is the useful direction, since a
        # large count is what saturates M0 and wastes the whole loop budget.
        big_lo, big_hi = 0.05, 0.35
        big_true = exact(big_lo, big_hi)
        @test big_true > 400
        big_est = feast_estimate_count(A, (big_lo, big_hi); nprobe = 32)
        @test isapprox(big_est, big_true; rtol = 0.15)

        # Sizing M0 from the estimate clears the saturation that made the same
        # interval return Feast_ERROR_M0 at M0 = 20. (This interval packs ~480
        # eigenvalues into a tight cluster, so the residual does not reach the
        # default 1e-12 -- the point here is that the subspace is no longer the
        # binding constraint.)
        M0 = ceil(Int, 1.5 * big_est)
        sized = feast_scsrev!(A, big_lo, big_hi, M0, fresh_fpm())
        @test sized.info != Int(Feast_ERROR_M0)
        @test sized.M < M0
        @test isapprox(sized.M, big_true; rtol = 0.05)

        # On a well-separated count the sized solve converges outright.
        lam(k) = 4 * sin(k * pi / (2 * (n + 1)))^2
        mid_lo, mid_hi = lam(100) - 1.0e-9, lam(140) + 1.0e-9
        mid_true = exact(mid_lo, mid_hi)
        mid_est = feast_estimate_count(A, (mid_lo, mid_hi); nprobe = 32)
        mid_M0 = ceil(Int, 1.5 * mid_est)
        mid = feast_scsrev!(A, mid_lo, mid_hi, mid_M0, fresh_fpm())
        @test mid.info == 0
        @test mid.M == mid_true

        # A small count is only good to roughly +-1, which the docstring says.
        lam(k) = 4 * sin(k * pi / (2 * (n + 1)))^2
        small_lo, small_hi = lam(100) - 1.0e-9, lam(108) + 1.0e-9
        small_est = feast_estimate_count(A, (small_lo, small_hi); nprobe = 32)
        @test abs(small_est - exact(small_lo, small_hi)) < 3
    end

    @testset "Iterative solves work at the default tolerance" begin
        # The inner GMRES tolerance used to be pinned at 10^-fpm[3] = 1e-12
        # from the very first sweep. Unpreconditioned GMRES cannot reach that
        # on a shifted system whose contour point sits near the spectrum, so
        # the driver gave up with M = 0 before completing a single loop -- the
        # reason every other iterative test in the suite has to pass an
        # explicit `solver_tol`. The inner tolerance now tracks the outer
        # residual, which is the property that makes IFEAST work.
        n = 100
        A = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        B = Matrix{Float64}(I, n, n)
        reference = eigvals(Symmetric(A))
        interval = (reference[1] - 1.0e-9, reference[8] + 1.0e-9)

        result = feast_sygv!(copy(A), copy(B), interval..., 16, fresh_fpm();
                             solver = :gmres, solver_maxiter = 2000,
                             solver_restart = 30)
        @test result.info == 0
        @test result.M == 8
        @test isapprox(sort(result.lambda[1:result.M]), reference[1:8]; atol = 1.0e-8)
        for j in 1:result.M
            q = result.q[:, j]
            @test isapprox(norm(q), 1.0; atol = 1.0e-8)
            @test norm(A * q - result.lambda[j] * q) / norm(q) < 1.0e-8
        end
    end

    @testset "Non-convergence is reported, not hidden" begin
        n = 60
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        reference = eigvals(Symmetric(A))
        interval = (reference[1] - 1.0e-9, reference[4] + 1.0e-9)

        # One refinement loop at a tolerance no single loop can reach must come
        # back as Feast_ERROR_NO_CONVERGENCE, not as success.
        fpm = fresh_fpm()
        fpm[4] = 1          # maxloop
        fpm[3] = 16         # 1e-16
        fpm[2] = 3          # a deliberately coarse contour
        result = feast_sbgv!(full_to_banded(A, 1),
                             full_to_banded(Matrix{Float64}(I, n, n), 0), 1, 0,
                             interval..., 8, fpm)
        @test result.info == Int(Feast_ERROR_NO_CONVERGENCE)
        @test result.epsout > 1.0e-16
    end

    @testset "fpm[10] chooses factorization storage without changing answers" begin
        n = 50
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        reference = eigvals(Symmetric(A))
        interval = (reference[1] - 1.0e-9, reference[4] + 1.0e-9)

        cached = fresh_fpm()
        cached[10] = 1
        uncached = fresh_fpm()
        uncached[10] = 0

        with_cache = feast_syev!(copy(A), interval..., 10, cached)
        without_cache = feast_syev!(copy(A), interval..., 10, uncached)
        @test with_cache.info == 0
        @test without_cache.info == 0
        @test with_cache.M == without_cache.M
        @test isapprox(with_cache.lambda, without_cache.lambda; atol = 1.0e-10)

        sparse_uncached = feast_scsrev!(sparse(A), interval..., 10, fresh_fpm())
        @test isapprox(sort(sparse_uncached.lambda), sort(with_cache.lambda);
                       atol = 1.0e-10)
    end

    @testset "Custom contour weights give a unit filter inside the contour" begin
        fpm = fresh_fpm()
        feastdefault!(fpm)
        center, radius = 5.0, 3.0
        for ne in (16, 32, 64)
            nodes = [complex(center + radius * cos(2π * (k - 1) / ne),
                             radius * sin(2π * (k - 1) / ne)) for k in 1:ne]
            contour = feast_customcontour(nodes, fpm)
            inside = feast_grationalx(contour.Zne, contour.Wne, [complex(center, 0.0)])[1]
            outside = feast_grationalx(contour.Zne, contour.Wne,
                                       [complex(center + 10 * radius, 0.0)])[1]
            # The Cauchy filter is ~1 inside and ~0 outside. Weights missing the
            # 1/(2πi) prefactor put this at ~0.38im for ne = 16.
            @test abs(inside - 1) < 5 / ne
            @test abs(outside) < 1.0e-8
        end
    end

    @testset "Contour builders resolve every parameter they read" begin
        # feastinit! fills fpm with the -111 sentinel; feastdefault! replaces it.
        # The builders used to trigger that resolution only when their point
        # count was still unset, so setting just fpm[2] (or fpm[8]) left
        # fpm[16], fpm[18] and fpm[19] at -111. fpm[18] is the ellipse aspect
        # ratio in percent, so -111 mirrors the contour, reverses its
        # orientation, and flips the sign of the rational filter.
        partial = fresh_fpm()
        partial[2] = 12
        resolved = fresh_fpm()
        feastdefault!(resolved)
        resolved[2] = 12

        cp = feast_contour(0.0, 1.0, partial)
        cr = feast_contour(0.0, 1.0, resolved)
        @test isapprox(cp.Zne, cr.Zne; atol = 1.0e-12)
        @test isapprox(cp.Wne, cr.Wne; atol = 1.0e-12)
        # The half-contour filter is ~+1 inside, never -1.
        @test isapprox(feast_rationalx(cp.Zne, cp.Wne, [0.5])[1], 1.0; atol = 1.0e-8)

        gpartial = fresh_fpm()
        gpartial[8] = 24
        gresolved = fresh_fpm()
        feastdefault!(gresolved)
        gresolved[8] = 24

        gp = feast_gcontour(0.0 + 0.0im, 1.0, gpartial)
        gr = feast_gcontour(0.0 + 0.0im, 1.0, gresolved)
        @test isapprox(gp.Zne, gr.Zne; atol = 1.0e-12)
        @test isapprox(gp.Wne, gr.Wne; atol = 1.0e-12)
        @test isapprox(feast_grationalx(gp.Zne, gp.Wne, [0.0 + 0.0im])[1], 1.0;
                       atol = 1.0e-8)
    end

    @testset "Custom contour drives a solve to the right eigenvalues" begin
        n = 40
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        reference = eigvals(Symmetric(A))
        Emin, Emax = reference[1] - 1.0e-6, reference[3] + 1.0e-6

        # Half-ellipse nodes matching what feast_contour would build, supplied
        # through the expert (x-suffix) entry point.
        ne = 12
        center = (Emin + Emax) / 2
        rad = (Emax - Emin) / 2
        nodes = ComplexF64[]
        weights = ComplexF64[]
        for k in 1:ne
            θ = π * (k - 0.5) / ne
            push!(nodes, center + rad * cos(θ) + im * rad * sin(θ))
            push!(weights, (rad * im * sin(θ) + rad * cos(θ)) / (2 * ne))
        end
        result = feast_syevx!(copy(A), Emin, Emax, 8, fresh_fpm(), nodes, weights)
        @test result.info == 0
        @test result.M == 3
        @test isapprox(sort(result.lambda[1:3]), reference[1:3]; atol = 1.0e-8)
    end

    @testset "Polynomial RCI finds the eigenvalues, not their reciprocals" begin
        # P(λ) = λ²I - diag(9, 16, 25), so the spectrum is ±3, ±4, ±5 with
        # eigenvectors e₁, e₂, e₃.
        #
        # The contour encloses 3, 4 and 5 but not the reciprocals 1/3, 1/4, 1/5.
        # That is the point: the moment pencil (A₀, A₁) must be solved as
        # A₁v = λA₀v, and assembling it the other way round returns 1/λ. A test
        # whose contour happens to contain both a value and its reciprocal --
        # anything centred on the origin, say -- cannot tell the two apart.
        roots = [3.0, 4.0, 5.0]
        coeffs_real = [Matrix(Diagonal(-roots .^ 2)), zeros(3, 3),
                       Matrix{Float64}(I, 3, 3)]
        coeffs = [ComplexF64.(C) for C in coeffs_real]
        center, radius = 4.0 + 0.0im, 1.5

        fpm = fresh_fpm()
        fpm[8] = 24     # full-contour integration points
        fpm[16] = 1     # trapezoidal, the accurate rule on a circle
        fpm[4] = 30     # refinement loops

        result = feast_grcipev!(coeffs, 2, center, radius, 3, copy(fpm))
        @test result.info == 0
        @test result.M == 3
        @test isapprox(sort(real.(result.lambda[1:result.M])), roots; atol = 1.0e-8)

        for j in 1:result.M
            q = result.q[:, j]
            @test isapprox(norm(q), 1.0; atol = 1.0e-8)
            λ = complex(result.lambda[j])
            Pq = (coeffs[1] + λ * coeffs[2] + λ^2 * coeffs[3]) * q
            @test norm(Pq) / norm(q) < 1.0e-8
        end

        # The real-coefficient entry point drives the same kernel.
        real_result = feast_srcipev!(coeffs_real, 2, center, radius, 3, copy(fpm))
        @test real_result.info == 0
        @test real_result.M == 3
        @test isapprox(sort(real.(real_result.lambda[1:real_result.M])), roots;
                       atol = 1.0e-8)
    end

    @testset "Polynomial RCI tolerates M0 above the count inside the contour" begin
        # P(λ) = λ²I - diag(1, 4, 9): spectrum ±1, ±2, ±3. The disc holds only
        # 2 and 3, so a trial subspace of width M0 = 3 is one wider than the
        # number of eigenvalues to be found -- the ordinary situation, since a
        # caller does not know the count in advance.
        #
        # Without rank truncation the reduced moment pencil is singular in that
        # case. The eigenvalues still come out roughly right, but the residual
        # sits on a floor far above the requested tolerance, and adding contour
        # points makes it worse rather than better, because the extra
        # resolution only sharpens a rank-deficient pencil.
        # P(λ) = λ²I - diag(1², 2², …, 40²), so the spectrum is ±1 … ±40 with
        # eigenvectors e₁ … e₄₀. N is well above M0 so the truncation is real:
        # S0 is 40x8 with a true rank of 3.
        nn = 40
        coeffs = [ComplexF64.(Matrix(Diagonal([-(float(j))^2 for j in 1:nn]))),
                  zeros(ComplexF64, nn, nn),
                  ComplexF64.(Matrix{Float64}(I, nn, nn))]
        center, radius = 5.0 + 0.0im, 1.2
        expected = [4.0, 5.0, 6.0]

        for ne in (16, 24, 32, 48, 64)
            fpm = fresh_fpm()
            fpm[8] = ne
            fpm[16] = 1
            fpm[4] = 20
            result = feast_grcipev!(coeffs, 2, center, radius, 8, fpm)
            @test result.M == 3
            @test isapprox(sort(real.(result.lambda[1:result.M])), expected;
                           atol = 1.0e-10)
            @test result.info == 0
            # Adding contour points must never make the answer worse. The
            # projected moment pencil this replaced degraded from 3e-9 to 3e-4
            # over exactly this sweep of ne.
            @test result.epsout < 1.0e-11
        end
    end

    @testset "Polynomial RCI keeps complex eigenvalues" begin
        # Damped oscillator P(λ) = K + λC + λ²I, diagonal, so each degree of
        # freedom contributes the roots of λ² + 2αλ + (α² + β²) = -α ± iβ.
        # Every eigenvalue is genuinely complex: a result type carrying only
        # real eigenvalues silently drops the oscillation frequency, which is
        # the whole quantity a damped PEP is solved for.
        α = 0.5
        βs = [2.0, 3.0, 4.0]
        K = Matrix(Diagonal(α^2 .+ βs .^ 2))
        C = Matrix(Diagonal(fill(2α, 3)))
        coeffs = [ComplexF64.(K), ComplexF64.(C),
                  ComplexF64.(Matrix{Float64}(I, 3, 3))]
        expected = [complex(-α, β) for β in βs]

        # A disc around the upper-half roots. The conjugates and the
        # reciprocals both sit well outside it.
        center, radius = complex(-α, 3.0), 1.5

        fpm = fresh_fpm()
        fpm[8] = 32
        fpm[16] = 1
        fpm[4] = 30

        result = feast_grcipev!(coeffs, 2, center, radius, 3, copy(fpm))
        @test result.info == 0
        @test result.M == 3
        @test eltype(result.lambda) <: Complex
        @test isapprox(sort(result.lambda[1:result.M]; by = imag), expected;
                       atol = 1.0e-8)
        # Guard the specific regression: real eigenvalues would pass an
        # isapprox on real parts alone.
        @test all(abs(imag(λ)) > 1 for λ in result.lambda[1:result.M])

        for j in 1:result.M
            q = result.q[:, j]
            λ = result.lambda[j]
            @test isapprox(norm(q), 1.0; atol = 1.0e-8)
            Pq = (coeffs[1] + λ * coeffs[2] + λ^2 * coeffs[3]) * q
            @test norm(Pq) / norm(q) < 1.0e-8
        end
    end
end

@testset "Threaded backend agrees with serial" begin
    # The dense threaded backend was disabled for disagreeing with serial. The
    # cause was a phase-collapsed real() of the Ritz vectors, not the threading;
    # this asserts the agreement that was missing.
    if Threads.nthreads() > 1
        n = 60
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        Bgen = Matrix(SymTridiagonal(fill(4.0, n), fill(1.0, n - 1)))

        for (Bmat, label) in ((Matrix{Float64}(I, n, n), "standard"),
                              (Bgen, "generalized"))
            reference = eigvals(Symmetric(A), Symmetric(Bmat))
            interval = (reference[3] - 1.0e-9, reference[6] + 1.0e-9)
            serial = feast(A, Bmat, interval; M0 = 10, parallel = :serial)
            threaded = feast(A, Bmat, interval; M0 = 10, parallel = :threads)
            @test threaded.info == serial.info
            @test threaded.M == serial.M
            @test threaded.M == 4
            @test isapprox(sort(threaded.lambda), sort(serial.lambda); atol = 1.0e-8)
            for j in 1:threaded.M
                q = threaded.q[:, j]
                @test isapprox(norm(q), 1.0; atol = 1.0e-8)
                @test norm(A * q - threaded.lambda[j] * (Bmat * q)) / norm(q) < 1.0e-8
            end
        end

        sparse_A = sparse(Matrix(SymTridiagonal(fill(2.0, 60), fill(-1.0, 59))))
        sparse_B = sparse(Matrix{Float64}(I, 60, 60))
        reference = eigvals(Symmetric(Matrix(sparse_A)))
        interval = (reference[1] - 1.0e-9, reference[4] + 1.0e-9)
        serial = feast(sparse_A, sparse_B, interval; M0 = 10, parallel = :serial)
        threaded = feast(sparse_A, sparse_B, interval; M0 = 10, parallel = :threads)
        @test threaded.M == serial.M
        @test isapprox(sort(threaded.lambda), sort(serial.lambda); atol = 1.0e-8)
    else
        @info "Skipping threaded/serial agreement (single thread)"
    end
end

@testset "Optional dependencies are wired through package extensions" begin
    # Krylov and MPI are weak dependencies. `using Krylov` at the top of
    # runtests.jl must be what turns the iterative paths on -- if the flag were
    # hard-coded true again, the unreachable-guard bug would be back.
    @test Base.get_extension(FeastKit, :FeastKitKrylovExt) !== nothing
    @test FeastKit.FEAST_KRYLOV_AVAILABLE[]

    project = read(joinpath(@__DIR__, "..", "Project.toml"), String)
    # Collect the package names declared under one Project.toml table, ignoring
    # comments (which mention both Krylov and MPI by name).
    function declared_packages(section::AbstractString)
        m = match(Regex("\\[" * section * "\\](.*?)(?:\\n\\[|\\z)", "s"), project)
        m === nothing && return String[]
        return [strip(match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line).captures[1])
                for line in split(m.captures[1], '\n')
                if occursin(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*\"", line)]
    end

    weak = declared_packages("weakdeps")
    deps = declared_packages("deps")
    @test "Krylov" in weak
    @test "MPI" in weak
    # Neither may reappear under [deps]; that is what made every user install MPI.
    @test !("Krylov" in deps)
    @test !("MPI" in deps)

    # The MPI seam exists whether or not MPI is loaded, so serial code can name
    # these without a hard dependency.
    for f in (FeastKit.mpi_feast, FeastKit.mpi_feast_general, FeastKit.feast_hybrid)
        @test f isa Function
    end
end

@testset "RCI kernels driven directly" begin
    # feast_hrci! is public API with no driver behind it, so without this it
    # would go completely unexercised. Both kernels are driven here exactly the
    # way the docs describe, which also pins the job sequence.
    n = 40

    @testset "feast_srci! real symmetric generalized" begin
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        B = Matrix(SymTridiagonal(fill(4.0, n), fill(1.0, n - 1)))
        reference = eigvals(Symmetric(A), Symmetric(B))
        Emin, Emax = reference[1] - 1.0e-9, reference[4] + 1.0e-9
        M0 = 10

        fpm = fresh_fpm()
        work = zeros(Float64, n, M0)
        workc = zeros(ComplexF64, n, M0)
        Aq = zeros(Float64, M0, M0)
        Sq = zeros(Float64, M0, M0)
        lambda = zeros(Float64, M0)
        q = zeros(Float64, n, M0)
        res = zeros(Float64, M0)
        ijob, Ze = Ref(-1), Ref(zero(ComplexF64))
        epsout, loop, mode, info = Ref(0.0), Ref(0), Ref(0), Ref(0)
        state = FeastSRCIState{Float64}()

        seen = Int[]
        steps = 0
        while true
            steps += 1
            @test steps < 2000
            feast_srci!(ijob, n, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                        Emin, Emax, M0, lambda, q, mode, res, info; state = state)
            push!(seen, ijob[])
            if ijob[] == Int(Feast_RCI_DONE)
                break
            elseif ijob[] == Int(Feast_RCI_FACTORIZE)
                continue                      # factorization is folded into SOLVE below
            elseif ijob[] == Int(Feast_RCI_SOLVE)
                workc .= (Ze[] * B - A) \ (B * ComplexF64.(work))
            elseif ijob[] == Int(Feast_RCI_MULT_A)
                mul!(view(work, :, 1:mode[]), A, view(q, :, 1:mode[]))
            elseif ijob[] == Int(Feast_RCI_MULT_B)
                mul!(view(work, :, 1:mode[]), B, view(q, :, 1:mode[]))
            else
                error("unexpected ijob $(ijob[])")
            end
        end

        @test info[] == 0
        @test mode[] == 4
        @test isapprox(sort(lambda[1:mode[]]), reference[1:4]; atol = 1.0e-8)
        for j in 1:mode[]
            @test isapprox(norm(q[:, j]), 1.0; atol = 1.0e-8)
            @test norm(A * q[:, j] - lambda[j] * (B * q[:, j])) / norm(q[:, j]) < 1.0e-8
        end
        # MULT_B must actually be requested; a caller that only handles MULT_A
        # would silently compute ||A q - λ q|| for a generalized problem.
        @test Int(Feast_RCI_MULT_B) in seen
        @test count(==(Int(Feast_RCI_MULT_A)), seen) ==
              count(==(Int(Feast_RCI_MULT_B)), seen)
    end

    @testset "feast_hrci! complex Hermitian generalized" begin
        rng = MersenneTwister(5)
        A = randn(rng, ComplexF64, n, n); A = (A + A') / 2
        B = randn(rng, ComplexF64, n, n); B = B * B' + n * I
        reference = eigvals(Hermitian(A), Hermitian(Matrix(B)))
        Emin, Emax = reference[1] - 1.0e-9, reference[3] + 1.0e-9
        M0 = 8

        fpm = fresh_fpm()
        work = zeros(Float64, n, M0)
        workc = zeros(ComplexF64, n, M0)
        zAq = zeros(ComplexF64, M0, M0)
        zSq = zeros(ComplexF64, M0, M0)
        lambda = zeros(Float64, M0)
        q = zeros(ComplexF64, n, M0)
        res = zeros(Float64, M0)
        ijob, Ze = Ref(-1), Ref(zero(ComplexF64))
        epsout, loop, mode, info = Ref(0.0), Ref(0), Ref(0), Ref(0)
        state = FeastHRCIState{Float64}()

        steps = 0
        while true
            steps += 1
            @test steps < 2000
            feast_hrci!(ijob, n, Ze, work, workc, zAq, zSq, fpm, epsout, loop,
                        Emin, Emax, M0, lambda, q, mode, res, info; state = state)
            if ijob[] == Int(Feast_RCI_DONE)
                break
            elseif ijob[] == Int(Feast_RCI_FACTORIZE)
                continue
            elseif ijob[] == Int(Feast_RCI_SOLVE)
                workc .= (Ze[] * B - A) \ (B * workc)
            elseif ijob[] == Int(Feast_RCI_MULT_A)
                mul!(view(workc, :, 1:mode[]), A, view(q, :, 1:mode[]))
            elseif ijob[] == Int(Feast_RCI_MULT_B)
                mul!(view(workc, :, 1:mode[]), B, view(q, :, 1:mode[]))
            else
                error("unexpected ijob $(ijob[])")
            end
        end

        @test info[] == 0
        @test mode[] == 3
        @test isapprox(sort(lambda[1:mode[]]), reference[1:3]; atol = 1.0e-7)
        for j in 1:mode[]
            @test isapprox(norm(q[:, j]), 1.0; atol = 1.0e-8)
            @test norm(A * q[:, j] - lambda[j] * (B * q[:, j])) / norm(q[:, j]) < 1.0e-7
        end
    end

    @testset "A rank-deficient user subspace is not reported as success" begin
        # KNOWN DEFECT, recorded so a fix flips these to passing.
        #
        # With fpm[5] = 1 the caller supplies the initial subspace. If that
        # subspace is nearly orthogonal to one eigenvector inside the interval,
        # the pivoted-QR compression (rank_tol = sqrt(eps) ~ 1.5e-8) drops that
        # direction on the first sweep. The Ritz pairs that survive are exact,
        # so the residual test passes immediately and FEAST returns
        # Feast_SUCCESS having found 2 of the 3 eigenvalues in the interval.
        #
        # Convergence is judged only on the pairs that were found; nothing
        # cross-checks the count. Refinement cannot rescue it either, because
        # the solve converges on loop 0 and never gets a second sweep.
        nd, M0d = 5, 4
        A = Matrix(Diagonal([1.0, 2.0, 3.0, 10.0, 11.0]))
        B = Matrix{Float64}(I, nd, nd)
        Emin, Emax = 0.5, 3.5

        fpm = fresh_fpm()
        fpm[5] = 1
        work = zeros(Float64, nd, M0d)
        work[1, 1] = 1.0
        work[2, 2] = 1.0
        work[3, 3] = 1.0e-12    # the e3 direction is present only at 1e-12
        work[4, 3] = 1.0
        work[5, 4] = 1.0

        workc = zeros(ComplexF64, nd, M0d)
        Aq = zeros(Float64, M0d, M0d)
        Sq = zeros(Float64, M0d, M0d)
        lambda = zeros(Float64, M0d)
        q = zeros(Float64, nd, M0d)
        res = zeros(Float64, M0d)
        ijob, Ze = Ref(-1), Ref(zero(ComplexF64))
        epsout, loop, mode, info = Ref(0.0), Ref(0), Ref(0), Ref(0)
        state = FeastSRCIState{Float64}()

        steps = 0
        while true
            steps += 1
            @test steps < 2000
            feast_srci!(ijob, nd, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                        Emin, Emax, M0d, lambda, q, mode, res, info; state = state)
            if ijob[] == Int(Feast_RCI_DONE)
                break
            elseif ijob[] == Int(Feast_RCI_FACTORIZE)
                continue
            elseif ijob[] == Int(Feast_RCI_SOLVE)
                workc .= (Ze[] * B - A) \ (B * ComplexF64.(work))
            elseif ijob[] == Int(Feast_RCI_MULT_A)
                mul!(view(work, :, 1:mode[]), A, view(q, :, 1:mode[]))
            elseif ijob[] == Int(Feast_RCI_MULT_B)
                mul!(view(work, :, 1:mode[]), B, view(q, :, 1:mode[]))
            end
        end

        # Three eigenvalues (1, 2, 3) lie in [0.5, 3.5].
        @test_broken mode[] == 3
        # Reporting success while undercounting is the harmful half: a caller
        # has no way to tell this apart from a correct answer.
        @test_broken !(info[] == 0 && mode[] < 3)
        # What it does today, so the record is unambiguous.
        @test mode[] == 2
        @test info[] == 0
    end

    @testset "Reusing the state object is mandatory" begin
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        reference = eigvals(Symmetric(A))
        M0 = 6
        fpm = fresh_fpm()
        work = zeros(Float64, n, M0)
        workc = zeros(ComplexF64, n, M0)
        Aq = zeros(Float64, M0, M0)
        Sq = zeros(Float64, M0, M0)
        lambda = zeros(Float64, M0)
        q = zeros(Float64, n, M0)
        res = zeros(Float64, M0)
        ijob, Ze = Ref(-1), Ref(zero(ComplexF64))
        epsout, loop, mode, info = Ref(0.0), Ref(0), Ref(0), Ref(0)

        state = FeastSRCIState{Float64}()
        feast_srci!(ijob, n, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                    reference[1] - 1.0e-9, reference[3] + 1.0e-9, M0,
                    lambda, q, mode, res, info; state = state)
        @test ijob[] == Int(Feast_RCI_FACTORIZE)

        # Continuing with a fresh state used to reset the machine silently and
        # return M = 0. It must now say so.
        @test_throws ArgumentError feast_srci!(ijob, n, Ze, work, workc, Aq, Sq,
                                               fpm, epsout, loop,
                                               reference[1] - 1.0e-9,
                                               reference[3] + 1.0e-9, M0, lambda,
                                               q, mode, res, info;
                                               state = FeastSRCIState{Float64}())
    end
end

@testset "Documented API surface behaves as the docs show" begin
    # Each of these was a doc example that did not run. They live here so the
    # signatures the manual advertises stay callable.
    n = 60
    A = Matrix(SymTridiagonal(2.0 * ones(n), -1.0 * ones(n - 1)))
    B = Matrix{Float64}(I, n, n)
    reference = eigvals(Symmetric(A))
    lo, hi = 0.0, reference[6] + 1.0e-9

    @testset "interval endpoints may be any Real" begin
        base = feast(A, (lo, hi); M0 = 12)
        # Mixed and integer endpoints must give the same answer as Float64 ones.
        # (A Float32 endpoint is deliberately not tested for equality: narrowing
        # moves the bound and may legitimately exclude a boundary eigenvalue.)
        for interval in ((0, hi), (0.0, hi))
            r = feast(A, interval; M0 = 12)
            @test r.info == 0
            @test r.M == base.M
        end
        @test feast(A, (0, Float32(hi)); M0 = 12).info == 0
        @test feast(A, B, (0, hi); M0 = 12).M == base.M

        # An integer-valued matrix has to be promoted, not rejected. Scaling A
        # by 10 scales its spectrum, so scale the interval to match.
        Ai = round.(Int, 10 .* A)
        integer_reference = eigvals(Symmetric(Float64.(Ai)))
        @test feast(Ai, (0, integer_reference[6] + 1.0e-9); M0 = 12).info == 0
    end

    @testset "contour builders accept integer bounds" begin
        c = feast_contour_expert(-2, 2, 16, 0, 100)
        @test length(c.Zne) == 16
        fpm = fresh_fpm(); feastdefault!(fpm)
        @test length(feast_contour(0, 4, fpm).Zne) == fpm[2]
        @test length(feast_gcontour(0, 2, fpm).Zne) == fpm[8]
    end

    @testset "rational filters accept any AbstractVector" begin
        c = feast_contour_expert(-2.0, 2.0, 16, 0, 100)
        vals = feast_rational_expert(c.Zne, c.Wne, range(-2.0, 2.0, length = 5))
        @test length(vals) == 5
        @test vals[3] > 0.9              # centre of the interval: filter ~ 1
        # The (Zne, Wne, lambda) order must reject mismatched node/weight lists
        # rather than silently reinterpreting the arguments.
        @test_throws ArgumentError feast_rationalx(c.Zne, c.Wne[1:3], [0.0])
    end

    @testset "feast_summary covers both result types" begin
        io = IOBuffer()
        feast_summary(io, feast(A, (lo, hi); M0 = 12))
        @test occursin("Eigenvalues found", String(take!(io)))

        Ac = randn(MersenneTwister(3), ComplexF64, 40, 40)
        general = feast_general(Ac, Matrix{ComplexF64}(I, 40, 40), 0.0 + 0.0im, 2.0; M0 = 12)
        feast_summary(io, general)
        out = String(take!(io))
        @test occursin("non-Hermitian", out)
        @test occursin("Eigenvalues found", out)
    end

    @testset "a bare zeros(64) fpm still describes a real contour" begin
        # `zeros(Int, 64)` without feastinit! used to leave fpm[18] = 0, which
        # collapses the ellipse onto a line segment and returns info = 5.
        bare = zeros(Int, 64)
        bare[2] = 16
        initialised = fresh_fpm()
        initialised[2] = 16
        r_bare = feast(A, (lo, hi); M0 = 12, fpm = bare)
        r_init = feast(A, (lo, hi); M0 = 12, fpm = copy(initialised))
        @test r_bare.info == 0
        @test r_bare.M == r_init.M
        @test isapprox(sort(r_bare.lambda), sort(r_init.lambda); atol = 1.0e-8)
    end

    @testset "feast_parallel_comparison runs without MPI" begin
        # It lives in the parent module, not the MPI extension: every MPI call
        # is behind an mpi_available() guard. It also used to interpolate
        # Python-style format specs (`$(t:.3f)`), which throw UndefVarError.
        @test !isempty(methods(feast_parallel_comparison))
        # redirect_stdout takes a stream, not an IOBuffer, so capture via a file.
        path, io = mktemp()
        results = try
            redirect_stdout(io) do
                feast_parallel_comparison(A, B, (lo, hi), 12)
            end
        finally
            close(io)
        end
        out = read(path, String)
        rm(path; force = true)
        @test occursin("Serial execution", out)
        @test occursin("seconds", out)
        @test haskey(results, :serial)
        @test results[:serial].info == 0
    end

    @testset "cg is rejected with an explanation" begin
        op = LinearOperator{Float64}((y, x) -> mul!(y, A, x), (n, n); issymmetric = true)
        id = LinearOperator{Float64}((y, x) -> copyto!(y, x), (n, n); issymmetric = true)
        err = try
            solver = FeastKit.create_iterative_solver(op, id, :cg)
            solver(zeros(ComplexF64, n, 1), 1.0 + 0.1im, zeros(ComplexF64, n, 1))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CG", sprint(showerror, err))
    end
end
