using Test, FeastKit, LinearAlgebra, SparseArrays

@testset "Projector, scaling, and contour regressions" begin
    @testset "Threaded half-contour projector" begin
        if Threads.nthreads() > 1
            for (n, interval, m, ne) in ((10, (0.5, 2.5), 5, 8), (20, (0.5, 1.5), 6, 4)),
                storage in (Matrix, sparse)
                A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
                expected = filter(x -> interval[1] <= x <= interval[2], eigvals(Symmetric(A)))
                f = feastinit().fpm; f[2] = ne
                r = feast(storage(A), interval; M0=m, fpm=f, backend=:threads)
                @test r.info == 0
                @test r.M == length(expected)
                @test r.lambda ≈ expected atol=1e-9
                @test maximum(norm(A*r.q[:,j] - r.lambda[j]*r.q[:,j]) for j in 1:r.M) < 1e-10
            end
        end
    end

    @testset "Pencil scaling preserves convergence" begin
        # M0 < N deliberately: a full basis hides premature convergence by
        # solving the exact eigenproblem in the first reduced pencil.
        n, m = 80, 10
        A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
        expected = filter(x -> 0.8 <= x <= 1.2, eigvals(Symmetric(A)))
        for kind in (:real, :hermitian, :general), storage in (Matrix, sparse), scale in (1e-12, 1.0, 1e12)
            D = Diagonal(cis.(range(0, 1; length=n)))
            kind == :general && (D = D * Diagonal(exp.(range(0, 1; length=n))))
            A0 = kind == :real ? A : D * A / D
            # Construct the Hermitian matrix exactly to satisfy input validation.
            kind == :hermitian && (A0 = Matrix(Hermitian(A0)))
            B0 = Matrix{eltype(A0)}(I, n, n)
            f = feastinit().fpm; f[2] = 3
            solver = kind == :real ? (storage == Matrix ? feast_sygv! : feast_scsrgv!) :
                     kind == :hermitian ? (storage == Matrix ? feast_hegv! : feast_hcsrgv!) :
                     (storage == Matrix ? feast_gegv! : feast_gcsrgv!)
            args = kind == :general ? (1.0+0im, 0.2) : (0.8, 1.2)
            r = solver(storage(scale*A0), storage(scale*B0), args..., m, f)
            @test r.info == 0
            @test r.M == length(expected)
            @test r.loop > 0
            @test sort(real.(r.lambda)) ≈ expected atol=1e-9
            # Residuals are floored at the pencil's spectral scale, which the
            # kernels measure on their seeded probe block before the first sweep.
            probes = kind == :real ? zeros(n, m) : zeros(ComplexF64, n, m)
            kind == :real ? FeastKit._feast_seeded_subspace!(probes) :
                            FeastKit._feast_seeded_subspace_complex!(probes)
            σ = norm(A0 * probes) / norm(B0 * probes)
            actual = [norm(A0*r.q[:,j] - r.lambda[j]*B0*r.q[:,j]) /
                      norm(B0*r.q[:,j]) / max(abs(r.lambda[j]), σ) for j in 1:r.M]
            @test maximum(actual) < 2e-12
            @test r.res ≈ actual atol=1e-14
        end
        # A zero eigenvalue still has a nonzero normalization through B*q.
        @test FeastKit._feast_scaled_residual(zeros(2), [1.0, 0.0], 0.0, 1.0) == 0
        @test isinf(FeastKit._feast_scaled_residual(zeros(2), zeros(2), 0.0, 1.0))
        # A change of units rescales the residual, the eigenvalue and the
        # spectral scale together, and leaves the measured residual unchanged.
        rvec, bq = [3e-9, -4e-9], [1.0, 2.0]
        for units in (1e-12, 1e12)
            @test FeastKit._feast_scaled_residual(units * rvec, bq, units * 0.2, units * 2.5) ≈
                  FeastKit._feast_scaled_residual(rvec, bq, 0.2, 2.5)
        end
    end

    @testset "Rectangle corners survive registration and precision conversion" begin
        for T in (Float32, Float64), points in (1, 32), storage in (Matrix, sparse)
            c = feast_rectangle(-1f0, 1f0, -1f0, 1f0; points_per_edge=points)
            expected = Complex{T}[0, 0.99+0.99im, -0.99-0.99im]
            A = storage(Diagonal(vcat(expected, Complex{T}[1.01+0.99im, 2])))
            f = feastinit().fpm; f[16] = 1
            r = FeastKit.with_custom_contour(f, c) do
                feast_general(A, Complex{T}(0), T(3); M0=5, fpm=f)
            end
            @test r.info == 0
            @test r.M == length(expected)
            @test sort(r.lambda; by=real) ≈ sort(expected; by=real) atol=100eps(T)
        end
    end
end
