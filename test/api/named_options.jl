using Test, FeastKit, LinearAlgebra, SparseArrays, Random, Krylov

@testset "Named public API" begin
    A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
    B = Matrix{Float64}(I, 4, 4)
    options = (; subspace_size=3, tol=1e-10, maxiter=30, quadrature_points=16)

    @testset "Assembled storage, precision, and solvers" begin
        for T in (Float64, ComplexF64), storage in (Matrix, sparse), solver in (:direct, :gmres)
            a, b = storage(T.(A)), storage(T.(B))
            inner = solver === :gmres ? (rtol=1e-13, maxiter=100, restart=8) : (;)
            for result in (feast(a, (0.5, 2.5); options..., solver=solver, solver_opts=inner),
                           feast(a, b, (0.5, 2.5); options..., solver=solver, solver_opts=inner))
                @test result.converged
                @test sort(result.values) ≈ [1.0, 2.0] atol=1e-9
                @test result.values === result.lambda
                @test result.vectors === result.q
                @test norm(a * result.vectors - result.vectors * Diagonal(result.values)) < 1e-8
            end
            for result in (feast_general(a, 1.5+0im, 0.8; options..., solver=solver, solver_opts=inner),
                           feast_general(a, b, 1.5+0im, 0.8; options..., solver=solver, solver_opts=inner))
                @test result.converged
                @test sort(real.(result.values)) ≈ [1.0, 2.0] atol=1e-9
            end
        end
        @test feast(Int.(A), (0, 2.5); options...).converged
        @test feast(Int.(A), Int.(B), (0, 2); options...).converged
        @test feast_general(Int.(A), 1+0im, 1; subspace_size=4).converged
        @test feast(Float32.(A), (0.5f0, 2.5f0); options...).converged
        for T in (Float64, ComplexF64), solver in (:direct, :gmres), generalized in (false, true)
            banded = full_to_banded(T.(A), 0)
            mass = generalized ? full_to_banded(T.(B), 0) : nothing
            inner = solver === :gmres ? (rtol=1e-13, maxiter=100, restart=8) : (;)
            result = feast_banded(banded, 0, (0.5, 2.5); options..., B=mass,
                                  solver=solver, solver_opts=inner)
            @test result.converged
            @test sort(result.values) ≈ [1.0, 2.0] atol=1e-9
        end
    end

    @testset "Legacy settings, conflicts, and validation" begin
        fpm = feastinit().fpm
        original = copy(fpm)
        @test feast(A, (0.5, 2.5); fpm=fpm, options...).converged
        @test fpm == original
        @test feast(A, (0.5, 2.5); M0=3, options...).converged
        @test_throws ArgumentError feast(A, (0.5, 2.5); M0=2, options...)
        for (index, keyword, value) in ((3, :tol, 1e-10), (4, :maxiter, 30), (2, :quadrature_points, 16))
            conflicting = feastinit().fpm
            conflicting[index] = index == 3 ? 8 : 12
            before = copy(conflicting)
            @test_throws ArgumentError feast(A, (0.5, 2.5); fpm=conflicting, (; keyword => value)...)
            @test conflicting == before
        end
        agreeing = feastinit()
        agreeing.fpm[3] = 10
        @test feast(A, (0.5, 2.5); fpm=agreeing, options..., tol=3e-10).converged
        for tol in (0, -1, Inf, NaN, 1e-20, 2, true)
            @test_throws ArgumentError feast(A, (0.5, 2.5); tol=tol)
        end
        for keyword in (:maxiter, :subspace_size, :quadrature_points), value in (0, -1, 2.5, true)
            @test_throws ArgumentError feast(A, (0.5, 2.5); (; keyword => value)...)
        end
        @test_throws ArgumentError feast(A, (0.5, 2.5); solver=:cg)
        @test_throws ArgumentError feast(A, (0.5, 2.5); solver_opts=(rtol=1e-10,))
        for opts in ((rtol=0,), (rtol=NaN,), (maxiter=0,), (restart=-1,), (tolerance=1e-8,))
            @test_throws ArgumentError feast(A, (0.5, 2.5); solver=:gmres, solver_opts=opts)
        end
        if Threads.nthreads() > 1
            @test_throws ArgumentError feast(A, (0.5, 2.5); solver=:gmres, backend=:threads)
            @test feast(A, (0.5, 2.5); options..., backend=:threads).converged
            @test feast(A, (0.5, 2.5); options..., solver=:gmres,
                        solver_opts=(rtol=1e-13,), backend=:auto).converged
        end
    end

    @testset "Matrix-free options reach the solver" begin
        op = LinearOperator{Float64}((y, x) -> mul!(y, A, x), size(A); issymmetric=true)
        shifts = Set{ComplexF64}()
        callback = (Y, z, X) -> begin
            push!(shifts, z)
            Y .= (z * I - A) \ X
        end
        result = feast(op, (0.5, 2.5); options..., solver=callback)
        @test result.converged
        @test sort(result.values) ≈ [1.0, 2.0] atol=1e-9
        @test length(shifts) == options.quadrature_points
        @test feast(op, (0.5, 2.5); options..., solver=:gmres,
                    solver_opts=(rtol=1e-13, maxiter=100, restart=8)).converged
        @test_throws ArgumentError feast(op, (0.5, 2.5); solver=:direct)
        @test_throws ArgumentError feast(op, (0.5, 2.5); solver=callback, solver_opts=(rtol=1e-10,))
        fpm = feastinit()
        fpm.fpm[3] = 8
        @test_throws ArgumentError feast(op, (0.5, 2.5); fpm=fpm, tol=1e-10, solver=callback)

        complex_op = LinearOperator{ComplexF64}((y, x) -> mul!(y, A, x), size(A))
        empty!(shifts)
        result = feast_general(complex_op, 1.5+0im, 0.8; options..., solver=callback)
        @test result.converged
        @test length(shifts) == options.quadrature_points
        @test feast(complex_op, feast_circle(1.5, 0.8; n=32); options...,
                    quadrature_points=32, solver=callback).converged
    end

    @testset "Full contours select eigenvalues and clean up" begin
        # Corners must be retained: a diamond through edge midpoints excludes
        # 0.9+0.9im, though it is inside the requested rectangle.
        C = Matrix(Diagonal(ComplexF64[0.9+0.9im, -0.4+0.2im, 1.2, 3]))
        contour = feast_rectangle(-1, 1, -1, 1; points_per_edge=32)
        registry_before = length(FeastKit.FEAST_CUSTOM_CONTOURS)
        fpm = feastinit().fpm
        before = copy(fpm)
        for storage in (Matrix, sparse)
            c, b = storage(C), storage(ComplexF64.(B))
            for result in (feast(c, contour; subspace_size=4, fpm=fpm),
                           feast(c, b, contour; subspace_size=4, fpm=fpm))
                @test result.converged
                @test sort(result.values; by=real) ≈ [-0.4+0.2im, 0.9+0.9im] atol=1e-9
            end
        end
        @test fpm == before
        @test length(FeastKit.FEAST_CUSTOM_CONTOURS) == registry_before
        @test_throws ArgumentError feast(C, contour; fpm=fpm, solver=:unsupported)
        @test fpm == before
        @test length(FeastKit.FEAST_CUSTOM_CONTOURS) == registry_before
        @test_throws ArgumentError feast(C, contour; quadrature_points=16)
        invalid = FeastKit.FeastContour{Float64}(ComplexF64[0, 1, NaN], ones(ComplexF64, 3))
        @test_throws ArgumentError feast(C, invalid)
        # A nested contour call must leave the caller's registration usable.
        outer = feast_circle(1.5, 0.8)
        FeastKit.with_custom_contour(fpm, outer) do
            id = fpm[29]
            feast(C, contour; subspace_size=4, fpm=fpm)
            @test fpm[29] == id
            @test FeastKit.feast_get_custom_contour(fpm).Zne == outer.Zne
        end
        @test length(FeastKit.FEAST_CUSTOM_CONTOURS) == registry_before
    end

    @testset "Results explain convergence and remain compact" begin
        result = feast(A, (0.5, 2.5); options...)
        @test all(name -> hasproperty(result, name), (:values, :vectors, :converged, :message))
        @test occursin("converged", repr(MIME"text/plain"(), result))
        @test result.message == "Success"
        # Saturation means completeness was not established even with small residuals.
        saturated = FeastResult{Float64,Float64}([1.0], ones(4, 1), 1, [1e-14], 2, 1e-14, 1)
        @test !saturated.converged
        @test occursin("subspace_size", saturated.message)
        @test occursin("subspace_size", repr(MIME"text/plain"(), saturated))
        io = IOBuffer()
        feast_summary(io, saturated)
        @test occursin("subspace_size", String(take!(io)))
        many = FeastGeneralResult{Float64}(ComplexF64.(1:100), ones(ComplexF64, 100, 100),
                                          100, zeros(100), 0, 0.0, 2)
        @test many.values === many.lambda && many.vectors === many.q
        @test length(repr(MIME"text/plain"(), many)) < 500
        @test occursin("92 more", repr(MIME"text/plain"(), many))
    end
end
