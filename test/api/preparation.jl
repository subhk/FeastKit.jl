using Test, FeastKit, LinearAlgebra, SparseArrays

@testset "Shared public option preparation" begin
    A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
    B = Matrix{Float64}(I, 4, 4)
    for T in (Float64, ComplexF64), storage in (Matrix, sparse)
        a, b = storage(T.(A)), storage(T.(B))
        for general in (false, true), generalized in (false, true)
            args = generalized ? (a, b) : (a,)
            solve = general ?
                (opts -> feast_general(args..., 1.5+0im, 0.8; opts...)) :
                (opts -> feast(args..., (0.5, 2.5); opts...))
            @testset "$T $storage general=$general generalized=$generalized" begin
                fpm = feastinit().fpm
                original = copy(fpm)
                opts = (; fpm, subspace_size=3, tol=1e-10, maxiter=30,
                         quadrature_points=16, backend=:serial)
                result = solve(opts)
                @test result.converged
                @test sort(real.(result.values)) ≈ [1.0, 2.0] atol=1e-9
                @test fpm == original
                @test_throws ArgumentError solve((; opts..., M0=2))
                @test_throws ArgumentError solve((; opts..., solver=:direct, solver_opts=(rtol=1e-8,)))
                @test_throws ArgumentError solve((; opts..., parallel=:threads))
                fpm[general ? 8 : 2] = 8
                @test_throws ArgumentError solve(opts)
                @test fpm[general ? 8 : 2] == 8
            end
        end
    end
end

@testset "Backend compatibility through public entry points" begin
    if Threads.nthreads() > 1
        A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
        B = Matrix{Float64}(I, 4, 4)
        H = complex.(A)
        options = (; subspace_size=3, tol=1e-10)
        # Both the high-level API and the exported routing helper enforce the
        # same restrictions; strict requests must never silently run serial.
        for (a, b) in ((H, complex.(B)), (A, sparse(B)))
            @test_throws ArgumentError feast(a, b, (0.5, 2.5); options..., backend=:threads)
            @test_throws ArgumentError feast_with_backend(a, b, (0.5, 2.5), :threads,
                                                          3, feastinit().fpm, nothing, true;
                                                          strict_backend=true)
        end
        for solve in (opts -> feast(H, (0.5, 2.5); opts...),
                      opts -> feast_general(A, 1.5+0im, 0.8; opts...),
                      opts -> feast(A, (0.5, 2.5); solver=:gmres, opts...))
            @test_throws ArgumentError solve((; options..., backend=:threads))
            result = solve((; options..., backend=:auto))
            @test result.converged
            @test sort(real.(result.values)) ≈ [1.0, 2.0] atol=1e-9
        end
    end
end
