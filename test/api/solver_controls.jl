using Test, FeastKit, LinearAlgebra, SparseArrays, Krylov

@testset "Warm starts and automatic sizing" begin
    N = 24
    base = Matrix(Diagonal(collect(1.0:N)))
    seed = Matrix{Float64}(I, N, N)[:, 2:4]
    for T in (Float64, ComplexF64), storage in (Matrix, sparse), general in (false, true)
        A, B = storage(T.(base)), storage(Matrix{T}(I, N, N))
        original_seed = T.(seed)
        for generalized in (false, true)
            args = generalized ? (A, B) : (A,)
            solve = general ? (opts -> feast_general(args..., 3.0+0im, 1.6; opts...)) :
                              (opts -> feast(args..., (1.4, 4.6); opts...))
            @testset "$T $storage general=$general generalized=$generalized" begin
                fpm = feastinit().fpm
                before = copy(fpm)
                for solver in (:direct, :gmres)
                    result = solve((; initial_subspace=original_seed, subspace_size=6,
                                    tol=1e-10, solver, fpm))
                    @test result.converged
                    @test sort(real.(result.values)) ≈ [2,3,4] atol=1e-9
                    @test original_seed == T.(seed)
                    @test fpm == before
                end
                automatic = solve((; subspace_size=:auto, tol=1e-10))
                @test automatic.converged
                @test sort(real.(automatic.values)) ≈ [2,3,4] atol=1e-9
                @test_throws ArgumentError solve((; subspace_size=:auto, M0=6))
                @test_throws ArgumentError solve((; subspace_size=6, max_subspace_size=8))
                @test_throws DimensionMismatch solve((; initial_subspace=ones(2,2)))
                @test_throws ArgumentError solve((; initial_subspace=zeros(N,2)))
                @test_throws ArgumentError solve((; initial_subspace=fill(NaN,N,2)))
            end
        end
    end
    # A supplied exact but incomplete eigenspace must not certify completeness.
    partial = feast(base, (1.4,4.6); initial_subspace=seed[:,1:1], subspace_size=6)
    @test partial.converged && partial.M == 3
    # Explicit memory cap is respected; saturation remains visible to callers.
    saturated = feast(base, (0.5,20.5); subspace_size=:auto, max_subspace_size=4)
    @test saturated.info == Int(Feast_ERROR_M0)
    @test saturated.M <= 4

    for general in (false,true)
        T = general ? ComplexF64 : Float64
        A = T.(base)
        op = LinearOperator{T}((y,x)->mul!(y,A,x),size(A); issymmetric=true)
        solve = general ? (opts -> feast_general(op, 3.0+0im, 1.6; opts...)) :
                          (opts -> feast(op, (1.4,4.6); opts...))
        for size_option in (6,:auto)
            result = solve((; initial_subspace=T.(seed), subspace_size=size_option,
                            tol=1e-10, solver_opts=(restart=N,)))
            @test result.converged
            @test sort(real.(result.values)) ≈ [2,3,4] atol=1e-9
        end
        # Matrix-free automatic sizing starts small and must grow on saturation.
        large = general ? feast_general(op, 10.5+0im, 10.0; subspace_size=:auto,
                                         solver_opts=(restart=N,), tol=1e-10) :
                          feast(op, (0.5,20.5); subspace_size=:auto,
                                solver_opts=(restart=N,), tol=1e-10)
        @test large.converged && large.M == 20
    end
    rectangle = feast_rectangle(1.4,4.6,-0.5,0.5; points_per_edge=24)
    result = feast(base, rectangle; subspace_size=:auto, initial_subspace=seed)
    @test result.converged && result.M == 3
end

@testset "Adaptive tolerances and persistent Krylov buffers" begin
    @test FeastKit._feast_inner_tol(false,1e-10,1.0,0) == 1e-10
    @test FeastKit._feast_inner_tol(true,1e-10,1.0,0) == 1e-3
    @test FeastKit._feast_inner_tol(true,1e-10,1e-6,2) ≈ 1e-7
    @test FeastKit._feast_inner_tol(true,1e-10,1e-14,3) == 1e-10
    n = 120
    A = Matrix(Diagonal(collect(1.0:n)))
    rhs = ones(ComplexF64,n,2)
    dest = similar(rhs)
    work = FeastKit._feast_krylov_workspace(n,Float64,n)
    solve(work) = FeastKit._feast_shifted_solve!(dest,rhs,A,nothing,2.0+1im,2,1e-10,2n,n;workspace=work)
    @test solve(work)
    @test solve(nothing)
    # Repeated nodes retain the O(N*restart) basis. Compare allocations instead
    # of relying on an absolute threshold tied to a particular Julia release.
    reusable = @allocated solve(work)
    fresh = @allocated solve(nothing)
    @test reusable < fresh / 2
    @test norm((2.0+1im)*dest-A*dest-rhs)/norm(rhs) < 1e-8
    band = reshape(collect(1.0:24),1,:)
    for T in (Float64,ComplexF64)
        r = feast_banded(T.(band),0,(1.4,4.6);subspace_size=6,solver=:gmres,tol=1e-10)
        @test r.converged && r.M == 3
    end
end
