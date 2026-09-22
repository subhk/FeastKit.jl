using Test, FeastKit, LinearAlgebra, SparseArrays, Random

@testset "Residual inverse mixed precision" begin
    rng = MersenneTwister(781)
    n = 32
    for kind in (:real, :hermitian, :general), generalized in (false,true), scale in (1e-20,1.0,1e20), cache in (0,1)
        T = kind === :real ? Float64 : ComplexF64
        U = Matrix(qr(randn(rng,T,n,n)).Q)
        H = U * Diagonal(collect(1.0:n)) * U'
        H = Matrix(Hermitian(H))
        if kind === :general
            D = Diagonal(exp.(range(0,1;length=n)))
            H = D * H / D
        end
        mass = generalized ? collect(range(0.8,1.2;length=n)) : ones(n)
        B = Matrix(Diagonal(T.(mass))) * scale
        A = (Diagonal(sqrt.(mass)) * H * Diagonal(sqrt.(mass))) * scale
        kind === :general || (A = Matrix(Hermitian(A)))
        # Common pencil scaling exercises Float32 factor/RHS normalization.
        fpm = feastinit().fpm
        fpm[10] = cache
        before = copy(fpm)
        args = !generalized && scale == 1.0 ? (A,) : (A,B)
        result = kind === :general ? feast_general(args...,3.0+0im,1.6;subspace_size=7,tol=1e-11,mixed_precision=true,fpm) :
                                     feast(args...,(1.4,4.6);subspace_size=7,tol=1e-11,mixed_precision=true,fpm)
        @test result.converged
        @test sort(real.(result.values)) ≈ [2,3,4] atol=1e-9
        @test maximum(norm((A/scale)*result.vectors[:,j] - result.values[j]*(B/scale)*result.vectors[:,j]) /
                      norm((B/scale)*result.vectors[:,j]) for j in 1:result.M) < 1e-9
        @test fpm == before
    end
    A = Matrix(Diagonal(collect(1.0:n)))
    for complex in (false,true)
        matrix = complex ? ComplexF64.(A) : A
        result = feast(matrix,(1.4,4.6);mixed_precision=true,subspace_size=:auto)
        @test result.converged && result.M == 3
        result = feast(matrix,feast_rectangle(1.4,4.6,-0.5,0.5;points_per_edge=24);
                       mixed_precision=true,initial_subspace=Matrix{Float64}(I,n,n)[:,2:4],subspace_size=6)
        @test result.converged && result.M == 3
    end
    @test_throws ArgumentError feast(sparse(A),(1.4,4.6);mixed_precision=true)
    @test_throws ArgumentError feast(Float32.(A),(1.4f0,4.6f0);mixed_precision=true)
    @test_throws ArgumentError feast(A,(1.4,4.6);mixed_precision=true,solver=:gmres)
    @test_throws ArgumentError feast(A,(1.4,4.6);mixed_precision=true,backend=:threads)
    fpm = feastinit().fpm
    fpm[42] = 0
    @test_throws ArgumentError feast(A,(1.4,4.6);mixed_precision=true,fpm)
    @test_throws ArgumentError feast(A,(1.4,4.6);mixed_precision=1)
    fpm[42] = 2
    @test_throws ArgumentError feastdefault!(fpm)

    # Exercise actual Float32 factors, per-shift reuse, zero inactive columns,
    # and fallback from a matrix that becomes singular after Float32 rounding.
    fpm = feastinit().fpm
    fpm[42] = 1
    feastdefault!(fpm)
    ws = FeastKit._feast_mixed_workspace(A,nothing,2,fpm,:direct)
    Q = hcat(ones(n),zeros(n))
    dest = zeros(ComplexF64,n,2)
    z = 3.0+0.5im
    FeastKit._feast_mixed_solve!(dest,ws,z,Q,1,1)
    @test eltype(ws.factors[1].low.factors) == ComplexF32
    @test ws.fallbacks == 0
    @test norm((z*I-A)*dest-Q)/norm(Q) < 1e-5
    @test iszero(norm(dest[:,2]))
    factor = ws.factors[1].low
    FeastKit._feast_mixed_solve!(dest,ws,z,Q,1,1)
    @test ws.factors[1].low === factor
    bad = 2.0I - [1.0 1.0; 1.0 1.0+1e-10]
    ws = FeastKit._feast_mixed_workspace(bad,nothing,2,fpm,:direct)
    Q = Matrix{ComplexF64}(I,2,2)
    dest = similar(Q)
    FeastKit._feast_mixed_solve!(dest,ws,2.0+0im,Q,1,1)
    @test ws.fallbacks == 1
    @test ws.factors[1].low === nothing
    @test ws.factors[1].high !== nothing
    @test dest ≈ (2.0I-bad)\Q
end
