using Test, FeastKit, LinearAlgebra, SparseArrays, Random

@testset "Generalized and specialized eigenproblems" begin
    @testset "Matrix-free mass-weighted projector $T mass=$mass" for T in (Float64, ComplexF64), mass in (1e-2, 1e-10)
        # The 1e10 mass contrast exposes dropped directions when the projector
        # omits B*Q. Its reduced pencil can amplify Float64 roundoff by 1e10,
        # so use a tolerance above eps(Float64)*cond(B) for that fixture.
        # Keep a better-conditioned case at the default 1e-12 tolerance.
        tol = mass == 1e-10 ? 1e-5 : 1e-12
        B = Matrix(Diagonal(T[1,mass,mass]))
        A = B * Diagonal(T[1,2,3])
        ao = LinearOperator{T}((y,x)->mul!(y,A,x),(3,3); issymmetric=true)
        bo = LinearOperator{T}((y,x)->mul!(y,B,x),(3,3); issymmetric=true)
        solve = (Y,z,X)->copyto!(Y,(z*B-A)\X)
        @testset "Starting subspace seed=$seed" for seed in (nothing, 13, 99)
            # Exercise the default subspace and complex starts that exposed
            # roundoff-limited convergence in the ill-conditioned fixture.
            initial = seed === nothing ? nothing : randn(MersenneTwister(seed), T, 3, 3)
            options = (; M0=3, solver=solve, initial_subspace=initial)
            mass == 1e-10 && (options = merge(options, (; tol)))
            r = T === Float64 ? feast(ao,bo,(0.5,2.5); options...) :
                feast_general(ao,bo,1.5+0im,1.0; options...)
            @test r.info == 0
            @test r.M == 2
            @test sort(r.lambda; by=real) ≈ T[1,2] atol=tol
            actual = map(1:r.M) do j
                Aq, Bq = A*r.q[:,j], B*r.q[:,j]
                norm(Aq-r.lambda[j]*Bq) / norm(Bq) / max(abs(r.lambda[j]),1)
            end
            @test maximum(actual) <= tol
            @test r.res ≈ actual atol=10eps(Float64)
        end
    end
    @testset "Specialized saturation $storage M0=$m" for storage in (:dense,:sparse,:banded), m in (1,3)
        A = Matrix{ComplexF64}(I,3,3)
        f = feastinit().fpm
        r = storage == :dense ? feast_geev_complex_sym!(A,1.0+0im,0.5,m,f) :
            storage == :sparse ? feast_scsrev_complex!(sparse(A),1.0+0im,0.5,m,f) :
            feast_sbev_complex!(ones(ComplexF64,1,3),0,1.0+0im,0.5,m,f)
        @test r.info == (m == 1 ? Int(Feast_ERROR_M0) : 0)
        @test r.M == m
    end
    @testset "Callback generalized residual and saturation" begin
        A = Matrix(Diagonal([2.,4.,6.])); B = 2Matrix{Float64}(I,3,3)
        r = feast_matvec((y,x)->mul!(y,A,x),(y,x)->mul!(y,B,x),3,(0.5,3.5);M0=3)
        @test r.info == 0
        actual = [norm(A*r.q[:,j]-r.lambda[j]*B*r.q[:,j])/max(abs(r.lambda[j]),1) for j=1:r.M]
        @test r.res ≈ actual atol=1e-12
        r = feast_matvec((y,x)->copyto!(y,x),(y,x)->copyto!(y,x),3,(0.5,1.5);M0=1)
        @test r.info == Int(Feast_ERROR_M0)
    end
    @testset "High-level parameter wrappers" begin
        coeffs = [-Matrix(Diagonal(ComplexF64[1,2,3])),Matrix{ComplexF64}(I,3,3)]
        r = FeastKit.feast_polynomial(coeffs,1.5+0im,0.75;M0=3,fpm=feastinit())
        @test r.info == 0
        @test r.M == 2
        r = feast_matvec((y,x)->copyto!(y,x),(y,x)->copyto!(y,x),3,(0.5,1.5);M0=3,fpm=feastinit())
        @test r.info == 0
        @test r.M == 3
    end
end
