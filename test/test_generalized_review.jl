using Test, FeastKit, LinearAlgebra, SparseArrays

@testset "Generalized and specialized review regressions" begin
    @testset "Matrix-free mass-weighted projector $T" for T in (Float64, ComplexF64)
        B = Matrix(Diagonal(T[1e10,1,1]))
        A = B * Diagonal(T[1,2,3])
        ao = LinearOperator{T}((y,x)->mul!(y,A,x),(3,3); issymmetric=true)
        bo = LinearOperator{T}((y,x)->mul!(y,B,x),(3,3); issymmetric=true)
        solve = (Y,z,X)->copyto!(Y,(z*B-A)\X)
        r = T === Float64 ? feast(ao,bo,(0.5,2.5); M0=3,solver=solve,tol=1e-5) :
            feast_general(ao,bo,1.5+0im,1.0; M0=3,solver=solve)
        @test r.info == 0
        @test r.M == 2
        @test sort(real.(r.lambda)) ≈ [1.,2.] atol=(T === Float64 ? 1e-5 : 1e-8)
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
