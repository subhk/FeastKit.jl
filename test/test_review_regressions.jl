using Test, FeastKit, LinearAlgebra, SparseArrays

review_fpm() = (f = zeros(Int, 64); feastinit!(f); f)

@testset "Review regressions" begin
    @testset "Single precision sparse solves" begin
        for CT in (Float32, ComplexF32), generalized in (false, true), cache in (0, 1)
            A = spdiagm(0 => CT[1, 2, 3])
            B = spdiagm(0 => ones(CT, 3))
            f = review_fpm(); f[10] = cache
            r = generalized ? feast(A, B, (0.5f0, 2.5f0); M0=3, fpm=f) :
                              feast(A, (0.5f0, 2.5f0); M0=3, fpm=f)
            @test r.info == 0
            @test r.M == 2
            @test eltype(r.lambda) == Float32
            @test eltype(r.q) == CT
            @test r.lambda ≈ Float32[1, 2] atol=1f-4
            # Float32 convergence has a sqrt(eps) floor. Check the actual
            # per-pair relative residual rather than a tighter absolute block
            # norm, whose value also grows with the number of eigenvectors.
            relative_residual = maximum(
                norm(A*r.q[:,j] - r.lambda[j]*r.q[:,j]) /
                (max(abs(r.lambda[j]), 1f0) * norm(r.q[:,j])) for j in 1:r.M)
            @test relative_residual <= 1.01f0 * FeastKit.feast_tolerance(f, Float32)
        end
        for solver in (FeastKit.feast_gcsrgv!, FeastKit._feast_sparse_complex_symmetric)
            A = spdiagm(0 => ComplexF32[1,2,3]); B = spdiagm(0 => ones(ComplexF32,3))
            f=review_fpm(); f[3]=5
            r=solver(A,B,ComplexF32(1.5),0.75f0,3,f)
            @test r.info == 0
            @test r.M == 2
            @test sort(real.(r.lambda)) ≈ Float32[1,2] atol=1f-4
        end
    end

    @testset "Empty intervals do not expose the projection rank" begin
        for CT in (Float64, ComplexF64), storage in (Matrix, sparse)
            A = storage(Diagonal(CT[1, 2, 3]))
            r = feast(A, (1.25, 1.75); M0=3)
            @test r.M == 0
            @test isempty(r.lambda)
            @test isempty(r.res)
            @test size(r.q, 2) == 0
        end
        r = feast_general(Matrix(Diagonal(ComplexF64[1,2,3])), 5.0+0im, 0.5; M0=3)
        @test r.M == 0
        @test isempty(r.lambda)
    end

    @testset "Expert RCI preserves explicit state" begin
        for kind in (:real, :hermitian, :general)
            n = 3; m = 3; f = review_fpm()
            general = kind == :general; complex_q = kind != :real
            c = general ? feast_gcontour(1.5+0im, 0.75, f) : feast_contour(0.5, 2.5, f)
            fn = kind == :real ? FeastKit.feast_srcix! :
                 kind == :hermitian ? FeastKit.feast_hrcix! : FeastKit.feast_grcix!
            state = kind == :real ? FeastKit.FeastSRCIState{Float64}() :
                    kind == :hermitian ? FeastKit.FeastHRCIState{Float64}() : FeastKit.FeastGRCIState{Float64}()
            w = zeros(n,m); wc = zeros(ComplexF64,n,m)
            aq = complex_q ? zeros(ComplexF64,m,m) : zeros(m,m); sq = copy(aq)
            q = complex_q ? zeros(ComplexF64,n,m) : zeros(n,m)
            lam = general ? zeros(ComplexF64,m) : zeros(m); res = zeros(m)
            job=Ref(-1); z=Ref(0.0+0im); ep=Ref(0.0); lp=Ref(0); mode=Ref(0); info=Ref(0)
            A = Diagonal([1.,2.,3.])
            for step in 1:1000
                fn(job,n,z,w,wc,aq,sq,f,ep,lp,general ? 1.5+0im : 0.5,
                   general ? 0.75 : 2.5,m,lam,q,mode,res,info,c.Zne,c.Wne; state=state)
                job[] == 0 && break
                if job[] == 11
                    wc .= (z[]*I-A) \ (complex_q ? wc : w)
                elseif job[] in (30,40)
                    dest = complex_q ? wc : w
                    dest[:,1:mode[]] .= job[] == 30 ? A*q[:,1:mode[]] : q[:,1:mode[]]
                end
            end
            @test job[] == 0
            @test info[] == 0
            @test mode[] == 2
            @test sort(real.(lam[1:mode[]])) ≈ [1.,2.] atol=1e-8
            @test f[29] == 0
        end
    end

    @testset "Polynomial probe saturation is not certified" begin
        for m in (1,3)
            n=3; f=review_fpm(); f[5]=1
            c=feast_gcontour(2.0+0im,2.0,f)
            w=Matrix{ComplexF64}(I,n,m); wc=copy(w)
            aq=zeros(ComplexF64,m,m); bq=copy(aq); q=copy(w)
            lam=zeros(ComplexF64,m); res=zeros(m)
            job=Ref(-1); z=Ref(0.0+0im); ep=Ref(0.0); lp=Ref(0); mode=Ref(0); info=Ref(0)
            state=FeastKit.FeastPolyRCIState{Float64}(); A=Diagonal(ComplexF64[1,2,3])
            for step in 1:1000
                FeastKit.feast_grcipevx!(job,1,n,z,w,wc,aq,bq,f,ep,lp,2.0+0im,2.0,m,
                    lam,q,mode,res,info,c.Zne,c.Wne; state=state)
                job[] == 0 && break
                if job[] == 11
                    wc .= (z[]*I-A) \ w
                elseif job[] == 30
                    for j in 1:mode[]
                        wc[:,j] .= (lam[j]*I-A)*q[:,j]
                    end
                end
            end
            @test info[] == (m == 1 ? Int(FeastKit.Feast_ERROR_M0) : 0)
            @test mode[] == m
            @test maximum(res) < 1e-8
        end
    end

    @testset "Structured count estimator inputs" begin
        A = Diagonal([1.,2.,3.]); f=review_fpm(); f[2]=32
        reference=feast_estimate_count(Matrix(A),(0.5,2.5); fpm=f)
        for wrapped in (A, Symmetric(Matrix(A)), Hermitian(sparse(A)), view(Matrix(A),:,:))
            @test feast_estimate_count(wrapped,(0.5,2.5); fpm=f) ≈ reference atol=1e-10
        end
        @test feast_estimate_count(Diagonal([1,2,3]),(0,4); fpm=f) ≈ 3 atol=1e-6
        @test feast_estimate_count(A,(0.5,2.5); B=Diagonal(ones(3)),fpm=f) ≈ reference atol=1e-10
        @test feast_estimate_count(sparse(A),(0.5,2.5); B=Diagonal(ones(Float32,3)),fpm=f) ≈ reference atol=1e-10
    end
end
