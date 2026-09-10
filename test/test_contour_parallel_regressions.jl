using Test, FeastKit, LinearAlgebra, SparseArrays

@testset "Contour membership and parallel RCI numerics" begin
    @testset "Parallel filtered eigenvectors and generalized residuals" begin
        for scale in (1.0, 2.0)
            A = Matrix(Diagonal(scale .* collect(1.0:6.0)))
            B = scale * Matrix{Float64}(I,6,6)
            r = feast_parallel(A,B,(0.5,2.5); M0=3)
            @test r.info == 0
            @test r.M == 2
            @test r.lambda ≈ [1.,2.] atol=1e-8
            for j in 1:r.M
                actual = norm(A*r.q[:,j]-r.lambda[j]*B*r.q[:,j]) /
                         (max(abs(r.lambda[j]),1.0)*norm(r.q[:,j]))
                @test actual < 1e-8
                @test r.res[j] ≈ actual atol=1e-12
            end
        end
    end
    @testset "Parallel exit status" begin
        A = Matrix{Float64}(I,3,3)
        for m in (1,3)
            r = feast_parallel(A,A,(0.5,1.5);M0=m)
            @test r.info == (m == 1 ? Int(Feast_ERROR_M0) : 0)
        end
        A = Matrix(Diagonal(collect(1.0:20.0)))
        f = feastinit().fpm; f[2]=4; f[4]=1
        r = feast_parallel(A,Matrix{Float64}(I,20,20),(2.5,5.5);M0=4,fpm=f)
        @test maximum(r.res) > 1e-8
        # With this deliberately coarse filter, platform-dependent Ritz pairs
        # can fill the trial subspace. Saturation takes precedence over the
        # iteration-limit status; neither case may report success.
        expected = r.M == 4 ? Int(Feast_ERROR_M0) : Int(Feast_ERROR_NO_CONVERGENCE)
        @test r.info == expected
    end
    @testset "Unbatched parallel RCI uses the same kernel" begin
        N=6; M0=3
        A=Matrix(Diagonal(2 .* collect(1.0:N))); B=2*Matrix{Float64}(I,N,N)
        f=feastinit().fpm; feastdefault!(f)
        s=ParallelFeastState{Float64}(f[2],M0,false,false)
        work=zeros(N,M0); FeastKit._feast_seeded_subspace!(work)
        workc=zeros(ComplexF64,N,M0); Aq=zeros(M0,M0); Sq=zeros(M0,M0)
        lambda=zeros(M0); q=zeros(N,M0); res=zeros(M0); factor=nothing
        for _ in 1:10000
            pfeast_srci!(s,N,work,workc,Aq,Sq,f,0.5,2.5,M0,lambda,q,res)
            s.ijob == Int(Feast_RCI_DONE) && break
            if s.ijob == Int(Feast_RCI_FACTORIZE)
                factor=lu(s.Ze*B-A)
            elseif s.ijob == Int(Feast_RCI_SOLVE)
                workc .= factor \ (B*work)
            elseif s.ijob == Int(Feast_RCI_MULT_A)
                work[:,1:s.mode] .= A*q[:,1:s.mode]
            elseif s.ijob == Int(Feast_RCI_MULT_B)
                work[:,1:s.mode] .= B*q[:,1:s.mode]
            else
                error("Unexpected unbatched RCI job $(s.ijob)")
            end
        end
        @test s.ijob == Int(Feast_RCI_DONE)
        @test s.info == 0
        @test s.mode == 2
        @test lambda[1:s.mode] ≈ [1.,2.] atol=1e-8
        @test maximum(res[1:s.mode]) < 1e-8
    end
    @testset "Polygon orientation, concavity, and boundary" begin
        nodes=ComplexF64[0,2,2+1im,1+1im,1+2im,2im]
        for polygon in (nodes,reverse(nodes))
            @test FeastKit._feast_inside_polygon(0.5+1.5im,polygon)
            @test !FeastKit._feast_inside_polygon(1.5+1.5im,polygon)
            @test FeastKit._feast_inside_polygon(1.0+1im,polygon)
            @test !FeastKit._feast_inside_polygon(ComplexF64(NaN),polygon)
        end
        for method in (0,1)
            f=feastinit().fpm; f[16]=method
            c=feast_gcontour(1.5+0im,0.75,f)
            @test FeastKit._feast_inside_polygon(1.0+0im,c.Zne)
            @test FeastKit._feast_inside_polygon(2.0+0im,c.Zne)
            @test !FeastKit._feast_inside_polygon(3.0+0im,c.Zne)
            # Real-valued Ritz arrays are platform-dependent LAPACK output.
            @test FeastKit._feast_inside_general_region(1.0,1.5+0im,0.75,f)
            FeastKit.with_custom_contour(f,c) do
                @test FeastKit._feast_inside_general_region(1.0,1.5+0im,0.75,f)
                @test !FeastKit._feast_inside_general_region(3.0,1.5+0im,0.75,f)
            end
        end
    end
    @testset "Custom rectangle excludes nominal-circle eigenvalues" begin
        nodes = ComplexF64[]; weights = ComplexF64[]
        corners = ComplexF64[-2-0.2im,2-0.2im,2+0.2im,-2+0.2im]
        for e in 1:4
            a = corners[e]; dz=(corners[mod1(e+1,4)]-a)/64
            for j in 1:64
                push!(nodes,a+(j-0.5)*dz); push!(weights,dz/(2π*im))
            end
        end
        for storage in (Matrix,sparse)
            A = storage(Matrix(Diagonal(ComplexF64[0.5,0.5+1im,4.])))
            f = feastinit().fpm; f[16]=1
            r = FeastKit.with_custom_contour(f,nodes,weights) do
                feast_general(A,0.0+0im,2.1;M0=3,fpm=f,backend=:serial)
            end
            @test r.info == 0
            @test r.M == 1
            @test r.lambda ≈ [0.5+0im] atol=1e-8
        end
        A=full_to_general_banded(Matrix(Diagonal(ComplexF64[0.5,0.5+1im,4.])),0)
        f=feastinit().fpm; f[16]=1
        r=feast_gbevx!(A,0,0.0+0im,2.1,3,f,nodes,weights)
        @test r.info == 0
        @test r.M == 1
        @test r.lambda ≈ [0.5+0im] atol=1e-8
    end
end
