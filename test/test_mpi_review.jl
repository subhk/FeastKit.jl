# Called with MPI initialized by the backend test driver.
function test_mpi_review(comm)
    test_mpi_safety(comm)
@testset "MPI review regressions" begin
    @testset "Partial Hermitian projector $storage $solver" for storage in (Matrix,sparse), solver in (:direct,:gmres)
        n = 20
        U = Matrix{ComplexF64}(I,n,n)
        U[[1,n],[1,n]] = [1 im; im 1] / sqrt(2)
        mass = collect(range(1.,2.;length=n))
        A = storage(Matrix(Hermitian(U*Diagonal((1:n).*mass)*U')))
        B = storage(Matrix(Hermitian(U*Diagonal(mass)*U')))
        f = feastinit().fpm; f[4] = 6
        # Only root registers the public half contour.
        root = MPI.Comm_size(comm)-1
        c = feast_contour(0.5,3.5,f)
        solve = () -> mpi_feast(A,B,(0.5,3.5);M0=4,fpm=f,comm=comm,root=root,
                                solver=solver,solver_tol=1e-13)
        r = MPI.Comm_rank(comm) == root ? FeastKit.with_custom_contour(solve,f,c) : solve()
        @test r.info == 0
        @test r.M == 3
        @test r.lambda ≈ [1.,2.,3.] atol=1e-9
        @test all(j -> norm(A*r.q[:,j]-r.lambda[j]*B*r.q[:,j]) < 1e-10, 1:r.M)
        @test all(j -> norm(r.q[:,j]) ≈ 1, 1:r.M)
    end
    @testset "Custom contour" begin
        A = Matrix(Diagonal(ComplexF64[1,2,5]))
        c = feast_gcontour(1.0+0im,0.25,feastinit().fpm)
        f = feastinit().fpm
        r = FeastKit.with_custom_contour(f,c) do
            feast_general(A,5.0+0im,0.4;M0=2,fpm=f,backend=:mpi,comm=comm)
        end
        @test r.info == 0
        @test r.M == 1
        @test r.lambda ≈ [1.0+0im] atol=1e-8
    end
    @testset "Root-only custom contour $storage $solver" for storage in (Matrix,sparse), solver in (:direct,:gmres)
        root = MPI.Comm_size(comm)-1
        A = storage(Matrix(Diagonal(ComplexF64[1,2,5])))
        c = feast_gcontour(1.0+0im,0.25,feastinit().fpm)
        f = feastinit().fpm
        run_solver = ()->mpi_feast_general(A,5.0+0im,0.4;M0=2,fpm=f,comm=comm,root=root,solver=solver)
        r = if MPI.Comm_rank(comm) == root
            FeastKit.with_custom_contour(run_solver,f,c)
        else
            run_solver()
        end
        @test r.info == 0
        @test r.M == 1
        @test r.lambda ≈ [1.0+0im] atol=1e-8
        @test FeastKit.feast_get_custom_contour(Float64,f) === nothing
    end
    @testset "Hybrid filtered subspace, threads=$threaded" for threaded in (false,true)
        A = Matrix(Diagonal(collect(1.0:6))); B = Matrix{Float64}(I,6,6)
        r = feast_hybrid(A,B,(0.5,2.5);M0=3,comm=comm,use_threads_per_rank=threaded)
        @test r.info == 0
        @test r.M == 2
        @test r.lambda ≈ [1.,2.] atol=1e-8
        @test maximum(r.res) < 1e-8
    end
    @testset "Hybrid generalized and saturation threads=$threaded" for threaded in (false,true)
        for m in (1,3)
            A = 2Matrix{Float64}(I,3,3); B = copy(A)
            r = feast_hybrid(A,B,(0.5,1.5);M0=m,fpm=feastinit(),comm=comm,use_threads_per_rank=threaded)
            @test r.info == (m == 1 ? Int(Feast_ERROR_M0) : 0)
            @test r.M == m
            @test maximum(r.res) < 1e-8
        end
    end
    @testset "Collective singular factorization" begin
        f = feastinit().fpm; f[16]=1
        c = feast_gcontour(0.0+0im,1.0,f)
        A = Matrix(Diagonal(ComplexF64[c.Zne[1],3,4]))
        r = try
            mpi_feast_general(A,0.0+0im,1.0;M0=2,fpm=f,comm=comm)
        catch
            nothing
        end
        @test r isa FeastGeneralResult
        if r isa FeastGeneralResult
            @test r.info == Int(Feast_ERROR_LAPACK)
            @test r.M == 0
        end
    end
    @testset "Collective singular symmetric shift $T $storage" for T in (Float64,ComplexF64), storage in (Matrix,sparse)
        # Only the rank owning the first node sees the singular shifted system.
        c = feast_contour(0.5,1.5,feastinit().fpm)
        c.Zne[1] = 1.0+0im
        f = feastinit().fpm
        A = storage(Matrix(Diagonal(T[1,3,4])))
        r = FeastKit.with_custom_contour(f,c) do
            mpi_feast(A,(0.5,1.5);M0=2,fpm=f,comm=comm)
        end
        @test r.info == Int(Feast_ERROR_LAPACK)
        @test r.M == 0
    end
    @testset "Sparse standard MPI keeps scalar type $T" for T in (Float32,Float64)
        A = sparse(Matrix(Diagonal(T[1,2,3])))
        r = mpi_feast(A,(T(0.5),T(2.5));M0=3,comm=comm)
        @test r.info == 0
        @test r.M == 2
        @test r.lambda ≈ T[1,2] atol=1e-4
    end
end
end
