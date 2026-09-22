using Logging

# Called with MPI initialized by the backend test driver.
function test_mpi_api(comm)
    test_mpi_safety(comm)
@testset "MPI public API" begin
    @testset "Named high-level MPI options" begin
        options = (; subspace_size=3, tol=1e-10, maxiter=30,
                     quadrature_points=16, backend=:mpi, comm=comm,
                     solver=:gmres, solver_opts=(rtol=1e-13, maxiter=100, restart=8))
        A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
        @test_throws ArgumentError feast(A, (0.5, 2.5); options...)
        for storage in (Matrix, sparse)
            H = storage(ComplexF64.(A))
            B = storage(Matrix{ComplexF64}(I, 4, 4))
            for r in (feast(H, (0.5, 2.5); options...),
                      feast(H, B, (0.5, 2.5); options...),
                      feast_general(storage(A), 1.5+0im, 0.8; options...),
                      feast_general(H, B, 1.5+0im, 0.8; options...))
                @test r.converged
                @test sort(real.(r.values)) ≈ [1.0, 2.0] atol=1e-9
            end
            contour = feast_rectangle(0.5, 2.5, -1, 1)
            for r in (feast(H, contour; options..., quadrature_points=128),
                      feast(H, B, contour; options..., quadrature_points=128))
                @test r.converged
                @test sort(real.(r.values)) ≈ [1.0, 2.0] atol=1e-9
            end
        end
    end
    @testset "Rectangle geometry broadcast from a nonzero root" begin
        root = MPI.Comm_size(comm)-1
        A = Matrix(Diagonal(ComplexF64[0, 0.99+0.99im, 2]))
        f = feastinit().fpm; f[16] = 1
        c = feast_rectangle(-1f0, 1f0, -1f0, 1f0)
        solve = () -> mpi_feast_general(A, 0.0+0im, 3.0; M0=3, fpm=f, comm=comm, root=root)
        r = MPI.Comm_rank(comm) == root ? FeastKit.with_custom_contour(solve, f, c) : solve()
        @test r.info == 0
        @test r.M == 2
        @test r.lambda ≈ [0, 0.99+0.99im] atol=1e-9
    end
    @testset "MPI scaled partial pencil $kind" for kind in (:real, :hermitian, :general)
        n = 20
        A0 = Matrix(SymTridiagonal(fill(2.0,n), fill(-1.0,n-1)))
        expected = filter(x -> 0.5 <= x <= 1.5, eigvals(Symmetric(A0)))
        CT = kind == :real ? Float64 : ComplexF64
        A = sparse(CT.(1e-12*A0)); B = spdiagm(0 => fill(CT(1e-12), n))
        r = kind == :general ? mpi_feast_general(A,B,1.0+0im,0.5; M0=6,comm=comm) :
                               mpi_feast(A,B,(0.5,1.5); M0=6,comm=comm)
        @test r.info == 0
        @test r.M == length(expected)
        @test sort(real.(r.lambda)) ≈ expected atol=1e-9
        @test maximum(norm(A0*r.q[:,j]-r.lambda[j]*r.q[:,j]) for j in 1:r.M) < 2e-12
    end
    @testset "Tiny general custom contour $T $storage" for T in (Float32,Float64), storage in (Matrix,sparse)
        scale = T == Float32 ? T(1e-8) : T(1e-16)
        center, radius = Complex{T}(T(1.5)*scale), T(0.75)*scale
        A = storage(Matrix(Diagonal(Complex{T}.(scale .* T[1,2,2.4]))))
        f = feastinit().fpm
        c = feast_gcontour(center,radius,f)
        root = MPI.Comm_size(comm)-1
        solve = () -> mpi_feast_general(A,center,radius;M0=3,fpm=f,comm=comm,root=root)
        r = with_logger(NullLogger()) do
            MPI.Comm_rank(comm) == root ? FeastKit.with_custom_contour(solve,f,c) : solve()
        end
        @test r.info == 0
        @test r.M == 2
        @test r.M == 2 && isapprox(sort(real.(r.lambda ./ scale)),T[1,2];rtol=1e-5)
    end
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
