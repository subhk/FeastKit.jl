# Called with MPI initialized. Keep rank-local fault injection in a separate
# process (test_mpi_faults.jl), so these numerical tests use ordinary methods.
function test_mpi_safety(comm)
    ext = Base.get_extension(FeastKit,:FeastKitMPIExt)
    @testset "MPI API and cache policy" begin
        @testset "Parameter wrapper $storage generalized=$generalized" for storage in (Matrix,sparse), generalized in (false,true)
            A = storage(Matrix(Diagonal([1.,2.,3.])))
            B = storage(Matrix{Float64}(I,3,3))
            r = generalized ? mpi_feast(A,B,(0.5,2.5);M0=3,fpm=feastinit(),comm=comm) :
                mpi_feast(A,(0.5,2.5);M0=3,fpm=feastinit(),comm=comm)
            @test r.info == 0
            @test r.lambda ≈ [1.,2.]
        end
        @testset "Uncached factors do not retain LU" begin
            A = Matrix(Diagonal([1.,2.,3.])); B = Matrix{Float64}(I,3,3)
            factors = ext._mpi_factorize_contour(A,B,ComplexF64[1+im,2+im],comm;store=false)
            @test factors !== nothing
            @test !(factors isa Vector)
            @test factors[1] !== factors[1]
        end
        @testset "Uncached $storage $kind" for storage in (Matrix,sparse), kind in (:real,:hermitian,:general)
            T = kind == :real ? Float64 : ComplexF64
            A = storage(Matrix(Diagonal(T.(1:20)))); B = storage(Matrix{T}(I,20,20))
            f = feastinit().fpm; f[10]=0
            r = kind == :general ? mpi_feast_general(A,B,2.0+0im,1.5;M0=5,fpm=f,comm=comm) :
                mpi_feast(A,B,(0.5,3.5);M0=5,fpm=f,comm=comm)
            @test r.info == 0
            @test r.M == 3
            @test sort(real.(r.lambda)) ≈ [1.,2.,3.] atol=1e-8
        end
    end
end
