using Test, FeastKit, LinearAlgebra, SparseArrays

@testset "Hermitian projector and API review" begin
    @testset "Partial Hermitian subspace $storage $solver cache=$cache" for
            storage in (:dense, :sparse, :banded), solver in (:direct, :gmres), cache in (0, 1)
        n = 20
        # A genuinely complex Hermitian pencil with eigenvalues 1:20.
        U = Matrix{ComplexF64}(I,n,n)
        U[[1,n],[1,n]] = [1 im; im 1] / sqrt(2)
        mass = collect(range(1.,2.;length=n))
        A = Matrix(Hermitian(U * Diagonal((1:n).*mass) * U'))
        B = Matrix(Hermitian(U * Diagonal(mass) * U'))
        f = feastinit().fpm
        f[4] = 6
        f[10] = cache
        r = if storage == :dense
            feast_hegv!(A,B,0.5,3.5,4,f;solver=solver,solver_tol=1e-13)
        elseif storage == :sparse
            feast_hcsrgv!(sparse(A),sparse(B),0.5,3.5,4,f;solver=solver,solver_tol=1e-13)
        else
            feast_hbgv!(full_to_banded(A,n-1),full_to_banded(B,n-1),n-1,n-1,
                        0.5,3.5,4,f;solver=solver,solver_tol=1e-13)
        end
        @test r.info == 0
        @test r.M == 3
        @test r.lambda ≈ [1.,2.,3.] atol=1e-9
        @test all(j -> norm(A*r.q[:,j]-r.lambda[j]*B*r.q[:,j]) < 1e-10, 1:r.M)
        @test all(j -> norm(r.q[:,j]) ≈ 1, 1:r.M)
    end
    @testset "Custom Hermitian half contour" begin
        A = Matrix(Diagonal(ComplexF64.(1:20)))
        f = feastinit().fpm; f[4] = 6
        c = feast_contour(0.5,3.5,f)
        r = FeastKit.with_custom_contour(f,c) do
            feast(A,(0.5,3.5);M0=4,fpm=f)
        end
        @test r.info == 0
        @test r.lambda ≈ [1.,2.,3.] atol=1e-9
    end
    @testset "High-level parallel wrapper rejects manual RCI" begin
        A = Matrix(Diagonal([1.,2.,3.]))
        @test_throws ArgumentError feast_parallel(A,Matrix{Float64}(I,3,3),
                                                  (0.5,2.5);M0=3,auto_rci=false)
    end
    @testset "Abstract polynomial coefficient storage $storage" for storage in (sparse, Diagonal, Hermitian)
        coeffs = [storage(-Matrix(Diagonal(ComplexF64[1,2,3]))),
                  storage(Matrix{ComplexF64}(I,3,3))]
        original = deepcopy(coeffs)
        r = FeastKit.feast_polynomial(coeffs,1.5+0im,0.75;M0=3)
        @test r.info == 0
        @test r.M == 2
        @test sort(real.(r.lambda)) ≈ [1.,2.] atol=1e-9
        @test coeffs == original
    end
end
