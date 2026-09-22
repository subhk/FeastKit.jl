using Test, FeastKit, Krylov, LinearAlgebra, SparseArrays

real_projection_callback(A::AbstractMatrix{T}) where T =
    (y::AbstractVector{T},x::AbstractVector{T})->mul!(y,A,x)

@testset "Serial projection and scaling regressions" begin
    @testset "GMRES near invariant-subspace breakdown" begin
        T = Float32
        A = T[2 1 0; 1 3 1; 0 1 4]
        B = T[2 .5 0; .5 1 0; 0 0 1]
        Q = zeros(T,3,3)
        FeastKit._feast_seeded_subspace!(Q)
        fpm = feastinit().fpm
        feastdefault!(fpm)
        z = first(feast_contour(T(.1),T(5),fpm).Zne)
        op = FeastKit.MatrixFreeShiftedOperator(3,z,
            real_projection_callback(A),real_projection_callback(B),T)
        rhs = Complex{T}.(B*Q[:,2])
        # Julia 1.10 previously reported breakdown before meeting 1e-6 here.
        for warm in (false,true)
            x, solved = warm ? FeastKit._feast_gmres(op,rhs,zeros(Complex{T},3);
                rtol=T(1e-6),atol=T(1e-12)) : FeastKit._feast_gmres(op,rhs;
                rtol=T(1e-6),atol=T(1e-12))
            @test solved
            @test norm((z*B-A)*x-rhs)/norm(rhs) <= T(1e-6)
        end
        workspace = FeastKit._feast_gmres_workspace(3,Complex{T})
        @test FeastKit._feast_gmres!(workspace,op,rhs;rtol=T(1e-6),atol=T(1e-12))
        x = FeastKit._feast_gmres_solution(workspace)
        @test norm((z*B-A)*x-rhs)/norm(rhs) <= T(1e-6)
    end
    @testset "Noncommuting callback pencil $T" for T in (Float32,Float64)
        A=T[2 1 0;1 3 1;0 1 4]
        B=T[2 .5 0;.5 1 0;0 0 1]
        # These callbacks intentionally accept only real vectors.
        amul=real_projection_callback(A)
        bmul=real_projection_callback(B)
        r=feast_matvec(amul,bmul,3,(T(0.1),T(5));M0=3)
        @test r.info == 0
        @test r.M == 3
        @test r.M == 3 && isapprox(sort(r.lambda),eigvals(Symmetric(A),Symmetric(B));rtol=1e-5)
        @test r.M == 3 && maximum(norm(A*r.q[:,j]-r.lambda[j]*B*r.q[:,j]) for j=1:3) < 1e-4
    end

    @testset "Assembled iterative pencil $T $storage scale=$scale" for
            T in (Float32,Float64), storage in (Matrix,sparse), scale in (1.,1e-9,1e-20)
        for complex in (false,true)
            CT=complex ? Complex{T} : T
            A=storage(Matrix(Diagonal(CT.(T(scale).*T[1,2,3]))))
            B=storage(T(scale)*Matrix{CT}(I,3,3))
            f=feastinit().fpm
            solver=complex ? (storage==Matrix ? feast_hegv! : feast_hcsrgv!) :
                (storage==Matrix ? feast_sygv! : feast_scsrgv!)
            r=solver(A,B,T(.5),T(2.5),3,f;solver=:gmres)
            @test r.info == 0
            @test r.M == 2
            @test r.M == 2 && isapprox(sort(r.lambda),T[1,2];rtol=1e-5)
        end
    end

    @testset "Shifted block zero RHS and normalized residual" begin
        for storage in (Matrix,sparse), s in (1.,1e-20)
            A=storage(Matrix(Diagonal(ComplexF64.(s.*[1,2,3]))))
            B=storage(s*Matrix{ComplexF64}(I,3,3))
            z=4.0+im
            rhs=s.*ComplexF64[1 0;2 0;3 0]
            dest=similar(rhs)
            success=FeastKit._feast_shifted_solve!(dest,rhs,A,B,z,2,1e-12,100,3)
            @test success
            @test norm((z*B-A)*dest-rhs)/norm(rhs) < 1e-10
            @test iszero(dest[:,2])
        end
    end

    @testset "Sparse standard shifted solve and callback failure status" begin
        A=spdiagm(0=>ComplexF64[1,2,3])
        rhs=ComplexF64[1 0;2 0;3 0]; dest=similar(rhs); z=4.0+im
        @test FeastKit.solve_shifted_iterative_identity!(dest,rhs,A,z,1e-12,100,3)
        @test norm((z*I-A)*dest-rhs)/norm(rhs) < 1e-10
        @test iszero(dest[:,2])
        fill!(rhs,ComplexF64(NaN))
        @test !FeastKit.solve_shifted_iterative_identity!(dest,rhs,A,z,1e-12,100,3)

        a=Matrix(Diagonal([1.,2,3,4]))
        r=FeastKit.feast_sparse_matvec!(real_projection_callback(a),
            (y,x)->copyto!(y,x),4,.5,3.5,4,feastinit().fpm;
            gmres_rtol=1e-14,gmres_atol=0.,gmres_maxiter=1)
        @test r.info == Int(Feast_ERROR_NO_CONVERGENCE)
    end
end
