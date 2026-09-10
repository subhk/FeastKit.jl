using Test, FeastKit, LinearAlgebra, Krylov

@testset "Serial solver edge regressions" begin
    @testset "Hermitian banded saturation $T generalized=$generalized M0=$m" for
            T in (Float32,Float64), generalized in (false,true), m in (1, 3)
        A = ones(Complex{T},1,3)
        r = generalized ? feast_hbgv!(A,copy(A),0,0,T(0.5),T(1.5),m,feastinit().fpm) :
            feast_hbev!(A,0,T(0.5),T(1.5),m,feastinit().fpm)
        @test r.info == (m == 1 ? Int(Feast_ERROR_M0) : 0)
        @test r.M == m
    end

    @testset "Scaled matrix-free pencil $T $method scale=$scale" for
            T in (Float32, ComplexF32, Float64, ComplexF64), method in (:gmres, :bicgstab), scale in (1.0, 1e-9, 1e-20)
        RT = typeof(real(zero(T)))
        A = Matrix(Diagonal(T.(scale .* [1,2,3])))
        B = RT(scale) * Matrix{T}(I,3,3)
        ao = LinearOperator{T}((y,x)->mul!(y,A,x),(3,3);issymmetric=true)
        bo = LinearOperator{T}((y,x)->mul!(y,B,x),(3,3);issymmetric=true)
        r = T <: Real ? feast(ao,bo,(RT(0.5),RT(2.5));M0=3,solver=method) :
            feast_general(ao,bo,Complex{RT}(1.5),RT(0.75);M0=3,solver=method)
        @test r.info == 0
        @test r.M == 2
        @test r.M == 2 && isapprox(sort(real.(r.lambda)),[1.,2.];atol=RT == Float32 ? 1e-5 : 1e-7)
    end

    @testset "Preconditioner is applied $T $method scale=$scale" for
            T in (Float32,Float64), method in (:gmres,:bicgstab), scale in (1.,1e-9)
        diagonal = Complex{T}[1,2,3]
        s = T(scale)
        A = LinearOperator{Complex{T}}((y,x)->(y .= s .* diagonal .* x),(3,3))
        B = LinearOperator{Complex{T}}((y,x)->(y .= s .* x),(3,3))
        z = Complex{T}(4,1)
        calls = Ref(0)
        P = LinearOperator{Complex{T}}((y,x)->(calls[]+=1; y .= x ./ (s .* (z .- diagonal))),(3,3))
        tol = T == Float32 ? 1e-5 : 1e-11
        solve = create_iterative_solver(A,B,method;preconditioner=P,maxiter=1,rtol=tol)
        Y = zeros(Complex{T},3,2)
        X = s .* Complex{T}[1 0; 2 0; 3 0]
        solve(Y,z,X)
        @test calls[] > 0
        @test norm(s .* (z .- diagonal).*Y-X) <= tol * norm(X)
        @test iszero(Y[:,2])
    end

    @testset "Complex user seed $kind overlap=$overlap budget=$budget" for
            kind in (:hermitian,:general), overlap in (0.,1e-12), budget in (1,20)
        n,m = 5,4
        A = Matrix(Diagonal(ComplexF64[1,2,3,10,11]))
        w = allocate_matfree_workspace(ComplexF64,n,m)
        w.workc[1,1]=1; w.workc[2,2]=1; w.workc[4,3]=1
        w.workc[3,3]=overlap; w.workc[5,4]=1
        f = feastinit().fpm; f[5]=1; f[4]=budget
        ij=Ref(-1); z=Ref(0.0+0im); ep=Ref(0.); loop=Ref(0); mode=Ref(0); info=Ref(0)
        state = kind == :hermitian ? FeastKit.FeastHRCIState{Float64}() : FeastKit.FeastGRCIState{Float64}()
        lam = kind == :hermitian ? zeros(m) : zeros(ComplexF64,m)
        for call in 1:10000
            if kind == :hermitian
                feast_hrci!(ij,n,z,w.work,w.workc,w.zAq,w.zSq,f,ep,loop,0.5,3.5,m,lam,w.q,mode,w.res,info;state=state)
            else
                feast_grci!(ij,n,z,w.work,w.workc,w.zAq,w.zSq,f,ep,loop,2.0+0im,1.5,m,lam,w.q,mode,w.res,info;state=state)
            end
            ij[] == Int(Feast_RCI_DONE) && break
            if ij[] == Int(Feast_RCI_SOLVE)
                w.workc .= (z[]*I-A)\w.workc
            elseif ij[] == Int(Feast_RCI_MULT_A)
                mul!(view(w.workc,:,1:mode[]),A,view(w.q,:,1:mode[]))
            elseif ij[] == Int(Feast_RCI_MULT_B)
                copyto!(view(w.workc,:,1:mode[]),view(w.q,:,1:mode[]))
            end
        end
        @test ij[] == Int(Feast_RCI_DONE)
        @test 0 < loop[] <= budget
        if budget == 20
            @test info[] == 0
            @test mode[] == 3
            @test mode[] == 3 && isapprox(sort(real.(lam[1:mode[]])),[1.,2.,3.];atol=1e-8)
        else
            # A verification sweep must not bypass the residual criterion or
            # the caller's refinement budget.
            @test info[] == (ep[] <= FeastKit.feast_tolerance(f,Float64) ? 0 : Int(Feast_ERROR_NO_CONVERGENCE))
            @test info[] != 0 || mode[] == 3
        end
    end
end
