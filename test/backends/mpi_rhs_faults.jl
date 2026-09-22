using Test, FeastKit, MPI, Krylov, LinearAlgebra, SparseArrays

# A valid AbstractMatrix that simulates a rank-local failure in B*Q. Its
# entries, dimensions and solver parameters are identical on every rank.
mutable struct FaultyMPIMass <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    fail::Bool
end
Base.size(B::FaultyMPIMass) = size(B.data)
Base.getindex(B::FaultyMPIMass,i::Int,j::Int) = B.data[i,j]
function LinearAlgebra.mul!(Y::AbstractMatrix,B::FaultyMPIMass,X::AbstractMatrix)
    B.fail && error("injected rank-local RHS failure")
    return mul!(Y,B.data,X)
end

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
@testset "MPI RHS failure synchronization" begin
    A = Matrix(Diagonal(2.0 .* (1:20)))
    B = FaultyMPIMass(2Matrix{Float64}(I,20,20),false)
    for threaded in (false,true)
        B.fail = false
        control = feast_hybrid(A,B,(0.5,3.5);M0=5,comm=comm,use_threads_per_rank=threaded)
        @test control.info == 0
        B.fail = rank == 1
        result = feast_hybrid(A,B,(0.5,3.5);M0=5,comm=comm,use_threads_per_rank=threaded)
        @test result.info == Int(Feast_ERROR_LAPACK)
        @test result.M == 0
        MPI.Barrier(comm)
    end
end

# Concrete complex/sparse public entry points do not accept custom matrix
# wrappers. In this dedicated process, intercept only multiplication by the
# selected B instance; all other calls retain the ordinary five-argument mul!.
const mpi_rhs_fault_target = Ref{Any}(nothing)
for BT in (Matrix{Float64}, SparseMatrixCSC{Float64,Int},
           Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64,Int})
    @eval function LinearAlgebra.mul!(Y::StridedMatrix{ComplexF64},B::$BT,X::StridedMatrix{ComplexF64})
        B === mpi_rhs_fault_target[] && error("injected concrete-matrix RHS failure")
        return mul!(Y,B,X,true,false)
    end
end
@testset "MPI concrete RHS failure $storage $kind $solver" for storage in (Matrix,sparse), kind in (:real,:hermitian,:general), solver in (:direct,:gmres)
    kind == :real && solver == :gmres && continue
    T = kind == :real ? Float64 : ComplexF64
    A = storage(Matrix(Diagonal(T.(2 .* (1:20)))))
    B = storage(2Matrix{T}(I,20,20))
    solve = () -> kind == :real ? mpi_feast(A,B,(0.5,3.5);M0=5,comm=comm) :
        kind == :hermitian ? mpi_feast(A,B,(0.5,3.5);M0=5,comm=comm,solver=solver) :
        mpi_feast_general(A,B,2.0+0im,1.5;M0=5,comm=comm,solver=solver)
    mpi_rhs_fault_target[] = nothing
    control = solve()
    @test control.info == 0
    @test control.M == 3
    @test control.epsout <= 1e-12
    mpi_rhs_fault_target[] = rank == 1 ? B : nothing
    result = solve()
    @test result.info == Int(solver == :direct ? Feast_ERROR_LAPACK : Feast_ERROR_NO_CONVERGENCE)
    @test result.M == 0
    mpi_rhs_fault_target[] = nothing
    MPI.Barrier(comm)
end
MPI.Finalize()
