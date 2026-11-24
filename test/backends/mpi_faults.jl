using Test, FeastKit, MPI, LinearAlgebra, SparseArrays
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

struct FaultyResidualMatrix <: AbstractMatrix{ComplexF64}
    data::Matrix{ComplexF64}
    fail::Bool
end
Base.size(A::FaultyResidualMatrix) = size(A.data)
Base.getindex(A::FaultyResidualMatrix,i::Int,j::Int) = A.data[i,j]
function LinearAlgebra.mul!(y::AbstractVector,A::FaultyResidualMatrix,x::AbstractVector)
    A.fail && error("injected rank-local residual failure")
    return mul!(y,A.data,x)
end
ext = Base.get_extension(FeastKit,:FeastKitMPIExt)

@testset "MPI empty-region residual failure" begin
    A_empty = Matrix(Diagonal(ComplexF64[3,4]))
    B_empty = Matrix{ComplexF64}(I,2,2)
    root = MPI.Comm_size(comm) - 1
    solve(fail) = ext._mpi_feast_complex_general!(
        FaultyResidualMatrix(A_empty, fail), B_empty, 0.0+0im, 1.0, 2,
        feastinit().fpm; comm=comm, root=root)
    control = solve(false)
    @test control.info == Int(Feast_SUCCESS)
    @test control.M == 0

    # Only root tests the residual of the nearest out-of-region Ritz pair.
    result = solve(rank == root)
    @test result.info == Int(Feast_ERROR_LAPACK)
    @test result.M == 0
end

# Inject only into the selected rank's QR, after a successful control solve.
# The real driver compresses its filtered subspace in real arithmetic.
A = Matrix(Diagonal(1.:20.)); B = Matrix{Float64}(I,20,20)
control = FeastKit.mpi_feast_sygv!(A,B,0.5,3.5,4,feastinit().fpm;comm=comm)
@test control.info == 0
if rank == 1
    @eval FeastKit function _feast_qr_compress!(dest::Matrix{Float64},src::AbstractMatrix{Float64},n::Int;kwargs...)
        error("injected rank-local projected QR failure")
    end
end
r = FeastKit.mpi_feast_sygv!(A,B,0.5,3.5,4,feastinit().fpm;comm=comm)
@test r.info == Int(Feast_ERROR_LAPACK)
@test r.M == 0
MPI.Barrier(comm)
# Complex drivers have a separate projected-QR path. Inject on one rank only,
# after the real-driver checks, and cover both public storage families.
if rank == 1
    @eval LinearAlgebra function qr!(A::Matrix{ComplexF64};kwargs...)
        error("injected rank-local complex QR failure")
    end
end
for storage in (Matrix,sparse), general in (false,true)
    Az = storage(complex.(A)); Bz = storage(complex.(B))
    local r = general ? mpi_feast_general(Az,Bz,2.0+0im,1.5;M0=4,comm=comm) :
        mpi_feast(Az,Bz,(0.5,3.5);M0=4,comm=comm)
    @test r.info == Int(Feast_ERROR_LAPACK)
    @test r.M == 0
    MPI.Barrier(comm)
end
# Residual work is distributed separately from the projected solve. A failure
# on a single rank must be reported before any rank enters its sum reduction.
q = Matrix{ComplexF64}(I,2,2)
res = zeros(2)
@test ext.mpi_compute_complex_residuals!(FaultyResidualMatrix(q,false),q,ones(2),q,res,2,comm)
@test res == zeros(2)
@test !ext.mpi_compute_complex_residuals!(FaultyResidualMatrix(q,rank==1),q,ones(2),q,res,2,comm)
MPI.Barrier(comm)
MPI.Finalize()
