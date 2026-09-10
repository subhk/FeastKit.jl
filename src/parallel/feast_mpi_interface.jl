# Hybrid execution shares the MPI solver's filtered-subspace/Rayleigh–Ritz
# algorithm. Only the rank-local contour solves gain a thread-level batch.
function feast_hybrid(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                     interval::Tuple{T,T}; M0::Int = 10,
                     fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                     comm::MPI.Comm = MPI.COMM_WORLD,
                     use_threads_per_rank::Bool = true) where T<:Real
    size(A,1) == size(A,2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))
    params = _ensure_feast_parameters(fpm)
    return mpi_feast_sygv!(A,B,interval[1],interval[2],M0,params;
                          comm=comm,use_threads=use_threads_per_rank)
end
