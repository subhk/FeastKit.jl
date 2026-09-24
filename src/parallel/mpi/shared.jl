# MPI-based parallel FeastKit implementation
# True MPI support for HPC clusters and distributed computing

# Note: MPI should already be loaded when this file is included
using LinearAlgebra
using SparseArrays

# The root owns contour configuration, including the process-local registry.
# Broadcast geometry explicitly; registry IDs are not meaningful on other ranks.
function _mpi_contour(::Type{T}, fpm, center, radius, root, comm; general=false) where T
    contour = nothing
    custom = false
    failure = nothing
    if MPI.Comm_rank(comm) == root
        try
            contour = feast_get_custom_contour(T, fpm)
            custom = contour !== nothing
            contour === nothing && (contour = general ? feast_gcontour(center,radius,fpm) :
                                                        feast_contour(center,radius,fpm))
        catch err
            failure = sprint(showerror,err)
        end
    end
    failure = MPI.bcast(failure,root,comm)
    failure === nothing || throw(ArgumentError(failure))
    contour = MPI.bcast(contour,root,comm)
    custom = MPI.bcast(custom,root,comm)
    return contour, custom
end

# Lazy factors implement the no-storage policy without retaining any LU. The
# projection loop drops each factor before moving to the next contour point.
struct _MPIUncachedFactors{TA,TB,TZ} <: AbstractVector{Any}
    A::TA
    B::TB
    nodes::TZ
end
Base.size(f::_MPIUncachedFactors) = size(f.nodes)
Base.getindex(f::_MPIUncachedFactors,i::Int) = lu(f.nodes[i]*f.B-f.A)

function _mpi_collective_try(f, comm)
    success = try
        f()
        true
    catch err
        @debug "MPI local computation failed" exception=err
        false
    end
    return _mpi_success_count(success,comm) == MPI.Comm_size(comm)
end

# Residual floor for the MPI drivers. The spectral-scale probe multiplies by A
# and B, so like every other rank-local operation it must fail collectively:
# returns `nothing` on every rank when it failed on any, otherwise root's value.
function _mpi_residual_floor(A, B, Q, region, root, comm)
    σ = zero(float(real(eltype(Q))))
    ok = _mpi_collective_try(comm) do
        σ = _feast_spectral_scale(A, B, Q)
    end
    ok || return nothing
    return MPI.bcast(_feast_residual_floor(σ, region), root, comm)
end

function _mpi_factorize_contour(A, B, nodes, comm; store=true)
    store || return _MPIUncachedFactors(A,B,nodes)
    factors = try
        [lu(z*B-A) for z in nodes]
    catch err
        @debug "MPI local factorization failed" exception=err
        nothing
    end
    return _mpi_success_count(factors !== nothing,comm) == MPI.Comm_size(comm) ? factors : nothing
end

# MPI collectives remain on the calling thread. Worker tasks own separate solve
# and accumulation buffers; any local failure is reported collectively by callers.
function _mpi_local_projection!(dest, factors, rhs, weights, scratch;
                                scale=1, use_threads=false)
    fill!(dest,zero(eltype(dest)))
    try
        if use_threads && !(factors isa _MPIUncachedFactors) && Threads.nthreads() > 1 && length(factors) > 1
            chunks = collect(Iterators.partition(eachindex(factors),
                             cld(length(factors),Threads.nthreads())))
            partials = [zeros(eltype(dest),size(dest)) for _ in chunks]
            Threads.@threads for ci in eachindex(chunks)
                Y = similar(scratch)
                for e in chunks[ci]
                    copyto!(Y,rhs)
                    ldiv!(factors[e],Y)
                    partials[ci] .+= (scale*weights[e]) .* Y
                end
            end
            for part in partials
                dest .+= part
            end
        else
            for e in eachindex(factors)
                copyto!(scratch,rhs)
                ldiv!(factors[e],scratch)
                dest .+= (scale*weights[e]) .* scratch
            end
        end
        return true
    catch err
        @debug "MPI local shifted solve failed" exception=err
        return false
    end
end
