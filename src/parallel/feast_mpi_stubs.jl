# MPI seam.
#
# MPI is a weak dependency: the implementations of everything declared here live
# in `ext/FeastKitMPIExt.jl` and only exist once the user has loaded MPI. The
# names are declared in the parent module so they can stay exported and so the
# serial code paths can reference them without MPI installed.
#
# Calling one of these without MPI loaded raises a MethodError; the backend
# selection in `feast_backend_utils.jl` guards every internal call site with
# `_mpi_backend_ready`, which is false unless the extension is active.

"""
    MPIFeastState{T}

Per-rank state for the MPI FEAST drivers: the communicator, this rank's slice of
the contour, and the convergence bookkeeping.

`comm` is typed `Any` rather than `MPI.Comm` because this struct has to be
constructible in the parent module, which does not know MPI's types. It is only
touched at loop boundaries (rank/size queries, collectives), never per element,
so the dynamic dispatch costs nothing measurable.
"""
mutable struct MPIFeastState{T<:Real}
    # MPI communication info
    comm::Any
    rank::Int
    size::Int
    root::Int

    # Feast parameters
    N::Int
    M0::Int
    ne::Int

    # Local contour points assigned to this rank
    local_points::Vector{Int}
    local_Zne::Vector{Complex{T}}
    local_Wne::Vector{Complex{T}}

    # Convergence state
    converged::Bool
    loop::Int
    epsout::T
    info::Int
end

"""
    _mpi_contour_slice(rank, nranks, ne)

Contiguous, load-balanced slice of `1:ne` owned by `rank` (0-based) out of
`nranks`. Shared by the state constructor and the distribution reporting so the
two cannot drift apart.
"""
function _mpi_contour_slice(rank::Int, nranks::Int, ne::Int)
    points_per_rank = div(ne, nranks)
    remainder = ne % nranks
    start_idx = rank * points_per_rank + min(rank, remainder) + 1
    local_count = points_per_rank + (rank < remainder ? 1 : 0)
    return collect(start_idx:(start_idx + local_count - 1))
end

function MPIFeastState{T}(comm, rank::Int, nranks::Int, N::Int, M0::Int,
                          ne::Int, root::Int = 0) where T<:Real
    local_points = _mpi_contour_slice(rank, nranks, ne)
    local_count = length(local_points)
    return MPIFeastState{T}(comm, rank, nranks, root, N, M0, ne, local_points,
                            Vector{Complex{T}}(undef, local_count),
                            Vector{Complex{T}}(undef, local_count),
                            false, 0, zero(T), 0)
end

# Runtime queries the extension implements. `_mpi_initialized` is called from
# __init__; the others back `feast_parallel_info`.
function _mpi_initialized end
function _mpi_world_comm end
function _mpi_comm_rank end
function _mpi_comm_size end

# Driver entry points implemented by FeastKitMPIExt.
function mpi_feast end
function mpi_feast_general end
function feast_hybrid end

function mpi_feast_sygv! end
function mpi_feast_scsrgv! end
function mpi_feast_heev! end
function mpi_feast_hegv! end
function mpi_feast_geev! end
function mpi_feast_gegv! end
function mpi_feast_hcsrev! end
function mpi_feast_hcsrgv! end
function mpi_feast_gcsrev! end
function mpi_feast_gcsrgv! end

function mpi_feast_available end
function mpi_feast_init end
function mpi_feast_finalize end
function mpi_feast_benchmark end
