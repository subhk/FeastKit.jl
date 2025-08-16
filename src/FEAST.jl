module FEAST

export feastinit, feastinit!, feastdefault!
export feast_srci!, feast_hrci!, feast_grci!
export feast_contour, feast_gcontour, feast_customcontour
export feast, feast_general, feast_matvec
export feast_parallel, pfeast_srci!, ParallelFeastState
export mpi_feast, feast_hybrid, MPIFeastState
export feast_summary, feast_validate_interval
export eigvals_feast, eigen_feast
export feast_parallel_info, feast_parallel_comparison
export determine_parallel_backend, mpi_available, feast_parallel_capabilities
export feast_with_backend, feast_serial
export FeastResult, FeastParameters, FeastWorkspaceReal, FeastWorkspaceComplex

using LinearAlgebra
using SparseArrays
using Distributed

include("core/feast_types.jl")
include("core/feast_parameters.jl")
include("core/feast_tools.jl")
include("core/feast_aux.jl")
include("core/feast_backend_utils.jl")
include("kernel/feast_kernel.jl")
include("dense/feast_dense.jl")
include("sparse/feast_sparse.jl")
include("banded/feast_banded.jl")
include("parallel/feast_parallel.jl")
include("parallel/feast_parallel_rci.jl")
include("interfaces/feast_interfaces.jl")

# Conditional MPI loading - include at end to allow proper loading
const MPI_AVAILABLE = Ref(false)

function __init__()
    try
        # Try to load MPI components
        eval(:(
            using MPI
            include(joinpath(@__DIR__, "parallel", "feast_mpi.jl"))
            include(joinpath(@__DIR__, "parallel", "feast_mpi_interface.jl"))
        ))
        MPI_AVAILABLE[] = true
    catch e
        @debug "MPI.jl not available or failed to load. MPI features will be disabled." exception=e
        MPI_AVAILABLE[] = false
    end
end

end
