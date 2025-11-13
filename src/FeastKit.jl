module FeastKit

export feastinit, feastinit!, feastinit_driver, feastdefault!
export feast_srci!, feast_srcix!, feast_hrci!, feast_hrcix!, feast_grci!, feast_grcix!
export feast_contour, feast_gcontour, feast_customcontour
export feast, feast_general, feast_matvec
export feast_parallel, pfeast_srci!, ParallelFeastState
export mpi_feast, feast_hybrid, MPIFeastState
export feast_summary, feast_validate_interval
export check_feast_srci_input, feast_inside_contour, feast_inside_gcontour
export feast_name, feast_memory_estimate
export full_to_banded, banded_to_full, feast_banded_info
export distribute_contour_points
# Dense matrix solver exports
export feast_sygv!, feast_sygvx!, feast_syev!, feast_syevx!
export feast_heev!, feast_heevx!, feast_hegv!, feast_hegvx!
export feast_gegv!, feast_gegvx!, feast_geev!, feast_geevx!
export feast_pep!, feast_gepev!, feast_gepevx!, feast_hepev!, feast_hepevx!,
       feast_sypev!, feast_sypevx!, feast_srcipev!, feast_srcipevx!,
       feast_grcipev!, feast_grcipevx!
# Sparse matrix solver exports
export feast_scsrgv!, feast_scsrgvx!, feast_scsrev!, feast_scsrevx!,
       feast_scsrgv_iterative!, difeast_scsrgv!, difeast_scsrgvx!
export feast_hcsrev!, feast_hcsrevx!, feast_hcsrgv!, feast_hcsrgvx!,
       feast_gcsrgv!, feast_gcsrgvx!
export feast_gcsrev!, feast_gcsrevx!
export feast_scsrpev!, feast_scsrpevx!, feast_hcsrpev!, feast_hcsrpevx!
export feast_gcsrpev!, feast_gcsrpevx!
# Banded matrix solver exports
export feast_sbgv!, feast_sbgvx!, feast_sbev!, feast_sbevx!
export feast_hbev!, feast_hbevx!, feast_hbgv!, feast_hbgvx!
export feast_gbgv!, feast_gbgvx!, feast_gbev!, feast_gbevx!
export Feast_SUCCESS, Feast_ERROR_N, Feast_ERROR_M0, Feast_ERROR_EMIN_EMAX,
       Feast_ERROR_EMID_R, Feast_ERROR_NO_CONVERGENCE, Feast_ERROR_MEMORY,
       Feast_ERROR_INTERNAL, Feast_ERROR_LAPACK, Feast_ERROR_FPM
export Feast_RCI_INIT, Feast_RCI_DONE, Feast_RCI_FACTORIZE, Feast_RCI_SOLVE,
       Feast_RCI_SOLVE_TRANSPOSE, Feast_RCI_MULT_A, Feast_RCI_MULT_B
export feast_sparse_info
export nworkers
export eigvals_feast, eigen_feast
export feast_parallel_info, feast_parallel_comparison
export determine_parallel_backend, mpi_available, feast_parallel_capabilities
export feast_with_backend, feast_serial
export FeastResult, FeastParameters, FeastWorkspaceReal, FeastWorkspaceComplex
# Matrix-free interface exports
export MatrixFreeOperator, MatrixVecFunction, LinearOperator
export feast_matfree_srci!, feast_matfree_grci!
export feast_contour_expert, feast_contour_custom_weights!, feast_rational_expert,
       feast_rational, feast_rationalx, feast_grational, feast_grationalx,
       feast_distribution_type
export feast_polynomial, create_iterative_solver, allocate_matfree_workspace

using LinearAlgebra
using SparseArrays
using Distributed
using FastGaussQuadrature

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
include("interfaces/feast_matfree.jl")
include("deprecations.jl")

# Conditional MPI loading - include at end to allow proper loading
const MPI_AVAILABLE = Ref(false)

function __init__()
    # Skip MPI initialization on CI environments to avoid hanging
    # MPI.Initialized() can hang when MPI runtime is not properly configured
    if get(ENV, "CI", "false") == "true"
        @debug "Running on CI, skipping MPI initialization"
        MPI_AVAILABLE[] = false
        return
    end

    # Only attempt MPI loading if explicitly enabled via environment variable
    # This prevents hanging when MPI is not properly configured
    if get(ENV, "FEASTKIT_ENABLE_MPI", "false") != "true"
        @debug "MPI not explicitly enabled (set FEASTKIT_ENABLE_MPI=true to enable), MPI features disabled"
        MPI_AVAILABLE[] = false
        return
    end

    try
        # Try to load MPI components
        @eval using MPI
        # Check if MPI is initialized with a timeout-safe approach
        # Only check if we're actually running under mpirun/mpiexec
        mpi_initialized = false
        try
            mpi_initialized = @eval MPI.Initialized()
        catch e
            @debug "MPI.Initialized() check failed, MPI features disabled" exception=e
            MPI_AVAILABLE[] = false
            return
        end

        if mpi_initialized
            include(joinpath(@__DIR__, "parallel", "feast_mpi.jl"))
            include(joinpath(@__DIR__, "parallel", "feast_mpi_interface.jl"))
            MPI_AVAILABLE[] = true
        else
            MPI_AVAILABLE[] = false
        end
    catch e
        @debug "MPI.jl not available or failed to load. MPI features will be disabled." exception=e
        MPI_AVAILABLE[] = false
    end
end

end
