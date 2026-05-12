module FeastKit

export feastinit, feastinit!, feastinit_driver, feastdefault!
export feast_srci!, feast_srcix!, feast_hrci!, feast_hrcix!, feast_grci!, feast_grcix!
export ifeast_srci!, ifeast_hrci!, ifeast_grci!
export feast_contour, feast_gcontour, feast_customcontour
export feast, feast_general, feast_matvec, feast_banded, feast_set_defaults!, feast_clear_all_contours!
export feast_parallel, pfeast_srci!, ParallelFeastState
export pfeast_sygv!, pfeast_scsrgv!, pfeast_compute_all_contour_points!, pfeast_show_distribution
export mpi_feast, feast_hybrid, MPIFeastState
export feast_summary, feast_validate_interval
export check_feast_srci_input, feast_inside_contour, feast_inside_gcontour
export feast_name, feast_memory_estimate
export full_to_banded, full_to_general_banded, banded_to_full, feast_banded_info
export distribute_contour_points
# Dense matrix solver exports
export feast_sygv!, feast_sygvx!, feast_syev!, feast_syevx!,
       difeast_sygv!, difeast_syev!
export feast_heev!, feast_heevx!, feast_hegv!, feast_hegvx!,
       zifeast_heev!, zifeast_hegv!
export feast_gegv!, feast_gegvx!, feast_geev!, feast_geevx!,
       feast_geev_complex_sym!, feast_gegv_complex_sym!,
       zifeast_gegv!, zifeast_geev!
export feast_pep!, feast_gepev!, feast_gepevx!, feast_hepev!, feast_hepevx!,
       feast_sypev!, feast_sypevx!, feast_srcipev!, feast_srcipevx!,
       feast_grcipev!, feast_grcipevx!
# Sparse matrix solver exports
export feast_scsrgv!, feast_scsrgvx!, feast_scsrev!, feast_scsrevx!,
       feast_scsrgv_iterative!, difeast_scsrgv!, difeast_scsrgvx!
export feast_hcsrev!, feast_hcsrevx!, feast_hcsrgv!, feast_hcsrgvx!,
       zifeast_hcsrev!, zifeast_hcsrevx!, zifeast_hcsrgv!, zifeast_hcsrgvx!,
       feast_gcsrgv!, feast_gcsrgvx!, zifeast_gcsrgv!, zifeast_gcsrgvx!,
       feast_scsrgv_complex!, feast_scsrgvx_complex!, feast_scsrev_complex!, feast_scsrevx_complex!,
       zifeast_scsrgv_complex!, zifeast_scsrgvx_complex!, zifeast_scsrev_complex!, zifeast_scsrevx_complex!
export feast_gcsrev!, feast_gcsrevx!, zifeast_gcsrev!, zifeast_gcsrevx!
export feast_scsrpev!, feast_scsrpevx!, feast_hcsrpev!, feast_hcsrpevx!
export feast_gcsrpev!, feast_gcsrpevx!
# Banded matrix solver exports
export feast_sbgv!, feast_sbgvx!, feast_sbev!, feast_sbevx!,
       difeast_sbgv!, difeast_sbev!
export feast_hbev!, feast_hbevx!, feast_hbgv!, feast_hbgvx!,
       zifeast_hbev!, zifeast_hbgv!
export feast_gbgv!, feast_gbgvx!, feast_gbev!, feast_gbevx!,
       feast_sbgv_complex!, feast_sbgvx_complex!, feast_sbev_complex!, feast_sbevx_complex!,
       zifeast_gbgv!, zifeast_gbev!, zifeast_sbgv_complex!, zifeast_sbev_complex!
# FEAST-compatible precision-prefixed aliases
export sfeast_syev!, dfeast_syev!, sfeast_sygv!, dfeast_sygv!,
       sfeast_syevx!, dfeast_syevx!, sfeast_sygvx!, dfeast_sygvx!,
       sfeast_sypev!, dfeast_sypev!, sfeast_sypevx!, dfeast_sypevx!,
       sfeast_srcipev!, dfeast_srcipev!, sfeast_srcipevx!, dfeast_srcipevx!,
       sfeast_scsrev!, dfeast_scsrev!, sfeast_scsrgv!, dfeast_scsrgv!,
       sfeast_scsrevx!, dfeast_scsrevx!, sfeast_scsrgvx!, dfeast_scsrgvx!,
       sfeast_scsrpev!, dfeast_scsrpev!, sfeast_scsrpevx!, dfeast_scsrpevx!,
       sfeast_sbev!, dfeast_sbev!, sfeast_sbgv!, dfeast_sbgv!,
       sfeast_sbevx!, dfeast_sbevx!, sfeast_sbgvx!, dfeast_sbgvx!
export cfeast_heev!, zfeast_heev!, cfeast_hegv!, zfeast_hegv!,
       cfeast_heevx!, zfeast_heevx!, cfeast_hegvx!, zfeast_hegvx!,
       cfeast_geev!, zfeast_geev!, cfeast_gegv!, zfeast_gegv!,
       cfeast_geevx!, zfeast_geevx!, cfeast_gegvx!, zfeast_gegvx!,
       cfeast_gepev!, zfeast_gepev!, cfeast_gepevx!, zfeast_gepevx!,
       cfeast_hepev!, zfeast_hepev!, cfeast_hepevx!, zfeast_hepevx!,
       cfeast_grcipev!, zfeast_grcipev!, cfeast_grcipevx!, zfeast_grcipevx!,
       cfeast_hcsrev!, zfeast_hcsrev!, cfeast_hcsrgv!, zfeast_hcsrgv!,
       cfeast_hcsrevx!, zfeast_hcsrevx!, cfeast_hcsrgvx!, zfeast_hcsrgvx!,
       cfeast_scsrev!, zfeast_scsrev!, cfeast_scsrgv!, zfeast_scsrgv!,
       cfeast_scsrevx!, zfeast_scsrevx!, cfeast_scsrgvx!, zfeast_scsrgvx!,
       cfeast_gcsrev!, zfeast_gcsrev!, cfeast_gcsrgv!, zfeast_gcsrgv!,
       cfeast_gcsrevx!, zfeast_gcsrevx!, cfeast_gcsrgvx!, zfeast_gcsrgvx!,
       cfeast_hcsrpev!, zfeast_hcsrpev!, cfeast_hcsrpevx!, zfeast_hcsrpevx!,
       cfeast_gcsrpev!, zfeast_gcsrpev!, cfeast_gcsrpevx!, zfeast_gcsrpevx!,
       cfeast_hbev!, zfeast_hbev!, cfeast_hbgv!, zfeast_hbgv!,
       cfeast_hbevx!, zfeast_hbevx!, cfeast_hbgvx!, zfeast_hbgvx!,
       cfeast_sbev!, zfeast_sbev!, cfeast_sbgv!, zfeast_sbgv!,
       cfeast_sbevx!, zfeast_sbevx!, cfeast_sbgvx!, zfeast_sbgvx!,
       cfeast_gbev!, zfeast_gbev!, cfeast_gbgv!, zfeast_gbgv!,
       cfeast_gbevx!, zfeast_gbevx!, cfeast_gbgvx!, zfeast_gbgvx!
export psfeast_syev!, pdfeast_syev!, psfeast_sygv!, pdfeast_sygv!,
       psfeast_scsrev!, pdfeast_scsrev!, psfeast_scsrgv!, pdfeast_scsrgv!,
       psfeast_srci!, pdfeast_srci!
export Feast_SUCCESS, Feast_ERROR_N, Feast_ERROR_M0, Feast_ERROR_EMIN_EMAX,
       Feast_ERROR_EMID_R, Feast_ERROR_NO_CONVERGENCE, Feast_ERROR_MEMORY,
       Feast_ERROR_INTERNAL, Feast_ERROR_LAPACK, Feast_ERROR_FPM
export Feast_RCI_INIT, Feast_RCI_DONE, Feast_RCI_FACTORIZE, Feast_RCI_SOLVE,
       Feast_RCI_FACTORIZE_T, Feast_RCI_SOLVE_T, Feast_RCI_SOLVE_TRANSPOSE,
       Feast_RCI_MULT_A, Feast_RCI_MULT_A_H, Feast_RCI_MULT_B, Feast_RCI_MULT_B_H,
       Feast_RCI_BIORTHOG, Feast_RCI_REDUCED_SYSTEM
export feast_sparse_info
export eigvals_feast, eigen_feast
export feast_parallel_info, feast_parallel_comparison
export determine_parallel_backend, mpi_available, feast_parallel_capabilities
export feast_with_backend, feast_serial
export FeastResult, FeastGeneralResult, FeastParameters, FeastWorkspaceReal, FeastWorkspaceComplex
export FeastSRCIState, FeastHRCIState, FeastGRCIState, FeastPolyRCIState
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

const FEAST_KRYLOV_AVAILABLE = Ref(false)
try
    using Krylov: gmres
    import Krylov
    FEAST_KRYLOV_AVAILABLE[] = true
catch e
    @debug "Krylov.jl not available; iterative FEAST variants requiring GMRES will be disabled." exception=e
    FEAST_KRYLOV_AVAILABLE[] = false
end

# Load MPI at compile time (it's in [deps]) but defer initialization check to __init__
const MPI_AVAILABLE = Ref(false)
const _MPI_LOADED = try
    using MPI
    true
catch
    false
end

include("core/feast_types.jl")
include("core/feast_parameters.jl")
include("core/feast_tools.jl")
include("core/feast_aux.jl")
include("core/feast_backend_utils.jl")
include("kernel/feast_kernel.jl")
include("dense/feast_dense.jl")
include("sparse/feast_sparse.jl")
include("banded/feast_banded.jl")
include("interfaces/feast_precision_aliases.jl")
include("parallel/feast_parallel.jl")
include("parallel/feast_parallel_rci.jl")
include("interfaces/feast_interfaces.jl")
include("interfaces/feast_matfree.jl")
include("deprecations.jl")

# Include MPI files at compile time (they define types/methods that need to exist);
# actual MPI usage is gated by MPI_AVAILABLE[] at runtime.
if _MPI_LOADED
    include("parallel/feast_mpi.jl")
    include("parallel/feast_mpi_interface.jl")
end

function __init__()
    if !_MPI_LOADED
        MPI_AVAILABLE[] = false
        return
    end

    # Skip MPI initialization on CI environments to avoid hanging
    if get(ENV, "CI", "false") == "true"
        @debug "Running on CI, skipping MPI initialization"
        MPI_AVAILABLE[] = false
        return
    end

    # Only attempt MPI usage if explicitly enabled via environment variable
    # This prevents hanging when MPI is not properly configured
    if get(ENV, "FEASTKIT_ENABLE_MPI", "false") != "true"
        @debug "MPI not explicitly enabled (set FEASTKIT_ENABLE_MPI=true to enable), MPI features disabled"
        MPI_AVAILABLE[] = false
        return
    end

    try
        MPI_AVAILABLE[] = MPI.Initialized()
    catch e
        @debug "MPI.Initialized() check failed, MPI features disabled" exception=e
        MPI_AVAILABLE[] = false
    end
end

end
