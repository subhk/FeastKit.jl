# MPI.jl package extension.
#
# The MPI drivers distribute contour points across ranks. They are only useful
# on a cluster, so MPI is a weak dependency: `using MPI` alongside FeastKit
# activates this extension and gives the `mpi_feast*` entry points their
# methods. Without it those names exist but have no methods, and every internal
# call site is gated by `_mpi_backend_ready`, which stays false.
module FeastKitMPIExt

using FeastKit
using MPI
using LinearAlgebra
using SparseArrays
using Printf

# Solver internals the MPI drivers build on. Imported explicitly so that a
# rename in the parent module fails loudly here rather than silently falling
# back to a different binding.
import FeastKit: MPIFeastState, FeastResult, FeastGeneralResult,
                 FeastWorkspaceReal, FeastParameters,
                 Feast_SUCCESS, Feast_ERROR_LAPACK, Feast_ERROR_NO_CONVERGENCE,
                 feastinit!, feastdefault!, feast_tolerance,
                 feast_contour, feast_gcontour, feast_inside_contour,
                 feast_residual!, feast_sort!, feast_sort_general!,
                 check_feast_srci_input, check_feast_grci_input,
                 _feast_qr_compress!, _feast_real_column!,
                 _feast_exit_info,
                 _feast_reorder_by_interval!, _feast_reorder_by_gcontour!,
                 _feast_seeded_subspace!, _feast_seeded_subspace_complex!,
                 solve_shifted_iterative!, solve_dense_shifted!,
                 FEAST_KRYLOV_AVAILABLE,
                 feast_get_custom_contour,
                 _ensure_feast_parameters,
                 mpi_available, feast, feast_parallel_info

# Entry points this extension implements.
import FeastKit: _mpi_initialized, _mpi_world_comm, _mpi_comm_rank, _mpi_comm_size,
                 mpi_feast, mpi_feast_general, feast_hybrid,
                 mpi_feast_sygv!, mpi_feast_scsrgv!,
                 mpi_feast_heev!, mpi_feast_hegv!, mpi_feast_geev!, mpi_feast_gegv!,
                 mpi_feast_hcsrev!, mpi_feast_hcsrgv!,
                 mpi_feast_gcsrev!, mpi_feast_gcsrgv!,
                 mpi_feast_available, mpi_feast_init, mpi_feast_finalize,
                 mpi_feast_benchmark

_mpi_initialized() = MPI.Initialized()
_mpi_world_comm() = MPI.COMM_WORLD
_mpi_comm_rank(comm) = MPI.Comm_rank(comm)
_mpi_comm_size(comm) = MPI.Comm_size(comm)

include(joinpath(@__DIR__, "..", "src", "parallel", "feast_mpi.jl"))
include(joinpath(@__DIR__, "..", "src", "parallel", "feast_mpi_interface.jl"))

end # module
