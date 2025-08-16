module FEAST

export feastinit, feast_srci, feast_hrci, feast_grci
export feast_contour, feast_customcontour
export feast_dense, feast_sparse, feast_banded
export feast, feast_general, feast_banded, feast_matvec
export feast_parallel, pfeast_srci!, ParallelFeastState
export feast_summary, feast_validate_interval
export eigvals_feast, eigen_feast

using LinearAlgebra
using SparseArrays
using Distributed

include("core/feast_types.jl")
include("core/feast_parameters.jl")
include("core/feast_tools.jl")
include("core/feast_aux.jl")
include("kernel/feast_kernel.jl")
include("dense/feast_dense.jl")
include("sparse/feast_sparse.jl")
include("banded/feast_banded.jl")
include("parallel/feast_parallel.jl")
include("parallel/feast_parallel_rci.jl")
include("interfaces/feast_interfaces.jl")

end
