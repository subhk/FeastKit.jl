module FEAST

export feastinit, feast_srci, feast_hrci, feast_grci
export feast_contour, feast_customcontour
export feast_dense, feast_sparse, feast_banded

using LinearAlgebra
using SparseArrays

include("core/feast_types.jl")
include("core/feast_parameters.jl")
include("core/feast_tools.jl")
include("core/feast_aux.jl")
include("kernel/feast_kernel.jl")
include("dense/feast_dense.jl")
include("sparse/feast_sparse.jl")
include("banded/feast_banded.jl")
include("interfaces/feast_interfaces.jl")

end
