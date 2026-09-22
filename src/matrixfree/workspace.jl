"""
    allocate_matfree_workspace(T, N, M0)

Allocate workspace arrays for matrix-free Feast operations.

Real problems use real search vectors plus complex shifted-solve buffers.
General complex problems use real FEAST bookkeeping with complex RHS/solution
buffers. The `rhs` field is explicit scratch for callbacks that write into
`workc`.
"""
function allocate_matfree_workspace(::Type{T}, N::Int, M0::Int) where T
    if T <: Real
        return (
            work = zeros(T, N, M0),
            workc = zeros(Complex{T}, N, M0),
            rhs = zeros(Complex{T}, N, M0),
            Aq = zeros(T, M0, M0),
            Sq = zeros(T, M0, M0),
            lambda = zeros(T, M0),
            q = zeros(T, N, M0),
            res = zeros(T, M0)
        )
    else # Complex
        RT = real(T)
        return (
            work = zeros(RT, N, M0),
            workc = zeros(T, N, M0),
            rhs = zeros(T, N, M0),
            zAq = zeros(T, M0, M0),
            zSq = zeros(T, M0, M0),
            lambda = zeros(T, M0),
            q = zeros(T, N, M0),
            res = zeros(RT, M0)
        )
    end
end
