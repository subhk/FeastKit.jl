# Parallel FeastKit implementation
# Each contour point is solved independently using distributed computing

using Distributed
using LinearAlgebra

function _pfeast_dense_shifted_system!(dest::AbstractMatrix{Complex{T}},
                                       z::Complex{T},
                                       A::AbstractMatrix{T},
                                       B::AbstractMatrix{T}) where T<:Real
    @boundscheck size(dest) == size(A) == size(B) || throw(DimensionMismatch("matrix sizes must match"))
    @inbounds @simd for i in eachindex(dest, A, B)
        dest[i] = z * B[i] - A[i]
    end
    return dest
end

function _pfeast_store_complex_moments!(Aq::AbstractMatrix{Complex{T}},
                                        Sq::AbstractMatrix{Complex{T}},
                                        Q_proj::AbstractMatrix{Complex{T}},
                                        temp::AbstractMatrix{Complex{T}},
                                        workc::AbstractMatrix{Complex{T}},
                                        weight::Complex{T},
                                        z::Complex{T}) where T<:Real
    weighted_z = weight * z
    @inbounds for j in axes(temp, 2), i in axes(temp, 1)
        val = temp[i, j]
        Aq[i, j] = weight * val
        Sq[i, j] = weighted_z * val
    end
    @inbounds for j in axes(workc, 2), i in axes(workc, 1)
        Q_proj[i, j] = weight * workc[i, j]
    end
    return Aq, Sq, Q_proj
end

function _pfeast_store_real_moments!(Aq::AbstractMatrix{T},
                                     Sq::AbstractMatrix{T},
                                     Q_proj::AbstractMatrix{T},
                                     temp::AbstractMatrix{Complex{T}},
                                     workc::AbstractMatrix{Complex{T}},
                                     weight::Complex{T},
                                     z::Complex{T}) where T<:Real
    weighted_z = weight * z
    @inbounds for j in axes(temp, 2), i in axes(temp, 1)
        val = temp[i, j]
        Aq[i, j] = real(weight * val)
        Sq[i, j] = real(weighted_z * val)
    end
    @inbounds for j in axes(workc, 2), i in axes(workc, 1)
        Q_proj[i, j] = real(weight * workc[i, j])
    end
    return Aq, Sq, Q_proj
end

# Distribute contour points among workers
function distribute_contour_points(ne::Int, nw::Int)
    points_per_worker = div(ne, nw)
    remainder = ne % nw

    chunks = Vector{Vector{Int}}(undef, nw)
    start_idx = 1

    for i in 1:nw
        chunk_size = points_per_worker + (i <= remainder ? 1 : 0)
        chunks[i] = collect(start_idx:(start_idx + chunk_size - 1))
        start_idx += chunk_size
    end

    return chunks
end
