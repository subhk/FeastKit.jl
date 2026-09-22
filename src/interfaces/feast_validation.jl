# Search interval validation and inexpensive spectral bounds.

# Gershgorin circle bounds: O(nnz) for sparse, O(N²) for dense
function _gershgorin_bounds(A::SparseMatrixCSC{TV, Ti}) where {TV, Ti}
    T = real(eltype(A))
    N = size(A, 1)
    # Accumulate off-diagonal row sums by iterating over stored nonzeros (O(nnz))
    radii = zeros(T, N)
    rv = rowvals(A)
    nz = nonzeros(A)
    for col in 1:N
        for idx in nzrange(A, col)
            row = rv[idx]
            if row != col
                radii[row] += abs(nz[idx])
            end
        end
    end
    min_est = typemax(T)
    max_est = typemin(T)
    for i in 1:N
        center = real(A[i, i])
        min_est = min(min_est, center - radii[i])
        max_est = max(max_est, center + radii[i])
    end
    return (min_est, max_est)
end

function _gershgorin_bounds(A::AbstractMatrix)
    T = real(eltype(A))
    N = size(A, 1)
    min_est = typemax(T)
    max_est = typemin(T)
    for i in 1:N
        center = real(A[i, i])
        radius = zero(T)
        for j in 1:N
            if j != i
                radius += abs(A[i, j])
            end
        end
        min_est = min(min_est, center - radius)
        max_est = max(max_est, center + radius)
    end
    return (min_est, max_est)
end

function feast_validate_interval(A::AbstractMatrix{T}, interval::Tuple{T,T}) where T<:Real
    Emin, Emax = interval
    if Emin >= Emax
        throw(ArgumentError("Invalid interval: Emin must be less than Emax"))
    end

    min_est, max_est = _gershgorin_bounds(A)

    if Emax < min_est || Emin > max_est
        @warn "Search interval [$Emin, $Emax] may not contain eigenvalues. " *
              "Estimated eigenvalue range: [$(min_est), $(max_est)]"
    end

    return (min_est, max_est)
end

function feast_validate_interval(A::AbstractMatrix{Complex{T}}, interval::Tuple{T,T}) where T<:Real
    Emin, Emax = interval
    if Emin >= Emax
        throw(ArgumentError("Invalid interval: Emin must be less than Emax"))
    end

    min_est, max_est = _gershgorin_bounds(A)

    if Emax < min_est || Emin > max_est
        @warn "Search interval [$Emin, $Emax] may not contain eigenvalues. " *
              "Estimated eigenvalue range: [$(min_est), $(max_est)]"
    end

    return (min_est, max_est)
end
