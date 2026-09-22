# Seed a private workspace, preserving the caller's vectors and adding
# independent directions when only some eigenvectors are supplied.
function _feast_initial_subspace!(dest::AbstractMatrix{T}, initial) where T
    initial === nothing && return dest
    initial isa AbstractMatrix || throw(ArgumentError("initial_subspace must be a matrix of column vectors"))
    size(initial, 1) == size(dest, 1) || throw(DimensionMismatch("initial_subspace must have one row per matrix row"))
    0 < size(initial, 2) <= size(dest, 2) || throw(ArgumentError("initial_subspace must have between 1 and subspace_size columns"))
    all(isfinite, initial) || throw(ArgumentError("initial_subspace must contain finite values"))
    if T <: Real && !(eltype(initial) <: Real)
        all(isreal, initial) || throw(ArgumentError("Real symmetric solves require a real initial_subspace"))
        initial = real.(initial)
    end
    seed = Matrix{T}(initial)
    all(isfinite, seed) || throw(ArgumentError("initial_subspace overflows the working precision"))
    for j in axes(seed, 2)
        nrm = norm(view(seed, :, j))
        isfinite(nrm) && nrm > 0 || throw(ArgumentError("initial_subspace columns must have finite, nonzero norm"))
        view(seed, :, j) ./= nrm
    end
    T <: Real ? _feast_seeded_subspace!(dest) : _feast_seeded_subspace_complex!(dest)
    copyto!(view(dest, :, 1:size(seed, 2)), seed)
    return dest
end

function _feast_initial_parameters(fpm, initial)
    initial === nothing && return fpm
    params = copy(fpm)
    params[5] = 1
    return params
end
