# Matrix materialization, scalar promotion, and shared public API preparation.

# All public storage/operator families translate named controls in one place.
# The returned NamedTuple keeps these implementation details out of the API.
function _feast_prepare_options(N; M0, subspace_size, fpm, tol, maxiter,
                                quadrature_points, solver, solver_opts,
                                general::Bool=false, matrix_free::Bool=false,
                                initial_subspace=nothing, mixed_precision=nothing)
    params = _feast_named_parameters(fpm; tol=tol, maxiter=maxiter,
                                     quadrature_points=quadrature_points, general=general, mixed_precision=mixed_precision)
    matrix_free && params[42] == 1 && throw(ArgumentError("mixed_precision requires assembled dense matrices"))
    solver_options = _feast_solver_options(solver, solver_opts; matrix_free=matrix_free)
    if initial_subspace !== nothing && M0 === nothing && subspace_size === nothing
        initial_subspace isa AbstractMatrix || throw(ArgumentError("initial_subspace must be a matrix"))
        subspace_size = max(10, size(initial_subspace, 2) + 2)
    end
    M0 = _feast_subspace_size(M0, subspace_size, N)
    params = _feast_initial_parameters(params, initial_subspace)
    return (; params, solver_options, M0)
end

function _feast_prepare_assembled(A; parallel, backend, strict_backend, comm,
                                  general::Bool=false, initial_subspace=nothing, kwargs...)
    options = _feast_prepare_options(size(A, 1); general=general, initial_subspace=initial_subspace, kwargs...)
    requested = _normalize_backend(parallel, backend)
    allow_backend_fallback = _allow_backend_fallback(parallel, backend, strict_backend)
    backend_choice = _select_parallel_backend(requested, comm;
                                              allow_fallback=allow_backend_fallback)
    backend_choice = _feast_solver_backend(backend_choice, options.solver_options,
                                           A, allow_backend_fallback; general=general)
    if options.params[42] == 1
        options.solver_options.solver === :direct || throw(ArgumentError("mixed_precision requires solver=:direct"))
        materialized = _materialize_matrix(A)
        materialized isa Matrix && eltype(materialized) in (Float64,ComplexF64) ||
            throw(ArgumentError("mixed_precision requires dense Float64 or ComplexF64 matrices"))
        backend_choice = _feast_backend_or_serial(backend_choice,
            backend_choice === :serial ? nothing : "mixed_precision currently requires backend=:serial",
            allow_backend_fallback)
    end
    if initial_subspace !== nothing
        backend_choice = _feast_backend_or_serial(backend_choice,
            backend_choice === :serial ? nothing : "initial_subspace currently requires backend=:serial",
            allow_backend_fallback)
        options = (; options..., solver_options=(; options.solver_options..., initial_subspace))
    end
    return (; options..., backend_choice, allow_backend_fallback)
end

# Use the same scalar promotion and contour validation for standard and
# generalized problems; B=nothing preserves the specialized standard driver.
function _feast_prepare_general_matrices(A, B, center, radius)
    A = _materialize_matrix(A)
    B = B === nothing ? nothing : _materialize_matrix(B)
    real_type = float(promote_type(_real_component_type(eltype(A)),
                                  B === nothing ? _real_component_type(eltype(A)) :
                                                  _real_component_type(eltype(B)),
                                  _real_component_type(typeof(center)), typeof(radius)))
    complex_type = Complex{real_type}
    A_exec = _materialize_matrix_eltype(A, complex_type)
    B_exec = B === nothing ? nothing : _materialize_matrix_eltype(B, complex_type)
    center_exec, radius_exec = complex_type(center), real_type(radius)
    isfinite(center_exec) && isfinite(radius_exec) && radius_exec > zero(real_type) ||
        throw(ArgumentError("Contour center must be finite and radius must be finite and positive"))
    return (; A_exec, B_exec, center_exec, radius_exec)
end

function _materialize_matrix(A::AbstractMatrix)
    # Kernels dispatch on Matrix and SparseMatrixCSC. Lazy wrappers such as
    # Symmetric/Hermitian are converted while preserving sparse storage.
    if A isa Matrix || A isa SparseMatrixCSC
        return A
    elseif A isa Symmetric
        parent(A) isa SparseMatrixCSC && return SparseMatrixCSC(A)
        return Matrix(A)
    elseif A isa Hermitian
        parent(A) isa SparseMatrixCSC && return SparseMatrixCSC(A)
        return Matrix(A)
    else
        return Matrix(A)
    end
end

@inline _real_component_type(::Type{Complex{T}}) where T<:Real = T
@inline _real_component_type(::Type{T}) where T<:Real = T

# Search-interval endpoints should not have to match the matrix element type
# exactly. `feast(A, (0, 1))` and `feast(A, (0.0, 1))` are the natural things to
# write, and feast_general already promotes its center/radius the same way.
@inline _feast_interval_type(::Type{T}) where T<:AbstractFloat = T
@inline _feast_interval_type(::Type{<:Integer}) = Float64
@inline _feast_interval_type(::Type{T}) where T<:Real = float(T)

function _feast_promote_interval(A::AbstractMatrix, interval::Tuple{Real,Real})
    T = _feast_interval_type(_real_component_type(eltype(A)))
    return T(interval[1]), T(interval[2])
end

function _ensure_complex_matrix(A::AbstractMatrix)
    materialized = _materialize_matrix(A)
    return eltype(materialized) <: Complex ? materialized : Complex.(materialized)
end

function _materialize_matrix_eltype(A::AbstractMatrix, ::Type{T}) where T
    materialized = _materialize_matrix(A)
    return eltype(materialized) === T ? materialized : T.(materialized)
end

# An integer-valued matrix has no typed method to fall through to, so promote
# its element type here as well -- forwarding only the interval would recurse
# straight back into this method.
@inline _feast_promote_eltype(A::AbstractMatrix) =
    eltype(A) <: Integer ? _materialize_matrix_eltype(A, Float64) :
    eltype(A) <: Complex{<:Integer} ? _materialize_matrix_eltype(A, ComplexF64) : A
