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
    # Integer and rational matrices are promoted to floating point later.
    _feast_check_supported_eltype(_feast_float_eltype(eltype(A)))
    options = _feast_prepare_options(size(A, 1); general=general, initial_subspace=initial_subspace, kwargs...)
    requested = _normalize_backend(parallel, backend)
    allow_backend_fallback = _allow_backend_fallback(parallel, backend, strict_backend)
    backend_choice = _select_parallel_backend(requested, comm;
                                              allow_fallback=allow_backend_fallback)
    backend_choice = _feast_solver_backend(backend_choice, options.solver_options,
                                           A, allow_backend_fallback; general=general)
    if options.params[42] == 1
        options.solver_options.solver === :direct || throw(ArgumentError("mixed_precision requires solver=:direct"))
        materialized = _materialize_matrix(A; dense=true)
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
function _feast_prepare_general_matrices(A, B, center, radius; dense::Bool=false)
    A = _materialize_matrix(A; dense=dense)
    B = B === nothing ? nothing : _materialize_matrix(B; dense=dense)
    real_type = float(promote_type(_real_component_type(eltype(A)),
                                  B === nothing ? _real_component_type(eltype(A)) :
                                                  _real_component_type(eltype(B)),
                                  _real_component_type(typeof(center)), typeof(radius)))
    _feast_check_supported_eltype(real_type)
    complex_type = Complex{real_type}
    A_exec = _materialize_matrix_eltype(A, complex_type)
    B_exec = B === nothing ? nothing : _materialize_matrix_eltype(B, complex_type)
    B_exec === nothing || ((A_exec, B_exec) = _feast_match_storage(A_exec, B_exec))
    center_exec, radius_exec = complex_type(center), real_type(radius)
    isfinite(center_exec) && isfinite(radius_exec) && radius_exec > zero(real_type) ||
        throw(ArgumentError("Contour center must be finite and radius must be finite and positive"))
    return (; A_exec, B_exec, center_exec, radius_exec)
end

function _materialize_matrix(A::AbstractMatrix; dense::Bool=false)
    # Kernels dispatch on Matrix and SparseMatrixCSC. `dense=true` is for the
    # paths that only exist for dense storage (mixed precision): banded
    # structured matrices are densified for them, while sparse storage stays
    # sparse so that those paths reject it.
    # The sparse kernels are written for `Int` column pointers and row indices.
    if A isa Matrix || A isa SparseMatrixCSC{<:Any,Int}
        return A
    elseif _feast_sparse_storage(A) || (!dense && _feast_banded_structured(A))
        # Banded structured matrices and wrappers of sparse storage keep their
        # O(N) storage: densifying a SymTridiagonal costs N² memory and a dense
        # LU at every contour point.
        return SparseMatrixCSC{eltype(A),Int}(sparse(A))
    else
        return Matrix(A)
    end
end

# Storage that is better kept sparse than densified.
_feast_sparse_like(A::AbstractMatrix) = _feast_sparse_storage(A) || _feast_banded_structured(A)
_feast_sparse_storage(::SparseMatrixCSC) = true
_feast_sparse_storage(A::Union{Symmetric,Hermitian,Adjoint,Transpose}) = _feast_sparse_storage(parent(A))
_feast_sparse_storage(A::AbstractMatrix) = issparse(A)
_feast_banded_structured(::Union{Diagonal,Bidiagonal,Tridiagonal,SymTridiagonal}) = true
_feast_banded_structured(A::Union{Symmetric,Hermitian,Adjoint,Transpose}) = _feast_banded_structured(parent(A))
_feast_banded_structured(::AbstractMatrix) = false

# The solvers take a pencil in one storage family. A structured or dense B
# next to a sparse A (`feast(A, Diagonal(b), ...)`), or the reverse, follows
# A's storage instead of being rejected.
function _materialize_pencil(A::AbstractMatrix, B::AbstractMatrix; dense::Bool=false)
    dense && return _feast_match_storage(_materialize_matrix(A; dense=true),
                                         _materialize_matrix(B; dense=true))
    A_exec = _materialize_matrix(A)
    B_exec = A_exec isa SparseMatrixCSC ?
        (B isa SparseMatrixCSC || B isa Symmetric || B isa Hermitian ?
             _materialize_matrix(B) : SparseMatrixCSC{eltype(B),Int}(sparse(B))) :
        _materialize_matrix(B)
    return _feast_match_storage(A_exec, B_exec)
end

function _feast_match_storage(A::AbstractMatrix, B::AbstractMatrix)
    A isa SparseMatrixCSC && !(B isa SparseMatrixCSC) &&
        return A, SparseMatrixCSC{eltype(B),Int}(sparse(B))
    A isa Matrix && !(B isa Matrix) && return A, Matrix(B)
    return A, B
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

function _materialize_matrix_eltype(A::AbstractMatrix, ::Type{T}; dense::Bool=false) where T
    materialized = _materialize_matrix(A; dense=dense)
    return eltype(materialized) === T ? materialized : T.(materialized)
end

# An integer-valued matrix has no typed method to fall through to, so promote
# its element type here as well -- forwarding only the interval would recurse
# straight back into this method.
@inline _feast_promote_eltype(A::AbstractMatrix) =
    _feast_promote_eltype(A, _feast_float_eltype(eltype(A)))
@inline _feast_promote_eltype(A::AbstractMatrix, ::Type{T}) where T =
    eltype(A) === T ? A : _materialize_matrix_eltype(A, T)

# The floating-point element type a matrix is solved in: integers, rationals
# and other exact reals have no solver of their own.
@inline _feast_float_eltype(::Type{T}) where T<:AbstractFloat = T
@inline _feast_float_eltype(::Type{T}) where T<:Real = float(T)
@inline _feast_float_eltype(::Type{Complex{T}}) where T<:Real = Complex{_feast_float_eltype(T)}
@inline _feast_float_eltype(::Type{T}) where T = T
