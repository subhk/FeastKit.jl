# Public assembled symmetric, Hermitian, and general eigenproblem entry points.

# Main Feast interface functions
"""
    feast(A, [B,] interval; subspace_size=10, tol=1e-12, maxiter=20,
          quadrature_points=8, solver=:direct, solver_opts=(;), kwargs...)

Compute symmetric/Hermitian eigenpairs in `interval`. `subspace_size` (legacy
alias `M0`) should exceed the expected eigenvalue count. `tol` rounds down to
a decimal power in `[1e-16, 1]`; Float32 retains its precision floor. Node counts
follow the integration rule configured in `fpm`. Named options cannot conflict
with explicit `fpm` entries and use a copy when overriding its settings.

Use `subspace_size=:auto` with an optional `max_subspace_size` cap for count
estimation and bounded growth. Pass `initial_subspace=previous.vectors` for
serial or matrix-free warm starts. `mixed_precision=true` enables Float32
residual correction solves with Float64 residual checks on serial dense
Float64/ComplexF64 direct problems (`fpm[42]=1`, default 0).

Use `solver=:gmres` after loading Krylov for iterative shifted solves, with
`solver_opts=(rtol=..., maxiter=..., restart=...)`. Outer `maxiter` limits FEAST
refinement; `solver_opts.maxiter` limits each inner solve. Serial assembled
drivers support both solvers; complex MPI drivers also support GMRES. Explicit
unsupported backends raise an error; `backend=:auto` permits serial fallback.

The result provides `values`, `vectors`, `converged`, and a status `message`,
alongside the original FEAST fields. Check `converged` before using eigenpairs.
"""
function feast(A::AbstractMatrix{T}, B::AbstractMatrix{T},
               interval::Tuple{T,T}; M0::Union{Int,Nothing} = nothing,
               fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
               subspace_size=nothing, tol=nothing, maxiter=nothing,
               initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
               quadrature_points=nothing, solver::Symbol=:direct,
               solver_opts::NamedTuple=NamedTuple(),
               backend::Union{Symbol, Nothing} = nothing,
               parallel::Union{Bool, Symbol, Nothing} = nothing,
               strict_backend::Bool = false,
               use_threads::Bool = true,
               comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, B, interval, (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=false)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Main Feast interface for real symmetric generalized eigenvalue problems
    T <: Integer && return feast(float.(A), float.(B),
                                 (float(interval[1]), float(interval[2]));
                                 M0=M0, fpm=fpm, backend=backend, parallel=parallel,
                                 subspace_size=subspace_size, tol=tol, maxiter=maxiter, initial_subspace=initial_subspace, max_subspace_size=max_subspace_size, mixed_precision=mixed_precision,
                                 quadrature_points=quadrature_points, solver=solver, solver_opts=solver_opts,
                                 strict_backend=strict_backend, use_threads=use_threads,
                                 comm=comm)
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))
    issymmetric(A) || throw(ArgumentError("feast expects a symmetric real matrix A; use feast_general for non-symmetric problems"))
    issymmetric(B) || throw(ArgumentError("B must be symmetric positive definite for real generalized problems"))

    feast_validate_interval(A, interval)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace)

    # Materialization happens after validation so user-facing errors still refer
    # to the original matrix shape and symmetry expectations.
    A_exec = _materialize_matrix(A)
    B_exec = _materialize_matrix(B)

    return _execute_feast(A_exec, B_exec, interval, backend_choice, M0, params,
                          comm, use_threads, allow_backend_fallback; solver_options=solver_options)
end

function feast(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}},
               interval::Tuple{T,T}; M0::Union{Int,Nothing} = nothing,
               fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
               subspace_size=nothing, tol=nothing, maxiter=nothing,
               initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
               quadrature_points=nothing, solver::Symbol=:direct,
               solver_opts::NamedTuple=NamedTuple(),
               backend::Union{Symbol, Nothing} = nothing,
               parallel::Union{Bool, Symbol, Nothing} = nothing,
               strict_backend::Bool = false,
               use_threads::Bool = true,
               comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, B, interval, (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=false)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Feast interface for complex Hermitian generalized eigenvalue problems
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))
    ishermitian(A) || throw(ArgumentError("feast expects a Hermitian matrix A when using real intervals; call feast_general for non-Hermitian problems"))
    ishermitian(B) || throw(ArgumentError("B must be Hermitian positive definite for complex generalized problems"))

    feast_validate_interval(A, interval)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace)

    A_exec = _materialize_matrix(A)
    B_exec = _materialize_matrix(B)

    return _execute_feast(A_exec, B_exec, interval, backend_choice, M0, params,
                          comm, use_threads, allow_backend_fallback; solver_options=solver_options)
end

function feast(A::AbstractMatrix{T}, interval::Tuple{T,T};
               M0::Union{Int,Nothing} = nothing, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
               subspace_size=nothing, tol=nothing, maxiter=nothing,
               initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
               quadrature_points=nothing, solver::Symbol=:direct,
               solver_opts::NamedTuple=NamedTuple(),
               backend::Union{Symbol, Nothing} = nothing,
               parallel::Union{Bool, Symbol, Nothing} = nothing,
               strict_backend::Bool = false,
               use_threads::Bool = true, comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, nothing, interval, (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=false)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Feast interface for standard real symmetric eigenvalue problems (B = I)
    # No FEAST path can run in integer arithmetic: the tolerance is 10^-fpm[3]
    # and the contour is complex. Promote and re-enter.
    T <: Integer && return feast(float.(A), (float(interval[1]), float(interval[2]));
                                 M0=M0, fpm=fpm, backend=backend, parallel=parallel,
                                 subspace_size=subspace_size, tol=tol, maxiter=maxiter, initial_subspace=initial_subspace, max_subspace_size=max_subspace_size, mixed_precision=mixed_precision,
                                 quadrature_points=quadrature_points, solver=solver, solver_opts=solver_opts,
                                 strict_backend=strict_backend, use_threads=use_threads,
                                 comm=comm)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    issymmetric(A) || throw(ArgumentError("feast expects a symmetric real matrix A; use feast_general for non-symmetric problems"))
    feast_validate_interval(A, interval)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace)
    A_exec = _materialize_matrix(A)

    if backend_choice == :serial
        if A_exec isa Matrix
            return feast_syev!(A_exec, interval[1], interval[2], M0, params; solver_options...)
        elseif A_exec isa SparseMatrixCSC
            return feast_scsrev!(A_exec, interval[1], interval[2], M0, params; solver_options...)
        end
    end

    B = A_exec isa SparseMatrixCSC ? spdiagm(0 => fill(one(T), N)) : Matrix{T}(I, N, N)
    return _execute_feast(A_exec, B, interval, backend_choice, M0, params,
                          comm, use_threads, allow_backend_fallback; solver_options=solver_options)
end

function feast(A::AbstractMatrix{Complex{T}}, interval::Tuple{T,T};
               M0::Union{Int,Nothing} = nothing, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
               subspace_size=nothing, tol=nothing, maxiter=nothing,
               initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
               quadrature_points=nothing, solver::Symbol=:direct,
               solver_opts::NamedTuple=NamedTuple(),
               backend::Union{Symbol, Nothing} = nothing,
               parallel::Union{Bool, Symbol, Nothing} = nothing,
               strict_backend::Bool = false,
               use_threads::Bool = true, comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, nothing, interval, (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=false)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Feast interface for standard complex Hermitian eigenvalue problems (B = I)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    ishermitian(A) || throw(ArgumentError("feast expects a Hermitian matrix A when using real intervals; call feast_general for non-Hermitian problems"))
    feast_validate_interval(A, interval)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace)
    A_exec = _materialize_matrix(A)

    if backend_choice == :serial
        if A_exec isa Matrix
            return feast_heev!(A_exec, interval[1], interval[2], M0, params; solver_options...)
        elseif A_exec isa SparseMatrixCSC
            return feast_hcsrev!(A_exec, interval[1], interval[2], M0, params; solver_options...)
        end
    end

    identity_vals = fill(one(Complex{T}), N)
    B = A_exec isa SparseMatrixCSC ? spdiagm(0 => identity_vals) : Matrix{Complex{T}}(I, N, N)
    return _execute_feast(A_exec, B, interval, backend_choice, M0, params,
                          comm, use_threads, allow_backend_fallback; solver_options=solver_options)
end

"""
    feast_general(A, [B,] center, radius; kwargs...)

Compute general eigenpairs inside a circular complex region. Accepts the same
named options as [`feast`](@ref), with `quadrature_points` setting the full
contour count (default 16). Returns `FeastGeneralResult`. For other full contour
shapes, pass a `FeastContour` directly to `feast(A, contour)`.
"""
function feast_general(A::AbstractMatrix, B::AbstractMatrix,
                       center::Complex{T}, radius::T; M0::Union{Int,Nothing} = nothing,
                       fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                       subspace_size=nothing, tol=nothing, maxiter=nothing,
                       initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
                       quadrature_points=nothing, solver::Symbol=:direct,
                       solver_opts::NamedTuple=NamedTuple(),
                       backend::Union{Symbol, Nothing} = nothing,
                       parallel::Union{Bool, Symbol, Nothing} = nothing,
                       strict_backend::Bool = false,
                       use_threads::Bool = true,
                       comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, B, (center, radius), (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=true)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Feast interface for general (non-Hermitian) eigenvalue problems
    # Uses circular contour in complex plane

    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))

    N = size(A, 1)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace, general=true)

    (; A_exec, B_exec, center_exec, radius_exec) = _feast_prepare_general_matrices(A, B, center, radius)

    return _execute_feast_general(A_exec, B_exec, center_exec, radius_exec,
                                  backend_choice, M0, params, comm, use_threads,
                                  allow_backend_fallback; solver_options=solver_options)
end

function feast_general(A::AbstractMatrix, B::AbstractMatrix,
                       center::Complex{Tc}, radius::Tr; kwargs...) where {Tc<:Real, Tr<:Real}
    T = promote_type(Tc, Tr)
    center_promoted = Complex{T}(center)
    radius_promoted = convert(T, radius)
    return feast_general(A, B, center_promoted, radius_promoted; kwargs...)
end

function feast_general(A::AbstractMatrix, center::Complex{T}, radius::T;
                       M0::Union{Int,Nothing} = nothing, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                       subspace_size=nothing, tol=nothing, maxiter=nothing,
                       initial_subspace=nothing, max_subspace_size=nothing, mixed_precision=nothing,
                       quadrature_points=nothing, solver::Symbol=:direct,
                       solver_opts::NamedTuple=NamedTuple(),
                       backend::Union{Symbol, Nothing} = nothing,
                       parallel::Union{Bool, Symbol, Nothing} = nothing,
                       strict_backend::Bool = false,
                       use_threads::Bool = true,
                       comm = nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A, nothing, (center, radius), (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace, mixed_precision, backend, parallel, strict_backend, use_threads, comm);
                                 M0=M0, max_subspace_size=max_subspace_size, general=true)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))
    # Feast interface for standard general eigenvalue problems (B = I)

    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))

    N = size(A, 1)

    (; params, solver_options, M0, backend_choice, allow_backend_fallback) =
        _feast_prepare_assembled(A; M0=M0, subspace_size=subspace_size, fpm=fpm,
                                tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                                solver=solver, solver_opts=solver_opts, parallel=parallel,
                                backend=backend, strict_backend=strict_backend, comm=comm, mixed_precision=mixed_precision, initial_subspace=initial_subspace, general=true)

    (; A_exec, center_exec, radius_exec) = _feast_prepare_general_matrices(A, nothing, center, radius)

    if backend_choice == :serial
        if A_exec isa Matrix
            return feast_geev!(A_exec, center_exec, radius_exec, M0, params; solver_options...)
        elseif A_exec isa SparseMatrixCSC
            return feast_gcsrev!(A_exec, center_exec, radius_exec, M0, params; solver_options...)
        end
    end

    B_exec = A_exec isa SparseMatrixCSC ?
             spdiagm(0 => fill(one(eltype(A_exec)), N)) :
             Matrix{eltype(A_exec)}(I, N, N)
    return _execute_feast_general(A_exec, B_exec, center_exec, radius_exec,
                                  backend_choice, M0, params, comm, use_threads,
                                  allow_backend_fallback; solver_options=solver_options)
end

function feast_general(A::AbstractMatrix, center::Complex{Tc}, radius::Tr; kwargs...) where {Tc<:Real, Tr<:Real}
    T = promote_type(Tc, Tr)
    center_promoted = Complex{T}(center)
    radius_promoted = convert(T, radius)
    return feast_general(A, center_promoted, radius_promoted; kwargs...)
end

function feast(A::AbstractMatrix, interval::Tuple{Real,Real}; kwargs...)
    Ap = _feast_promote_eltype(A)
    return feast(Ap, _feast_promote_interval(Ap, interval); kwargs...)
end

function feast(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple{Real,Real}; kwargs...)
    Ap = _feast_promote_eltype(A)
    Bp = _feast_promote_eltype(B)
    if eltype(Ap) !== eltype(Bp)
        TE = promote_type(eltype(Ap), eltype(Bp))
        Ap = _materialize_matrix_eltype(Ap, TE)
        Bp = _materialize_matrix_eltype(Bp, TE)
    end
    return feast(Ap, Bp, _feast_promote_interval(Ap, interval); kwargs...)
end
