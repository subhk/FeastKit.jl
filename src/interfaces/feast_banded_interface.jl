# Public banded-storage entry points.

function feast_banded(A::Matrix{T}, kla::Int, interval::Tuple{T,T};
                     B::Union{Matrix{T}, Nothing} = nothing, klb::Int = 0,
                     M0::Union{Int,Nothing} = nothing, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                     subspace_size=nothing, tol=nothing, maxiter=nothing,
                     quadrature_points=nothing, solver::Symbol=:direct,
                     solver_opts::NamedTuple=NamedTuple()) where T<:Real
    # Feast interface for real symmetric banded matrices

    Emin, Emax = interval

    # Initialize Feast parameters if not provided
    (; params, solver_options, M0) =
        _feast_prepare_options(size(A, 2); M0=M0, subspace_size=subspace_size, fpm=fpm,
                               tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                               solver=solver, solver_opts=solver_opts)

    if B === nothing
        # Standard eigenvalue problem - create identity in banded format
        N = size(A, 2)
        B_banded = zeros(T, 1, N)
        B_banded[1, :] .= one(T)
        return feast_sbgv!(copy(A), B_banded, kla, 0, Emin, Emax, M0, params; solver_options...)
    else
        # Generalized eigenvalue problem
        return feast_sbgv!(copy(A), copy(B), kla, klb, Emin, Emax, M0, params; solver_options...)
    end
end

function feast_banded(A::Matrix{Complex{T}}, kla::Int, interval::Tuple{T,T};
                     B::Union{Matrix{Complex{T}}, Nothing} = nothing, klb::Int = 0,
                     M0::Union{Int,Nothing} = nothing, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                     subspace_size=nothing, tol=nothing, maxiter=nothing,
                     quadrature_points=nothing, solver::Symbol=:direct,
                     solver_opts::NamedTuple=NamedTuple()) where T<:Real
    # Feast interface for complex Hermitian banded matrices

    Emin, Emax = interval

    # Initialize Feast parameters if not provided
    (; params, solver_options, M0) =
        _feast_prepare_options(size(A, 2); M0=M0, subspace_size=subspace_size, fpm=fpm,
                               tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                               solver=solver, solver_opts=solver_opts)

    if B === nothing
        # Standard eigenvalue problem
        return feast_hbev!(copy(A), kla, Emin, Emax, M0, params; solver_options...)
    else
        # Generalized eigenvalue problem
        return feast_hbgv!(copy(A), copy(B), kla, klb, Emin, Emax, M0, params; solver_options...)
    end
end

function feast_banded(A::Matrix, kla::Int, interval::Tuple{Real,Real}; kwargs...)
    Ap = _feast_promote_eltype(A)
    return feast_banded(Ap, kla, _feast_promote_interval(Ap, interval); kwargs...)
end
