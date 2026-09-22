# High-level matrix-free Feast interfaces

"""
    feast(A_op, B_op, interval; kwargs...)

High-level matrix-free Feast interface for symmetric/Hermitian problems.

# Arguments
- `A_op`: Matrix-free operator for A
- `B_op`: Matrix-free operator for B
- `interval`: Search interval (Emin, Emax) for real problems

# Keyword Arguments
- `M0`: Maximum number of eigenvalues (default: 10)
- `subspace_size`: Alias for `M0`
- `solver`: Linear solver (:gmres, :bicgstab, or custom function)
- `solver_opts`: Options for iterative solver
- `fpm`: Feast parameters
- `tol`: Convergence tolerance
- `maxiter`: Maximum refinement iterations
- `quadrature_points`: Half-contour node count (default: 8)

Named controls and explicit `fpm` entries must agree. A custom solver callback
must be configured directly; `solver_opts` is only for built-in iterative solvers.

# Returns
- `FeastResult` with eigenvalues and eigenvectors
"""
function feast(A_op::MatrixFreeOperator{T}, B_op::MatrixFreeOperator{T},
               interval::Tuple{T,T};
               M0::Union{Int,Nothing} = nothing,
               solver::Union{Symbol, Function} = :gmres,
               solver_opts::NamedTuple = NamedTuple(),
               fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
               tol=nothing, maxiter=nothing,
               subspace_size=nothing, quadrature_points=nothing,
               initial_subspace=nothing, max_subspace_size=nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A_op, B_op, interval, (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace);
                                 M0=M0, max_subspace_size=max_subspace_size, general=false)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))

    # Validate operators are compatible
    if !issymmetric(A_op) && !ishermitian(A_op)
        throw(ArgumentError("A_op must be symmetric or Hermitian for this interface"))
    end

    (; params, solver_options, M0) =
        _feast_prepare_options(size(A_op, 1); M0=M0, subspace_size=subspace_size, fpm=fpm,
                               tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                               solver=solver, solver_opts=solver_opts, initial_subspace=initial_subspace, matrix_free=true)

    # Only built-in solvers with an implicit tolerance follow outer accuracy.
    inner_tolerance = !(solver isa Function) && !haskey(solver_opts, :rtol) ? Ref(1e-3) : nothing

    # Create linear solver if needed
    linear_solver = if isa(solver, Function)
        solver
    else
        create_iterative_solver(A_op, B_op, solver; solver_options..., tolerance_ref=inner_tolerance)
    end

    # Call matrix-free RCI
    return feast_matfree_srci!(A_op, B_op, interval, M0;
                              linear_solver=linear_solver,
                              fpm=params, inner_tolerance=inner_tolerance, initial_subspace=initial_subspace)
end

"""
    feast(A_op, interval; kwargs...)

Matrix-free Feast for standard eigenvalue problems (B = I).
"""
function feast(A_op::MatrixFreeOperator{T}, interval::Tuple{T,T}; kwargs...) where T<:Real
    # Create identity operator
    N = size(A_op, 1)
    B_op = LinearOperator{T}((y, x) -> copy!(y, x), (N, N),
                           issymmetric=true, ishermitian=true, isposdef=true)

    return feast(A_op, B_op, interval; kwargs...)
end

"""
    feast_general(A_op, B_op, center, radius; kwargs...)

Matrix-free Feast for general (non-Hermitian) eigenvalue problems.
"""
function feast_general(A_op::MatrixFreeOperator{Complex{T}},
                      B_op::MatrixFreeOperator{Complex{T}},
                      center::Complex{T}, radius::T;
                      M0::Union{Int,Nothing} = nothing,
                      solver::Union{Symbol, Function} = :gmres,
                      solver_opts::NamedTuple = NamedTuple(),
                      fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                      tol=nothing, maxiter=nothing,
                      subspace_size=nothing, quadrature_points=nothing,
                      initial_subspace=nothing, max_subspace_size=nothing) where T<:Real
    if subspace_size === :auto
        return _feast_auto_solve(A_op, B_op, (center, radius), (; fpm, tol, maxiter, quadrature_points, solver, solver_opts, initial_subspace);
                                 M0=M0, max_subspace_size=max_subspace_size, general=true)
    end
    max_subspace_size === nothing || throw(ArgumentError("max_subspace_size requires subspace_size=:auto"))

    (; params, solver_options, M0) =
        _feast_prepare_options(size(A_op, 1); M0=M0, subspace_size=subspace_size, fpm=fpm,
                               tol=tol, maxiter=maxiter, quadrature_points=quadrature_points,
                               solver=solver, solver_opts=solver_opts, initial_subspace=initial_subspace, matrix_free=true, general=true)

    # Only built-in solvers with an implicit tolerance follow outer accuracy.
    inner_tolerance = !(solver isa Function) && !haskey(solver_opts, :rtol) ? Ref(1e-3) : nothing

    # Create linear solver
    linear_solver = if isa(solver, Function)
        solver
    else
        create_iterative_solver(A_op, B_op, solver; solver_options..., tolerance_ref=inner_tolerance)
    end

    # Call matrix-free RCI for general problems
    return feast_matfree_grci!(A_op, B_op, center, radius, M0;
                              linear_solver=linear_solver,
                              fpm=params, inner_tolerance=inner_tolerance, initial_subspace=initial_subspace)
end

function feast_general(A_op::MatrixFreeOperator{Complex{T}}, center::Complex, radius::Real;
                       kwargs...) where T<:Real
    N = size(A_op, 1)
    B_op = LinearOperator{Complex{T}}((y, x) -> copy!(y, x), (N, N),
                                     ishermitian=true, isposdef=true)
    return feast_general(A_op, B_op, Complex{T}(center), T(radius); kwargs...)
end

function feast_general(A_op::MatrixFreeOperator{Complex{T}}, B_op::MatrixFreeOperator{Complex{T}},
                       center::Complex, radius::Real; kwargs...) where T<:Real
    return feast_general(A_op, B_op, Complex{T}(center), T(radius); kwargs...)
end


# Matrix-free interfaces
function feast_matvec(A_mul!::Function, B_mul!::Function, N::Int,
                     interval::Tuple{T,T}; M0::Int = 10,
                     fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing) where T<:Real
    # Feast with matrix-free operations
    # A_mul!(y, x) computes y = A*x
    # B_mul!(y, x) computes y = B*x

    Emin, Emax = interval

    # Initialize Feast parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end

    return feast_sparse_matvec!(A_mul!, B_mul!, N, Emin, Emax, M0, _ensure_feast_parameters(fpm))
end
