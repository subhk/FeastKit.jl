# Automatic sizing is opt-in. A count estimate chooses the first width; a
# saturation status or nonconvergence with inadequate oversampling triggers
# bounded growth. A failed linear solve without Ritz pairs does not trigger it.
function _feast_auto_solve(A, B, region, options; M0=nothing,
                           max_subspace_size=nothing, general::Bool=false)
    M0 === nothing || throw(ArgumentError("M0 cannot be combined with subspace_size=:auto"))
    N = size(A, 1)
    size(A) == (N, N) || throw(ArgumentError("A must be square"))
    B === nothing || size(B) == size(A) || throw(DimensionMismatch("B must match A"))
    limit = max_subspace_size === nothing ? N : max_subspace_size
    limit isa Integer && !(limit isa Bool) && 0 < limit <= N ||
        throw(ArgumentError("max_subspace_size must be an integer between 1 and the matrix dimension"))
    params = _feast_named_parameters(options.fpm; tol=options.tol, maxiter=options.maxiter,
                                     quadrature_points=options.quadrature_points, general=general,
                                     mixed_precision=get(options, :mixed_precision, nothing))
    _feast_solver_options(options.solver, options.solver_opts; matrix_free=!(A isa AbstractMatrix))
    initial = options.initial_subspace
    initial === nothing || initial isa AbstractMatrix ||
        throw(ArgumentError("initial_subspace must be a matrix"))
    seed_width = initial === nothing ? 0 : size(initial, 2)
    seed_width <= limit || throw(ArgumentError("initial_subspace exceeds max_subspace_size"))
    estimate = if A isa AbstractMatrix
        general ? feast_estimate_count(A, region...; B=B, fpm=params, nprobe=min(8,N)) :
                  feast_estimate_count(A, region; B=B, fpm=params, nprobe=min(8,N))
    else
        # Matrix-free operators have no direct factorization for a cheap count
        # estimate. Start with the usual width, then use saturation feedback.
        6.0
    end
    isfinite(estimate) || throw(ArgumentError("Eigenvalue count estimate is not finite; specify subspace_size explicitly"))
    width = min(limit, max(4, seed_width + 2, ceil(Int, 1.5 * clamp(estimate, 0, N)) + 2))
    while true
        opts = (; options..., fpm=copy(params), subspace_size=width)
        args = B === nothing ? (A,) : (A, B)
        result = general ? feast_general(args..., region...; opts...) : feast(args..., region; opts...)
        saturated = result.info == Int(Feast_ERROR_M0) ||
                    (result.info == Int(Feast_ERROR_NO_CONVERGENCE) && result.M > 0 &&
                     width < ceil(Int, 1.5result.M) + 2)
        saturated && width < limit || return result
        width = min(limit, max(width + 1, 2width))
    end
end
