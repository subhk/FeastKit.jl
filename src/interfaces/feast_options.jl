const FEAST_PARAMETERS_LENGTH = 64

@inline function _ensure_feast_parameters(fpm::Union{Vector{Int},FeastParameters,Nothing})
    # High-level APIs accept nothing, a raw fpm vector, or the wrapper type. The
    # solver kernels always receive the concrete Vector{Int} expected by FEAST.
    if fpm === nothing
        params = zeros(Int, FEAST_PARAMETERS_LENGTH)
        feastinit!(params)
        return params
    end
    vec = fpm isa FeastParameters ? fpm.fpm : fpm
    length(vec) >= FEAST_PARAMETERS_LENGTH ||
        throw(ArgumentError("fpm vector must have length ≥ $(FEAST_PARAMETERS_LENGTH)"))
    return vec
end

# Shared translation from named public options to the legacy FEAST controls.
function _feast_subspace_size(M0, subspace_size, N)
    for (name, value) in ((:M0, M0), (:subspace_size, subspace_size))
        value === nothing && continue
        value isa Integer && !(value isa Bool) && value > 0 ||
            throw(ArgumentError("$name must be a positive integer"))
    end
    if M0 !== nothing && subspace_size !== nothing && M0 != subspace_size
        throw(ArgumentError("Conflicting M0=$M0 and subspace_size=$subspace_size; use either name, or the same value for both"))
    end
    requested = something(subspace_size, M0, 10)
    return Int(min(requested, N))
end

function _feast_named_parameters(fpm; tol=nothing, maxiter=nothing,
                                 quadrature_points=nothing, general::Bool=false, mixed_precision=nothing)
    params = _ensure_feast_parameters(fpm)
    if tol === nothing && maxiter === nothing && quadrature_points === nothing && mixed_precision === nothing
        return params
    end
    # Named settings must not change a caller's reusable parameter vector.
    params = copy(params)
    if tol !== nothing
        tol isa Real && !(tol isa Bool) && isfinite(tol) && 1e-16 <= tol <= 1 ||
            throw(ArgumentError("tol must be finite and between 1e-16 and 1; FEAST uses decimal tolerance exponents"))
        exponent = ceil(Int, -log10(tol))
        _feast_set_named_parameter!(params, 3, exponent, :tol)
    end
    for (name, value, index) in ((:maxiter, maxiter, 4),
                                 (:quadrature_points, quadrature_points, general ? 8 : 2))
        value === nothing && continue
        value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) ||
            throw(ArgumentError("$name must be a positive integer"))
        name === :quadrature_points && general && value < 2 &&
            throw(ArgumentError("quadrature_points must be at least 2 for a full contour"))
        _feast_set_named_parameter!(params, index, Int(value), name)
    end
    if mixed_precision !== nothing
        mixed_precision isa Bool || throw(ArgumentError("mixed_precision must be true or false"))
        old = params[42]
        (old == FEAST_UNINITIALIZED || old == Int(mixed_precision)) ||
            throw(ArgumentError("Conflicting mixed_precision and fpm[42]=$old"))
        params[42] = Int(mixed_precision)
    end
    # Validate node counts against the selected quadrature rule now, before a
    # backend can interpret an invalid configuration as a runtime failure.
    feastdefault!(params)
    return params
end

function _feast_set_named_parameter!(params, index, value, name)
    old = params[index]
    unset = old == FEAST_UNINITIALIZED || (index != 3 && old <= 0)
    unset || old == value || throw(ArgumentError(
        "Conflicting $name and fpm[$index]=$old; omit one setting or make them agree"))
    params[index] = value
    return params
end

function _feast_solver_options(solver, opts::NamedTuple; matrix_free::Bool=false)
    allowed = matrix_free ? (:gmres, :bicgstab) : (:direct, :gmres, :iterative)
    if matrix_free && solver isa Function
        isempty(opts) || throw(ArgumentError("solver_opts cannot be used with a custom solver callback; configure the callback directly"))
        return opts
    end
    solver in allowed || throw(ArgumentError(matrix_free ?
        "Matrix-free solves require solver=:gmres, :bicgstab, or a callback; :direct requires assembled matrices" :
        "Use solver=:direct or :gmres for assembled matrices"))
    solver == :direct && !isempty(opts) &&
        throw(ArgumentError("solver_opts applies to iterative solves; use solver=:gmres"))
    keys_allowed = matrix_free ? (:rtol, :maxiter, :restart, :preconditioner) : (:rtol, :maxiter, :restart)
    for name in keys(opts)
        name in keys_allowed || throw(ArgumentError("Unsupported solver_opts key :$name; supported keys are $keys_allowed"))
    end
    if haskey(opts, :rtol)
        rtol = opts.rtol
        rtol isa Real && isfinite(rtol) && 0 < rtol < 1 ||
            throw(ArgumentError("solver_opts.rtol must be finite and between 0 and 1"))
    end
    for name in (:maxiter, :restart)
        haskey(opts, name) || continue
        value = opts[name]
        value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) ||
            throw(ArgumentError("solver_opts.$name must be a positive integer"))
    end
    if matrix_free
        # Keep the established matrix-free defaults and preconditioner API.
        return merge(opts, (; rtol=Float64(get(opts, :rtol, 1e-6)),
                             maxiter=Int(get(opts, :maxiter, 1000)),
                             restart=Int(get(opts, :restart, 30))))
    end
    return (; solver=solver === :iterative ? :gmres : solver,
              solver_tol=get(opts, :rtol, 0.0),
              solver_maxiter=Int(get(opts, :maxiter, 500)),
              solver_restart=Int(get(opts, :restart, 30)))
end


# Advanced configuration functions
function feast_set_defaults!(fpm::Vector{Int};
                            print_level::Int = 1,
                            integration_points::Int = 8,
                            tolerance_exp::Int = 12,
                            max_refinement::Int = 20)
    # Set common Feast parameters with user-friendly names
    # Validate against the same constraints as feastdefault!

    length(fpm) >= 64 || throw(ArgumentError("fpm array must have at least 64 elements"))

    print_level <= 1 ||
        throw(ArgumentError("print_level must be 0, 1, or negative for file output, got $print_level"))

    integration_points > 0 ||
        throw(ArgumentError("integration_points must be positive, got $integration_points"))

    0 <= tolerance_exp <= 16 ||
        throw(ArgumentError("tolerance_exp must be between 0 and 16, got $tolerance_exp"))

    max_refinement > 0 ||
        throw(ArgumentError("max_refinement must be positive, got $max_refinement"))

    fpm[1] = print_level
    fpm[2] = integration_points
    fpm[3] = tolerance_exp
    fpm[4] = max_refinement

    return fpm
end
