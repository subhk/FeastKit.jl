const _FeastResultLike = Union{FeastResult,FeastGeneralResult}

# Aliases keep the original fields and constructors compatible with existing
# callers while matching LinearAlgebra.Eigen's values/vectors vocabulary.
function Base.getproperty(result::_FeastResultLike, name::Symbol)
    name === :values && return getfield(result, :lambda)
    name === :vectors && return getfield(result, :q)
    name === :converged && return getfield(result, :info) == 0
    name === :message && return _feast_status_message(getfield(result, :info),
                                                      getfield(result, :M))
    return getfield(result, name)
end

Base.propertynames(result::_FeastResultLike, private::Bool=false) =
    (fieldnames(typeof(result))..., :values, :vectors, :converged, :message)

function _feast_require_convergence(result::_FeastResultLike)
    result.converged || error("FEAST solve failed (info=$(result.info)): $(result.message). " *
                              "Use feast(...) to inspect the full result.")
    return nothing
end

# `check=true` turns a failed solve into an error. Without it the values are
# still returned, as they always were, but never silently: a non-converged
# result can contain spurious or inaccurate values.
function _feast_checked(result::_FeastResultLike, check::Bool)
    if check
        _feast_require_convergence(result)
    elseif !result.converged
        @warn "FEAST did not converge (info=$(result.info)): $(result.message). " *
              "Returning unverified values; pass check=true to throw instead, or " *
              "call feast(...) to inspect the full result."
    end
    return result
end

function _feast_status_message(info::Int)
    info == 0 && return "Success"
    info == 1 && return "Invalid matrix dimension; use a nonempty square matrix"
    info == 2 && return "Subspace too small or invalid; increase subspace_size (M0) or narrow the search region"
    info == 3 && return "Invalid interval; use finite endpoints with Emin < Emax"
    info == 4 && return "Invalid contour; use a finite center and a positive radius"
    info == 5 && return "Not converged; increase maxiter, check subspace_size, or tighten the shifted solver tolerance"
    info == 6 && return "Allocation failed; reduce subspace_size or the problem size"
    info == 7 && return "Solver failed; check the shifted systems and solver settings"
    info == 8 && return "Factorization or reduced eigensolve failed; check the matrix pencil and contour"
    info == 9 && return "Invalid FEAST parameters; check fpm or the named options"
    return "Unrecognized FEAST status code $info"
end

function _feast_status_message(info::Int, M::Int)
    info == 0 && M == 0 && return "Success; the search region contains no eigenvalues"
    return _feast_status_message(info)
end

function Base.show(io::IO, result::_FeastResultLike)
    print(io, nameof(typeof(result)), "(", result.M, " eigenvalues, ",
          result.converged ? "converged" : "info=$(result.info)",
          ", residual=", result.epsout, ", iterations=", result.loop, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::_FeastResultLike)
    show(io, result)
    if !result.converged
        print(io, '\n', result.message)
    end
    if !isempty(result.values)
        print(io, "\nvalues: ")
        # Bound the display even when a large eigenvalue set is returned.
        shown = min(length(result.values), 8)
        show(IOContext(io, :compact => true, :limit => true), result.values[1:shown])
        shown < length(result.values) && print(io, " … (", length(result.values) - shown, " more)")
    end
end

# Convenience functions with different interfaces
"""
    eigvals_feast(A, interval; check=false, kwargs...)
    eigvals_feast(A, B, interval; check=false, kwargs...)

Return only the eigenvalues from [`feast`](@ref). With `check=true`, throw an
`ErrorException` if the returned FEAST status is nonzero, including subspace
saturation. The error includes the status code and recovery guidance.

The default `check=false` still returns the eigenvalues after an unsuccessful
solve, but logs a warning with the status, because such values may be
inaccurate or spurious. It does not suppress exceptions raised by `feast`. Use
`feast` directly to inspect convergence, residuals, and any partial results.
Other keywords are forwarded to `feast`.
"""
function eigvals_feast(A::AbstractMatrix, interval::Tuple; check::Bool=false, kwargs...)
    # Return only eigenvalues
    result = feast(A, interval; kwargs...)
    _feast_checked(result, check)
    return result.lambda
end

"""
    eigen_feast(A, interval; check=false, kwargs...)
    eigen_feast(A, B, interval; check=false, kwargs...)

Return a `LinearAlgebra.Eigen` object with `values` and `vectors` from
[`feast`](@ref). With `check=true`, throw an `ErrorException` for any nonzero
FEAST status, including subspace saturation, with the status code and recovery
guidance. The default `check=false` still returns the decomposition after an
unsuccessful solve, but logs a warning with the status. It does not suppress
exceptions raised by `feast`. Use `feast` directly to inspect convergence,
residuals, and any partial results. Other keywords are forwarded to `feast`.
"""
function eigen_feast(A::AbstractMatrix, interval::Tuple; check::Bool=false, kwargs...)
    # Return eigenvalues and eigenvectors as Eigen object
    result = feast(A, interval; kwargs...)
    _feast_checked(result, check)
    return Eigen(result.lambda, result.q)
end

function eigvals_feast(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple;
                      check::Bool=false, kwargs...)
    # Return only eigenvalues for generalized problem
    result = feast(A, B, interval; kwargs...)
    _feast_checked(result, check)
    return result.lambda
end

function eigen_feast(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple;
                     check::Bool=false, kwargs...)
    # Return eigenvalues and eigenvectors for generalized problem
    result = feast(A, B, interval; kwargs...)
    _feast_checked(result, check)
    return Eigen(result.lambda, result.q)
end

# Utility functions for result analysis
function feast_summary(io::IO, result::FeastResult)
    # Print summary of Feast results to the provided IO
    println(io, "FeastKit Eigenvalue Solution Summary")
    println(io, "="^40)
    println(io, "Eigenvalues found: ", result.M)
    println(io, "Final residual: ", result.epsout)
    println(io, "Refinement loops: ", result.loop)
    println(io, "Exit status: ", result.message, " (info=", result.info, ")")
    if result.M > 0
        println(io, "\nEigenvalues:")
        for i in 1:result.M
            println(io, "  λ[$i] = ", result.lambda[i], "  (residual: ", result.res[i], ")")
        end
    end
    return nothing
end

function feast_summary(result::FeastResult)
    feast_summary(stdout, result)
end

# Non-Hermitian solves return a FeastGeneralResult, whose eigenvalues are
# complex. Without these methods `feast_summary(feast_general(...))` was a
# MethodError even though the docs present the function generically.
function feast_summary(io::IO, result::FeastGeneralResult)
    println(io, "FeastKit Eigenvalue Solution Summary (non-Hermitian)")
    println(io, "="^40)
    println(io, "Eigenvalues found: ", result.M)
    println(io, "Final residual: ", result.epsout)
    println(io, "Refinement loops: ", result.loop)
    println(io, "Exit status: ", result.message, " (info=", result.info, ")")
    if result.M > 0
        println(io, "\nEigenvalues:")
        for i in 1:result.M
            println(io, "  λ[$i] = ", result.lambda[i], "  (residual: ", result.res[i], ")")
        end
    end
    return nothing
end

function feast_summary(result::FeastGeneralResult)
    feast_summary(stdout, result)
end
