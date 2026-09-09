# Shared FEAST drivers over the RCI kernels.
#
# `_feast_symmetric_real` runs `feast_srci!` and `_feast_hermitian_complex` runs
# `feast_hrci!`. Between them they cover dense and sparse, standard and
# generalized, direct and iterative -- storage only shows up in the shifted
# matrix, the factorization cache and the mat-vecs, which dispatch on the
# argument types.
#
# This replaced four near-identical contour loops (one per storage x symmetry
# combination). They had already drifted: only two of them rank-compressed the
# filtered subspace, and only some reported non-convergence.
#
# Real problems stay real. The earlier real paths promoted A and B to `Complex`
# up front, which doubled storage and matmul cost and let the imaginary part of
# the filtered subspace -- pure quadrature noise for a real pencil -- survive
# rank compression, producing spurious Ritz pairs.

# Build (z*B - A), or (z*I - A) when B is nothing, in complex arithmetic from
# real storage without materializing complex copies of A and B.
function _feast_shifted_complex(A::Matrix{T}, ::Nothing, z::Complex{T}) where T<:Real
    dest = Matrix{Complex{T}}(undef, size(A))
    @inbounds @simd for i in eachindex(dest, A)
        dest[i] = -A[i]
    end
    @inbounds for i in axes(A, 1)
        dest[i, i] += z
    end
    return dest
end

function _feast_shifted_complex(A::Matrix{T}, B::Matrix{T}, z::Complex{T}) where T<:Real
    dest = Matrix{Complex{T}}(undef, size(A))
    @inbounds @simd for i in eachindex(dest, A, B)
        dest[i] = z * B[i] - A[i]
    end
    return dest
end

# z*I - A for real sparse storage, built straight into the complex result so no
# sparse identity is materialized on every contour point.
function _feast_shifted_complex(A::SparseMatrixCSC{T,Int}, ::Nothing,
                                z::Complex{T}) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(DimensionMismatch("A must be square"))

    shifted = SparseMatrixCSC{Complex{T},Int}(A.m, A.n, copy(A.colptr),
                                              copy(A.rowval),
                                              Vector{Complex{T}}(undef, nnz(A)))
    @inbounds @simd for i in eachindex(shifted.nzval)
        shifted.nzval[i] = -A.nzval[i]
    end

    @inbounds for col in 1:N
        diagonal_found = false
        for p in shifted.colptr[col]:(shifted.colptr[col + 1] - 1)
            row = shifted.rowval[p]
            if row == col
                shifted.nzval[p] += z
                diagonal_found = true
                break
            elseif row > col
                break
            end
        end
        # A structurally missing diagonal entry cannot be filled in place, so
        # fall back to the general path for that (rare) sparsity pattern.
        diagonal_found || return spdiagm(0 => fill(z, N)) - A
    end

    return shifted
end

_feast_shifted_complex(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                       z::Complex{T}) where T<:Real = z * B - A

# Same two builders for storage that is already complex.
function _feast_shifted_complex(A::Matrix{Complex{T}}, ::Nothing,
                                z::Complex{T}) where T<:Real
    dest = Matrix{Complex{T}}(undef, size(A))
    return _feast_dense_shifted_identity_minus!(dest, z, A)
end

function _feast_shifted_complex(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                                z::Complex{T}) where T<:Real
    dest = Matrix{Complex{T}}(undef, size(A))
    @inbounds @simd for i in eachindex(dest, A, B)
        dest[i] = z * B[i] - A[i]
    end
    return dest
end

_feast_shifted_complex(A::SparseMatrixCSC{Complex{T},Int}, ::Nothing,
                       z::Complex{T}) where T<:Real =
    _feast_sparse_shifted_identity_minus(A, z)

_feast_shifted_complex(A::SparseMatrixCSC{Complex{T},Int},
                       B::SparseMatrixCSC{Complex{T},Int},
                       z::Complex{T}) where T<:Real = z * B - A

# Concretely typed factorization caches keep the shifted solve inferrable.
_feast_factor_cache(::Matrix{T}, n::Int) where T<:Real =
    Vector{Union{Nothing, LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}}}(nothing, n)

_feast_factor_cache(::Matrix{Complex{T}}, n::Int) where T<:Real =
    Vector{Union{Nothing, LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}}}(nothing, n)

_feast_factor_cache(::SparseMatrixCSC{T,Int}, n::Int) where T<:Real =
    Vector{Union{Nothing, SparseArrays.UMFPACK.UmfpackLU{Complex{T}, Int}}}(nothing, n)

_feast_factor_cache(::SparseMatrixCSC{Complex{T},Int}, n::Int) where T<:Real =
    Vector{Union{Nothing, SparseArrays.UMFPACK.UmfpackLU{Complex{T}, Int}}}(nothing, n)

# Shifted matrix-vector product used by the iterative (GMRES) solver path.
# `A` and `B` may be real or already complex; the Krylov vectors are complex
# either way because the contour shift is.
function _feast_shifted_solve!(dest::Matrix{Complex{T}},
                               rhs::Matrix{Complex{T}},
                               A::Matrix, B::Union{Matrix,Nothing},
                               z::Complex{T}, ncols::Int, tol::T,
                               maxiter::Int, restart::Int) where T<:Real
    N = size(A, 1)
    tmpA = Vector{Complex{T}}(undef, N)
    tmpB = Vector{Complex{T}}(undef, N)
    function apply_shift!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B === nothing
            @. tmpB = z * x
        else
            mul!(tmpB, B, x)
            @. tmpB = z * tmpB
        end
        mul!(tmpA, A, x)
        @. y = tmpB - tmpA
        return y
    end
    return solve_dense_shifted!(view(dest, :, 1:ncols), view(rhs, :, 1:ncols),
                                apply_shift!, :gmres, tol, maxiter, restart)
end

function _feast_shifted_solve!(dest::Matrix{Complex{T}},
                               rhs::Matrix{Complex{T}},
                               A::SparseMatrixCSC,
                               B::Union{SparseMatrixCSC,Nothing},
                               z::Complex{T}, ncols::Int, tol::T,
                               maxiter::Int, restart::Int) where T<:Real
    if B === nothing
        return solve_shifted_iterative_identity!(view(dest, :, 1:ncols),
                                                 view(rhs, :, 1:ncols),
                                                 A, z, tol, maxiter, restart)
    end
    return solve_shifted_iterative!(view(dest, :, 1:ncols), view(rhs, :, 1:ncols),
                                    A, B, z, tol, maxiter, restart)
end


# Inner tolerance for the iterative (IFEAST) shifted solves.
#
# FEAST's defining property is that it tolerates inexact solves: the contour
# filter is applied to a trial subspace that the outer loop refines anyway, so
# an inner solve only has to be accurate relative to how far the outer
# iteration still has to go. Driving GMRES to 1e-12 on the first sweep, while
# the outer residual is still O(1), buys nothing and costs Krylov iterations.
#
# The inner tolerance therefore tracks the outer residual, floored at what the
# user actually asked for. An explicitly supplied `solver_tol` is honoured
# exactly and never adapted.
@inline function _feast_inner_tol(adaptive::Bool, final_tol::T, epsout::T,
                                  loop::Int) where T<:Real
    adaptive || return final_tol
    loop <= 0 && return max(final_tol, T(1e-3))
    epsout > zero(T) || return max(final_tol, T(1e-3))
    return clamp(epsout * T(0.1), final_tol, T(1e-3))
end

# Refactorize the shifted system into `slot`, reusing the previous
# factorization's symbolic analysis where that is possible.
#
# Every contour point shares one sparsity pattern -- `z*B - A` is a structural
# union that does not depend on `z` -- so UMFPACK's symbolic analysis only has
# to run once. `lu!` reuses it and is ~30% cheaper than a fresh `lu` on a 2D
# Laplacian. It does, however, overwrite the numeric factorization it is given,
# so it can only be used when the caller is not holding on to the previous
# factorization: that is the `fpm[10] = 0` policy, one slot refactorized at
# every contour point. With `fpm[10] = 1` all `ne` factorizations must stay
# live at once and each needs its own symbolic analysis.
function _feast_refactorize(previous, shifted::SparseMatrixCSC, reuse_symbolic::Bool)
    if reuse_symbolic && previous !== nothing
        try
            return lu!(previous, shifted)
        catch err
            # A pattern change (the structurally-missing-diagonal fallback in
            # _feast_shifted_complex) makes the stored symbolic analysis
            # invalid; start over rather than fail the solve.
            @debug "Symbolic reuse rejected, refactorizing from scratch" exception=err
        end
    end
    return lu(shifted)
end

# Dense storage owns the freshly built shifted matrix, so it can be factorized
# in place instead of copied first.
_feast_refactorize(::Any, shifted::Matrix, ::Bool) = lu!(shifted)

"""
    _feast_symmetric_real(A, B, Emin, Emax, M0, fpm; solver=:direct, ...)

Drive `feast_srci!` for a real symmetric (generalized) eigenproblem stored
densely or sparsely. `B === nothing` selects the standard problem `A q = λ q`.

`fpm[10]` selects whether the shifted factorizations are cached across
refinement loops (1, the default) or discarded after each contour point (0).
Caching costs `fpm[2]` factorizations of the shifted matrix at once, which for
large `N` is the dominant memory term.
"""
function _feast_symmetric_real(A::AbstractMatrix{T},
                               B::Union{AbstractMatrix{T},Nothing},
                               Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                               solver::Symbol = :direct,
                               solver_tol::Real = 0.0,
                               solver_maxiter::Int = 500,
                               solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    B === nothing || size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    issymmetric(A) || throw(ArgumentError("Matrix A must be symmetric"))
    B !== nothing && !issymmetric(B) &&
        throw(ArgumentError("Matrix B must be symmetric positive definite"))

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_direct || FEAST_KRYLOV_AVAILABLE[] ||
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Run `using Krylov` to load the FeastKitKrylovExt extension."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    # Only the default tolerance is adapted; an explicit request is obeyed.
    adaptive_tol = solver_tol == 0.0

    workspace = FeastWorkspaceReal{T}(N, M0)
    rci_state = FeastSRCIState{T}()

    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)

    # fpm[10] = 0 keeps a single factorization slot; the shifted matrix is then
    # refactorized on every visit to a contour point.
    store_factors = fpm[10] == 1
    factor_cache = _feast_factor_cache(A, 0)
    current_slot = 1
    rhs_real = Matrix{T}(undef, N, M0)
    rhs_complex = solver_is_direct ? Matrix{Complex{T}}(undef, 0, 0) :
                                     Matrix{Complex{T}}(undef, N, M0)

    # The kernel issues at most (ne solves + 4 multiplies + 1 factorize) jobs per
    # refinement loop; the bound just stops a malformed state machine spinning.
    max_rci_iterations = max(fpm[2], 1) * 4 * (fpm[4] + 2) + 64
    rci_iterations = 0
    completed = false

    while true
        rci_iterations += 1
        if rci_iterations > max_rci_iterations
            info[] = Int(Feast_ERROR_INTERNAL)
            @warn "FEAST RCI loop exceeded its job budget" max_rci_iterations
            break
        end

        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.Aq, workspace.Sq, fpm, epsout, loop,
                    Emin, Emax, M0, workspace.lambda, workspace.q,
                    mode, workspace.res, info; state=rci_state)

        if ijob[] == Int(Feast_RCI_DONE)
            completed = true
            break

        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            solver_is_direct || continue

            if isempty(factor_cache)
                factor_cache = _feast_factor_cache(A, store_factors ? fpm[51] : 1)
            end
            current_slot = store_factors ? fpm[50] : 1
            if !(1 <= current_slot <= length(factor_cache))
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end

            if factor_cache[current_slot] === nothing || !store_factors
                try
                    factor_cache[current_slot] =
                        _feast_refactorize(factor_cache[current_slot],
                                           _feast_shifted_complex(A, B, Ze[]),
                                           !store_factors)
                catch err
                    @debug "Shifted factorization failed" shift=Ze[] exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Right-hand side is B * (current trial subspace), kept real until
            # the complex solve needs it.
            #
            # The kernel re-copies the same trial subspace into `work` before
            # every contour point, so B * work is constant across one sweep and
            # only has to be formed at its first point (fpm[50] == 1). Forming
            # it per point cost ne - 1 redundant N x N by N x M0 products per
            # refinement loop. The solve below reads `rhs_real` but never writes
            # it, so the cached value survives the sweep.
            if fpm[50] == 1
                if B === nothing
                    copyto!(rhs_real, workspace.work)
                else
                    mul!(rhs_real, B, workspace.work)
                end
            end

            if solver_is_direct
                factor = factor_cache[current_slot]
                if factor === nothing
                    info[] = Int(Feast_ERROR_INTERNAL)
                    break
                end
                copyto!(workspace.workc, rhs_real)
                try
                    ldiv!(factor, workspace.workc)
                catch err
                    @debug "Shifted solve failed" shift=Ze[] exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
                store_factors || (factor_cache[current_slot] = nothing)
            else
                copyto!(rhs_complex, rhs_real)
                success = _feast_shifted_solve!(workspace.workc, rhs_complex,
                                                A, B, Ze[], M0,
                                                _feast_inner_tol(adaptive_tol, tol_value,
                                                                 epsout[], loop[]),
                                                solver_maxiter, solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            ncols = mode[]
            mul!(view(workspace.work, :, 1:ncols), A, view(workspace.q, :, 1:ncols))

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            ncols = mode[]
            if B === nothing
                copyto!(view(workspace.work, :, 1:ncols), view(workspace.q, :, 1:ncols))
            else
                mul!(view(workspace.work, :, 1:ncols), B, view(workspace.q, :, 1:ncols))
            end

        else
            error("Unexpected FEAST RCI job code: ijob=$(ijob[])")
        end
    end

    # Leaving the loop without reaching DONE means mode[] still holds whatever
    # the last outstanding request asked for, and lambda/q are mid-iteration.
    # Reporting those as results would hand back garbage under an error code.
    M = completed ? mode[] : 0
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                             workspace.res[1:M], info[], epsout[], loop[])
end

"""
    _feast_hermitian_complex(A, B, Emin, Emax, M0, fpm; solver=:direct, ...)

Drive `feast_hrci!` for a complex Hermitian (generalized) eigenproblem stored
densely or sparsely. `B === nothing` selects the standard problem `A q = λ q`.

This is the Hermitian twin of [`_feast_symmetric_real`](@ref): identical control
flow, complex trial subspace and Ritz vectors, and the same `fpm[10]`
factorization-cache policy.
"""
function _feast_hermitian_complex(A::AbstractMatrix{Complex{T}},
                                  B::Union{AbstractMatrix{Complex{T}},Nothing},
                                  Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                  solver::Symbol = :direct,
                                  solver_tol::Real = 0.0,
                                  solver_maxiter::Int = 500,
                                  solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    B === nothing || size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    ishermitian(A) || throw(ArgumentError("Matrix A must be Hermitian"))
    B !== nothing && !ishermitian(B) &&
        throw(ArgumentError("Matrix B must be Hermitian positive definite"))

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_direct || FEAST_KRYLOV_AVAILABLE[] ||
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Run `using Krylov` to load the FeastKitKrylovExt extension."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    # Only the default tolerance is adapted; an explicit request is obeyed.
    adaptive_tol = solver_tol == 0.0

    workspace = FeastWorkspaceComplex{T}(N, M0)
    rci_state = FeastHRCIState{T}()

    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)

    store_factors = fpm[10] == 1
    factor_cache = _feast_factor_cache(A, 0)
    current_slot = 1
    # feast_hrci! hands the trial subspace over in workc and expects the solved
    # system back in the same buffer, so the right-hand side needs its own room.
    rhs = Matrix{Complex{T}}(undef, N, M0)

    max_rci_iterations = max(fpm[2], 1) * 4 * (fpm[4] + 2) + 64
    rci_iterations = 0
    completed = false

    while true
        rci_iterations += 1
        if rci_iterations > max_rci_iterations
            info[] = Int(Feast_ERROR_INTERNAL)
            @warn "FEAST RCI loop exceeded its job budget" max_rci_iterations
            break
        end

        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.zAq, workspace.zSq, fpm, epsout, loop,
                    Emin, Emax, M0, workspace.lambda, workspace.q,
                    mode, workspace.res, info; state=rci_state)

        if ijob[] == Int(Feast_RCI_DONE)
            completed = true
            break

        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            solver_is_direct || continue

            if isempty(factor_cache)
                factor_cache = _feast_factor_cache(A, store_factors ? fpm[51] : 1)
            end
            current_slot = store_factors ? fpm[50] : 1
            if !(1 <= current_slot <= length(factor_cache))
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end

            if factor_cache[current_slot] === nothing || !store_factors
                try
                    factor_cache[current_slot] =
                        _feast_refactorize(factor_cache[current_slot],
                                           _feast_shifted_complex(A, B, Ze[]),
                                           !store_factors)
                catch err
                    @debug "Shifted factorization failed" shift=Ze[] exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # As in _feast_symmetric_real: the kernel restores the trial
            # subspace into workc before each contour point, so B * workc is
            # the same at every point of one sweep. workc is overwritten by the
            # solve, `rhs` is not, so caching it here is safe.
            if fpm[50] == 1
                if B === nothing
                    copyto!(rhs, workspace.workc)
                else
                    mul!(rhs, B, workspace.workc)
                end
            end

            if solver_is_direct
                factor = factor_cache[current_slot]
                if factor === nothing
                    info[] = Int(Feast_ERROR_INTERNAL)
                    break
                end
                copyto!(workspace.workc, rhs)
                try
                    ldiv!(factor, workspace.workc)
                catch err
                    @debug "Shifted solve failed" shift=Ze[] exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
                store_factors || (factor_cache[current_slot] = nothing)
            else
                success = _feast_shifted_solve!(workspace.workc, rhs, A, B, Ze[],
                                                M0,
                                                _feast_inner_tol(adaptive_tol, tol_value,
                                                                 epsout[], loop[]),
                                                solver_maxiter,
                                                solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            ncols = mode[]
            mul!(view(workspace.workc, :, 1:ncols), A, view(workspace.q, :, 1:ncols))

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            ncols = mode[]
            if B === nothing
                copyto!(view(workspace.workc, :, 1:ncols), view(workspace.q, :, 1:ncols))
            else
                mul!(view(workspace.workc, :, 1:ncols), B, view(workspace.q, :, 1:ncols))
            end

        else
            error("Unexpected FEAST RCI job code: ijob=$(ijob[])")
        end
    end

    M = completed ? mode[] : 0
    return FeastResult{T, Complex{T}}(workspace.lambda[1:M], workspace.q[:, 1:M],
                                      M, workspace.res[1:M], info[], epsout[],
                                      loop[])
end

"""
    feast_estimate_count(A, interval; B=nothing, nprobe=16, fpm=nothing)

Estimate how many eigenvalues of `A` (or of the pencil `(A, B)`) lie in
`interval`, without solving the eigenproblem.

FEAST requires a trial subspace wider than the number of eigenvalues in the
search region, but that number is usually what you are trying to find out. Too
small an `M0` saturates the subspace: FEAST burns every refinement loop and
returns `Feast_ERROR_M0` having silently missed eigenvalues. This sizes `M0`
before the fact — take roughly `1.5x` the estimate, which is the usual FEAST
guidance.

The estimate is Hutchinson's trace estimator applied to the spectral projector
`P = (1/2πi) ∮ (zB - A)⁻¹ B dz`, whose trace is exactly the eigenvalue count.
It costs one contour sweep with `nprobe` right-hand sides — comparable to a
single refinement loop, and far cheaper than discovering the problem by
exhausting the loop budget.

Accuracy improves as the count grows: the relative error scales like
`sqrt(2M/nprobe)/M`, so a handful of probes place a large count within a few
percent while a count of two or three is only good to about ±1. That is the
useful direction, since it is the large counts that cause the failure.

Returns a `Float64`; it is a statistical estimate, not an exact count.
"""
function feast_estimate_count(A::AbstractMatrix, interval::Tuple{Real,Real};
                              B::Union{AbstractMatrix,Nothing} = nothing,
                              nprobe::Int = 16,
                              fpm::Union{Vector{Int},Nothing} = nothing)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    B === nothing || size(B) == (N, N) ||
        throw(ArgumentError("B must match the size of A"))
    nprobe > 0 || throw(ArgumentError("nprobe must be positive, got $nprobe"))

    params = if fpm === nothing
        v = zeros(Int, 64); feastinit!(v); v
    else
        copy(fpm)
    end
    feastdefault!(params)

    RT = real(float(eltype(A)))
    Emin, Emax = RT(interval[1]), RT(interval[2])
    Emin < Emax || throw(ArgumentError("interval must satisfy Emin < Emax"))

    contour = feast_get_custom_contour(RT, params)
    contour === nothing && (contour = feast_contour(Emin, Emax, params))

    p = min(nprobe, N)
    # Rademacher probes: E[v vᵀ] = I, so E[vᵀ P v] = trace(P).
    rng = MersenneTwister(hash((N, p, :feast_estimate_count)))
    V = Matrix{RT}(undef, N, p)
    @inbounds for i in eachindex(V)
        V[i] = rand(rng, Bool) ? one(RT) : -one(RT)
    end

    rhs = B === nothing ? Matrix{Complex{RT}}(V) : Matrix{Complex{RT}}(B * V)
    acc = zeros(Complex{RT}, N, p)
    X = Matrix{Complex{RT}}(undef, N, p)
    factor = nothing
    for e in eachindex(contour.Zne)
        shifted = _feast_shifted_complex(A, B, contour.Zne[e])
        factor = _feast_refactorize(factor, shifted, true)
        copyto!(X, rhs)
        ldiv!(factor, X)
        weight = 2 * contour.Wne[e]   # the omitted conjugate half
        @inbounds for i in eachindex(acc, X)
            acc[i] += weight * X[i]
        end
    end

    total = zero(RT)
    @inbounds for j in 1:p, i in 1:N
        total += V[i, j] * real(acc[i, j])
    end
    return Float64(total / p)
end
