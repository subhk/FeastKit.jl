# Feast sparse matrix routines
# Translated from dzfeast_sparse.f90 and related files

using SparseArrays
using LinearAlgebra
using Random

# Shifted operator for GMRES solves on explicit sparse matrices.
struct SparseShiftedOperator{CT<:Complex,TA<:AbstractMatrix,TB<:AbstractMatrix}
    A::TA
    B::TB
    z::CT
    tmpB::Vector{CT}
    tmpA::Vector{CT}
end

Base.size(op::SparseShiftedOperator) = size(op.A)
Base.eltype(::SparseShiftedOperator{CT}) where {CT} = CT

function LinearAlgebra.mul!(y::AbstractVector{CT}, op::SparseShiftedOperator{CT}, x::AbstractVector{CT}) where CT
    # Compute y = z * B * x - A * x using the operator-owned scratch buffers.
    mul!(op.tmpB, op.B, x)
    @. op.tmpB = op.z * op.tmpB
    mul!(op.tmpA, op.A, x)
    @. y = op.tmpB - op.tmpA
    return y
end

"""
    MatrixFreeShiftedOperator(N, z, A_matvec!, B_matvec!, T)

Matrix-free representation of `(zB - A)` for Krylov solves. The real and
imaginary scratch vectors let real-valued user callbacks operate on complex
vectors without allocating split views on every `mul!` call.
"""
mutable struct MatrixFreeShiftedOperator{T<:Real,CT<:Complex,FA,FB}
    N::Int
    z::CT
    A_matvec!::FA
    B_matvec!::FB
    tmp_real::Vector{T}
    tmp_imag::Vector{T}
    result_real::Vector{T}
    result_imag::Vector{T}
    x_real::Vector{T}
    x_imag::Vector{T}
end

MatrixFreeShiftedOperator(N::Int, z::CT, A_matvec!::FA,
                          B_matvec!::FB, ::Type{T}) where {T<:Real,CT<:Complex,FA,FB} =
    MatrixFreeShiftedOperator{T,CT,FA,FB}(N, z, A_matvec!, B_matvec!,
                                          zeros(T, N), zeros(T, N),
                                          zeros(T, N), zeros(T, N),
                                          zeros(T, N), zeros(T, N))

Base.size(op::MatrixFreeShiftedOperator) = (op.N, op.N)
Base.eltype(::MatrixFreeShiftedOperator{T,CT}) where {T,CT} = CT

function LinearAlgebra.mul!(y::AbstractVector{CT}, op::MatrixFreeShiftedOperator{T,CT}, x::AbstractVector{CT}) where {T<:Real,CT<:Complex}
    # User callbacks are real-valued, so split x into real/imaginary parts,
    # apply A and B to each part, and recombine z * Bx - Ax in complex form.
    @inbounds for i in 1:op.N
        xi = x[i]
        op.x_real[i] = real(xi)
        op.x_imag[i] = imag(xi)
    end

    op.B_matvec!(op.tmp_real, op.x_real)
    op.B_matvec!(op.tmp_imag, op.x_imag)

    z_real = real(op.z)
    z_imag = imag(op.z)

    @inbounds for i in 1:op.N
        br = op.tmp_real[i]
        bi = op.tmp_imag[i]
        op.result_real[i] = z_real * br - z_imag * bi
        op.result_imag[i] = z_real * bi + z_imag * br
    end

    op.A_matvec!(op.tmp_real, op.x_real)
    op.A_matvec!(op.tmp_imag, op.x_imag)

    @inbounds for i in 1:op.N
        op.result_real[i] -= op.tmp_real[i]
        op.result_imag[i] -= op.tmp_imag[i]
        y[i] = Complex{T}(op.result_real[i], op.result_imag[i])
    end

    return y
end

@inline function _check_complex_symmetric(A::SparseMatrixCSC)
    issymmetric(A) || throw(ArgumentError("Matrix must be complex symmetric (equal to its transpose)"))
end

@inline function _convert_sparse_complex(A::SparseMatrixCSC{T,Int}, ::Type{Complex{T}}) where T<:Real
    return SparseMatrixCSC{Complex{T},Int}(A.m, A.n, copy(A.colptr), copy(A.rowval),
                                           Complex{T}.(A.nzval))
end

function _feast_sparse_shifted_identity_minus(A::SparseMatrixCSC{Complex{T},Int},
                                              z::Complex{T}) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(DimensionMismatch("A must be square"))

    shifted = copy(A)
    @inbounds @simd for i in eachindex(shifted.nzval)
        shifted.nzval[i] = -shifted.nzval[i]
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

        if !diagonal_found
            return spdiagm(0 => fill(z, N)) - A
        end
    end

    return shifted
end

struct SparseIdentityShiftedOperator{CT<:Complex,TA<:SparseMatrixCSC}
    A::TA
    z::CT
    tmpA::Vector{CT}
end

Base.size(op::SparseIdentityShiftedOperator) = size(op.A)
Base.eltype(::SparseIdentityShiftedOperator{CT}) where {CT} = CT

function LinearAlgebra.mul!(y::AbstractVector{CT},
                            op::SparseIdentityShiftedOperator{CT},
                            x::AbstractVector{CT}) where CT
    mul!(op.tmpA, op.A, x)
    @. y = op.z * x - op.tmpA
    return y
end

@inline function _copyto_complex!(dest::AbstractVector{Complex{T}}, src::AbstractVector{T}) where T<:Real
    @inbounds for i in eachindex(src)
        dest[i] = Complex{T}(src[i])
    end
    return dest
end

"""
    solve_shifted_iterative!(dest, rhs, A, B, z, tol, maxiter, gmres_restart)

Solve the sparse shifted system `(zB - A) * X = rhs` column by column with
GMRES. The operator and residual buffers are reused for all right-hand sides,
while Krylov owns the per-column iterate it returns.
"""
function solve_shifted_iterative!(dest::AbstractMatrix{CT},
                                  rhs::AbstractMatrix{CT},
                                  A::SparseMatrixCSC,
                                  B::SparseMatrixCSC,
                                  z::CT, tol::TR,
                                  maxiter::Int, gmres_restart::Int) where {CT<:Complex, TR<:Real}
    N = size(A, 1)
    ncols = size(rhs, 2)

    # Create temporary vectors for the operator
    tmpB = Vector{CT}(undef, N)
    tmpA = Vector{CT}(undef, N)

    op = SparseShiftedOperator(A, B, z, tmpB, tmpA)
    residual = Vector{CT}(undef, N)

    for j in 1:ncols
        b = view(rhs, :, j)
        # Note: Krylov.gmres doesn't support initial guess, starts from zero
        x_sol, stats = gmres(op, b;
                             restart=true,
                             memory=max(gmres_restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        mul!(residual, op, x_sol)
        @. residual -= b
        res_norm = norm(residual)
        b_norm = norm(b)
        # Keep this independent check aligned with Krylov's convergence test;
        # the explicit residual may be slightly larger from roundoff alone.
        residual_limit = 10 * tol * max(b_norm, one(b_norm))
        if !stats.solved || res_norm > residual_limit
            return false
        end
        dest[:, j] .= x_sol
    end

    return true
end

function solve_shifted_iterative_identity!(dest::AbstractMatrix{CT},
                                           rhs::AbstractMatrix{CT},
                                           A::SparseMatrixCSC,
                                           z::CT, tol::TR,
                                           maxiter::Int, gmres_restart::Int) where {CT<:Complex, TR<:Real}
    N = size(A, 1)
    ncols = size(rhs, 2)
    tmpA = Vector{CT}(undef, N)
    op = SparseIdentityShiftedOperator(A, z, tmpA)
    residual = Vector{CT}(undef, N)

    for j in 1:ncols
        b = view(rhs, :, j)
        x_sol, stats = gmres(op, b;
                             restart=true,
                             memory=max(gmres_restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        mul!(residual, op, x_sol)
        @. residual -= b
        res_norm = norm(residual)
        b_norm = norm(b)
        residual_limit = 10 * tol * max(b_norm, one(b_norm))
        if !stats.solved || res_norm > residual_limit
            return false
        end
        dest[:, j] .= x_sol
    end

    return true
end

"""
    _feast_sparse_hermitian(A, B, Emin, Emax, M0, fpm; solver=:direct)

Shared sparse complex Hermitian FEAST implementation. It mirrors the dense
Hermitian path but uses sparse factorizations or sparse GMRES for the shifted
systems. Work arrays are kept outside the contour loop to avoid repeated
allocation during refinement.
"""
function _feast_sparse_hermitian(A::SparseMatrixCSC{Complex{T},Int},
                                 B::Union{SparseMatrixCSC{Complex{T},Int},Nothing},
                                 Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    N = size(A, 1)

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    # Workspace fields are reused for the basis, shifted solves, Ritz vectors,
    # residuals, and interval reordering scratch across all FEAST loops.
    workspace = FeastWorkspaceComplex{T}(N, M0)
    Q_basis = view(workspace.workc, :, 1:M0)
    _feast_seeded_subspace_complex!(Q_basis)
    solutions = workspace.q
    lambda_vec = workspace.lambda
    res_vec = workspace.res

    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)
    Aq_herm = similar(zAq)
    Sq_herm = similar(zSq)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    # Always allocate a separate rhs_buffer to avoid aliasing Q_basis
    # (aliased views are fragile — any future in-place write to one corrupts the other)
    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(solutions)
    residual_vec = zeros(Complex{T}, N)
    Bq_vec = B === nothing ? nothing : zeros(Complex{T}, N)

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    factor_cache = Vector{Union{Nothing, SparseArrays.UMFPACK.UmfpackLU{Complex{T}, Int}}}(undef, length(Zne))
    fill!(factor_cache, nothing)

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)
    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0

    # Allocate buffer for accumulated filtered subspace
    Q_proj = zeros(Complex{T}, N, M0)

    for loop_idx in 0:maxloop
        # Reset contour accumulators before applying the spectral projector to
        # the current basis.
        loop_count = loop_idx
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(Q_proj, zero(Complex{T}))

        gmres_failed = false

        for e in 1:length(Zne)
            # Each contour point contributes a shifted solve plus weighted
            # moment matrices for the reduced eigenproblem.
            z = Zne[e]
            weight = 2 * Wne[e]

            if B === nothing
                rhs_buffer .= Q_basis
            else
                mul!(rhs_buffer, B, Q_basis)
            end

            if solver_is_direct
                solver_factor = factor_cache[e]
                try
                    if solver_factor === nothing
                        shifted_matrix = B === nothing ? _feast_sparse_shifted_identity_minus(A, z) : z * B - A
                        solver_factor = lu(shifted_matrix)
                        factor_cache[e] = solver_factor
                    end
                    ldiv!(solutions, solver_factor, rhs_buffer)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Sparse direct solve failed for shift $z" exception=err
                    gmres_failed = true
                    break
                end
            else
                success = if B === nothing
                    solve_shifted_iterative_identity!(solutions, rhs_buffer, A, z,
                                                      tol_value, solver_maxiter,
                                                      solver_restart)
                else
                    solve_shifted_iterative!(solutions, rhs_buffer, A, B,
                                             z, tol_value, solver_maxiter,
                                             solver_restart)
                end
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    gmres_failed = true
                    break
                end
            end

            # Accumulate filtered subspace
            @. Q_proj += weight * solutions

            mul!(moment, adjoint(Q_basis), solutions)
            @inbounds zAq .+= weight .* moment
            @inbounds zSq .+= (weight * z) .* moment
        end

        if gmres_failed
            break
        end

        try
            # For half-contour integration of Hermitian problems, the full contour
            # integral yields Hermitian reduced matrices (R(z̄)^H = R(z) for A = A^H).
            # Extract the Hermitian part via (M + M^H)/2, NOT real/symmetric part.
            _feast_hermitian_part!(Aq_herm, zAq)
            _feast_hermitian_part!(Sq_herm, zSq)

            # Solve Hermitian generalized eigenproblem: Sq*x = lambda*Aq*x
            # Eigenvalues are real; eigenvectors are complex.
            lambda_red = Vector{T}(undef, 0)
            v_red = Array{Complex{T}}(undef, 0, 0)
            try
                F = eigen(Hermitian(Sq_herm), Hermitian(Aq_herm))
                lambda_red = Vector{T}(F.values)
                v_red = Matrix{Complex{T}}(F.vectors)
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general complex eigenvalue solver
                    F = eigen(Sq_herm, Aq_herm)
                    lambda_red = Vector{T}(real.(F.values))
                    v_red = Matrix{Complex{T}}(F.vectors)
                else
                    rethrow(e)
                end
            end

            # Project eigenvectors using filtered subspace Q_proj (complex coefficients)
            for idx in 1:M0
                mul!(view(solutions, :, idx), Q_proj, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda_vec, solutions, perm,
                                             lambda_tmp, solutions_tmp,
                                             Emin, Emax, M0)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Normalize only eigenvectors inside interval (for residual computation)
            for j in 1:M
                vec = view(solutions, :, j)
                nrm = norm(vec)
                nrm > 0 && (vec ./= nrm)
            end

            # Compute residuals only for eigenvalues inside interval
            max_res = zero(T)
            for j in 1:M
                q_col = view(solutions, :, j)
                mul!(residual_vec, A, q_col)
                if B === nothing
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    mul!(Bq_vec, B, q_col)
                    @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                end
                # Relative residual: normalize by max(|λ|, 1)
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M

            if epsout_val <= eps_tol
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Use full M0 subspace for next iteration
            copyto!(Q_basis, solutions)
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced eigenvalue problem failed during sparse Hermitian FEAST" exception=err
            break
        end
    end

    workspace.lambda[1:M_found] .= lambda_vec[1:M_found]
    workspace.res[1:M_found] .= res_vec[1:M_found]

    if M_found == 0
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end

    lambda = lambda_vec[1:M_found]
    q = solutions[:, 1:M_found]
    res = res_vec[1:M_found]

    return FeastResult{T, Complex{T}}(lambda, q, M_found, res,
                                      info_code, epsout_val, loop_count)
end

"""
    _feast_sparse_complex_symmetric(A, B, Emid, r, M0, fpm; solver=:direct)

Shared sparse complex-symmetric FEAST implementation. The shifted systems still
use sparse LU or GMRES, while the reduced Ritz pencil is formed with the
transpose bilinear form `Qᵀ A Q` and `Qᵀ B Q` required by complex-symmetric
problems.
"""
@views function _feast_sparse_complex_symmetric(A::SparseMatrixCSC{Complex{T},Int},
                                                B::SparseMatrixCSC{Complex{T},Int},
                                                Emid::Complex{T}, r::T,
                                                M0::Int, fpm::Vector{Int};
                                                solver::Symbol = :direct,
                                                solver_tol::Real = 0.0,
                                                solver_maxiter::Int = 500,
                                                solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))
    _check_complex_symmetric(A)
    _check_complex_symmetric(B)

    feastdefault!(fpm)
    check_feast_grci_input(N, M0, Emid, r, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    workspace = FeastWorkspaceComplex{T}(N, M0)
    Q_basis = view(workspace.workc, :, 1:M0)
    _feast_seeded_subspace_complex!(Q_basis)
    shifted_solutions = workspace.q
    lambda_vec = Vector{Complex{T}}(undef, M0)
    res_vec = workspace.res

    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)
    rhs_iterative = solver_is_iterative ? similar(rhs_buffer) : nothing
    Q_proj = zeros(Complex{T}, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQ = Matrix{Complex{T}}(undef, N, M0)
    Ared = Matrix{Complex{T}}(undef, M0, M0)
    Bred = Matrix{Complex{T}}(undef, M0, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(shifted_solutions)
    residual_vec = Vector{Complex{T}}(undef, N)
    Bq_vec = Vector{Complex{T}}(undef, N)

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_gcontour(Emid, r, fpm))
    Zne = contour.Zne
    Wne = contour.Wne

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)
    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false
        for e in eachindex(Zne)
            z = Zne[e]
            weight = Wne[e]
            mul!(rhs_buffer, B, Q_basis)

            if solver_is_direct
                shifted_matrix = z * B - A
                try
                    solver_factor = lu(shifted_matrix)
                    ldiv!(shifted_solutions, solver_factor, rhs_buffer)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Sparse complex-symmetric direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                copyto!(rhs_iterative, rhs_buffer)
                success = solve_shifted_iterative!(shifted_solutions, rhs_iterative,
                                                   A, B, z, tol_value,
                                                   solver_maxiter, solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            @. Q_proj += weight * shifted_solutions
        end
        solve_failed && break

        try
            # Complex-symmetric problems use the bilinear transpose form.
            # Using adjoint here would turn this path back into the general
            # non-Hermitian projection and lose the structure we validated.
            mul!(AQ, A, Q_proj)
            mul!(BQ, B, Q_proj)
            mul!(Ared, transpose(Q_proj), AQ)
            mul!(Bred, transpose(Q_proj), BQ)

            F = eigen(Ared, Bred)
            lambda_red = F.values
            v_red = F.vectors

            for idx in 1:M0
                mul!(view(shifted_solutions, :, idx), Q_proj, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_gcontour!(lambda_vec, shifted_solutions, perm,
                                            lambda_tmp, solutions_tmp,
                                            Emid, r, fpm, M0)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for idx in 1:M0
                vec = view(shifted_solutions, :, idx)
                nrm = norm(vec)
                if nrm > zero(T)
                    vec ./= nrm
                else
                    fill!(vec, zero(Complex{T}))
                    vec[mod1(idx, N)] = one(Complex{T})
                end
            end

            max_res = zero(T)
            for j in 1:M
                q_col = view(shifted_solutions, :, j)
                mul!(residual_vec, A, q_col)
                mul!(Bq_vec, B, q_col)
                @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M

            if epsout_val <= eps_tol
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            copyto!(Q_basis, shifted_solutions)
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced eigenvalue problem failed during sparse complex-symmetric FEAST" exception=err
            break
        end
    end

    if M_found == 0 && info_code == Int(Feast_SUCCESS)
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    M_found > 1 && feast_sort_general!(lambda_vec, shifted_solutions, res_vec, M_found)

    lambda = lambda_vec[1:M_found]
    q = shifted_solutions[:, 1:M_found]
    res = res_vec[1:M_found]

    return FeastGeneralResult{T}(lambda, q, M_found, res,
                                 info_code, epsout_val, loop_count)
end

function feast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))

    complex_A = _convert_sparse_complex(A, Complex{T})
    complex_B = _convert_sparse_complex(B, Complex{T})
    complex_result = _feast_sparse_hermitian(complex_A, complex_B,
                                             Emin, Emax, M0, fpm;
                                             solver=solver, solver_tol=solver_tol,
                                             solver_maxiter=solver_maxiter,
                                             solver_restart=solver_restart)
    return _complex_to_real_result(complex_result)
end

function feast_scsrgvx!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}};
                        solver::Symbol = :direct,
                        solver_tol::Real = 0.0,
                        solver_maxiter::Int = 500,
                        solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_scsrgv!(A, B, Emin, Emax, M0, fpm;
                      solver=solver,
                      solver_tol=solver_tol,
                      solver_maxiter=solver_maxiter,
                      solver_restart=solver_restart)
    end
end

function feast_scsrevx!(A::SparseMatrixCSC{T,Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_scsrev!(A, Emin, Emax, M0, fpm)
    end
end

function feast_hcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    ishermitian(A) || throw(ArgumentError("Matrix A must be Hermitian for feast_hcsrev!"))

    return _feast_sparse_hermitian(A, nothing, Emin, Emax, M0, fpm;
                                   solver=solver, solver_tol=solver_tol,
                                   solver_maxiter=solver_maxiter,
                                   solver_restart=solver_restart)
end

function feast_hcsrevx!(A::SparseMatrixCSC{Complex{T},Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}};
                        solver::Symbol = :direct,
                        solver_tol::Real = 0.0,
                        solver_maxiter::Int = 500,
                        solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hcsrev!(A, Emin, Emax, M0, fpm;
                      solver=solver, solver_tol=solver_tol,
                      solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_hcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         solver_tol::Real = 0.0,
                         solver_maxiter::Int = 500,
                         solver_restart::Int = 30) where T<:Real
    return feast_hcsrev!(A, Emin, Emax, M0, fpm;
                         solver=:gmres, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_hcsrevx!(A::SparseMatrixCSC{Complex{T},Int},
                          Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}};
                          solver_tol::Real = 0.0,
                          solver_maxiter::Int = 500,
                          solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_hcsrev!(A, Emin, Emax, M0, fpm;
                        solver_tol=solver_tol,
                        solver_maxiter=solver_maxiter,
                        solver_restart=solver_restart)
    end
end

function feast_hcsrgv!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must match the size of A"))
    ishermitian(A) || throw(ArgumentError("A must be Hermitian for feast_hcsrgv!"))
    ishermitian(B) || throw(ArgumentError("B must be Hermitian positive definite for feast_hcsrgv!"))

    return _feast_sparse_hermitian(A, B, Emin, Emax, M0, fpm;
                                   solver=solver, solver_tol=solver_tol,
                                   solver_maxiter=solver_maxiter,
                                   solver_restart=solver_restart)
end

function feast_hcsrgvx!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}};
                        solver::Symbol = :direct,
                        solver_tol::Real = 0.0,
                        solver_maxiter::Int = 500,
                        solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hcsrgv!(A, B, Emin, Emax, M0, fpm;
                      solver=solver, solver_tol=solver_tol,
                      solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_hcsrgv!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         solver_tol::Real = 0.0,
                         solver_maxiter::Int = 500,
                         solver_restart::Int = 30) where T<:Real
    return feast_hcsrgv!(A, B, Emin, Emax, M0, fpm;
                         solver=:gmres, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_hcsrgvx!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                          Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}};
                          solver_tol::Real = 0.0,
                          solver_maxiter::Int = 500,
                          solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_hcsrgv!(A, B, Emin, Emax, M0, fpm;
                        solver_tol=solver_tol,
                        solver_maxiter=solver_maxiter,
                        solver_restart=solver_restart)
    end
end

function feast_gcsrgv!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    # Feast for sparse complex general eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are general sparse matrices
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_grci_input(N, M0, Emid, r, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    current_shift = Ref(zero(Complex{T}))
    rhs_iterative = solver_is_iterative ? zeros(Complex{T}, N, M0) : nothing
    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)

    # Initialize workspace
    workspace = FeastWorkspaceComplex{T}(N, M0)
    
    # Initialize variables for RCI
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
    # Results will be complex eigenvalues
    lambda_complex = Vector{Complex{T}}(undef, M0)
    q_complex = Matrix{Complex{T}}(undef, N, M0)
    
    # Sparse linear solver workspace
    sparse_solver = nothing
    factor_cache = Dict{Complex{T}, SparseArrays.UMFPACK.UmfpackLU{Complex{T}, Int}}()

    # Persistent RCI state (must be reused across calls in the loop)
    grci_state = FeastGRCIState{T}()

    @views while true
        # Call Feast RCI kernel for general problems
        feast_grci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emid, r, M0, lambda_complex, q_complex,
                   mode, workspace.res, info; state=grci_state)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A for sparse matrices
            z = Ze[]
            if solver_is_direct
                # LU factorization for sparse matrix
                try
                    sparse_solver = get(factor_cache, z, nothing)
                    if sparse_solver === nothing
                        sparse_matrix = z * B - A
                        sparse_solver = lu(sparse_matrix)
                        factor_cache[z] = sparse_solver
                    end
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = z
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = view(rhs_buffer, :, 1:M0)
            workc_block = view(workspace.workc, :, 1:M0)
            mul!(rhs, B, workc_block)
            
            if solver_is_direct
                try
                    ldiv!(workc_block, sparse_solver, rhs)
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                copyto!(rhs_iterative, rhs)
                success = solve_shifted_iterative!(workc_block, rhs_iterative,
                                                   A, B, current_shift[],
                                                   tol_value, solver_maxiter, solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            # Compute B * q (for forming reduced matrix zBq = Q^H * B * Q)
            M = mode[]
            mul!(view(workspace.workc, :, 1:M), B, view(q_complex, :, 1:M))

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q (for forming zAq or computing residuals)
            M = mode[]
            mul!(view(workspace.workc, :, 1:M), A, view(q_complex, :, 1:M))

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_B($(Int(Feast_RCI_MULT_B))), MULT_A($(Int(Feast_RCI_MULT_A))), " *
                  "DONE($(Int(Feast_RCI_DONE)))")
        end
    end

    # Extract results
    M = mode[]
    lambda = lambda_complex[1:M]
    q = q_complex[:, 1:M]
    res = workspace.res[1:M]

    return FeastGeneralResult{T}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_gcsrgvx!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}};
                        solver::Symbol = :direct,
                        solver_tol::Real = 0.0,
                        solver_maxiter::Int = 500,
                        solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gcsrgv!(A, B, Emid, r, M0, fpm;
                      solver=solver, solver_tol=solver_tol,
                      solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function feast_gcsrevx!(A::SparseMatrixCSC{Complex{T},Int},
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                        Zne::AbstractVector{Complex{TZ}},
                        Wne::AbstractVector{Complex{TW}};
                        solver::Symbol = :direct,
                        solver_tol::Real = 0.0,
                        solver_maxiter::Int = 500,
                        solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gcsrev!(A, Emid, r, M0, fpm;
                      solver=solver, solver_tol=solver_tol,
                      solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function feast_scsrgv_complex!(A::SparseMatrixCSC{Complex{T},Int},
                               B::SparseMatrixCSC{Complex{T},Int},
                               Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                               solver::Symbol = :direct,
                               solver_tol::Real = 0.0,
                               solver_maxiter::Int = 500,
                               solver_restart::Int = 30) where T<:Real
    _check_complex_symmetric(A)
    _check_complex_symmetric(B)
    return _feast_sparse_complex_symmetric(A, B, Emid, r, M0, fpm;
                                           solver=solver, solver_tol=solver_tol,
                                           solver_maxiter=solver_maxiter,
                                           solver_restart=solver_restart)
end

function feast_scsrgvx_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                B::SparseMatrixCSC{Complex{T},Int},
                                Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                                Zne::AbstractVector{Complex{TZ}},
                                Wne::AbstractVector{Complex{TW}};
                                solver::Symbol = :direct,
                                solver_tol::Real = 0.0,
                                solver_maxiter::Int = 500,
                                solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_scsrgv_complex!(A, B, Emid, r, M0, fpm;
                              solver=solver, solver_tol=solver_tol,
                              solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function feast_scsrev_complex!(A::SparseMatrixCSC{Complex{T},Int},
                               Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                               solver::Symbol = :direct,
                               solver_tol::Real = 0.0,
                               solver_maxiter::Int = 500,
                               solver_restart::Int = 30) where T<:Real
    _check_complex_symmetric(A)
    B = spdiagm(0 => fill(Complex{T}(1), size(A, 1)))
    return feast_scsrgv_complex!(A, B, Emid, r, M0, fpm;
                                 solver=solver, solver_tol=solver_tol,
                                 solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_scsrevx_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                                Zne::AbstractVector{Complex{TZ}},
                                Wne::AbstractVector{Complex{TW}};
                                solver::Symbol = :direct,
                                solver_tol::Real = 0.0,
                                solver_maxiter::Int = 500,
                                solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_scsrev_complex!(A, Emid, r, M0, fpm;
                              solver=solver, solver_tol=solver_tol,
                              solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_scsrgv_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                 B::SparseMatrixCSC{Complex{T},Int},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    return feast_scsrgv_complex!(A, B, Emid, r, M0, fpm;
                                 solver=:gmres, solver_tol=solver_tol,
                                 solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_scsrgvx_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                  B::SparseMatrixCSC{Complex{T},Int},
                                  Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                                  Zne::AbstractVector{Complex{TZ}},
                                  Wne::AbstractVector{Complex{TW}};
                                  solver_tol::Real = 0.0,
                                  solver_maxiter::Int = 500,
                                  solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_scsrgv_complex!(A, B, Emid, r, M0, fpm;
                                solver_tol=solver_tol,
                                solver_maxiter=solver_maxiter,
                                solver_restart=solver_restart)
    end
end

function zifeast_scsrev_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    return feast_scsrev_complex!(A, Emid, r, M0, fpm;
                                 solver=:gmres, solver_tol=solver_tol,
                                 solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_scsrevx_complex!(A::SparseMatrixCSC{Complex{T},Int},
                                  Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                                  Zne::AbstractVector{Complex{TZ}},
                                  Wne::AbstractVector{Complex{TW}};
                                  solver_tol::Real = 0.0,
                                  solver_maxiter::Int = 500,
                                  solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_scsrev_complex!(A, Emid, r, M0, fpm;
                                solver_tol=solver_tol,
                                solver_maxiter=solver_maxiter,
                                solver_restart=solver_restart)
    end
end

function zifeast_gcsrevx!(A::SparseMatrixCSC{Complex{T},Int},
                          Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}};
                          solver_tol::Real = 0.0,
                          solver_maxiter::Int = 500,
                          solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_gcsrev!(A, Emid, r, M0, fpm;
                        solver_tol=solver_tol,
                        solver_maxiter=solver_maxiter,
                        solver_restart=solver_restart)
    end
end

function zifeast_gcsrgv!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                         solver_tol::Real = 0.0,
                         solver_maxiter::Int = 500,
                         solver_restart::Int = 30) where T<:Real
    return feast_gcsrgv!(A, B, Emid, r, M0, fpm;
                         solver=:gmres, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_gcsrgvx!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                          Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}};
                          solver_tol::Real = 0.0,
                          solver_maxiter::Int = 500,
                          solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        zifeast_gcsrgv!(A, B, Emid, r, M0, fpm;
                        solver_tol=solver_tol,
                        solver_maxiter=solver_maxiter,
                        solver_restart=solver_restart)
    end
end

# Iterative refinement for sparse problems
function feast_scsrgv_iterative!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                 Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                                 max_iter::Int = 3) where T<:Real
    # Feast with iterative refinement for better accuracy
    
    return feast_scsrgv!(A, B, Emin, Emax, M0, fpm;
                         solver=:gmres,
                         solver_tol=10.0^(-fpm[3]),
                         solver_maxiter=400,
                         solver_restart=30)
end

function difeast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         solver_tol::Real = 0.0,
                         solver_maxiter::Int = 500,
                         solver_restart::Int = 30) where T<:Real
    return feast_scsrgv!(A, B, Emin, Emax, M0, fpm;
                         solver=:gmres,
                         solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter,
                         solver_restart=solver_restart)
end

function difeast_scsrgvx!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                          Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}};
                          solver_tol::Real = 0.0,
                          solver_maxiter::Int = 500,
                          solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return feast_scsrgvx!(A, B, Emin, Emax, M0, fpm, Zne, Wne;
                          solver=:gmres,
                          solver_tol=solver_tol,
                          solver_maxiter=solver_maxiter,
                          solver_restart=solver_restart)
end

function feast_scsrpev!(A::Vector{SparseMatrixCSC{T,Int}}, d::Int,
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    length(A) == d + 1 || throw(ArgumentError("Need d+1 coefficient matrices"))
    dense_coeffs = [Matrix{T}(Ai) for Ai in A]
    return feast_sypev!(dense_coeffs, d, Emid, r, M0, fpm)
end

function feast_scsrpevx!(A::Vector{SparseMatrixCSC{T,Int}}, d::Int,
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_scsrpev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_hcsrpev!(A::Vector{SparseMatrixCSC{Complex{T},Int}}, d::Int,
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    length(A) == d + 1 || throw(ArgumentError("Need d+1 coefficient matrices"))
    dense_coeffs = [Matrix{Complex{T}}(Ai) for Ai in A]
    return feast_hepev!(dense_coeffs, d, Emid, r, M0, fpm)
end

function feast_hcsrpevx!(A::Vector{SparseMatrixCSC{Complex{T},Int}}, d::Int,
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hcsrpev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_gcsrpev!(A::Vector{SparseMatrixCSC{Complex{T},Int}}, d::Int,
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    length(A) == d + 1 || throw(ArgumentError("Need d+1 coefficient matrices"))
    dense_coeffs = [Matrix{Complex{T}}(Ai) for Ai in A]
    return feast_gepev!(dense_coeffs, d, Emid, r, M0, fpm)
end

function feast_gcsrpevx!(A::Vector{SparseMatrixCSC{Complex{T},Int}}, d::Int,
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gcsrpev!(A, d, Emid, r, M0, fpm)
    end
end

"""
    feast_sparse_matvec!(A_matvec!, B_matvec!, N, Emin, Emax, M0, fpm; kwargs...)

Matrix-free sparse-style FEAST using GMRES for shifted solves. The user supplies
real `A_matvec!` and `B_matvec!` callbacks; internally the shifted operator
splits complex vectors into real and imaginary work buffers so the callbacks do
not need to handle complex arithmetic themselves.
"""
function feast_sparse_matvec!(A_matvec!::Function, B_matvec!::Function,
                             N::Int, Emin::T, Emax::T, M0::Int,
                             fpm::Vector{Int};
                             gmres_rtol::T = T(1e-6),
                             gmres_atol::T = T(1e-12),
                             gmres_restart::Int = 20,
                             gmres_maxiter::Int = 200) where T<:Real
    FEAST_KRYLOV_AVAILABLE[] ||
        throw(ArgumentError("Krylov.jl is required for matrix-free GMRES solves"))

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    workspace = FeastWorkspaceReal{T}(N, M0)
    Q = workspace.work
    _feast_seeded_subspace!(Q)

    Aq_block = view(workspace.Aq, 1:M0, 1:M0)
    Sq_block = view(workspace.Sq, 1:M0, 1:M0)
    workc_block = view(workspace.workc, :, 1:M0)
    q_vectors = view(workspace.q, :, 1:M0)
    Q_block = view(Q, :, 1:M0)
    lambda_vec = workspace.lambda
    res_vec = workspace.res

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    ne = length(Zne)

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)

    shifted_operator = MatrixFreeShiftedOperator(N, zero(Complex{T}), A_matvec!, B_matvec!, T)

    # Scratch buffers for applying B to each real basis vector and promoting
    # the result to the complex RHS expected by the shifted Krylov solve.
    rhs_real = zeros(T, N)
    rhs_complex = zeros(Complex{T}, N)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    Q_proj_real = zeros(T, N, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    q_tmp = similar(q_vectors)
    residual_vec = zeros(T, N)

    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0

    # Allocate buffer for accumulated filtered subspace
    Q_proj = zeros(Complex{T}, N, M0)

    @views for loop_idx in 0:maxloop
        # The real reduced matrices are accumulated from complex contour
        # moments. Q_proj is kept complex until projection back to real vectors.
        loop_count = loop_idx
        fill!(Aq_block, zero(T))
        fill!(Sq_block, zero(T))
        fill!(Q_proj, zero(Complex{T}))

        gmres_failed = false
        for e in 1:ne
            shifted_operator.z = Zne[e]
            weight = 2 * Wne[e]

            for j in 1:M0
                # Build one complex RHS at a time: B * q_j is real, then copied
                # into the complex Krylov vector without allocating.
                B_matvec!(rhs_real, view(Q_block, :, j))
                _copyto_complex!(rhs_complex, rhs_real)

                # Note: Krylov.gmres doesn't support initial guess, starts from zero
                sol, stats = gmres(shifted_operator, rhs_complex;
                                   rtol = gmres_rtol,
                                   atol = gmres_atol,
                                   memory = gmres_restart,
                                   itmax = gmres_maxiter)

                if !stats.solved
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    @warn "GMRES did not converge for contour point $e, column $j" residual=stats.residuals[end]
                    gmres_failed = true
                    break
                end

                workc_block[:, j] .= sol
            end

            gmres_failed && break

            # Accumulate filtered subspace
            @. Q_proj += weight * workc_block

            mul!(moment, transpose(Q_block), workc_block)
            Aq_block .+= real.(weight .* moment)
            Sq_block .+= real.((weight * Zne[e]) .* moment)
        end

        if gmres_failed
            break
        end

        try
            # For half-contour integration, the moment matrices are already real.
            # Use Symmetric wrapper directly - no need for 0.5*(A+A') symmetrization.
            # IMPORTANT: Solve Sq*v = lambda*Aq*v (consistent with _feast_sparse_hermitian)
            # Aq = sum(w * Q' * Y), Sq = sum(w * z * Q' * Y)
            lambda_red = T[]
            v_red = Matrix{T}(undef, 0, 0)
            try
                F = eigen(Symmetric(Sq_block), Symmetric(Aq_block))
                lambda_red = F.values
                v_red = F.vectors
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general eigenvalue solver if not positive definite
                    F = eigen(Sq_block, Aq_block)
                    lambda_red = real.(F.values)
                    v_red = real.(F.vectors)
                else
                    rethrow(e)
                end
            end

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q
            _feast_copy_real!(Q_proj_real, Q_proj)
            for idx in 1:M0
                mul!(view(q_vectors, :, idx), Q_proj_real, view(v_red, :, idx))
                lambda_vec[idx] = convert(T, lambda_red[idx])
            end

            M = _feast_reorder_by_interval!(lambda_vec, q_vectors, perm,
                                             lambda_tmp, q_tmp, Emin, Emax, M0)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Normalize only eigenvectors inside interval (for residual computation)
            for j in 1:M
                vec = view(q_vectors, :, j)
                nrm = norm(vec)
                nrm > 0 && (vec ./= nrm)
            end

            # Compute residuals only for eigenvalues inside interval
            max_res = zero(T)
            for j in 1:M
                q_col = view(q_vectors, :, j)
                A_matvec!(residual_vec, q_col)
                @. residual_vec = residual_vec - lambda_vec[j] * q_col
                # Relative residual: normalize by max(|λ|, 1)
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M

            if epsout_val <= eps_tol
                info_code = Int(Feast_SUCCESS)
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Use full M0 subspace for next iteration
            Q[:, 1:M0] .= q_vectors[:, 1:M0]
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced eigenvalue problem failed in matrix-free FEAST" exception=err
            break
        end
    end

    lambda = workspace.lambda[1:M_found]
    q = workspace.q[:, 1:M_found]
    res = workspace.res[1:M_found]

    return FeastResult{T, T}(lambda, q, M_found, res, info_code, epsout_val, loop_count)
end

# Convenience wrapper for sparse matrices using GMRES
function feast_sparse_matvec!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                             Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                             gmres_rtol::T = T(1e-6),
                             gmres_atol::T = T(1e-12), 
                             gmres_restart::Int = 20,
                             gmres_maxiter::Int = 200) where T<:Real
    # Wrapper that creates matvec functions from sparse matrices
    N = size(A, 1)
    
    # Define matrix-vector multiplication functions
    function A_matvec!(y::AbstractVector{T}, x::AbstractVector{T})
        mul!(y, A, x)
    end
    
    function B_matvec!(y::AbstractVector{T}, x::AbstractVector{T})
        mul!(y, B, x)
    end
    
    # Call the matrix-free version
    return feast_sparse_matvec!(A_matvec!, B_matvec!, N, Emin, Emax, M0, fpm;
                               gmres_rtol=gmres_rtol, gmres_atol=gmres_atol,
                               gmres_restart=gmres_restart, gmres_maxiter=gmres_maxiter)
end

# Utility functions for sparse matrix operations
function feast_sparse_info(A::SparseMatrixCSC)
    # Print information about sparse matrix
    N = size(A, 1)
    nnz_A = nnz(A)
    density = nnz_A / (N^2) * 100
    
    println("Sparse Matrix Information:")
    println("  Size: $(N) x $(N)")
    println("  Non-zeros: $(nnz_A)")
    # Use Printf for formatted percentage
    println("  Density: ", Printf.@sprintf("%.2f", density), "%")

    return (N, nnz_A, density)
end

# Standard eigenvalue problem variants (B = I)

function feast_scsrev!(A::SparseMatrixCSC{T,Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for sparse real symmetric standard eigenvalue problem
    # Solves: A*q = lambda*q where A is symmetric
    # This is equivalent to feast_scsrgv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    complex_A = _convert_sparse_complex(A, Complex{T})
    complex_result = _feast_sparse_hermitian(complex_A, nothing,
                                             Emin, Emax, M0, fpm)
    return _complex_to_real_result(complex_result)
end

function feast_gcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    # Feast for sparse complex general standard eigenvalue problem
    # Solves: A*q = lambda*q where A is a general matrix
    # This is equivalent to feast_gcsrgv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    # Create sparse identity matrix for B
    B = sparse(Complex{T}(1.0) * I, N, N)

    # Call generalized version with B = I
    return feast_gcsrgv!(A, B, Emid, r, M0, fpm;
                         solver=solver, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_gcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                         solver_tol::Real = 0.0,
                         solver_maxiter::Int = 500,
                         solver_restart::Int = 30) where T<:Real
    return feast_gcsrev!(A, Emid, r, M0, fpm;
                         solver=:gmres, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end
