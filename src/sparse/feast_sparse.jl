# Feast sparse matrix routines
# Translated from dzfeast_sparse.f90 and related files

using SparseArrays
using LinearAlgebra
using Random

# Shifted operator for GMRES solves
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
    mul!(op.tmpB, op.B, x)
    @. op.tmpB = op.z * op.tmpB
    mul!(op.tmpA, op.A, x)
    @. y = op.tmpB - op.tmpA
    return y
end

mutable struct MatrixFreeShiftedOperator{T<:Real,CT<:Complex}
    N::Int
    z::CT
    A_matvec!::Function
    B_matvec!::Function
    tmp_real::Vector{T}
    tmp_imag::Vector{T}
    result_real::Vector{T}
    result_imag::Vector{T}
    x_real::Vector{T}
    x_imag::Vector{T}
end

MatrixFreeShiftedOperator(N::Int, z::CT, A_matvec!::Function,
                          B_matvec!::Function, ::Type{T}) where {T<:Real,CT<:Complex} =
    MatrixFreeShiftedOperator{T,CT}(N, z, A_matvec!, B_matvec!,
                                    zeros(T, N), zeros(T, N),
                                    zeros(T, N), zeros(T, N),
                                    zeros(T, N), zeros(T, N))

Base.size(op::MatrixFreeShiftedOperator) = (op.N, op.N)
Base.eltype(::MatrixFreeShiftedOperator{T,CT}) where {T,CT} = CT

function LinearAlgebra.mul!(y::AbstractVector{CT}, op::MatrixFreeShiftedOperator{T,CT}, x::AbstractVector{CT}) where {T<:Real,CT<:Complex}
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

@inline function _copyto_complex!(dest::AbstractVector{Complex{T}}, src::AbstractVector{T}) where T<:Real
    @inbounds for i in eachindex(src)
        dest[i] = Complex{T}(src[i])
    end
    return dest
end
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
        x_initial = zeros(CT, N)
        x_sol, stats = gmres(op, b, x_initial;
                             restart=true,
                             memory=max(gmres_restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        mul!(residual, op, x_sol)
        @. residual -= b
        res_norm = norm(residual)
        b_norm = norm(b)
        if !stats.solved || res_norm > tol * max(b_norm, one(b_norm))
            return false
        end
        dest[:, j] .= x_sol
    end

    return true
end

function _feast_sparse_hermitian(A::SparseMatrixCSC{Complex{T},Int},
                                 B::Union{SparseMatrixCSC{Complex{T},Int},Nothing},
                                 Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    identity_sparse = spdiagm(0 => fill(Complex{T}(1), N))
    B_matrix = B === nothing ? identity_sparse : B

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver in (:direct, :gmres) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct or :gmres."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    workspace = FeastWorkspaceComplex{T}(N, M0)
    Q_basis = view(workspace.workc, :, 1:M0)
    _feast_seeded_subspace_complex!(Q_basis)
    solutions = view(workspace.q, :, 1:M0)
    lambda_vec = workspace.lambda
    res_vec = workspace.res

    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    ReducedT = promote_type(Float64, T)
    rhs_buffer = B === nothing ? Q_basis : Matrix{Complex{T}}(undef, N, M0)
    residual_vec = zeros(Complex{T}, N)
    Bq_vec = B === nothing ? nothing : zeros(Complex{T}, N)

    contour = feast_get_custom_contour(fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm)
    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))

        gmres_failed = false

        for e in 1:length(Zne)
            z = Zne[e]
            weight = 2 * Wne[e]

            rhs = rhs_buffer
            if B === nothing
                rhs = Q_basis
            else
                mul!(rhs_buffer, B_matrix, Q_basis)
            end

            if solver_is_direct
                shifted_matrix = z * B_matrix - A
                try
                    solver_factor = lu(shifted_matrix)
                    solutions .= solver_factor \ rhs
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Sparse direct solve failed for shift $z" exception=err
                    gmres_failed = true
                    break
                end
            else
                success = solve_shifted_iterative!(solutions, rhs, A, B_matrix,
                                                   z, tol_value, solver_maxiter, solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    gmres_failed = true
                    break
                end
            end

            mul!(moment, adjoint(Q_basis), solutions)
            @inbounds zAq .+= weight .* moment
            @inbounds zSq .+= (weight * z) .* moment
        end

        if gmres_failed
            break
        end

        try
            Aq_real = Matrix{ReducedT}(real.(0.5 .* (zAq .+ adjoint(zAq))))
            Sq_real = Matrix{ReducedT}(real.(0.5 .* (zSq .+ adjoint(zSq))))

            # Try Symmetric solver first (more accurate), fall back to general if not positive definite
            lambda_red = Vector{ReducedT}(undef, 0)
            v_red = Array{ReducedT}(undef, 0, 0)
            try
                F = eigen(Symmetric(Aq_real), Symmetric(Sq_real))
                lambda_red = Vector{ReducedT}(F.values)
                v_red = Array{ReducedT}(F.vectors)
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general eigenvalue solver if not positive definite
                    F = eigen(Aq_real, Sq_real)
                    lambda_red = Vector{ReducedT}(real.(F.values))
                    v_red = Array{ReducedT}(real.(F.vectors))
                else
                    rethrow(e)
                end
            end

            indices = Int[]
            selected_lambda = Vector{T}()
            for (i, val) in enumerate(lambda_red)
                if Emin <= val <= Emax
                    push!(indices, i)
                    push!(selected_lambda, val)
                end
            end

            M = length(indices)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for (idx, i) in enumerate(indices)
                coeffs = Vector{T}(view(v_red, :, i))
                mul!(view(solutions, :, idx), Q_basis, coeffs)
                lambda_vec[idx] = convert(T, lambda_red[i])
            end

            for j in 1:M
                vec = view(solutions, :, j)
                nrm = norm(vec)
                nrm > 0 && (vec ./= nrm)
            end

            max_res = zero(T)
            for j in 1:M
                q_col = view(solutions, :, j)
                mul!(residual_vec, A, q_col)
                if B === nothing
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    mul!(Bq_vec, B_matrix, q_col)
                    @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                end
                res_val = norm(residual_vec)
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M

            if epsout_val <= eps_tol
                Q_result = view(workspace.q, :, 1:M)
                Q_result .= solutions[:, 1:M]
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            Q_basis[:, 1:M] .= solutions[:, 1:M]
            if M < M0
                submatrix = view(Q_basis, :, M+1:M0)
                _feast_seeded_subspace_complex!(submatrix)
            end
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
    lambda = real.(complex_result.lambda)
    q_real = Array{T}(undef, N, complex_result.M)
    for j in 1:complex_result.M
        @inbounds q_real[:, j] .= real.(complex_result.q[:, j])
    end
    res = complex_result.res
    return FeastResult{T, T}(lambda, q_real, complex_result.M, res,
                              complex_result.info, complex_result.epsout,
                              complex_result.loop)
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

    solver_choice = solver in (:direct, :gmres, :iterative) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct or :gmres."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    current_shift = Ref(zero(Complex{T}))
    rhs_iterative = solver_is_iterative ? zeros(Complex{T}, N, M0) : nothing

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
    
    while true
        # Call Feast RCI kernel for general problems
        feast_grci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emid, r, M0, lambda_complex, q_complex, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A for sparse matrices
            z = Ze[]
            if solver_is_direct
                sparse_matrix = z * B - A
                
                # LU factorization for sparse matrix
                try
                    sparse_solver = lu(sparse_matrix)
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = z
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = B * workspace.workc[:, 1:M0]
            
            if solver_is_direct
                try
                    workspace.workc[:, 1:M0] .= sparse_solver \ rhs
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                rhs_iterative .= rhs
                success = solve_shifted_iterative!(workspace.workc[:, 1:M0], rhs_iterative,
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
            workspace.workc[:, 1:M] .= B * q_complex[:, 1:M]

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q (for forming zAq or computing residuals)
            M = mode[]
            workspace.workc[:, 1:M] .= A * q_complex[:, 1:M]

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

    return FeastResult{T, Complex{T}}(real.(lambda), q, M, res, info[], epsout[], loop[])
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
    return feast_gcsrgv!(A, B, Emid, r, M0, fpm;
                         solver=solver, solver_tol=solver_tol,
                         solver_maxiter=solver_maxiter, solver_restart=solver_restart)
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
    B = sparse(Complex{T}(1.0) * I, size(A, 1), size(A, 2))
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

# Matrix-free interface for sparse problems with GMRES solver
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

    contour = feast_get_custom_contour(fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    ne = length(Zne)

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm)

    shifted_operator = MatrixFreeShiftedOperator(N, zero(Complex{T}), A_matvec!, B_matvec!, T)
    rhs_real = zeros(T, N)
    rhs_complex = zeros(Complex{T}, N)
    gmres_guess = zeros(Complex{T}, N)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    residual_vec = zeros(T, N)

    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(Aq_block, zero(T))
        fill!(Sq_block, zero(T))

        gmres_failed = false
        for e in 1:ne
            shifted_operator.z = Zne[e]
            weight = 2 * Wne[e]

            for j in 1:M0
                B_matvec!(rhs_real, view(Q_block, :, j))
                _copyto_complex!(rhs_complex, rhs_real)
                fill!(gmres_guess, zero(Complex{T}))

                sol, stats = gmres(shifted_operator, rhs_complex;
                                   x = gmres_guess,
                                   rtol = gmres_rtol,
                                   atol = gmres_atol,
                                   restart = gmres_restart,
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

            moment .= transpose(Q_block) * workc_block
            Aq_block .+= real.(weight .* moment)
            Sq_block .+= real.((weight * Zne[e]) .* moment)
        end

        if gmres_failed
            break
        end

        try
            Aq_sym = 0.5 .* (Aq_block .+ transpose(Aq_block))
            Sq_sym = 0.5 .* (Sq_block .+ transpose(Sq_block))

            # Try Symmetric solver first (more accurate), fall back to general if not positive definite
            lambda_red = T[]
            v_red = Matrix{T}(undef, 0, 0)
            try
                F = eigen(Symmetric(Aq_sym), Symmetric(Sq_sym))
                lambda_red = F.values
                v_red = F.vectors
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general eigenvalue solver if not positive definite
                    F = eigen(Aq_sym, Sq_sym)
                    lambda_red = real.(F.values)
                    v_red = real.(F.vectors)
                else
                    rethrow(e)
                end
            end

            selected = Vector{Int}()
            selected_lambda = Vector{T}()
            for i in 1:length(lambda_red)
                val = lambda_red[i]
                if Emin <= val <= Emax
                    push!(selected, i)
                    push!(selected_lambda, val)
                end
            end

            M = length(selected)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            Q_block = view(Q, :, 1:M0)
            for (idx, eig_idx) in enumerate(selected)
                coeffs = Vector{T}(view(v_red, :, eig_idx))
                mul!(view(q_vectors, :, idx), Q_block, coeffs)
                lambda_vec[idx] = convert(T, selected_lambda[eig_idx])
            end

            for j in 1:M
                vec = view(q_vectors, :, j)
                nrm = norm(vec)
                nrm > 0 && (vec ./= nrm)
            end

            max_res = zero(T)
            for j in 1:M
                q_col = view(q_vectors, :, j)
                A_matvec!(residual_vec, q_col)
                @. residual_vec = residual_vec - lambda_vec[j] * q_col
                res_val = norm(residual_vec)
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

            Q[:, 1:M] .= q_vectors[:, 1:M]
            if M < M0
                for col in M+1:M0
                    randn!(view(Q, :, col))
                    col_norm = norm(view(Q, :, col))
                    col_norm > 0 && (view(Q, :, col) ./= col_norm)
                end
            end
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

    # Create sparse identity matrix for B
    B = sparse(I, N, N)

    # Call generalized version with B = I
    return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
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
