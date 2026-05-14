# Feast dense matrix routines
# Translated from dzfeast_dense.f90 and dzfeast_pev_dense.f90

struct DenseShiftOperator{F,T<:Real}
    mulfun::F
    N::Int
end

Base.size(op::DenseShiftOperator) = (op.N, op.N)
Base.eltype(::DenseShiftOperator{F,T}) where {F,T} = Complex{T}

function LinearAlgebra.mul!(y::AbstractVector{Complex{T}},
                            op::DenseShiftOperator{F,T},
                            x::AbstractVector{Complex{T}}) where {F,T<:Real}
    op.mulfun(y, x)
    return y
end

"""
    solve_dense_shifted!(dest, rhs, apply_shift!, solver, tol, maxiter, restart)

Solve `(zB - A) * X = rhs` one right-hand side at a time with an iterative
Krylov method. `apply_shift!` is a closure over the current shift `z`; keeping
it concrete lets Julia specialize the matrix-vector product used by GMRES.
"""
function solve_dense_shifted!(dest::AbstractMatrix{Complex{T}},
                              rhs::AbstractMatrix{Complex{T}},
                              apply_shift!::F,
                              solver::Symbol, tol::T,
                              maxiter::Int, restart::Int) where {T<:Real,F}
    solver == :direct && error("Direct solve should be handled before calling solve_dense_shifted!")

    if !FEAST_KRYLOV_AVAILABLE[]
        error("Krylov.jl required for iterative dense FEAST solves")
    end

    N = size(rhs, 1)
    op = DenseShiftOperator{typeof(apply_shift!), T}(apply_shift!, N)

    residual = zeros(Complex{T}, N)
    x0 = zeros(Complex{T}, N)
    @views for j in 1:size(rhs, 2)
        b = view(rhs, :, j)
        fill!(x0, zero(Complex{T}))
        x_sol, stats = gmres(op, b, x0;
                             restart=true,
                             memory=max(restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        apply_shift!(residual, x_sol)
        @. residual -= b
        res_norm = norm(residual)
        b_norm = norm(b)
        # Krylov reports convergence using its own recurrence. The explicit
        # residual recomputation can differ by a few ulps, so validate it with a
        # small slack instead of making this check stricter than the solver.
        residual_limit = T(10) * tol * max(b_norm, one(T))
        if !stats.solved || res_norm > residual_limit
            @warn "GMRES failed to converge" iteration_stats=stats residual=res_norm rhs_norm=b_norm
            return false
        end
        dest[:, j] .= x_sol
    end

    return true
end

"""
    _feast_dense_complex_hermitian(A, B, Emin, Emax, M0, fpm; solver=:direct)

Shared implementation for dense complex Hermitian FEAST. The loop applies the
contour spectral projector to a trial subspace, solves a small reduced
Hermitian eigenproblem, then keeps only eigenpairs inside the requested
interval. Scratch arrays are allocated once here and reused across contour
points and refinement loops.
"""
function _feast_dense_complex_hermitian(A::Matrix{Complex{T}},
                                        B::Union{Matrix{Complex{T}},Nothing},
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
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_is_identity = B === nothing
    B_matrix = B_is_identity ? Matrix{Complex{T}}(undef, 0, 0) : copy(B)

    # Main FEAST workspaces. The rhs/reorder buffers look redundant, but they
    # prevent per-contour copies and slice allocations in the inner iteration.
    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    solutions = similar(Q_basis)
    rhs_buffer = zeros(Complex{T}, N, M0)
    rhs_copy = similar(rhs_buffer)
    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)
    Aq_herm = similar(zAq)
    Sq_herm = similar(zSq)
    lambda_vec = zeros(T, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(solutions)
    res_vec = zeros(T, M0)
    residual_vec = zeros(Complex{T}, N)
    Bq_vec = B_is_identity ? nothing : zeros(Complex{T}, N)
    shifted_matrix = similar(A)
    current_shift = Ref(zero(Complex{T}))
    tmpAx = solver_is_iterative ? zeros(Complex{T}, N) : nothing
    tmpBx = solver_is_iterative ? zeros(Complex{T}, N) : nothing

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_is_identity
            @. tmpBx = current_shift[] * x
        else
            mul!(tmpBx, B_matrix, x)
            @. tmpBx = current_shift[] * tmpBx
        end
        mul!(tmpAx, A, x)
        @. y = tmpBx - tmpAx
        return y
    end

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    factor_cache = Vector{Union{Nothing, LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}}}(undef, length(Zne))
    fill!(factor_cache, nothing)

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    loop_count = 0
    M_found = 0
    active_dim = M0

    # Allocate buffer for accumulated filtered subspace
    Q_proj = zeros(Complex{T}, N, M0)

    @views for loop_idx in 0:maxloop
        # Each refinement starts from a fresh projected subspace accumulator; the
        # basis itself is updated only after convergence checks.
        loop_count = loop_idx
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false

        for (idx, z) in enumerate(Zne)
            # Apply each contour resolvent `(zB - A)^-1 B` to the current basis.
            # The half-contour weights are doubled for Hermitian symmetry.
            weight = 2 * Wne[idx]
            basis_block = view(Q_basis, :, 1:active_dim)
            rhs_block = view(rhs_buffer, :, 1:active_dim)
            rhs_copy_block = view(rhs_copy, :, 1:active_dim)
            solutions_block = view(solutions, :, 1:active_dim)
            qproj_block = view(Q_proj, :, 1:active_dim)

            if B_is_identity
                copyto!(rhs_block, basis_block)
            else
                mul!(rhs_block, B_matrix, basis_block)
            end

            if solver_is_direct
                factor = factor_cache[idx]
                if factor === nothing
                    if B_is_identity
                        _feast_dense_shifted_identity_minus!(shifted_matrix, z, A)
                    else
                        @. shifted_matrix = z * B_matrix - A
                    end
                    try
                        factor = lu(shifted_matrix)
                        factor_cache[idx] = factor
                    catch err
                        info_code = Int(Feast_ERROR_LAPACK)
                        @warn "Dense direct solve failed for shift $z" exception=err
                        solve_failed = true
                        break
                    end
                end
                copyto!(rhs_copy_block, rhs_block)
                try
                    ldiv!(solutions_block, factor, rhs_copy_block)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Dense direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                copyto!(rhs_copy_block, rhs_block)
                current_shift[] = z
                success = solve_dense_shifted!(solutions_block, rhs_copy_block,
                                               shifted_mul!, solver_choice,
                                               tol_value, solver_maxiter,
                                               solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            # Accumulate the filtered subspace. It is orthonormalized before
            # Rayleigh-Ritz so oversized trial spaces do not make the reduced
            # problem rank deficient.
            @. qproj_block += weight * solutions_block
        end

        solve_failed && break

        try
            rank = _feast_qr_compress!(solutions_tmp, Q_proj, active_dim;
                                       rank_tol=sqrt(eps(T)))
            if rank == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            q_rank = view(solutions_tmp, :, 1:rank)
            aq_work = view(rhs_buffer, :, 1:rank)
            bq_work = view(rhs_copy, :, 1:rank)
            zAq_rank = view(zAq, 1:rank, 1:rank)
            zSq_rank = view(zSq, 1:rank, 1:rank)
            Aq_rank = view(Aq_herm, 1:rank, 1:rank)
            Sq_rank = view(Sq_herm, 1:rank, 1:rank)

            mul!(aq_work, A, q_rank)
            mul!(zSq_rank, adjoint(q_rank), aq_work)
            _feast_hermitian_part!(Sq_rank, zSq_rank)

            if B_is_identity
                fill!(Aq_rank, zero(Complex{T}))
                for i in 1:rank
                    Aq_rank[i, i] = one(Complex{T})
                end
            else
                mul!(bq_work, B_matrix, q_rank)
                mul!(zAq_rank, adjoint(q_rank), bq_work)
                _feast_hermitian_part!(Aq_rank, zAq_rank)
            end

            # Solve Hermitian generalized eigenproblem: Sq*x = lambda*Aq*x
            # Eigenvalues are real; eigenvectors are complex.
            lambda_red = Vector{T}(undef, 0)
            v_red = Array{Complex{T}}(undef, 0, 0)
            try
                F = eigen(Hermitian(Sq_rank), Hermitian(Aq_rank))
                lambda_red = Vector{T}(F.values)
                v_red = Matrix{Complex{T}}(F.vectors)
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general complex eigenvalue solver
                    F = eigen(Sq_rank, Aq_rank)
                    lambda_red = Vector{T}(real.(F.values))
                    v_red = Matrix{Complex{T}}(F.vectors)
                else
                    rethrow(e)
                end
            end

            # Project eigenvectors using the orthonormal filtered subspace.
            for idx in 1:rank
                mul!(view(solutions, :, idx), q_rank, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda_vec, solutions, perm,
                                             lambda_tmp, solutions_tmp,
                                             Emin, Emax, rank)
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
                if B_is_identity
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    mul!(Bq_vec, B_matrix, q_col)
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

            active_dim = rank
            Q_basis[:, 1:active_dim] .= solutions[:, 1:active_dim]
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced dense Hermitian eigenproblem failed" exception=err
            break
        end
    end

    lambda = lambda_vec[1:M_found]
    q = solutions[:, 1:M_found]
    res = res_vec[1:M_found]

    return FeastResult{T, Complex{T}}(lambda, q, M_found, res,
                                      info_code, epsout_val, loop_count)
end




function feast_sygv!(A::Matrix{T}, B::Matrix{T},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    complex_A = Complex{T}.(A)
    complex_B = Complex{T}.(B)
    complex_result = _feast_dense_complex_hermitian(complex_A, complex_B,
                                                   Emin, Emax, M0, fpm;
                                                   solver=solver, solver_tol=solver_tol,
                                                   solver_maxiter=solver_maxiter,
                                                   solver_restart=solver_restart)
    return _complex_to_real_result(complex_result)
end

@inline function _complex_to_real_result(result::FeastResult{T, Complex{T}}) where T<:Real
    M = result.M
    N = size(result.q, 1)
    q_real = Array{T}(undef, N, M)
    @inbounds for j in 1:M, i in 1:N
        q_real[i, j] = real(result.q[i, j])
    end
    lambda_real = Vector{T}(undef, M)
    res_real = Vector{T}(undef, M)
    @inbounds for i in 1:M
        lambda_real[i] = result.lambda[i]
        res_real[i] = result.res[i]
    end
    return FeastResult{T, T}(lambda_real, q_real, M, res_real,
                             result.info, result.epsout, result.loop)
end


function feast_heev!(A::Matrix{Complex{T}},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    return _feast_dense_complex_hermitian(A, nothing, Emin, Emax, M0, fpm;
                                          solver=solver, solver_tol=solver_tol,
                                          solver_maxiter=solver_maxiter,
                                          solver_restart=solver_restart)
end

function feast_gegv!(A::Matrix{Complex{T}}, B::Union{Matrix{Complex{T}},Nothing},
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense complex general eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are general matrices
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    B === nothing || size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))
    B_is_identity = B === nothing

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_grci_input(N, M0, Emid, r, fpm)
    
    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    use_direct = solver_choice == :direct
    use_iterative = !use_direct
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    use_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))

    A_iter = use_iterative ? Matrix{Complex{T}}(A) : nothing
    B_iter = use_iterative && !B_is_identity ? Matrix{Complex{T}}(B) : nothing
    tmpAx = use_iterative ? zeros(Complex{T}, N) : nothing
    tmpBx = use_iterative ? zeros(Complex{T}, N) : nothing
    current_shift = Ref(zero(Complex{T}))

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
    
    # LU factorization workspace
    LU_factorization = Ref{LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}}()
    temp_matrix = Matrix{Complex{T}}(undef, N, N)
    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)
    rhs_copy = Matrix{Complex{T}}(undef, N, M0)
    factor_cache = Dict{Complex{T}, LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}}()

    # Persistent RCI state (must be reused across calls in the loop)
    grci_state = FeastGRCIState{T}()

    # Safety counter to prevent infinite loops
    max_rci_iterations = fpm[2] * (fpm[4] + 1) * 10  # num_points * (max_loops + 1) * safety_factor
    rci_iteration_count = 0

    @views while true
        rci_iteration_count += 1
        if rci_iteration_count > max_rci_iterations
            info[] = Int(Feast_ERROR_NO_CONVERGENCE)
            @warn "FEAST RCI loop exceeded maximum iterations ($max_rci_iterations). " *
                  "This may indicate a bug in the algorithm or numerical issues. " *
                  "Current ijob=$(ijob[]), loop=$(loop[])"
            break
        end

        # Call Feast RCI kernel for general problems
        feast_grci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emid, r, M0, lambda_complex, q_complex,
                   mode, workspace.res, info; state=grci_state)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A
            z = Ze[]
            if use_direct
                try
                    factor = get(factor_cache, z, nothing)
                    if factor === nothing
                        if B_is_identity
                            _feast_dense_shifted_identity_minus!(temp_matrix, z, A)
                        else
                            @. temp_matrix = z * B - A
                        end
                        factor = lu(temp_matrix)
                        factor_cache[z] = factor
                    end
                    LU_factorization[] = factor
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = z
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = view(rhs_buffer, :, 1:M0)
            workc_block = view(workspace.workc, :, 1:M0)
            if B_is_identity
                copyto!(rhs, workc_block)
            else
                mul!(rhs, B, workc_block)
            end

            if use_direct
                try
                    copyto!(workc_block, rhs)
                    ldiv!(LU_factorization[], workc_block)
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                copyto!(rhs_copy, rhs)
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    if B_is_identity
                        @. tmpBx = current_shift[] * x
                    else
                        mul!(tmpBx, B_iter, x)
                        @. tmpBx = current_shift[] * tmpBx
                    end
                    mul!(tmpAx, A_iter, x)
                    @. y = tmpBx - tmpAx
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], rhs_copy,
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                if !success
                    # Fall back to direct solve if iterative solver failed
                    if B_is_identity
                        _feast_dense_shifted_identity_minus!(temp_matrix, current_shift[], A_iter)
                    else
                        @. temp_matrix = current_shift[] * B_iter - A_iter
                    end
                    try
                        LU_factorization[] = lu!(temp_matrix)
                        copyto!(workc_block, rhs_copy)
                        ldiv!(LU_factorization[], workc_block)
                    catch e
                        info[] = Int(Feast_ERROR_LAPACK)
                        break
                    end
                end
                # If iterative solve succeeded, solution is already in workspace.workc
            end

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            # Compute B * q (for forming reduced matrix zBq = Q^H * B * Q)
            M = mode[]
            if B_is_identity
                copyto!(view(workspace.workc, :, 1:M), view(q_complex, :, 1:M))
            else
                mul!(view(workspace.workc, :, 1:M), B, view(q_complex, :, 1:M))
            end

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

# Polynomial helpers

function _check_polynomial_coeffs(coeffs::Vector{Matrix{Complex{T}}}, d::Int) where T<:Real
    length(coeffs) == d + 1 ||
        throw(ArgumentError("Need d+1 coefficient matrices, got $(length(coeffs)) for degree $d"))
    N = size(coeffs[1], 1)
    size(coeffs[1], 2) == N ||
        throw(ArgumentError("Coefficient matrices must be square"))
    for (idx, mat) in enumerate(coeffs)
        size(mat) == (N, N) ||
            throw(ArgumentError("Coefficient matrix $idx must be size ($N, $N)"))
    end
    return N
end

function _evaluate_polynomial_matrix!(dest::AbstractMatrix{Complex{T}},
                                      coeffs::Vector{Matrix{Complex{T}}}, z::Complex{T}) where T<:Real
    dest .= coeffs[end]
    for k in length(coeffs)-1:-1:1
        @. dest = z * dest
        dest .+= coeffs[k]
    end
    return dest
end

function _apply_polynomial!(dest::AbstractVector{Complex{T}},
                            coeffs::Vector{Matrix{Complex{T}}}, λ::Complex{T},
                            vec::AbstractVector{Complex{T}},
                            scratch::AbstractVector{Complex{T}}) where T<:Real
    fill!(dest, zero(Complex{T}))
    λpow = one(Complex{T})
    for mat in coeffs
        mul!(scratch, mat, vec)
        @. dest += λpow * scratch
        λpow *= λ
    end
    return dest
end

function _feast_polynomial_rci!(coeffs::Vector{Matrix{Complex{T}}}, d::Int,
                                Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    N = _check_polynomial_coeffs(coeffs, d)
    check_feast_grci_input(N, M0, Emid, r, fpm)

    contour = feast_get_custom_contour(T, fpm)
    if contour === nothing
        contour = feast_gcontour(Emid, r, fpm)
    end
    Zne = Complex{T}.(contour.Zne)
    Wne = Complex{T}.(contour.Wne)

    work = zeros(Complex{T}, N, M0)
    workc = similar(work)
    Aq = zeros(Complex{T}, M0, M0)
    Bq = similar(Aq)
    lambda = zeros(Complex{T}, M0)
    q = similar(work)
    res = zeros(T, M0)

    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)

    factorization = nothing
    poly_matrix = similar(coeffs[1])
    scratch_vec = zeros(Complex{T}, N)

    while true
        feast_grcipevx!(ijob, d, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                        Emid, r, M0, lambda, q, mode, res, info, Zne, Wne)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            _evaluate_polynomial_matrix!(poly_matrix, coeffs, Ze[])
            try
                factorization = lu!(poly_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if factorization === nothing
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end
            try
                workc[:, 1:M0] .= factorization \ work[:, 1:M0]
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            M = mode[]
            for j in 1:M
                vec = view(q, :, j)
                dest = view(workc, :, j)
                _apply_polynomial!(dest, coeffs, lambda[j], vec, scratch_vec)
            end
        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            error("Unexpected FEAST polynomial RCI job code: ijob=$(ijob[])")
        end
    end

    M = mode[]
    lambda_real = real.(lambda[1:M])
    q_res = q[:, 1:M]
    res_res = res[1:M]

    return FeastResult{T, Complex{T}}(lambda_real, q_res, M, res_res,
                                      info[], epsout[], loop[])
end

# Polynomial eigenvalue problem support
function feast_pep!(A::Vector{Matrix{Complex{T}}}, d::Int,
                    Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for polynomial eigenvalue problems
    # Solves: P(lambda)*q = 0 where P(lambda) = A[1] + lambda*A[2] + ... + lambda^d*A[d+1]
    
    length(A) == d + 1 || throw(ArgumentError("Need d+1 coefficient matrices"))
    
    N = size(A[1], 1)
    for i in 1:d+1
        size(A[i]) == (N, N) || throw(ArgumentError("All matrices must be same size"))
    end
    
    # Linearize the polynomial eigenvalue problem
    # Convert to generalized eigenvalue problem of size d*N
    DN = d * N
    
    # Companion matrix form
    A_lin = zeros(Complex{T}, DN, DN)
    B_lin = zeros(Complex{T}, DN, DN)
    
    # Fill companion matrices using first companion linearization
    # For P(λ) = A[1] + λ*A[2] + ... + λ^d*A[d+1] where A[k] corresponds to λ^{k-1}
    #
    # A_lin has identity blocks on super-diagonal (blocks (i-1, i) for i=1:d-1)
    # and -A[1], -A[2], ..., -A[d] (i.e., -A_0 to -A_{d-1}) in the last block row
    #
    # B_lin has identity blocks on diagonal (blocks (i, i) for i=0:d-2)
    # and A[d+1] (i.e., A_d) in the last diagonal block

    # Super-diagonal identity blocks in A_lin
    for i in 1:d-1
        A_lin[(i-1)*N+1:i*N, i*N+1:(i+1)*N] .= I(N)
    end

    # Last row blocks: -A[1] through -A[d] (coefficients of λ^0 through λ^{d-1})
    for j in 1:d
        A_lin[(d-1)*N+1:d*N, (j-1)*N+1:j*N] .= -A[j]
    end

    # Diagonal identity blocks in B_lin (first d-1 blocks)
    for i in 1:d-1
        B_lin[(i-1)*N+1:i*N, (i-1)*N+1:i*N] .= I(N)
    end

    # Last diagonal block: A[d+1] (coefficient of λ^d)
    B_lin[(d-1)*N+1:d*N, (d-1)*N+1:d*N] .= A[d+1]
    
    # Solve linearized problem
    result = feast_gegv!(A_lin, B_lin, Emid, r, M0*d, fpm)
    
    # Extract original eigenvectors (first N components)
    M = result.M
    lambda = result.lambda[1:M]
    q_orig = result.q[1:N, 1:M]

    return FeastGeneralResult{T}(lambda, q_orig, M, result.res[1:M],
                                 result.info, result.epsout, result.loop)
end

# Standard eigenvalue problem variants (B = I)

function feast_syev!(A::Matrix{T},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense real symmetric standard eigenvalue problem
    # Solves: A*q = lambda*q where A is symmetric
    # This is equivalent to feast_sygv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    complex_A = Complex{T}.(A)
    complex_result = _feast_dense_complex_hermitian(complex_A, nothing,
                                                   Emin, Emax, M0, fpm;
                                                   solver=solver,
                                                   solver_tol=solver_tol,
                                                   solver_maxiter=solver_maxiter,
                                                   solver_restart=solver_restart)
    return _complex_to_real_result(complex_result)
end

function feast_hegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    return _feast_dense_complex_hermitian(A, B, Emin, Emax, M0, fpm;
                                          solver=solver, solver_tol=solver_tol,
                                          solver_maxiter=solver_maxiter,
                                          solver_restart=solver_restart)
end


function feast_geev!(A::Matrix{Complex{T}},
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense complex general standard eigenvalue problem
    # Solves: A*q = lambda*q where A is a general matrix
    # This is equivalent to feast_gegv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    return feast_gegv!(A, nothing, Emid, r, M0, fpm;
                       solver=solver, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_gegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_gegv!(A, B, Emid, r, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_geev!(A::Matrix{Complex{T}},
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_geev!(A, Emid, r, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function difeast_sygv!(A::Matrix{T}, B::Matrix{T},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_sygv!(A, B, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function difeast_syev!(A::Matrix{T},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_syev!(A, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_heev!(A::Matrix{Complex{T}},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_heev!(A, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_hegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_hegv!(A, B, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

# Custom contour (x-suffix) variants
function feast_sygvx!(A::Matrix{T}, B::Matrix{T},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sygv!(A, B, Emin, Emax, M0, fpm)
    end
end

function feast_syevx!(A::Matrix{T},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_syev!(A, Emin, Emax, M0, fpm)
    end
end

function feast_hegvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hegv!(A, B, Emin, Emax, M0, fpm)
    end
end

function feast_heevx!(A::Matrix{Complex{T}},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_heev!(A, Emin, Emax, M0, fpm)
    end
end

function feast_gegvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gegv!(A, B, Emid, r, M0, fpm)
    end
end

function feast_geevx!(A::Matrix{Complex{T}},
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_geev!(A, Emid, r, M0, fpm)
    end
end

# Polynomial eigenvalue wrappers
function feast_gepev!(A::Vector{Matrix{Complex{T}}}, d::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    return feast_pep!(A, d, Emid, r, M0, fpm)
end

function feast_gepevx!(A::Vector{Matrix{Complex{T}}}, d::Int,
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                       Zne::AbstractVector{Complex{TZ}},
                       Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gepev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_hepev!(A::Vector{Matrix{Complex{T}}}, d::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    return feast_gepev!(A, d, Emid, r, M0, fpm)
end

function feast_hepevx!(A::Vector{Matrix{Complex{T}}}, d::Int,
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                       Zne::AbstractVector{Complex{TZ}},
                       Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hepev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_sypev!(A::Vector{Matrix{T}}, d::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    coeffs = [Complex{T}.(A[i]) for i in eachindex(A)]
    return feast_gepev!(coeffs, d, Emid, r, M0, fpm)
end

function feast_sypevx!(A::Vector{Matrix{T}}, d::Int,
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                       Zne::AbstractVector{Complex{TZ}},
                       Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sypev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_srcipev!(A::Vector{Matrix{T}}, d::Int,
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    coeffs = [Complex{T}.(mat) for mat in A]
    return _feast_polynomial_rci!(coeffs, d, Emid, r, M0, fpm)
end

function feast_srcipevx!(A::Vector{Matrix{T}}, d::Int,
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_srcipev!(A, d, Emid, r, M0, fpm)
    end
end

function feast_grcipev!(A::Vector{Matrix{Complex{T}}}, d::Int,
                        Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    return _feast_polynomial_rci!(A, d, Emid, r, M0, fpm)
end

function feast_grcipevx!(A::Vector{Matrix{Complex{T}}}, d::Int,
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_grcipev!(A, d, Emid, r, M0, fpm)
    end
end

"""
    _feast_dense_complex_symmetric(A, B, Emid, r, M0, fpm; solver=:direct)

Dense complex-symmetric FEAST implementation. The shifted systems are solved
with dense LU or GMRES, and the reduced Ritz pencil is formed with the
transpose bilinear form `Qᵀ A Q`, not the conjugate-adjoint form used by
Hermitian/general dense paths.
"""
@views function _feast_dense_complex_symmetric(A::Matrix{Complex{T}},
                                               B::Union{Matrix{Complex{T}},Nothing},
                                               Emid::Complex{T}, r::T,
                                               M0::Int, fpm::Vector{Int};
                                               solver::Symbol = :direct,
                                               solver_tol::Real = 0.0,
                                               solver_maxiter::Int = 500,
                                               solver_restart::Int = 30) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    B === nothing || size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))
    check_complex_symmetric(A)
    B !== nothing && check_complex_symmetric(B)

    feastdefault!(fpm)
    check_feast_grci_input(N, M0, Emid, r, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_is_identity = B === nothing
    B_matrix = B_is_identity ? Matrix{Complex{T}}(undef, 0, 0) : copy(B)

    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    shifted_solutions = similar(Q_basis)
    rhs_buffer = similar(Q_basis)
    rhs_copy = similar(rhs_buffer)
    Q_proj = zeros(Complex{T}, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQ = Matrix{Complex{T}}(undef, N, M0)
    Ared = Matrix{Complex{T}}(undef, M0, M0)
    Bred = Matrix{Complex{T}}(undef, M0, M0)
    lambda_vec = Vector{Complex{T}}(undef, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(shifted_solutions)
    res_vec = zeros(T, M0)
    residual_vec = Vector{Complex{T}}(undef, N)
    Bq_vec = Vector{Complex{T}}(undef, N)
    shifted_matrix = similar(A)
    current_shift = Ref(zero(Complex{T}))
    tmpAx = Vector{Complex{T}}(undef, N)
    tmpBx = Vector{Complex{T}}(undef, N)

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_is_identity
            @. tmpBx = current_shift[] * x
        else
            mul!(tmpBx, B_matrix, x)
            @. tmpBx = current_shift[] * tmpBx
        end
        mul!(tmpAx, A, x)
        @. y = tmpBx - tmpAx
        return y
    end

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
    active_dim = M0

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false
        for e in eachindex(Zne)
            z = Zne[e]
            weight = Wne[e]
            basis_block = view(Q_basis, :, 1:active_dim)
            rhs_block = view(rhs_buffer, :, 1:active_dim)
            rhs_copy_block = view(rhs_copy, :, 1:active_dim)
            shifted_block = view(shifted_solutions, :, 1:active_dim)
            qproj_block = view(Q_proj, :, 1:active_dim)

            if B_is_identity
                copyto!(rhs_block, basis_block)
            else
                mul!(rhs_block, B_matrix, basis_block)
            end

            if solver_is_direct
                if B_is_identity
                    @. shifted_matrix = -A
                    for i in 1:N
                        shifted_matrix[i, i] += z
                    end
                else
                    @. shifted_matrix = z * B_matrix - A
                end
                copyto!(rhs_copy_block, rhs_block)
                try
                    factor = lu!(shifted_matrix)
                    ldiv!(shifted_block, factor, rhs_copy_block)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Dense complex-symmetric direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                copyto!(rhs_copy_block, rhs_block)
                current_shift[] = z
                success = solve_dense_shifted!(shifted_block, rhs_copy_block,
                                               shifted_mul!, solver_choice,
                                               tol_value, solver_maxiter,
                                               solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            @. qproj_block += weight * shifted_block
        end
        solve_failed && break

        try
            rank = _feast_qr_compress!(solutions_tmp, Q_proj, active_dim;
                                       rank_tol=sqrt(eps(T)))
            if rank == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            q_rank = view(solutions_tmp, :, 1:rank)
            aq_work = view(AQ, :, 1:rank)
            bq_work = view(BQ, :, 1:rank)
            Ared_rank = view(Ared, 1:rank, 1:rank)
            Bred_rank = view(Bred, 1:rank, 1:rank)

            mul!(aq_work, A, q_rank)
            if B_is_identity
                copyto!(bq_work, q_rank)
            else
                mul!(bq_work, B_matrix, q_rank)
            end
            mul!(Ared_rank, transpose(q_rank), aq_work)
            mul!(Bred_rank, transpose(q_rank), bq_work)

            F = eigen(Ared_rank, Bred_rank)
            lambda_red = F.values
            v_red = F.vectors

            for idx in 1:rank
                mul!(view(shifted_solutions, :, idx), q_rank, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_gcontour!(lambda_vec, shifted_solutions, perm,
                                            lambda_tmp, solutions_tmp,
                                            Emid, r, fpm, rank)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for idx in 1:rank
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
                if B_is_identity
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    mul!(Bq_vec, B_matrix, q_col)
                    @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                end
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M
            epsout_val <= eps_tol && break

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            active_dim = rank
            copyto!(view(Q_basis, :, 1:active_dim),
                    view(shifted_solutions, :, 1:active_dim))
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced eigenvalue problem failed during dense complex-symmetric FEAST" exception=err
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

function feast_geev_complex_sym!(A::Matrix{Complex{T}},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    return _feast_dense_complex_symmetric(A, nothing, Emid, r, M0, fpm;
                                          solver=solver,
                                          solver_tol=solver_tol,
                                          solver_maxiter=solver_maxiter,
                                          solver_restart=solver_restart)
end

function feast_gegv_complex_sym!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    return _feast_dense_complex_symmetric(A, B, Emid, r, M0, fpm;
                                          solver=solver,
                                          solver_tol=solver_tol,
                                          solver_maxiter=solver_maxiter,
                                          solver_restart=solver_restart)
end
