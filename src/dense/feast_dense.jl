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

function solve_dense_shifted!(dest::AbstractMatrix{Complex{T}},
                              rhs::AbstractMatrix{Complex{T}},
                              apply_shift!::Function,
                              solver::Symbol, tol::T,
                              maxiter::Int, restart::Int) where T<:Real
    solver == :direct && error("Direct solve should be handled before calling solve_dense_shifted!")

    if !FEAST_KRYLOV_AVAILABLE[]
        error("Krylov.jl required for iterative dense FEAST solves")
    end

    N = size(rhs, 1)
    op = DenseShiftOperator{typeof(apply_shift!), T}(apply_shift!, N)

    residual = zeros(Complex{T}, N)
    for j in 1:size(rhs, 2)
        b = view(rhs, :, j)
        x0 = zeros(Complex{T}, N)
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
        if !stats.solved || res_norm > tol * max(b_norm, one(T))
            @warn "GMRES failed to converge" iteration_stats=stats residual=res_norm rhs_norm=b_norm
            return false
        end
        dest[:, j] .= x_sol
    end

    return true
end

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

    solver_choice = solver in (:direct, :gmres) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct or :gmres."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_matrix = B === nothing ? nothing : copy(B)
    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    solutions = similar(Q_basis)
    rhs_buffer = zeros(Complex{T}, N, M0)
    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    ReducedT = promote_type(Float64, T)
    lambda_vec = zeros(T, M0)
    res_vec = zeros(T, M0)
    residual_vec = zeros(Complex{T}, N)
    Bq_vec = B_matrix === nothing ? nothing : zeros(Complex{T}, N)
    shifted_matrix = similar(A)
    current_shift = Ref(zero(Complex{T}))
    tmpAx = solver_is_iterative ? zeros(Complex{T}, N) : nothing
    tmpBx = solver_is_iterative ? zeros(Complex{T}, N) : nothing

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_matrix === nothing
            @. tmpBx = current_shift[] * x
        else
            mul!(tmpBx, B_matrix, x)
            @. tmpBx = current_shift[] * tmpBx
        end
        mul!(tmpAx, A, x)
        @. y = tmpBx - tmpAx
        return y
    end

    contour = feast_get_custom_contour(fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    loop_count = 0
    M_found = 0

    # Allocate buffer for accumulated filtered subspace
    Q_proj = zeros(Complex{T}, N, M0)

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false

        for (idx, z) in enumerate(Zne)
            weight = 2 * Wne[idx]

            if B_matrix === nothing
                rhs_buffer .= Q_basis
            else
                mul!(rhs_buffer, B_matrix, Q_basis)
            end

            if solver_is_direct
                if B_matrix === nothing
                    for j in 1:N, i in 1:N
                        shifted_matrix[i, j] = (i == j ? z : zero(z)) - A[i, j]
                    end
                else
                    @. shifted_matrix = z * B_matrix - A
                end
                rhs_copy = copy(rhs_buffer)
                try
                    factor = lu(shifted_matrix)
                    solutions .= factor \ rhs_copy
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Dense direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                rhs_copy = copy(rhs_buffer)
                current_shift[] = z
                success = solve_dense_shifted!(solutions, rhs_copy,
                                               shifted_mul!, solver_choice,
                                               tol_value, solver_maxiter,
                                               solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            # Accumulate filtered subspace (spectral projector applied to Q_basis)
            @. Q_proj += weight * solutions

            mul!(moment, adjoint(Q_basis), solutions)
            @. zAq += weight * moment
            @. zSq += weight * z * moment
        end

        solve_failed && break

        try
            # For half-contour integration, take real part directly
            # The half-contour exploits symmetry: contribution from conjugate points
            # combines to give a purely real result when taking Re(zAq)
            Aq_real = Matrix{ReducedT}(real.(zAq))
            Sq_real = Matrix{ReducedT}(real.(zSq))

            # Try Symmetric solver first (more accurate), fall back to general if not positive definite
            # IMPORTANT: Solve Sq*x = lambda*Aq*x (not Aq*x = lambda*Sq*x)
            # This is because zAq ≈ Q'*P*Q and zSq ≈ Q'*A*P*Q where P is the spectral projector
            lambda_red = Vector{ReducedT}(undef, 0)
            v_red = Array{ReducedT}(undef, 0, 0)
            try
                F = eigen(Symmetric(Sq_real), Symmetric(Aq_real))
                lambda_red = Vector{ReducedT}(F.values)
                v_red = Array{ReducedT}(F.vectors)
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    # Fall back to general eigenvalue solver if not positive definite
                    F = eigen(Sq_real, Aq_real)
                    lambda_red = Vector{ReducedT}(real.(F.values))
                    v_red = Array{ReducedT}(real.(F.vectors))
                else
                    rethrow(e)
                end
            end

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q_basis
            # Q_proj = spectral projector applied to Q_basis via contour integration
            Q_proj_real = real.(Q_proj)
            for idx in 1:M0
                coeffs = Vector{T}(view(v_red, :, idx))
                mul!(view(solutions, :, idx), Q_proj_real, coeffs)
                lambda_vec[idx] = convert(T, lambda_red[idx])
            end

            # Reorder: put eigenvalues inside interval first while maintaining pairing
            inside_mask = [Emin <= lambda_vec[i] <= Emax for i in 1:M0]
            inside_indices = findall(inside_mask)
            outside_indices = findall(.!inside_mask)
            perm = vcat(inside_indices, outside_indices)

            # Apply permutation to maintain eigenvalue/eigenvector pairing
            lambda_vec[1:M0] = lambda_vec[perm]
            solutions[:, 1:M0] = solutions[:, perm]

            M = length(inside_indices)
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
                if B_matrix === nothing
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
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Use full M0 subspace for next iteration
            Q_basis[:, 1:M0] .= solutions[:, 1:M0]
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
    for j in 1:M
        @inbounds q_real[:, j] .= real.(result.q[:, j])
    end
    lambda_real = real.(result.lambda[1:M])
    res_real = result.res[1:M]
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

function feast_gegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense complex general eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are general matrices
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_grci_input(N, M0, Emid, r, fpm)
    
    solver_choice = solver in (:direct, :gmres, :iterative) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct or :gmres."))
    use_direct = solver_choice == :direct
    use_iterative = !use_direct
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    use_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))

    A_iter = use_iterative ? Matrix{Complex{T}}(A) : nothing
    B_iter = use_iterative ? Matrix{Complex{T}}(B) : nothing
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
    LU_factorization = nothing
    temp_matrix = Matrix{Complex{T}}(undef, N, N)

    # Safety counter to prevent infinite loops
    max_rci_iterations = fpm[2] * (fpm[4] + 1) * 10  # num_points * (max_loops + 1) * safety_factor
    rci_iteration_count = 0

    while true
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
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A
            z = Ze[]
            temp_matrix .= z .* B .- A

            if use_direct
                try
                    LU_factorization = lu!(temp_matrix)
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = z
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = B * workspace.workc[:, 1:M0]

            if use_direct
                try
                    workspace.workc[:, 1:M0] .= LU_factorization \ rhs
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                rhs_copy = copy(rhs)
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    mul!(tmpBx, B_iter, x)
                    @. tmpBx = current_shift[] * tmpBx
                    mul!(tmpAx, A_iter, x)
                    @. y = tmpBx - tmpAx
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], rhs_copy,
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                if !success
                    # Fall back to direct solve if iterative solver failed
                    temp_matrix .= current_shift[] .* B_iter .- A_iter
                    try
                        LU_factorization = lu!(temp_matrix)
                        workspace.workc[:, 1:M0] .= LU_factorization \ rhs_copy
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

    contour = feast_get_custom_contour(fpm)
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

    return FeastResult{T, Complex{T}}(lambda, q_orig, M, result.res[1:M],
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

    # Create identity matrix for B
    B = Matrix{T}(I, N, N)

    # Call generalized version with B = I
    return feast_sygv!(A, B, Emin, Emax, M0, fpm;
                       solver=solver, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
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

    # Create identity matrix for B
    B = Matrix{Complex{T}}(I, N, N)

    # Call generalized version with B = I
    return feast_gegv!(A, B, Emid, r, M0, fpm;
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
function feast_geev_complex_sym!(A::Matrix{Complex{T}},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    check_complex_symmetric(A)
    return feast_geev!(A, Emid, r, M0, fpm;
                       solver=solver, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_gegv_complex_sym!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                                 Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                 solver::Symbol = :direct,
                                 solver_tol::Real = 0.0,
                                 solver_maxiter::Int = 500,
                                 solver_restart::Int = 30) where T<:Real
    check_complex_symmetric(A)
    check_complex_symmetric(B)
    return feast_gegv!(A, B, Emid, r, M0, fpm;
                       solver=solver, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end
