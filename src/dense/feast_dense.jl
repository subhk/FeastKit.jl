# Feast dense matrix routines
# Translated from dzfeast_dense.f90 and dzfeast_pev_dense.f90

struct DenseShiftOperator{F,T<:Real}
    mulfun::F
    N::Int
end

Base.size(op::DenseShiftOperator) = (op.N, op.N)

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

    for j in 1:size(rhs, 2)
        b = view(rhs, :, j)
        x0 = zeros(Complex{T}, N)
        x_sol, stats = gmres(op, b;
                             x=x0,
                             restart=true,
                             memory=max(restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        stats.solved || return false
        dest[:, j] .= x_sol
    end

    return true
end

function feast_sygv!(A::Matrix{T}, B::Matrix{T},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense real symmetric generalized eigenvalue problem
    # Solves: A*q = lambda*B*q where A is symmetric, B is symmetric positive definite

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver in (:direct, :gmres, :iterative) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct or :gmres."))
    use_direct = solver_choice == :direct
    use_iterative = !use_direct
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    use_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))

    A_iter = use_iterative ? Complex{T}.(A) : nothing
    B_iter = use_iterative ? Complex{T}.(B) : nothing
    tmpAx = use_iterative ? zeros(Complex{T}, N) : nothing
    tmpBx = use_iterative ? zeros(Complex{T}, N) : nothing
    current_shift = Ref(zero(Complex{T}))

    # Initialize workspace
    workspace = FeastWorkspaceReal{T}(N, M0)
    
    # Initialize variables for RCI
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
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

        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q,
                   mode, workspace.res, info)
        if ijob[] == Int(Feast_RCI_FACTORIZE)
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
            rhs = B * workspace.work[:, 1:M0]

            if use_direct
                try
                    workspace.workc[:, 1:M0] .= LU_factorization \ rhs
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                rhs_complex = Complex{T}.(rhs)
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    mul!(tmpBx, B_iter, x)
                    @. tmpBx = current_shift[] * tmpBx
                    mul!(tmpAx, A_iter, x)
                    @. y = tmpBx - tmpAx
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], rhs_complex,
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                success || begin
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            workspace.work[:, 1:M] .= A * workspace.q[:, 1:M]

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, T}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_heev!(A::Matrix{Complex{T}}, 
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense complex Hermitian eigenvalue problem
    # Solves: A*q = lambda*q where A is Hermitian
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver in (:direct, :gmres, :iterative) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct or :gmres."))
    use_direct = solver_choice == :direct
    use_iterative = !use_direct
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    use_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative dense FEAST solves."))

    A_iter = use_iterative ? Matrix{Complex{T}}(A) : nothing
    tmpAx = use_iterative ? zeros(Complex{T}, N) : nothing
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

        # Call Feast RCI kernel
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q,
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            z = Ze[]
            # Compute z*I - A (don't broadcast I, it's UniformScaling)
            for i in 1:N
                for j in 1:N
                    temp_matrix[i,j] = (i == j ? z : zero(Complex{T})) - A[i,j]
                end
            end

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
            if use_direct
                try
                    workspace.workc[:, 1:M0] .= LU_factorization \ workspace.workc[:, 1:M0]
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    copy!(tmpAx, x)
                    @. tmpAx = current_shift[] * tmpAx
                    mul!(y, A_iter, x)
                    @. y = tmpAx - y
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], workspace.workc[:, 1:M0],
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                success || begin
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            workspace.workc[:, 1:M] .= A * workspace.q[:, 1:M]

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, Complex{T}}(lambda, q, M, res, info[], epsout[], loop[])
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
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    mul!(tmpBx, B_iter, x)
                    @. tmpBx = current_shift[] * tmpBx
                    mul!(tmpAx, A_iter, x)
                    @. y = tmpBx - tmpAx
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], rhs,
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                success || begin
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            workspace.workc[:, 1:M] .= A * q_complex[:, 1:M]

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end
    
    # Extract results
    M = mode[]
    lambda = lambda_complex[1:M]
    q = q_complex[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, Complex{T}}(real.(lambda), q, M, res, info[], epsout[], loop[])
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
    
    # Fill companion matrices
    for i in 1:d-1
        A_lin[(i-1)*N+1:i*N, i*N+1:(i+1)*N] .= I(N)
    end
    
    for j in 1:d+1
        A_lin[(d-1)*N+1:d*N, (j-1)*N+1:j*N] .= -A[j]
    end
    
    # B matrix
    for i in 1:d-1
        B_lin[i*N+1:(i+1)*N, i*N+1:(i+1)*N] .= I(N)
    end
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
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for dense real symmetric standard eigenvalue problem
    # Solves: A*q = lambda*q where A is symmetric
    # This is equivalent to feast_sygv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    # Create identity matrix for B
    B = Matrix{T}(I, N, N)

    # Call generalized version with B = I
    return feast_sygv!(A, B, Emin, Emax, M0, fpm)
end

function feast_hegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for dense complex Hermitian generalized eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are Hermitian, B is positive definite

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

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

        # Call Feast RCI kernel
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q,
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
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
                rhs_complex = Complex{T}.(rhs)
                function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                    mul!(tmpBx, B_iter, x)
                    @. tmpBx = current_shift[] * tmpBx
                    mul!(tmpAx, A_iter, x)
                    @. y = tmpBx - tmpAx
                    return y
                end
                success = solve_dense_shifted!(workspace.workc[:, 1:M0], rhs_complex,
                                               shifted_mul!, solver_choice, tol_value,
                                               solver_maxiter, solver_restart)
                success || begin
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            workspace.work[:, 1:M] .= real.(A * workspace.q[:, 1:M])

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end

    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]

    return FeastResult{T, Complex{T}}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_geev!(A::Matrix{Complex{T}},
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for dense complex general standard eigenvalue problem
    # Solves: A*q = lambda*q where A is a general matrix
    # This is equivalent to feast_gegv! with B = I

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    # Create identity matrix for B
    B = Matrix{Complex{T}}(I, N, N)

    # Call generalized version with B = I
    return feast_gegv!(A, B, Emid, r, M0, fpm)
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
