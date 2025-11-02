# Feast dense matrix routines
# Translated from dzfeast_dense.f90 and dzfeast_pev_dense.f90

function feast_sygv!(A::Matrix{T}, B::Matrix{T}, 
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for dense real symmetric generalized eigenvalue problem
    # Solves: A*q = lambda*B*q where A is symmetric, B is symmetric positive definite
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))
    
    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
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

    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q,
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A
            z = Ze[]
            temp_matrix .= z .* B .- A

            # LU factorization
            try
                LU_factorization = lu!(temp_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze*B - A) * X = B * workspace.work
            rhs = B * workspace.work[:, 1:M0]

            try
                # Solve with LU factors
                workspace.workc[:, 1:M0] .= LU_factorization \ rhs
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
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
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for dense complex Hermitian eigenvalue problem
    # Solves: A*q = lambda*q where A is Hermitian
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    
    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
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

    while true
        # Call Feast RCI kernel
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q,
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*I - A
            z = Ze[]
            temp_matrix .= z .* I .- A

            # LU factorization
            try
                LU_factorization = lu!(temp_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze*I - A) * X = workspace.workc
            try
                workspace.workc[:, 1:M0] .= LU_factorization \ workspace.workc[:, 1:M0]
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
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

function feast_gegv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for dense complex general eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are general matrices
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must be same size as A"))
    
    # Check inputs
    check_feast_grci_input(N, M0, Emid, r, fpm)
    
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

    while true
        # Call Feast RCI kernel for general problems
        feast_grci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emid, r, M0, lambda_complex, q_complex,
                   mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A
            z = Ze[]
            temp_matrix .= z .* B .- A

            # LU factorization
            try
                LU_factorization = lu!(temp_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = B * workspace.workc[:, 1:M0]

            try
                workspace.workc[:, 1:M0] .= LU_factorization \ rhs
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
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