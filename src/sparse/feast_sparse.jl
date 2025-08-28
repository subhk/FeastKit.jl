# Feast sparse matrix routines
# Translated from dzfeast_sparse.f90 and related files

using SparseArrays

function feast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for sparse real symmetric generalized eigenvalue problem in CSR format
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
    
    # Sparse linear solver workspace
    sparse_solver = nothing
    
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A for sparse matrices
            z = Ze[]
            sparse_matrix = z * B - A
            
            # LU factorization for sparse matrix
            try
                sparse_solver = lu(sparse_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems: (Ze*B - A) * X = B * workspace.work
            rhs = B * workspace.work[:, 1:M0]
            
            try
                # Solve with sparse LU factors
                workspace.workc[:, 1:M0] .= sparse_solver \ rhs
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
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, T}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_hcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for sparse complex Hermitian eigenvalue problem
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
    
    # Sparse linear solver workspace
    sparse_solver = nothing
    
    while true
        # Call Feast RCI kernel
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*I - A for sparse matrices
            z = Ze[]
            I_sparse = sparse(I, N, N)
            sparse_matrix = z * I_sparse - A
            
            # LU factorization for sparse matrix
            try
                sparse_solver = lu(sparse_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems
            try
                workspace.workc[:, 1:M0] .= sparse_solver \ workspace.workc[:, 1:M0]
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
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, Complex{T}}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_gcsrgv!(A::SparseMatrixCSC{Complex{T},Int}, B::SparseMatrixCSC{Complex{T},Int},
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for sparse complex general eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are general sparse matrices
    
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
            sparse_matrix = z * B - A
            
            # LU factorization for sparse matrix
            try
                sparse_solver = lu(sparse_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems: (Ze*B - A) * X = B * workspace.workc
            rhs = B * workspace.workc[:, 1:M0]
            
            try
                workspace.workc[:, 1:M0] .= sparse_solver \ rhs
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
        end
    end
    
    # Extract results
    M = mode[]
    lambda = lambda_complex[1:M]
    q = q_complex[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, Complex{T}}(real.(lambda), q, M, res, info[], epsout[], loop[])
end

# Iterative refinement for sparse problems
function feast_scsrgv_iterative!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                 Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                                 max_iter::Int = 3) where T<:Real
    # Feast with iterative refinement for better accuracy
    
    result = feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
    
    # Perform iterative refinement if converged
    if result.info == 0 && max_iter > 1
        for iter in 2:max_iter
            # Use previous result as initial guess
            fpm[5] = 1  # Use initial guess
            
            # Refine the solution
            result_new = feast_scsrgv!(A, B, Emin, Emax, result.M, fpm)
            
            # Check if refinement improved the solution
            if result_new.epsout < result.epsout
                result = result_new
            else
                break  # No improvement, stop refinement
            end
        end
    end
    
    return result
end

# Matrix-free interface for sparse problems
function feast_sparse_matvec!(A_matvec!::Function, B_matvec!::Function,
                             N::Int, Emin::T, Emax::T, M0::Int, 
                             fpm::Vector{Int}) where T<:Real
    # Feast with matrix-free operations
    # A_matvec!(y, x) computes y = A*x
    # B_matvec!(y, x) computes y = B*x
    
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
    
    # For matrix-free, we need an iterative solver
    # This is a simplified version - in practice, you'd use GMRES, BiCGSTAB, etc.
    
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # For matrix-free, we don't factorize but prepare for iterative solve
            # Store the shift for the iterative solver
            # In practice, you'd set up preconditioners here
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve (Ze*B - A) * X = B * workspace.work using iterative method
            z = Ze[]
            
            # Right-hand side
            rhs = zeros(T, N, M0)
            for j in 1:M0
                B_matvec!(view(rhs, :, j), view(workspace.work, :, j))
            end
            
            # Solve each system iteratively (simplified - use proper iterative solver)
            for j in 1:M0
                x = zeros(Complex{T}, N)
                r = complex.(rhs[:, j])
                
                # Simple fixed-point iteration (replace with proper solver)
                for iter in 1:10
                    # Compute residual and update
                    temp = zeros(T, N)
                    A_matvec!(temp, real.(x))
                    r_new = complex.(rhs[:, j]) .- temp .+ z .* r
                    x .= r_new ./ (z + 1.0)  # Simplified update
                end
                
                workspace.workc[:, j] .= x
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            for j in 1:M
                A_matvec!(view(workspace.work, :, j), view(workspace.q, :, j))
            end
            
        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, T}(lambda, q, M, res, info[], epsout[], loop[])
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
