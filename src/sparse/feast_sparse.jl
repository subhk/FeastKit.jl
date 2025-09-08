# Feast sparse matrix routines
# Translated from dzfeast_sparse.f90 and related files

using SparseArrays
using LinearAlgebra

# Try to import Krylov, but provide fallback if not available
const KRYLOV_AVAILABLE = try
    using Krylov
    true
catch
    false
end

# Simple GMRES fallback implementation (for demonstration)
function simple_gmres(A_op!::Function, b::Vector{Complex{T}}, x0::Vector{Complex{T}};
                     rtol::T = T(1e-6), atol::T = T(1e-12), 
                     restart::Int = 20, maxiter::Int = 200) where T<:Real
    
    n = length(b)
    x = copy(x0)
    
    # Simple BiCGSTAB-like iteration as fallback
    r = copy(b)
    A_op!(similar(r), x)
    r .-= similar(r)
    
    rho = norm(r)
    initial_residual = rho
    
    for iter in 1:maxiter
        if rho <= atol || rho <= rtol * initial_residual
            return x, (solved=true, residuals=[rho])
        end
        
        # Very simplified iterative step
        # In a real implementation, this would be proper GMRES/BiCGSTAB
        z = r / (norm(r) + 1e-14)
        A_op!(r, z)
        alpha = dot(r, z) / (dot(r, r) + 1e-14)
        x .+= alpha * z
        
        # Update residual
        A_op!(r, x)
        r .= b .- r
        rho = norm(r)
    end
    
    return x, (solved=false, residuals=[rho])
end

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

# Matrix-free interface for sparse problems with GMRES solver
function feast_sparse_matvec!(A_matvec!::Function, B_matvec!::Function,
                             N::Int, Emin::T, Emax::T, M0::Int, 
                             fpm::Vector{Int}; 
                             gmres_rtol::T = T(1e-6),
                             gmres_atol::T = T(1e-12),
                             gmres_restart::Int = 20,
                             gmres_maxiter::Int = 200) where T<:Real
    # Feast with matrix-free operations using GMRES for linear system solves
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
    
    # Storage for current shift value
    current_shift = Ref(zero(Complex{T}))
    
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Store the current shift for GMRES
            current_shift[] = Ze[]
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve (Ze*B - A) * X = B * workspace.work using GMRES
            z = current_shift[]
            
            # Right-hand side: B * workspace.work
            rhs = zeros(T, N, M0)
            for j in 1:M0
                B_matvec!(view(rhs, :, j), view(workspace.work, :, j))
            end
            
            # Define linear operator for (Ze*B - A)
            # For complex shift z, we need to handle real and imaginary parts separately
            function shifted_matvec!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                # Split complex vectors into real and imaginary parts
                x_real = real.(x)
                x_imag = imag.(x)
                
                # Temporary storage
                temp_real = zeros(T, N)
                temp_imag = zeros(T, N)
                
                # Compute A*x_real and A*x_imag
                A_matvec!(temp_real, x_real)
                A_matvec!(temp_imag, x_imag)
                y_A_real = temp_real
                y_A_imag = temp_imag
                
                # Compute B*x_real and B*x_imag  
                B_matvec!(temp_real, x_real)
                B_matvec!(temp_imag, x_imag)
                y_B_real = temp_real
                y_B_imag = temp_imag
                
                # y = z*B*x - A*x = (z_real + i*z_imag)*(B*x) - A*x
                z_real = real(z)
                z_imag = imag(z)
                
                y_real = z_real * y_B_real - z_imag * y_B_imag - y_A_real
                y_imag = z_real * y_B_imag + z_imag * y_B_real - y_A_imag
                
                y .= complex.(y_real, y_imag)
            end
            
            # Solve each system using GMRES
            for j in 1:M0
                # Convert RHS to complex
                b = complex.(rhs[:, j])
                
                # Initial guess (zero)
                x0 = zeros(Complex{T}, N)
                
                # Solve using GMRES (Krylov.jl if available, otherwise fallback)
                if KRYLOV_AVAILABLE
                    # Create linear operator for Krylov.jl
                    op = LinearOperator{Complex{T}}(N, N, false, false, shifted_matvec!)
                    
                    sol, stats = gmres(op, b; 
                                     x=x0,
                                     rtol=gmres_rtol,
                                     atol=gmres_atol, 
                                     restart=gmres_restart,
                                     itmax=gmres_maxiter)
                    
                    # Check convergence
                    if !stats.solved
                        @warn "GMRES did not converge for system $j, residual = $(stats.residuals[end])"
                        info[] = Int(Feast_ERROR_LAPACK)
                    end
                    
                    workspace.workc[:, j] .= sol
                else
                    # Use fallback solver
                    @warn "Krylov.jl not available, using simplified iterative solver"
                    
                    # Create function wrapper for simple solver
                    function A_op_wrapper!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
                        shifted_matvec!(y, x)
                    end
                    
                    sol, stats = simple_gmres(A_op_wrapper!, b, x0;
                                            rtol=gmres_rtol, atol=gmres_atol,
                                            restart=gmres_restart, maxiter=gmres_maxiter)
                    
                    if !stats.solved
                        @warn "Simple iterative solver did not converge for system $j"
                        info[] = Int(Feast_ERROR_LAPACK)
                    end
                    
                    workspace.workc[:, j] .= sol
                end
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
