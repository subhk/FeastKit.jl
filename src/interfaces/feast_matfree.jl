# Matrix-Free Feast Interface
# Provides callbacks for matrix-vector operations instead of explicit matrices

using LinearAlgebra
using SparseArrays

"""
    MatrixFreeOperator{T}

Abstract type for matrix-free operators.
"""
abstract type MatrixFreeOperator{T} end

"""
    MatrixVecFunction{T}

Matrix-free operator defined by a matrix-vector multiplication function.

# Fields
- `mul!`: Function with signature `mul!(y, op, x)` that computes `y = op * x`
- `size`: Size of the operator as `(m, n)`
- `issymmetric`: Whether the operator is symmetric
- `ishermitian`: Whether the operator is Hermitian  
- `isposdef`: Whether the operator is positive definite
"""
struct MatrixVecFunction{T} <: MatrixFreeOperator{T}
    mul!::Function
    size::Tuple{Int, Int}
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
    
    function MatrixVecFunction{T}(mul!::Function, size::Tuple{Int, Int};
                                 issymmetric::Bool = false,
                                 ishermitian::Bool = false,
                                 isposdef::Bool = false) where T
        new{T}(mul!, size, issymmetric, ishermitian, isposdef)
    end
end

# Convenience constructors
MatrixVecFunction(mul!::Function, size::Tuple{Int, Int}; kwargs...) = 
    MatrixVecFunction{Float64}(mul!, size; kwargs...)

"""
    LinearOperator{T}

Matrix-free operator that supports multiple operations.

# Fields
- `A_mul!`: Function `(y, x) -> y = A*x`
- `At_mul!`: Function `(y, x) -> y = A'*x` (optional)
- `Ac_mul!`: Function `(y, x) -> y = A†*x` (optional)
- `solve!`: Function `(y, z, x) -> y = (z*I - A)\\x` (linear solver)
- `size`: Operator size
- `issymmetric`, `ishermitian`, `isposdef`: Properties
"""
struct LinearOperator{T} <: MatrixFreeOperator{T}
    A_mul!::Function
    At_mul!::Union{Function, Nothing}
    Ac_mul!::Union{Function, Nothing}
    solve!::Union{Function, Nothing}
    size::Tuple{Int, Int}
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
    
    function LinearOperator{T}(A_mul!::Function, size::Tuple{Int, Int};
                              At_mul!::Union{Function, Nothing} = nothing,
                              Ac_mul!::Union{Function, Nothing} = nothing,
                              solve!::Union{Function, Nothing} = nothing,
                              issymmetric::Bool = false,
                              ishermitian::Bool = false,
                              isposdef::Bool = false) where T
        new{T}(A_mul!, At_mul!, Ac_mul!, solve!, size, 
               issymmetric, ishermitian, isposdef)
    end
end

LinearOperator(A_mul!::Function, size::Tuple{Int, Int}; kwargs...) =
    LinearOperator{Float64}(A_mul!, size; kwargs...)

# Interface functions
Base.size(op::MatrixFreeOperator) = op.size
Base.size(op::MatrixFreeOperator, dim::Int) = op.size[dim]
LinearAlgebra.issymmetric(op::MatrixFreeOperator) = op.issymmetric
LinearAlgebra.ishermitian(op::MatrixFreeOperator) = op.ishermitian
LinearAlgebra.isposdef(op::MatrixFreeOperator) = op.isposdef

# Matrix-vector multiplication
function LinearAlgebra.mul!(y::AbstractVector, op::MatrixVecFunction, x::AbstractVector)
    op.mul!(y, op, x)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector)
    op.A_mul!(y, x)
    return y
end

# Transpose multiplication
function LinearAlgebra.mul!(y::AbstractVector, 
                           At::LinearAlgebra.Transpose{T, <:LinearOperator{T}}, 
                           x::AbstractVector) where T
    op = At.parent
    if op.At_mul! !== nothing
        op.At_mul!(y, x)
    elseif op.issymmetric
        op.A_mul!(y, x)
    else
        throw(ArgumentError("Transpose not available for this operator"))
    end
    return y
end

# Adjoint multiplication  
function LinearAlgebra.mul!(y::AbstractVector,
                           Ac::LinearAlgebra.Adjoint{T, <:LinearOperator{T}},
                           x::AbstractVector) where T
    op = Ac.parent
    if op.Ac_mul! !== nothing
        op.Ac_mul!(y, x)
    elseif op.ishermitian
        op.A_mul!(y, x)
    elseif op.issymmetric && T <: Real
        op.A_mul!(y, x)
    else
        throw(ArgumentError("Adjoint not available for this operator"))
    end
    return y
end

"""
    feast_matfree_srci!(A_op, B_op, interval, M0; kwargs...)

Matrix-free Feast RCI for real symmetric eigenvalue problems.

# Arguments
- `A_op`: Matrix-free operator for A
- `B_op`: Matrix-free operator for B  
- `interval`: Search interval (Emin, Emax)
- `M0`: Maximum number of eigenvalues to find

# Keyword Arguments
- `fpm`: Feast parameters vector
- `linear_solver`: Function `(y, z, X) -> Y` where `Y = (z*B - A)\\X`
- `workspace`: Pre-allocated workspace matrices
- `maxiter`: Maximum refinement iterations
- `tol`: Convergence tolerance

# Returns
- `FeastResult` with eigenvalues and eigenvectors
"""
function feast_matfree_srci!(A_op::MatrixFreeOperator{T}, 
                            B_op::MatrixFreeOperator{T},
                            interval::Tuple{T, T}, M0::Int;
                            fpm::Union{Vector{Int}, Nothing} = nothing,
                            linear_solver::Union{Function, Nothing} = nothing,
                            workspace::Union{NamedTuple, Nothing} = nothing,
                            maxiter::Int = 20,
                            tol::T = 1e-12) where T<:Real
    
    Emin, Emax = interval
    N = size(A_op, 1)
    
    # Validate operators
    if size(A_op) != size(B_op) || size(A_op, 1) != size(A_op, 2)
        throw(DimensionMismatch("A_op and B_op must be square and same size"))
    end
    
    # Initialize Feast parameters
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[3] = round(Int, -log10(tol))  # Set tolerance
        fpm[4] = maxiter  # Set max iterations
    end
    
    # Allocate workspace if not provided
    if workspace === nothing
        workspace = allocate_matfree_workspace(T, N, M0)
    end
    
    work = workspace.work
    workc = workspace.workc
    Aq = workspace.Aq
    Sq = workspace.Sq
    lambda = workspace.lambda
    q = workspace.q
    res = workspace.res
    
    # Initialize RCI variables
    ijob = Ref(-1)  # Initialize
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
    # Matrix-free RCI loop
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, work, workc, Aq, Sq, fpm,
                   epsout, loop, Emin, Emax, M0, lambda, q, mode, res, info)
        
        if ijob[] == Int(Feast_RCI_DONE)
            break
        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            # User should prepare linear solver for (Ze[]*B - A)
            if linear_solver === nothing
                throw(ArgumentError("Linear solver callback required for matrix-free operation"))
            end
            continue
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems: (Ze[]*B - A) * X = work
            # Result should be stored in workc
            try
                linear_solver(workc, Ze[], work)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q, store result in work
            if mode[] == 1
                # Compute A * q[:, 1:M] where M = mode[]
                M = mode[]  # This will be updated by RCI to actual number found
                for j in 1:M
                    mul!(view(work, :, j), A_op, view(q, :, j))
                end
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_B) 
            # Compute B * q, store result in work
            if mode[] >= 1
                M = mode[]
                for j in 1:M  
                    mul!(view(work, :, j), B_op, view(q, :, j))
                end
            end
            
        else
            # Unknown RCI code
            throw(ArgumentError("Unknown Feast RCI code: $(ijob[])"))
        end
    end
    
    # Return results
    M_found = mode[]
    return FeastResult(
        lambda = lambda[1:M_found],
        q = q[:, 1:M_found],
        res = res[1:M_found],
        M = M_found,
        epsout = epsout[],
        loop = loop[],
        info = info[]
    )
end

"""
    feast_matfree_grci!(A_op, B_op, center, radius, M0; kwargs...)

Matrix-free Feast RCI for general (non-Hermitian) eigenvalue problems.
"""
function feast_matfree_grci!(A_op::MatrixFreeOperator{Complex{T}},
                            B_op::MatrixFreeOperator{Complex{T}},
                            center::Complex{T}, radius::T, M0::Int;
                            fpm::Union{Vector{Int}, Nothing} = nothing,
                            linear_solver::Union{Function, Nothing} = nothing,
                            workspace::Union{NamedTuple, Nothing} = nothing,
                            maxiter::Int = 20,
                            tol::T = 1e-12) where T<:Real
    
    N = size(A_op, 1)
    
    # Validate operators
    if size(A_op) != size(B_op) || size(A_op, 1) != size(A_op, 2)
        throw(DimensionMismatch("A_op and B_op must be square and same size"))
    end
    
    # Initialize Feast parameters
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[3] = round(Int, -log10(tol))
        fpm[4] = maxiter
    end
    
    # Allocate workspace if not provided
    if workspace === nothing
        workspace = allocate_matfree_workspace(Complex{T}, N, M0)
    end
    
    work = workspace.work
    workc = workspace.workc
    zAq = workspace.zAq
    zSq = workspace.zSq
    lambda = workspace.lambda
    q = workspace.q
    res = workspace.res
    
    # Initialize RCI variables
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
    # Matrix-free RCI loop for general problems
    while true
        feast_grci!(ijob, N, Ze, work, workc, zAq, zSq, fpm,
                   epsout, loop, center, radius, M0, lambda, q, mode, res, info)
        
        if ijob[] == Int(Feast_RCI_DONE)
            break
        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            if linear_solver === nothing
                throw(ArgumentError("Linear solver callback required"))
            end
            continue
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            try
                linear_solver(workc, Ze[], work)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break  
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            M = mode[]
            for j in 1:M
                mul!(view(work, :, j), A_op, view(q, :, j))
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_B)
            M = mode[]
            for j in 1:M
                mul!(view(work, :, j), B_op, view(q, :, j))
            end
            
        else
            throw(ArgumentError("Unknown Feast RCI code: $(ijob[])"))
        end
    end
    
    M_found = mode[]
    return FeastResult(
        lambda = lambda[1:M_found],
        q = q[:, 1:M_found], 
        res = res[1:M_found],
        M = M_found,
        epsout = epsout[],
        loop = loop[],
        info = info[]
    )
end

"""
    allocate_matfree_workspace(T, N, M0)

Allocate workspace arrays for matrix-free Feast operations.
"""
function allocate_matfree_workspace(::Type{T}, N::Int, M0::Int) where T
    if T <: Real
        return (
            work = zeros(T, N, M0),
            workc = zeros(Complex{T}, N, M0), 
            Aq = zeros(T, M0, M0),
            Sq = zeros(T, M0, M0),
            lambda = zeros(T, M0),
            q = zeros(T, N, M0),
            res = zeros(T, M0)
        )
    else # Complex
        RT = real(T)
        return (
            work = zeros(RT, N, M0),
            workc = zeros(T, N, M0),
            zAq = zeros(T, M0, M0),
            zSq = zeros(T, M0, M0), 
            lambda = zeros(T, M0),
            q = zeros(T, N, M0),
            res = zeros(RT, M0)
        )
    end
end

# High-level matrix-free Feast interfaces

"""
    feast(A_op, B_op, interval; kwargs...)

High-level matrix-free Feast interface for symmetric/Hermitian problems.

# Arguments
- `A_op`: Matrix-free operator for A
- `B_op`: Matrix-free operator for B
- `interval`: Search interval (Emin, Emax) for real problems

# Keyword Arguments
- `M0`: Maximum number of eigenvalues (default: 10)
- `solver`: Linear solver (:gmres, :bicgstab, :cg, or custom function)
- `solver_opts`: Options for iterative solver
- `fpm`: Feast parameters
- `tol`: Convergence tolerance
- `maxiter`: Maximum refinement iterations

# Returns
- `FeastResult` with eigenvalues and eigenvectors
"""
function feast(A_op::MatrixFreeOperator{T}, B_op::MatrixFreeOperator{T},
               interval::Tuple{T,T};
               M0::Int = 10,
               solver::Union{Symbol, Function} = :gmres,
               solver_opts::NamedTuple = NamedTuple(),
               fpm::Union{Vector{Int}, Nothing} = nothing,
               tol::T = T(1e-12),
               maxiter::Int = 20) where T<:Real
    
    # Validate operators are compatible
    if !issymmetric(A_op) && !ishermitian(A_op)
        throw(ArgumentError("A_op must be symmetric or Hermitian for this interface"))
    end
    
    # Create linear solver if needed
    linear_solver = if isa(solver, Function)
        solver
    else
        create_iterative_solver(A_op, B_op, solver; solver_opts...)
    end
    
    # Call matrix-free RCI
    return feast_matfree_srci!(A_op, B_op, interval, M0;
                              linear_solver=linear_solver,
                              fpm=fpm, tol=tol, maxiter=maxiter)
end

"""
    feast(A_op, interval; kwargs...)

Matrix-free Feast for standard eigenvalue problems (B = I).
"""
function feast(A_op::MatrixFreeOperator{T}, interval::Tuple{T,T}; kwargs...) where T<:Real
    # Create identity operator
    N = size(A_op, 1)
    B_op = LinearOperator{T}((y, x) -> copy!(y, x), (N, N), 
                           issymmetric=true, ishermitian=true, isposdef=true)
    
    return feast(A_op, B_op, interval; kwargs...)
end

"""
    feast_general(A_op, B_op, center, radius; kwargs...)

Matrix-free Feast for general (non-Hermitian) eigenvalue problems.
"""
function feast_general(A_op::MatrixFreeOperator{Complex{T}}, 
                      B_op::MatrixFreeOperator{Complex{T}},
                      center::Complex{T}, radius::T; 
                      M0::Int = 10,
                      solver::Union{Symbol, Function} = :gmres,
                      solver_opts::NamedTuple = NamedTuple(),
                      fpm::Union{Vector{Int}, Nothing} = nothing,
                      tol::T = T(1e-12),
                      maxiter::Int = 20) where T<:Real
    
    # Create linear solver
    linear_solver = if isa(solver, Function)
        solver
    else
        create_iterative_solver(A_op, B_op, solver; solver_opts...)
    end
    
    # Call matrix-free RCI for general problems
    return feast_matfree_grci!(A_op, B_op, center, radius, M0;
                              linear_solver=linear_solver,
                              fpm=fpm, tol=tol, maxiter=maxiter)
end

"""
    feast_polynomial(coeffs_ops, center, radius; kwargs...)

Matrix-free Feast for polynomial eigenvalue problems.
P(λ) = coeffs_ops[1] + λ*coeffs_ops[2] + λ²*coeffs_ops[3] + ...
"""
function feast_polynomial(coeffs_ops::Vector{<:MatrixFreeOperator{Complex{T}}},
                         center::Complex{T}, radius::T;
                         M0::Int = 10,
                         solver::Union{Symbol, Function} = :gmres,
                         kwargs...) where T<:Real
    
    d = length(coeffs_ops) - 1  # Polynomial degree
    N = size(coeffs_ops[1], 1)
    
    # Linearize polynomial eigenvalue problem
    # Convert P(λ)x = 0 to generalized eigenvalue problem (A - λB)y = 0
    
    # Create companion matrix operators
    function A_companion_mul!(y, x)
        # Implementation of companion matrix multiplication
        # This is a simplified version - full implementation would be more complex
        n = N
        
        # First block row: -coeffs[1] * x[1:n] + coeffs[end] * x[(d-1)*n+1:d*n]
        mul!(view(y, 1:n), coeffs_ops[1], view(x, 1:n))
        y[1:n] .*= -1
        if d > 0
            temp = similar(view(y, 1:n))
            mul!(temp, coeffs_ops[end], view(x, (d-1)*n+1:d*n))
            y[1:n] .+= temp
        end
        
        # Identity blocks for other rows
        for i in 1:d-1
            y[i*n+1:(i+1)*n] .= view(x, (i-1)*n+1:i*n)
        end
    end
    
    function B_companion_mul!(y, x)
        # B matrix for companion form
        fill!(y, 0)
        
        # Last coefficient block in first row
        if d > 1
            mul!(view(y, 1:N), coeffs_ops[end], view(x, (d-1)*N+1:d*N))
        end
        
        # Identity blocks
        for i in 1:d-1
            mul!(view(y, i*N+1:(i+1)*N), coeffs_ops[i+1], view(x, (i-1)*N+1:i*N))
        end
    end
    
    # Create linearized operators
    companion_size = (d * N, d * N)
    A_comp = LinearOperator{Complex{T}}(A_companion_mul!, companion_size)
    B_comp = LinearOperator{Complex{T}}(B_companion_mul!, companion_size)
    
    # Solve linearized problem
    result = feast_general(A_comp, B_comp, center, radius; M0=M0, solver=solver, kwargs...)
    
    # Extract original eigenvectors (first N components)
    if result.M > 0
        q_original = result.q[1:N, :]
        return FeastResult(
            lambda = result.lambda,
            q = q_original,
            res = result.res,
            M = result.M,
            epsout = result.epsout,
            loop = result.loop,
            info = result.info
        )
    else
        return result
    end
end

# Convenience functions for common linear solvers
"""
    create_iterative_solver(A_op, B_op, solver_type=:gmres; kwargs...)

Create iterative linear solver for matrix-free Feast.

# Arguments
- `A_op`, `B_op`: Matrix-free operators
- `solver_type`: `:gmres`, `:bicgstab`, `:cg`, etc.
- `kwargs`: Options passed to iterative solver

# Returns
- Function `(Y, z, X) -> solve (z*B - A) * Y = X`
"""
function create_iterative_solver(A_op::MatrixFreeOperator{T}, 
                                B_op::MatrixFreeOperator{T},
                                solver_type::Symbol = :gmres;
                                rtol::Float64 = 1e-6,
                                maxiter::Int = 1000,
                                restart::Int = 30,
                                preconditioner = nothing) where T
    
    N = size(A_op, 1)
    
    function linear_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
        # Create shifted operator: (z*B - A)
        function shifted_mul!(y, x)
            # y = (z*B - A) * x = z*(B*x) - A*x
            temp = similar(x)
            mul!(temp, B_op, x)
            temp .*= z
            mul!(y, A_op, x)
            y .= temp .- y
        end
        
        shifted_op = LinearOperator{T}(shifted_mul!, (N, N))
        
        M0 = size(X, 2)
        for j in 1:M0
            if solver_type == :gmres
                Y[:, j], info = gmres!(view(Y, :, j), shifted_op, view(X, :, j),
                                      restart=restart, rtol=rtol, maxiter=maxiter)
            elseif solver_type == :bicgstab
                Y[:, j], info = bicgstabl!(view(Y, :, j), shifted_op, view(X, :, j),
                                          l=2, rtol=rtol, maxiter=maxiter)
            elseif solver_type == :cg && (issymmetric(A_op) && isposdef(shifted_op))
                Y[:, j], info = cg!(view(Y, :, j), shifted_op, view(X, :, j),
                                   rtol=rtol, maxiter=maxiter)
            else
                throw(ArgumentError("Unsupported solver type: $solver_type"))
            end
            
            if !info.isconverged
                @warn "Linear solver did not converge for column $j"
            end
        end
    end
    
    return linear_solver
end

"""
    create_direct_solver(A_op, B_op; factorization=:lu)

Create direct linear solver using sparse factorization.
Only works if operators can be converted to sparse matrices.
"""
function create_direct_solver(A_op::MatrixFreeOperator{T}, 
                             B_op::MatrixFreeOperator{T};
                             factorization::Symbol = :lu) where T
    
    # This requires the operators to support conversion to sparse matrices
    # Implementation would depend on specific operator types
    throw(ArgumentError("Direct solver for general matrix-free operators not implemented. " *
                       "Use create_iterative_solver instead."))
end