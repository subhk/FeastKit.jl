# Matrix-Free Feast Interface
# Provides callbacks for matrix-vector operations instead of explicit matrices

using LinearAlgebra
using SparseArrays
using Krylov

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
Base.eltype(::MatrixFreeOperator{T}) where T = T
Base.eltype(::Type{<:MatrixFreeOperator{T}}) where T = T

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
            # mode[] contains M = number of eigenvalues found
            M = mode[]
            if M >= 1
                for j in 1:M
                    mul!(view(work, :, j), A_op, view(q, :, j))
                end
            end
            
        # Note: Feast_RCI_MULT_B is NOT issued by feast_srci! for symmetric problems.
        # The B matrix is handled implicitly through the linear solver callback
        # (z*B - A)^{-1}. The residual uses ||Aq - λq|| which assumes B=I or
        # that the eigenvalue problem has been transformed appropriately.

        else
            # Unknown RCI code
            throw(ArgumentError("Unknown Feast RCI code: $(ijob[])"))
        end
    end
    
    # Return results
    M_found = mode[]
    return FeastResult{T, T}(
        lambda[1:M_found],
        q[:, 1:M_found],
        M_found,
        res[1:M_found],
        info[],
        epsout[],
        loop[]
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
            # For general problems, workc contains Q0 (the RHS) and result goes back to workc
            # Need a temporary to avoid overwriting input before reading it
            try
                rhs_copy = copy(workc)
                linear_solver(workc, Ze[], rhs_copy)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q, store result in workc (complex for general problems)
            M = mode[]
            for j in 1:M
                mul!(view(workc, :, j), A_op, view(q, :, j))
            end

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            # Compute B * q, store result in workc (complex for general problems)
            M = mode[]
            for j in 1:M
                mul!(view(workc, :, j), B_op, view(q, :, j))
            end
            
        else
            throw(ArgumentError("Unknown Feast RCI code: $(ijob[])"))
        end
    end
    
    M_found = mode[]
    # Note: For general eigenvalue problems, eigenvalues are complex
    # but the lambda vector stores Complex{T}, so we extract real part for FeastResult
    lambda_real = real.(lambda[1:M_found])
    return FeastResult{T, Complex{T}}(
        lambda_real,
        q[:, 1:M_found],
        M_found,
        res[1:M_found],
        info[],
        epsout[],
        loop[]
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
- `solver`: Linear solver (:gmres, :bicgstab, or custom function)
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

Solves the polynomial eigenvalue problem P(λ)x = 0 where:
P(λ) = coeffs_ops[1] + λ*coeffs_ops[2] + λ²*coeffs_ops[3] + ... + λᵈ*coeffs_ops[d+1]

The polynomial is linearized using companion matrices to form a generalized 
eigenvalue problem (A - λB)y = 0 of size (d*N × d*N), where y = [x; λx; λ²x; ...; λᵈ⁻¹x].

# Arguments
- `coeffs_ops`: Vector of matrix operators [C₀, C₁, C₂, ..., Cᵈ] where P(λ) = Σᵢ λⁱCᵢ
- `center`: Center of circular search region in complex plane
- `radius`: Radius of circular search region
- `M0`: Maximum number of eigenvalues to find
- `solver`: Linear solver type or custom function
- `kwargs`: Additional options passed to feast_general

# Returns
- `FeastResult` with eigenvalues λ and corresponding eigenvectors x (first N components of full eigenvector)
"""
function feast_polynomial(coeffs_ops::Vector{<:MatrixFreeOperator{Complex{T}}},
                         center::Complex{T}, radius::T;
                         M0::Int = 10,
                         solver::Union{Symbol, Function} = :gmres,
                         kwargs...) where T<:Real
    
    d = length(coeffs_ops) - 1  # Polynomial degree
    N = size(coeffs_ops[1], 1)
    
    # Validate input
    if d < 1
        throw(ArgumentError("Need at least 2 coefficient operators (degree ≥ 1)"))
    end
    
    for i in 1:length(coeffs_ops)
        if size(coeffs_ops[i]) != (N, N)
            throw(DimensionMismatch("All coefficient operators must have size ($N, $N)"))
        end
    end
    
    # Linearize polynomial eigenvalue problem
    # Convert P(λ)x = 0 to generalized eigenvalue problem (A - λB)y = 0
    
    # Create companion matrix operators
    # For polynomial P(λ) = C_0 + λC_1 + λ²C_2 + ... + λᵈC_d
    # The companion matrix is of size (d*N × d*N) and has the structure:
    #   A = [   0    I    0  ...   0  ]
    #       [   0    0    I  ...   0  ]
    #       [   :    :    :  ⋱    :  ]
    #       [   0    0    0  ...   I  ]
    #       [-C_0  -C_1  -C_2 ... -C_{d-1}]
    
    function A_companion_mul!(y::AbstractVector, x::AbstractVector)
        n = N
        fill!(y, 0)
        
        # Identity blocks in super-diagonal: x[i*n+1:(i+1)*n] -> y[(i-1)*n+1:i*n]
        for i in 1:d-1
            # Block (i-1, i): I_n
            y[(i-1)*n+1:i*n] .= view(x, i*n+1:(i+1)*n)
        end
        
        # Last block row: -C_0*x[1:n] - C_1*x[n+1:2n] - ... - C_{d-1}*x[(d-1)*n+1:d*n]
        # Accumulate directly into output to avoid temporary allocations
        for i in 0:d-1
            # Create a temporary view for the matrix-vector product
            temp_view = view(y, (d-1)*n+1:d*n)
            temp_storage = similar(temp_view)
            mul!(temp_storage, coeffs_ops[i+1], view(x, i*n+1:(i+1)*n))
            temp_view .-= temp_storage
        end
    end
    
    function B_companion_mul!(y::AbstractVector, x::AbstractVector)
        # B matrix for companion linearization has the structure:
        #   B = [ I   0   0  ...  0  ]
        #       [ 0   I   0  ...  0  ]
        #       [ :   :   :  ⋱   :  ]
        #       [ 0   0   0  ...  I  ]
        #       [ 0   0   0  ... C_d ]
        
        n = N
        fill!(y, 0)
        
        # Identity blocks on diagonal for first d-1 block rows
        for i in 0:d-2
            # Block (i, i): I_n
            y[i*n+1:(i+1)*n] .= view(x, i*n+1:(i+1)*n)
        end
        
        # Last block row, last block: C_d * x[(d-1)*n+1:d*n]
        if d >= 1
            mul!(view(y, (d-1)*n+1:d*n), coeffs_ops[end], view(x, (d-1)*n+1:d*n))
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
        return FeastResult{T, Complex{T}}(
            result.lambda,
            q_original,
            result.M,
            result.res,
            result.info,
            result.epsout,
            result.loop
        )
    else
        return result
    end
end

# Convenience functions for common linear solvers
"""
    create_iterative_solver(A_op, B_op, solver_type=:gmres; kwargs...)

Create iterative linear solver for matrix-free Feast using Krylov.jl.

Note: The shifted system `(z*B - A)` has complex `z` (contour integration points),
so the linear solver must handle complex arithmetic.

# Arguments
- `A_op`, `B_op`: Matrix-free operators
- `solver_type`: `:gmres` (default, recommended) or `:bicgstab`
- `rtol`: Relative tolerance for convergence (default: 1e-6)
- `maxiter`: Maximum iterations (default: 1000)
- `restart`: GMRES restart parameter (default: 30)

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
        # Note: z is always complex in FEAST, so output must be complex
        CT = promote_type(T, typeof(z))

        function shifted_mul!(y, x)
            # y = (z*B - A) * x = z*(B*x) - A*x
            # Use complex temporary since z is complex
            temp = zeros(CT, length(x))
            mul!(temp, B_op, x)
            temp .*= z
            temp_A = zeros(CT, length(x))
            mul!(temp_A, A_op, x)
            y .= temp .- temp_A
        end

        shifted_op = LinearOperator{CT}(shifted_mul!, (N, N))

        M0 = size(X, 2)
        for j in 1:M0
            # Convert RHS to complex since z is complex (Krylov needs matching types)
            xj = CT.(X[:, j])
            if solver_type == :gmres
                result, stats = Krylov.gmres(shifted_op, xj, restart=restart,
                                             rtol=rtol, itmax=maxiter)
                Y[:, j] .= result
                converged = stats.solved
            elseif solver_type == :bicgstab
                result, stats = Krylov.bicgstab(shifted_op, xj,
                                                rtol=rtol, itmax=maxiter)
                Y[:, j] .= result
                converged = stats.solved
            elseif solver_type == :cg
                # CG only works for SPD systems - but (z*B - A) is NOT SPD for complex z
                # CG should not be used with FEAST's complex contour points
                throw(ArgumentError("CG solver cannot be used with FEAST: " *
                                   "the shifted system (z*B - A) is not SPD for complex z. " *
                                   "Use :gmres or :bicgstab instead."))
            else
                throw(ArgumentError("Unsupported solver type: $solver_type. " *
                                   "Use :gmres or :bicgstab"))
            end

            if !converged
                @warn "Linear solver did not converge for column $j"
            end
        end
    end

    return linear_solver
end

"""
    validate_companion_matrices(A_companion_mul!, B_companion_mul!, coeffs_ops, test_lambda, test_x)

Validate that the companion matrices correctly linearize the polynomial eigenvalue problem.

Tests that if P(λ)x = 0, then (A - λB)y = 0 where y = [x; λx; λ²x; ...; λᵈ⁻¹x].
"""
function validate_companion_matrices(A_companion_mul!::Function, 
                                   B_companion_mul!::Function,
                                   coeffs_ops::Vector{<:MatrixFreeOperator{Complex{T}}},
                                   test_lambda::Complex{T}, 
                                   test_x::AbstractVector{Complex{T}}) where T<:Real
    
    d = length(coeffs_ops) - 1
    N = length(test_x)
    
    # Construct companion eigenvector: y = [x; λx; λ²x; ...; λᵈ⁻¹x]
    y = zeros(Complex{T}, d * N)
    lambda_power = one(Complex{T})
    for i in 0:d-1
        y[i*N+1:(i+1)*N] .= lambda_power .* test_x
        lambda_power *= test_lambda
    end
    
    # Test (A - λB)y = 0
    Ay = similar(y)
    By = similar(y)
    
    A_companion_mul!(Ay, y)
    B_companion_mul!(By, y)
    
    residual = Ay - test_lambda * By
    residual_norm = norm(residual)
    
    # Also verify that P(λ)x = 0
    Px = zeros(Complex{T}, N)
    temp = similar(test_x)
    lambda_power = one(Complex{T})
    
    for i in 0:d
        mul!(temp, coeffs_ops[i+1], test_x)
        Px .+= lambda_power .* temp
        lambda_power *= test_lambda
    end
    
    polynomial_residual = norm(Px)
    
    return (
        companion_residual = residual_norm,
        polynomial_residual = polynomial_residual,
        companion_valid = residual_norm < 1e-12,
        polynomial_valid = polynomial_residual < 1e-12
    )
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
