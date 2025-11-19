# Feast sparse matrix routines
# Translated from dzfeast_sparse.f90 and related files

using SparseArrays
using LinearAlgebra

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

# Matrix-free shifted operator for (z*B - A) with matvec functions
struct MatrixFreeShiftedOperator{T<:Real,CT<:Complex}
    N::Int
    z::CT
    A_matvec!::Function
    B_matvec!::Function
    temp_real::Vector{T}
    temp_imag::Vector{T}
end

Base.size(op::MatrixFreeShiftedOperator) = (op.N, op.N)
Base.eltype(::MatrixFreeShiftedOperator{T,CT}) where {T,CT} = CT

function LinearAlgebra.mul!(y::AbstractVector{CT}, op::MatrixFreeShiftedOperator{T,CT}, x::AbstractVector{CT}) where {T<:Real,CT<:Complex}
    # Split complex vectors into real and imaginary parts
    x_real = real.(x)
    x_imag = imag.(x)

    # Compute A*x_real and A*x_imag
    op.A_matvec!(op.temp_real, x_real)
    y_A_real = copy(op.temp_real)
    op.A_matvec!(op.temp_imag, x_imag)
    y_A_imag = copy(op.temp_imag)

    # Compute B*x_real and B*x_imag
    op.B_matvec!(op.temp_real, x_real)
    y_B_real = copy(op.temp_real)
    op.B_matvec!(op.temp_imag, x_imag)
    y_B_imag = copy(op.temp_imag)

    # y = z*B*x - A*x = (z_real + i*z_imag)*(B*x) - A*x
    z_real = real(op.z)
    z_imag = imag(op.z)

    y_real = z_real * y_B_real - z_imag * y_B_imag - y_A_real
    y_imag = z_real * y_B_imag + z_imag * y_B_real - y_A_imag

    y .= complex.(y_real, y_imag)
    return y
end

@inline function _check_complex_symmetric(A::SparseMatrixCSC)
    issymmetric(A) || throw(ArgumentError("Matrix must be complex symmetric (equal to its transpose)"))
end

@inline function _convert_sparse_complex(A::SparseMatrixCSC{T,Int}, ::Type{Complex{T}}) where T<:Real
    return SparseMatrixCSC{Complex{T},Int}(A.m, A.n, copy(A.colptr), copy(A.rowval),
                                           Complex{T}.(A.nzval))
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
    fallback_solver = nothing

    for j in 1:ncols
        b = view(rhs, :, j)
        x_initial = zeros(CT, N)
        x_sol, stats = gmres(op, b, x_initial;
                             restart=true,
                             memory=max(gmres_restart, 2),
                             rtol=tol,
                             atol=tol,
                             itmax=maxiter)
        if stats.solved
            dest[:, j] .= x_sol
        else
            @info "GMRES column failed" column=j residuals=stats.residuals
            if fallback_solver === nothing
                # GMRES failed for this shift; build a sparse direct solver
                # Convert to complex if needed
                complex_B = eltype(B) <: Complex ? B : Complex{real(eltype(B))}.(B)
                complex_A = eltype(A) <: Complex ? A : Complex{real(eltype(A))}.(A)
                shifted_matrix = z .* complex_B .- complex_A
                try
                    fallback_solver = lu(shifted_matrix)
                catch err
                    @warn "Sparse fallback factorization failed for shift $z" exception=err
                    return false
                end
            end
            dest[:, j] .= fallback_solver \ b
        end
    end

    return true
end

function feast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver::Symbol = :direct,
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    # Feast for sparse real symmetric generalized eigenvalue problem in CSR format
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
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct or :gmres."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    if solver_is_iterative && solver_tol > 0
        solver_digits = max(1, ceil(Int, -log10(solver_tol)))
        relaxed_digits = max(2, solver_digits - 2)
        fpm[3] = min(fpm[3], relaxed_digits)
        fpm[5] = max(fpm[5], 1)
    end

    # Initialize workspace
    workspace = FeastWorkspaceReal{T}(N, M0)
    if fpm[5] == 0
        _feast_seeded_subspace!(workspace.work)
        fpm[5] = 1
    end
    
    # Initialize variables for RCI
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
    # Sparse linear solver / iterative workspace
    sparse_solver = nothing
    current_shift = Ref(zero(Complex{T}))
    rhs_iterative = solver_is_iterative ? zeros(Complex{T}, N, M0) : nothing
    
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.Aq, workspace.Sq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*B - A for sparse matrices
            if solver_is_direct
                z = Ze[]
                sparse_matrix = z * B - A

                # LU factorization for sparse matrix
                try
                    sparse_solver = lu(sparse_matrix)
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = Ze[]
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve sparse linear systems: (Ze*B - A) * X = B * workspace.work
            rhs = B * workspace.work[:, 1:M0]

            if solver_is_direct
                try
                    # Solve with sparse LU factors
                    workspace.workc[:, 1:M0] .= sparse_solver \ rhs
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                rhs_iterative .= rhs
                success = solve_shifted_iterative!(workspace.workc[:, 1:M0],
                                                   rhs_iterative, A, B,
                                                   current_shift[], tol_value,
                                                   solver_maxiter, solver_restart)
                if !success
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
    # Feast for sparse complex Hermitian eigenvalue problem
    # Solves: A*q = lambda*q where A is Hermitian
    
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver in (:direct, :gmres, :iterative) ? solver : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver option '$solver'. Use :direct or :gmres."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative FEAST solves. Please ensure it is in the environment."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)
    identity_sparse = solver_is_iterative ? sparse(I, N, N) : nothing
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
            if solver_is_direct
                I_sparse = sparse(I, N, N)
                sparse_matrix = z * I_sparse - A

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
            # Solve sparse linear systems
            if solver_is_direct
                try
                    workspace.workc[:, 1:M0] .= sparse_solver \ workspace.workc[:, 1:M0]
                catch e
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                success = solve_shifted_iterative!(workspace.workc[:, 1:M0], workspace.workc[:, 1:M0],
                                                   A, identity_sparse, current_shift[],
                                                   tol_value, solver_maxiter, solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            Aq = A * workspace.q[:, 1:M]
            workspace.workc[:, 1:M] .= Aq
            workspace.work[:, 1:M] .= real.(Aq)

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
    # Feast for sparse complex Hermitian generalized eigenvalue problems
    # Solves: A*q = lambda*B*q where A and B are Hermitian (B positive definite)

    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("A must be square"))
    size(B) == (N, N) || throw(ArgumentError("B must match the size of A"))
    ishermitian(A) || throw(ArgumentError("A must be Hermitian for feast_hcsrgv!"))
    ishermitian(B) || throw(ArgumentError("B must be Hermitian positive definite for feast_hcsrgv!"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    check_feast_srci_input(N, M0, Emin, Emax, fpm)

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

    workspace = FeastWorkspaceComplex{T}(N, M0)
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)

    sparse_solver = nothing

    while true
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.zAq, workspace.zSq, fpm, epsout, loop,
                    Emin, Emax, M0, workspace.lambda, workspace.q,
                    mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            z = Ze[]
            if solver_is_direct
                shifted_matrix = z * B - A
                try
                    sparse_solver = lu(shifted_matrix)
                catch
                    info[] = Int(Feast_ERROR_LAPACK)
                    break
                end
            else
                current_shift[] = z
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            rhs = B * workspace.workc[:, 1:M0]
            if solver_is_direct
                try
                    workspace.workc[:, 1:M0] .= sparse_solver \ rhs
                catch
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

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            M = mode[]
            Aq = A * workspace.q[:, 1:M]
            workspace.workc[:, 1:M] .= Aq
            workspace.work[:, 1:M] .= real.(Aq)

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end

    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]

    return FeastResult{T, Complex{T}}(lambda, q, M, res, info[], epsout[], loop[])
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

            # Create matrix-free operator for (z*B - A)
            temp_real = zeros(T, N)
            temp_imag = zeros(T, N)
            op = MatrixFreeShiftedOperator(N, z, A_matvec!, B_matvec!, temp_real, temp_imag)

            # Solve each system using GMRES
            for j in 1:M0
                # Convert RHS to complex
                b = complex.(rhs[:, j])

                # Initial guess (zero)
                x0 = zeros(Complex{T}, N)

                # Solve using GMRES (Krylov.jl if available, otherwise fallback)
                if FEAST_KRYLOV_AVAILABLE[]
                    # Use GMRES with our properly-typed operator
                    
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
