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
                            fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                            linear_solver::Union{Function, Nothing} = nothing,
                            workspace::Union{NamedTuple, Nothing} = nothing,
                            maxiter::Int = 20,
                            tol::T = T(1e-12),
                            inner_tolerance=nothing, initial_subspace=nothing) where T<:Real

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

    # Reject a bad N/M0/interval here, the way every other entry point does. The
    # kernel also refuses them, but by reporting an info code rather than
    # throwing, which would make this interface disagree with the rest.
    fpm_vec = _feast_initial_parameters(fpm isa FeastParameters ? fpm.fpm : fpm, initial_subspace)
    fpm_vec[42] == 1 && throw(ArgumentError("mixed_precision requires assembled dense matrices"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm_vec)

    # Allocate workspace if not provided
    if workspace === nothing
        workspace = allocate_matfree_workspace(T, N, M0)
    end

    work = workspace.work
    workc = workspace.workc
    _feast_initial_subspace!(work, initial_subspace)
    mass_rhs = similar(work)
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

    # Persistent RCI state (must be reused across calls in the loop)
    srci_state = FeastSRCIState{T}()

    # Matrix-free RCI loop
    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, work, workc, Aq, Sq, fpm_vec,
                   epsout, loop, Emin, Emax, M0, lambda, q, mode, res, info; state=srci_state)

        if ijob[] == Int(Feast_RCI_DONE)
            break
        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            # User should prepare linear solver for (Ze[]*B - A)
            if linear_solver === nothing
                throw(ArgumentError("Linear solver callback required for matrix-free operation"))
            end
            continue

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if inner_tolerance !== nothing
                final_tol = max(eps(T), T(0.1) * feast_tolerance(fpm_vec, T))
                inner_tolerance[] = _feast_inner_tol(true, final_tol, epsout[], loop[])
            end
            # Solve (Ze[]*B - A) * Y = B*Q and store Y in workc.
            try
                # The kernel supplies Q; generalized contour projection needs B*Q.
                for j in axes(work, 2)
                    mul!(view(mass_rhs,:,j), B_op, view(work,:,j))
                end
                linear_solver(workc, Ze[], mass_rhs)
            catch err
                @debug "Matrix-free linear solver callback failed" exception=err
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q, store result in work.
            # mode[] is how many columns of q the kernel wants multiplied.
            M = mode[]
            for j in 1:M
                mul!(view(work, :, j), A_op, view(q, :, j))
            end

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            # Compute B * q, store result in work. The kernel needs this both to
            # build the reduced mass matrix and to form the generalized residual
            # ||A q - lambda B q||.
            M = mode[]
            for j in 1:M
                mul!(view(work, :, j), B_op, view(q, :, j))
            end

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

The user-provided `linear_solver` must solve `(Ze * B - A) * Y = X` for the
current contour point `Ze`. The RCI kernel decides which operation is needed;
this wrapper only supplies matrix-vector products and shifted solves through
the matrix-free operators.
"""
function feast_matfree_grci!(A_op::MatrixFreeOperator{Complex{T}},
                            B_op::MatrixFreeOperator{Complex{T}},
                            center::Complex{T}, radius::T, M0::Int;
                            fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                            linear_solver::Union{Function, Nothing} = nothing,
                            workspace::Union{NamedTuple, Nothing} = nothing,
                            maxiter::Int = 20,
                            tol::T = T(1e-12),
                            inner_tolerance=nothing, initial_subspace=nothing) where T<:Real

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

    # Same up-front validation as the Hermitian matrix-free entry point.
    fpm_vec = _feast_initial_parameters(fpm isa FeastParameters ? fpm.fpm : fpm, initial_subspace)
    fpm_vec[42] == 1 && throw(ArgumentError("mixed_precision requires assembled dense matrices"))
    check_feast_grci_input(N, M0, center, radius, fpm_vec)

    # Allocate workspace if not provided
    if workspace === nothing
        workspace = allocate_matfree_workspace(Complex{T}, N, M0)
    end

    work = workspace.work
    workc = workspace.workc
    _feast_initial_subspace!(workc, initial_subspace)
    # `workc` is both the input RHS and output solution for solve requests.
    # Keep a separate RHS buffer so callbacks can overwrite `workc` safely.
    rhs = hasproperty(workspace, :rhs) ? workspace.rhs : similar(workc)
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

    # Persistent RCI state (must be reused across calls in the loop)
    grci_state = FeastGRCIState{T}()

    # Matrix-free RCI loop for general problems. `ijob` tells us which user
    # operation the core FEAST state machine needs next.
    while true
        feast_grci!(ijob, N, Ze, work, workc, zAq, zSq, fpm_vec,
                   epsout, loop, center, radius, M0, lambda, q, mode, res, info; state=grci_state)

        if ijob[] == Int(Feast_RCI_DONE)
            break
        elseif ijob[] == Int(Feast_RCI_FACTORIZE)
            if linear_solver === nothing
                throw(ArgumentError("Linear solver callback required"))
            end
            continue

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if inner_tolerance !== nothing
                final_tol = max(eps(T), T(0.1) * feast_tolerance(fpm_vec, T))
                inner_tolerance[] = _feast_inner_tol(true, final_tol, epsout[], loop[])
            end
            # workc contains Q0. Form B*Q0 in separate scratch before the
            # callback overwrites workc with the shifted solution.
            try
                for j in axes(workc, 2)
                    mul!(view(rhs,:,j), B_op, view(workc,:,j))
                end
                linear_solver(workc, Ze[], rhs)
            catch e
                @debug "Matrix-free linear solver callback failed" exception=e
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
    return FeastGeneralResult{T}(
        lambda[1:M_found],
        q[:, 1:M_found],
        M_found,
        res[1:M_found],
        info[],
        epsout[],
        loop[]
    )
end
