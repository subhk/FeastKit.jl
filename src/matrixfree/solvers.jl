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
- `preconditioner`: Optional left inverse-action operator, applied with
  `mul!(y, preconditioner, x)` on complex vectors (default: no preconditioning).

The inner stopping criterion is relative to the initial residual (`atol=0`),
so rescaling both matrices does not turn small nonzero right-hand sides into
accepted zero solutions.

# Returns
- Function `(Y, z, X) -> solve (z*B - A) * Y = X`
"""
function create_iterative_solver(A_op::MatrixFreeOperator{T},
                                B_op::MatrixFreeOperator{T},
                                solver_type::Symbol = :gmres;
                                rtol::Float64 = 1e-6,
                                maxiter::Int = 1000,
                                restart::Int = 30,
                                preconditioner = nothing,
                                tolerance_ref = nothing) where T

    FEAST_KRYLOV_AVAILABLE[] ||
        throw(ArgumentError("create_iterative_solver needs Krylov.jl. Run `using Krylov` to load the FeastKitKrylovExt extension."))

    # Validate before any zero-RHS shortcut; unsupported solvers must never
    # appear to work just because a particular trial column is zero.
    if solver_type == :cg
        throw(ArgumentError("CG solver cannot be used with FEAST: " *
                            "the shifted system (z*B - A) is not SPD for complex z. " *
                            "Use :gmres or :bicgstab instead."))
    elseif solver_type ∉ (:gmres, :bicgstab)
        throw(ArgumentError("Unsupported solver type: $solver_type. " *
                            "Use :gmres or :bicgstab"))
    end

    N = size(A_op, 1)
    CT = T <: Real ? Complex{T} : T
    RT = typeof(real(zero(CT)))
    rtol_value = RT(rtol)
    rhs_scale = Ref(one(RT))
    current_shift = Ref(zero(CT))
    temp = Vector{CT}(undef, N)
    temp_A = Vector{CT}(undef, N)
    xj = Vector{CT}(undef, N)
    gmres_workspace = _feast_gmres_workspace(N, CT; memory=max(restart, 2))
    bicgstab_workspace = solver_type === :bicgstab ? _feast_bicgstab_workspace(N, CT) : nothing

    function shifted_mul!(y, x)
        # y = (z*B - A) * x
        z = current_shift[]
        mul!(temp, B_op, x)
        @. temp = z * temp
        mul!(temp_A, A_op, x)
        @. y = (temp - temp_A) / rhs_scale[]
        return y
    end

    shifted_op = LinearOperator{CT}(shifted_mul!, (N, N))
    # Scaling K and b by 1/s leaves K*y=b unchanged. The inverse action
    # for K/s is s*P, keeping preconditioned systems on the same scale too.
    scaled_preconditioner = if preconditioner === nothing
        nothing
    else
        LinearOperator{CT}((y,x) -> begin
            mul!(y, preconditioner, x)
            y .*= rhs_scale[]
            y
        end, (N,N))
    end

    function linear_solver(Y::AbstractMatrix, z::Number, X::AbstractMatrix)
        # FEAST contour shifts are complex; keep the Krylov operator and
        # conversion scratch alive across contour points.
        current_shift[] = CT(z)
        inner_tol = tolerance_ref === nothing ? rtol_value : RT(tolerance_ref[])
        M0 = size(X, 2)
        for j in 1:M0
            # Convert RHS to complex since z is complex (Krylov needs matching types)
            @inbounds for i in 1:N
                xj[i] = CT(X[i, j])
            end
            # Besides relative stopping, Krylov has absolute breakdown tests.
            # Normalize both sides to avoid accepting/discarding tiny RHSs
            # solely because the entire pencil was rescaled.
            rhs_scale[] = norm(xj)
            if iszero(rhs_scale[])
                fill!(view(Y, :, j), zero(CT))
                continue
            end
            xj ./= rhs_scale[]
            if solver_type == :gmres
                converged = _feast_gmres!(gmres_workspace, shifted_op, xj;
                                          restart=true,
                                          rtol=inner_tol,
                                          atol=zero(RT),
                                          preconditioner=scaled_preconditioner,
                                          itmax=maxiter)
                copyto!(view(Y, :, j), _feast_gmres_solution(gmres_workspace))
            elseif solver_type == :bicgstab
                converged = _feast_bicgstab!(bicgstab_workspace, shifted_op, xj;
                                                    rtol=inner_tol, atol=zero(RT),
                                                    preconditioner=scaled_preconditioner,
                                                    itmax=maxiter)
                result = _feast_bicgstab_solution(bicgstab_workspace)
                if all(isfinite, result)
                    copyto!(view(Y, :, j), result)
                else
                    # BiCGSTAB can break down after an exact first step
                    # (including with an exact inverse preconditioner),
                    # forming 0/0 in its stabilization step. Retry that RHS
                    # with GMRES rather than feeding NaNs to the projector.
                    converged = _feast_gmres!(gmres_workspace, shifted_op, xj;
                                              restart=true, rtol=inner_tol, atol=zero(RT),
                                              preconditioner=scaled_preconditioner,
                                              itmax=maxiter)
                    copyto!(view(Y, :, j), _feast_gmres_solution(gmres_workspace))
                end
            end

            if !converged
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
