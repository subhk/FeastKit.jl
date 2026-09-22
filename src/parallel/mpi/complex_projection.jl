function _mpi_solver_choice(solver::Symbol)
    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice in (:direct, :gmres) ||
        throw(ArgumentError("Unsupported MPI FEAST solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_choice == :gmres && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative MPI FEAST solves. Run `using Krylov` to load the FeastKitKrylovExt extension."))
    return solver_choice
end

function _mpi_success_count(local_success::Bool, comm::MPI.Comm)
    flag = [local_success ? 1 : 0]
    return MPI.Allreduce(flag, MPI.SUM, comm)[1]
end

# Inner solves need headroom below the requested outer eigenpair residual.
# Use the same precision-aware outer target as the driver, and never ask a
# default inner solve for accuracy below machine epsilon. Explicit tolerances
# remain a caller-controlled override.
function _mpi_solver_tolerance(fpm, ::Type{T}, solver_tol) where T<:AbstractFloat
    return solver_tol == 0 ? max(T(0.01)*feast_tolerance(fpm,T),eps(T)) : T(solver_tol)
end

function _mpi_distribute_complex_contour!(mpi_state::MPIFeastState{T},
                                          Zne_global::Vector{Complex{T}},
                                          Wne_global::Vector{Complex{T}}) where T<:Real
    for (i, global_idx) in enumerate(mpi_state.local_points)
        mpi_state.local_Zne[i] = Zne_global[global_idx]
        mpi_state.local_Wne[i] = Wne_global[global_idx]
    end
    return mpi_state
end

function mpi_compute_complex_hermitian_moments(
    A::SparseMatrixCSC{Complex{T},Int},
    B::SparseMatrixCSC{Complex{T},Int},
    Q_basis::Matrix{Complex{T}},
    local_Zne::Vector{Complex{T}},
    local_Wne::Vector{Complex{T}},
    M0::Int,
    solver_choice::Symbol,
    tol::T,
    maxiter::Int,
    restart::Int,
    comm::MPI.Comm; workspace=nothing,
) where T<:Real
    N = size(A, 1)
    zAq_local = zeros(Complex{T}, M0, M0)
    zSq_local = zeros(Complex{T}, M0, M0)
    Q_proj_local = zeros(Complex{T}, N, M0)
    rhs = Matrix{Complex{T}}(undef, N, M0)
    solutions = similar(rhs)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    local_success = true

    mul!(rhs, B, Q_basis)
    for e in eachindex(local_Zne)
        z = local_Zne[e]
        weight = local_Wne[e]
        try
            if solver_choice == :direct
                F = lu(z * B - A)
                ldiv!(solutions, F, rhs)
            else
                local_success = solve_shifted_iterative!(solutions, rhs, A, B, z,
                                                         tol, maxiter, restart; workspace=workspace)
                local_success || break
            end
        catch err
            @warn "MPI rank $(MPI.Comm_rank(comm)): complex Hermitian solve failed for contour point $e" exception=err
            local_success = false
            break
        end

        mul!(moment, adjoint(Q_basis), solutions)
        @. zAq_local += weight * moment
        @. zSq_local += (weight * z) * moment
        @. Q_proj_local += weight * solutions
    end

    return zAq_local, zSq_local, Q_proj_local, local_success
end

function mpi_compute_complex_hermitian_moments(
    A::Matrix{Complex{T}},
    B::Matrix{Complex{T}},
    Q_basis::Matrix{Complex{T}},
    local_Zne::Vector{Complex{T}},
    local_Wne::Vector{Complex{T}},
    M0::Int,
    solver_choice::Symbol,
    tol::T,
    maxiter::Int,
    restart::Int,
    comm::MPI.Comm; workspace=nothing,
) where T<:Real
    N = size(A, 1)
    zAq_local = zeros(Complex{T}, M0, M0)
    zSq_local = zeros(Complex{T}, M0, M0)
    Q_proj_local = zeros(Complex{T}, N, M0)
    rhs = Matrix{Complex{T}}(undef, N, M0)
    solutions = similar(rhs)
    moment = Matrix{Complex{T}}(undef, M0, M0)
    local_success = true

    mul!(rhs, B, Q_basis)
    if solver_choice == :direct
        shifted = Matrix{Complex{T}}(undef, N, N)
        for e in eachindex(local_Zne)
            z = local_Zne[e]
            weight = local_Wne[e]
            try
                @. shifted = z * B - A
                F = lu!(shifted)
                ldiv!(solutions, F, rhs)
            catch err
                @warn "MPI rank $(MPI.Comm_rank(comm)): dense complex Hermitian solve failed for contour point $e" exception=err
                local_success = false
                break
            end

            mul!(moment, adjoint(Q_basis), solutions)
            @. zAq_local += weight * moment
            @. zSq_local += (weight * z) * moment
            @. Q_proj_local += weight * solutions
        end
    else
        rhs_copy = similar(rhs)
        tmpAx = Vector{Complex{T}}(undef, N)
        tmpBx = Vector{Complex{T}}(undef, N)
        current_shift = Ref(zero(Complex{T}))

        function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
            mul!(tmpBx, B, x)
            @. tmpBx = current_shift[] * tmpBx
            mul!(tmpAx, A, x)
            @. y = tmpBx - tmpAx
            return y
        end

        for e in eachindex(local_Zne)
            z = local_Zne[e]
            weight = local_Wne[e]
            try
                current_shift[] = z
                copyto!(rhs_copy, rhs)
                local_success = solve_dense_shifted!(solutions, rhs_copy,
                                                     shifted_mul!, solver_choice,
                                                     tol, maxiter, restart; workspace=workspace)
                local_success || break
            catch err
                @warn "MPI rank $(MPI.Comm_rank(comm)): dense iterative complex Hermitian solve failed for contour point $e" exception=err
                local_success = false
                break
            end

            mul!(moment, adjoint(Q_basis), solutions)
            @. zAq_local += weight * moment
            @. zSq_local += (weight * z) * moment
            @. Q_proj_local += weight * solutions
        end
    end

    return zAq_local, zSq_local, Q_proj_local, local_success
end

function mpi_compute_complex_general_projection(
    A::SparseMatrixCSC{Complex{T},Int},
    B::SparseMatrixCSC{Complex{T},Int},
    Q_basis::Matrix{Complex{T}},
    local_Zne::Vector{Complex{T}},
    local_Wne::Vector{Complex{T}},
    M0::Int,
    solver_choice::Symbol,
    tol::T,
    maxiter::Int,
    restart::Int,
    comm::MPI.Comm; workspace=nothing,
) where T<:Real
    N = size(A, 1)
    Q_proj_local = zeros(Complex{T}, N, M0)
    rhs = Matrix{Complex{T}}(undef, N, M0)
    solutions = similar(rhs)
    local_success = true

    mul!(rhs, B, Q_basis)
    for e in eachindex(local_Zne)
        z = local_Zne[e]
        try
            if solver_choice == :direct
                F = lu(z * B - A)
                ldiv!(solutions, F, rhs)
            else
                local_success = solve_shifted_iterative!(solutions, rhs, A, B, z,
                                                         tol, maxiter, restart; workspace=workspace)
                local_success || break
            end
        catch err
            @warn "MPI rank $(MPI.Comm_rank(comm)): complex general solve failed for contour point $e" exception=err
            local_success = false
            break
        end
        @. Q_proj_local += local_Wne[e] * solutions
    end

    return Q_proj_local, local_success
end

function mpi_compute_complex_general_projection(
    A::Matrix{Complex{T}},
    B::Matrix{Complex{T}},
    Q_basis::Matrix{Complex{T}},
    local_Zne::Vector{Complex{T}},
    local_Wne::Vector{Complex{T}},
    M0::Int,
    solver_choice::Symbol,
    tol::T,
    maxiter::Int,
    restart::Int,
    comm::MPI.Comm; workspace=nothing,
) where T<:Real
    N = size(A, 1)
    Q_proj_local = zeros(Complex{T}, N, M0)
    rhs = Matrix{Complex{T}}(undef, N, M0)
    solutions = similar(rhs)
    local_success = true

    mul!(rhs, B, Q_basis)
    if solver_choice == :direct
        shifted = Matrix{Complex{T}}(undef, N, N)
        for e in eachindex(local_Zne)
            z = local_Zne[e]
            try
                @. shifted = z * B - A
                F = lu!(shifted)
                ldiv!(solutions, F, rhs)
            catch err
                @warn "MPI rank $(MPI.Comm_rank(comm)): dense complex general solve failed for contour point $e" exception=err
                local_success = false
                break
            end
            @. Q_proj_local += local_Wne[e] * solutions
        end
    else
        rhs_copy = similar(rhs)
        tmpAx = Vector{Complex{T}}(undef, N)
        tmpBx = Vector{Complex{T}}(undef, N)
        current_shift = Ref(zero(Complex{T}))

        function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
            mul!(tmpBx, B, x)
            @. tmpBx = current_shift[] * tmpBx
            mul!(tmpAx, A, x)
            @. y = tmpBx - tmpAx
            return y
        end

        for e in eachindex(local_Zne)
            z = local_Zne[e]
            try
                current_shift[] = z
                copyto!(rhs_copy, rhs)
                local_success = solve_dense_shifted!(solutions, rhs_copy,
                                                     shifted_mul!, solver_choice,
                                                     tol, maxiter, restart; workspace=workspace)
                local_success || break
            catch err
                @warn "MPI rank $(MPI.Comm_rank(comm)): dense iterative complex general solve failed for contour point $e" exception=err
                local_success = false
                break
            end
            @. Q_proj_local += local_Wne[e] * solutions
        end
    end

    return Q_proj_local, local_success
end

function mpi_compute_complex_residuals!(A::AbstractMatrix{Complex{T}},
                                        B::AbstractMatrix{Complex{T}},
                                        lambda::AbstractVector,
                                        q::Matrix{Complex{T}},
                                        res::Vector{T},
                                        M::Int,
                                        comm::MPI.Comm) where T<:Real
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    eigs_per_rank = div(M, nprocs)
    remainder = M % nprocs
    start_idx = rank * eigs_per_rank + min(rank, remainder) + 1
    local_count = eigs_per_rank + (rank < remainder ? 1 : 0)
    end_idx = start_idx + local_count - 1

    local_res = zeros(T, length(res))
    residual = Vector{Complex{T}}(undef, size(A, 1))
    Bq = similar(residual)
    success = _mpi_collective_try(comm) do
        for j in start_idx:min(end_idx, M)
            qj = view(q, :, j)
            mul!(residual, A, qj)
            mul!(Bq, B, qj)
            @. residual = residual - lambda[j] * Bq
            local_res[j] = _feast_scaled_residual(residual, Bq, lambda[j])
        end
    end
    success || return false
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
    return true
end
