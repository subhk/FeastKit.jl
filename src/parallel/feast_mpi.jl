# MPI-based parallel FeastKit implementation
# True MPI support for HPC clusters and distributed computing

# Note: MPI should already be loaded when this file is included
using LinearAlgebra
using SparseArrays

# MPI-specific FeastKit state
mutable struct MPIFeastState{T<:Real}
    # MPI communication info
    comm::MPI.Comm
    rank::Int
    size::Int
    root::Int

    # Feast parameters
    N::Int
    M0::Int
    ne::Int

    # Local contour points assigned to this rank
    local_points::Vector{Int}
    local_Zne::Vector{Complex{T}}
    local_Wne::Vector{Complex{T}}

    # Convergence state
    converged::Bool
    loop::Int
    epsout::T
    info::Int

    function MPIFeastState{T}(comm::MPI.Comm, N::Int, M0::Int, ne::Int, root::Int=0) where T<:Real
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)

        # Distribute contour points among MPI ranks
        points_per_rank = div(ne, size)
        remainder = ne % size

        # Calculate local points for this rank
        start_idx = rank * points_per_rank + min(rank, remainder) + 1
        local_count = points_per_rank + (rank < remainder ? 1 : 0)
        local_points = collect(start_idx:(start_idx + local_count - 1))

        new(
            comm, rank, size, root,
            N, M0, ne,
            local_points,
            Vector{Complex{T}}(undef, local_count),
            Vector{Complex{T}}(undef, local_count),
            false, 0, zero(T), 0
        )
    end
end

# Main MPI FeastKit interface
function mpi_feast_sygv!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         comm::MPI.Comm = MPI.COMM_WORLD,
                         root::Int = 0) where T<:Real
    # MPI parallel Feast for real symmetric generalized eigenvalue problems

    # Initialize MPI if not already done
    if !MPI.Initialized()
        MPI.Init()
    end

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    N = Base.size(A, 1)

    # Input validation (on all ranks)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    # Generate integration contour (on root, then broadcast)
    contour = nothing
    if rank == root
        contour = feast_contour(Emin, Emax, fpm)
    end

    # Broadcast contour to all ranks
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)

    # Create MPI state
    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)

    # Distribute contour points
    for (i, global_idx) in enumerate(mpi_state.local_points)
        mpi_state.local_Zne[i] = Zne_global[global_idx]
        mpi_state.local_Wne[i] = Wne_global[global_idx]
    end

    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm, T)
    max_loops = fpm[4]

    # Trial subspace: deterministic complex seed (identical on every rank),
    # broadcast from root so all ranks start bit-for-bit identical.
    Q = Matrix{Complex{T}}(undef, N, M0)
    _feast_seeded_subspace_complex!(Q)
    MPI.Bcast!(Q, root, comm)
    active_dim = M0

    # Each rank factorizes ONLY its local contour shifts, once, and reuses them
    # across refinement loops. This is the work MPI parallelizes across ranks.
    local_factors = [lu(z * B - A) for z in mpi_state.local_Zne]

    # Scratch reused across loops. The reduced Rayleigh-Ritz problem mirrors the
    # serial dense Hermitian path (complex QR rank-compression + Hermitian RR),
    # which the old moment-based MPI path lacked — that is why it returned wrong
    # eigenpairs for M0 larger than the number of eigenvalues in the interval.
    Q_proj_local = Matrix{Complex{T}}(undef, N, M0)
    BQ_loop = Matrix{Complex{T}}(undef, N, M0)
    q_basis = Matrix{Complex{T}}(undef, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQm = Matrix{Complex{T}}(undef, N, M0)
    Sq = Matrix{Complex{T}}(undef, M0, M0)
    Aq = Matrix{Complex{T}}(undef, M0, M0)
    qcol = Vector{Complex{T}}(undef, N)
    lambda = zeros(T, M0)
    q = zeros(T, N, M0)
    res = zeros(T, M0)
    lambda_tmp = similar(lambda)
    perm = Vector{Int}(undef, M0)
    q_tmp = similar(q)
    residual_Aq = Vector{T}(undef, N)
    residual_Bq = Vector{T}(undef, N)
    residual = Vector{T}(undef, N)

    epsout = T(Inf)
    info_code = Int(Feast_SUCCESS)
    loop_done = 0
    M_found = 0

    for loop in 1:max_loops
        loop_done = loop
        mpi_state.loop = loop

        # Each rank applies its local contour resolvents to the trial subspace,
        # then the partial filtered subspaces are summed across ranks. For dense
        # systems the per-loop solves are heavy and distribute well, so the
        # reduced Rayleigh-Ritz is left replicated on every rank (an Allreduce,
        # not root-only) — that scales better here than idling ranks on root.
        qblk = view(Q, :, 1:active_dim)
        bq = view(BQ_loop, :, 1:active_dim)
        mul!(bq, B, qblk)
        qpl = view(Q_proj_local, :, 1:active_dim)
        fill!(qpl, zero(Complex{T}))
        for (e, Fe) in enumerate(local_factors)
            Y = Fe \ bq
            w = 2 * mpi_state.local_Wne[e]
            @. qpl += w * Y
        end
        global_Q_proj = MPI.Allreduce(Q_proj_local[:, 1:active_dim], MPI.SUM, comm)

        try
            rank_r = _feast_qr_compress!(q_basis, global_Q_proj, active_dim;
                                         rank_tol=sqrt(eps(T)))
            if rank_r == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            q_rank = view(q_basis, :, 1:rank_r)
            AQ_r = view(AQ, :, 1:rank_r)
            BQ_r = view(BQm, :, 1:rank_r)
            Sq_r = view(Sq, 1:rank_r, 1:rank_r)
            Aq_r = view(Aq, 1:rank_r, 1:rank_r)

            mul!(AQ_r, A, q_rank)
            mul!(Sq_r, adjoint(q_rank), AQ_r)
            mul!(BQ_r, B, q_rank)
            mul!(Aq_r, adjoint(q_rank), BQ_r)

            local lambda_red, v_red
            try
                Fr = eigen(Hermitian(Sq_r), Hermitian(Aq_r))
                lambda_red = Fr.values
                v_red = Fr.vectors
            catch err
                (isa(err, PosDefException) || isa(err, LinearAlgebra.LAPACKException)) || rethrow(err)
                Fr = eigen(Sq_r, Aq_r)
                lambda_red = real.(Fr.values)
                v_red = Fr.vectors
            end

            for idx in 1:rank_r
                mul!(qcol, q_rank, view(v_red, :, idx))
                @inbounds for i in 1:N
                    q[i, idx] = real(qcol[i])
                end
                lambda[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                            Emin, Emax, rank_r)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for j in 1:M
                nrm = norm(view(q, :, j))
                nrm > 0 && (view(q, :, j) ./= nrm)
            end

            feast_residual!(A, B, lambda, q, res, M,
                            residual_Aq, residual_Bq, residual)
            epsout = maximum(view(res, 1:M))
            mpi_state.epsout = epsout
            M_found = M

            if epsout <= eps_tolerance
                mpi_state.converged = true
                mpi_state.info = Int(Feast_SUCCESS)
                feast_sort!(lambda, q, res, M)
                return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                         Int(Feast_SUCCESS), epsout, loop)
            end

            active_dim = rank_r
            copyto!(view(Q, :, 1:active_dim), view(q, :, 1:active_dim))
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            break
        end
    end

    if info_code == Int(Feast_SUCCESS)
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    mpi_state.info = info_code
    M = M_found
    M > 1 && feast_sort!(lambda, q, res, M)
    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                             info_code, epsout, loop_done)
end

# Compute local moment contributions on each MPI rank (returns moments and Q_proj)
function mpi_compute_local_moments(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                  work::Matrix{T}, local_Zne::Vector{Complex{T}},
                                  local_Wne::Vector{Complex{T}}, M0::Int) where T<:Real

    local_ne = length(local_Zne)
    N = size(A, 1)
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    Q_proj_local = zeros(T, N, M0)

    # Process each local contour point
    for e in 1:local_ne
        z = local_Zne[e]
        w = local_Wne[e]

        try
            # Form and factorize (z*B - A)
            system_matrix = z * B - A
            F = lu(system_matrix)

            # Right-hand side
            rhs = B * work[:, 1:M0]

            # Solve linear systems
            workc_local = F \ rhs

            # Factor of 2 for half-contour symmetry
            weight = 2 * w

            # Accumulate moment contribution
            for j in 1:M0
                for i in 1:M0
                    inner_product = dot(work[:, i], workc_local[:, j])
                    Aq_local[i, j] += real(weight * inner_product)
                    Sq_local[i, j] += real(weight * z * inner_product)
                end
            end

            # Accumulate filtered subspace contribution
            Q_proj_local .+= real.(weight .* workc_local)

        catch err
            @warn "MPI rank $(MPI.Comm_rank(MPI.COMM_WORLD)): Linear solve failed for contour point $e: $err"
        end
    end

    return Aq_local, Sq_local, Q_proj_local
end

# Distributed residual computation
function mpi_compute_residuals!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                               lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                               M::Int, comm::MPI.Comm) where T<:Real

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Distribute eigenvalue computations among ranks
    eigs_per_rank = div(M, nprocs)
    remainder = M % nprocs

    start_idx = rank * eigs_per_rank + min(rank, remainder) + 1
    local_count = eigs_per_rank + (rank < remainder ? 1 : 0)
    end_idx = start_idx + local_count - 1

    # `res` is the full M0 workspace vector. MPI reductions require send and
    # receive buffers with matching lengths, even though only the first M
    # entries are active for the current iteration.
    local_res = zeros(T, length(res))
    for j in start_idx:min(end_idx, M)
        # Relative residual: ||A*q - λ*B*q|| / max(|λ|, 1)
        Aq = A * q[:, j]
        Bq = B * q[:, j]
        residual = Aq - lambda[j] * Bq
        local_res[j] = norm(residual) / max(abs(lambda[j]), one(eltype(lambda)))
    end

    # Reduce residuals across all ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end

# MPI sparse FeastKit
function mpi_feast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                          Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                          comm::MPI.Comm = MPI.COMM_WORLD,
                          root::Int = 0) where T<:Real
    # MPI parallel sparse real-symmetric generalized FEAST. Same corrected
    # algorithm as mpi_feast_sygv! (cached per-rank sparse factorizations,
    # Allreduce'd complex filtered subspace, QR-compressed Hermitian Rayleigh-Ritz).
    # The old moment-based path returned wrong eigenpairs for oversized M0.
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    N = Base.size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    contour = rank == root ? feast_contour(Emin, Emax, fpm) : nothing
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)

    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm, T)
    max_loops = fpm[4]

    Q = Matrix{Complex{T}}(undef, N, M0)
    _feast_seeded_subspace_complex!(Q)
    MPI.Bcast!(Q, root, comm)
    active_dim = M0

    # Each rank factorizes only its local contour shifts, once, reused across loops.
    local_factors = [lu(z * B - A) for z in mpi_state.local_Zne]

    Q_proj_local = Matrix{Complex{T}}(undef, N, M0)
    BQ_loop = Matrix{Complex{T}}(undef, N, M0)
    q_basis = Matrix{Complex{T}}(undef, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQm = Matrix{Complex{T}}(undef, N, M0)
    Sq = Matrix{Complex{T}}(undef, M0, M0)
    Aq = Matrix{Complex{T}}(undef, M0, M0)
    qcol = Vector{Complex{T}}(undef, N)
    lambda = zeros(T, M0)
    q = zeros(T, N, M0)
    res = zeros(T, M0)
    lambda_tmp = similar(lambda)
    perm = Vector{Int}(undef, M0)
    q_tmp = similar(q)
    residual_Aq = Vector{T}(undef, N)
    residual_Bq = Vector{T}(undef, N)
    residual = Vector{T}(undef, N)

    epsout = T(Inf)
    info_code = Int(Feast_SUCCESS)
    loop_done = 0
    M_found = 0
    flags = Vector{Int}(undef, 4)   # [rank_r, M, status, info]; status: 0=continue 1=converged 2=stop
    epsbuf = Vector{T}(undef, 1)

    for loop in 1:max_loops
        loop_done = loop
        mpi_state.loop = loop

        # Distributed contour solves: each rank applies only its local resolvents.
        qblk = view(Q, :, 1:active_dim)
        bq = view(BQ_loop, :, 1:active_dim)
        mul!(bq, B, qblk)
        qpl = view(Q_proj_local, :, 1:active_dim)
        fill!(qpl, zero(Complex{T}))
        for (e, Fe) in enumerate(local_factors)
            Y = Fe \ bq
            w = 2 * mpi_state.local_Wne[e]
            @. qpl += w * Y
        end

        # Sum the partial filtered subspaces to ROOT ONLY. The reduced
        # Rayleigh-Ritz (QR compress + dense N×M0 work) then runs once on root
        # instead of being replicated on every rank (which contended for memory
        # bandwidth and capped scaling). Results are broadcast back.
        global_Q_proj = MPI.Reduce(Q_proj_local[:, 1:active_dim], MPI.SUM, root, comm)

        rank_r = active_dim
        M = 0
        status = 0
        if rank == root
            gqp = global_Q_proj::Matrix{Complex{T}}
            try
                rank_r = _feast_qr_compress!(q_basis, gqp, active_dim;
                                             rank_tol=sqrt(eps(T)))
                if rank_r == 0
                    status = 2
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                else
                    q_rank = view(q_basis, :, 1:rank_r)
                    AQ_r = view(AQ, :, 1:rank_r)
                    BQ_r = view(BQm, :, 1:rank_r)
                    Sq_r = view(Sq, 1:rank_r, 1:rank_r)
                    Aq_r = view(Aq, 1:rank_r, 1:rank_r)
                    mul!(AQ_r, A, q_rank)
                    mul!(Sq_r, adjoint(q_rank), AQ_r)
                    mul!(BQ_r, B, q_rank)
                    mul!(Aq_r, adjoint(q_rank), BQ_r)
                    local lambda_red, v_red
                    try
                        Fr = eigen(Hermitian(Sq_r), Hermitian(Aq_r))
                        lambda_red = Fr.values
                        v_red = Fr.vectors
                    catch err
                        (isa(err, PosDefException) || isa(err, LinearAlgebra.LAPACKException)) || rethrow(err)
                        Fr = eigen(Sq_r, Aq_r)
                        lambda_red = real.(Fr.values)
                        v_red = Fr.vectors
                    end
                    for idx in 1:rank_r
                        mul!(qcol, q_rank, view(v_red, :, idx))
                        @inbounds for i in 1:N
                            q[i, idx] = real(qcol[i])
                        end
                        lambda[idx] = lambda_red[idx]
                    end
                    M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                                    Emin, Emax, rank_r)
                    if M == 0
                        status = 2
                        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    else
                        for j in 1:M
                            nrm = norm(view(q, :, j))
                            nrm > 0 && (view(q, :, j) ./= nrm)
                        end
                        feast_residual!(A, B, lambda, q, res, M,
                                        residual_Aq, residual_Bq, residual)
                        epsout = maximum(view(res, 1:M))
                        status = epsout <= eps_tolerance ? 1 : 0
                    end
                end
            catch err
                status = 2
                info_code = Int(Feast_ERROR_LAPACK)
            end
            flags[1] = rank_r; flags[2] = M; flags[3] = status; flags[4] = info_code
        end

        # Broadcast the control word, then the result data the other ranks need.
        MPI.Bcast!(flags, root, comm)
        rank_r = flags[1]; M = flags[2]; status = flags[3]; info_code = flags[4]
        if status == 2
            M_found = 0
            break
        end
        MPI.Bcast!(lambda, root, comm)
        MPI.Bcast!(q, root, comm)
        MPI.Bcast!(res, root, comm)
        epsbuf[1] = epsout
        MPI.Bcast!(epsbuf, root, comm)
        epsout = epsbuf[1]
        mpi_state.epsout = epsout
        M_found = M

        if status == 1
            mpi_state.converged = true
            mpi_state.info = Int(Feast_SUCCESS)
            feast_sort!(lambda, q, res, M)
            return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                     Int(Feast_SUCCESS), epsout, loop)
        end

        active_dim = rank_r
        copyto!(view(Q, :, 1:active_dim), view(q, :, 1:active_dim))
    end

    if info_code == Int(Feast_SUCCESS)
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    mpi_state.info = info_code
    M = M_found
    M > 1 && feast_sort!(lambda, q, res, M)
    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                             info_code, epsout, loop_done)
end

# Sparse moment computation for MPI (returns moments and Q_proj)
function mpi_compute_sparse_moments(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                   work::Matrix{T}, local_Zne::Vector{Complex{T}},
                                   local_Wne::Vector{Complex{T}}, M0::Int) where T<:Real

    local_ne = length(local_Zne)
    N = size(A, 1)
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    Q_proj_local = zeros(T, N, M0)

    for e in 1:local_ne
        z = local_Zne[e]
        w = local_Wne[e]

        try
            # Sparse system formation and solve
            system_matrix = z * B - A
            F = lu(system_matrix)

            rhs = B * work[:, 1:M0]
            workc_local = F \ rhs

            # Factor of 2 for half-contour symmetry
            weight = 2 * w

            # Accumulate moments
            for j in 1:M0
                for i in 1:M0
                    inner_product = dot(work[:, i], workc_local[:, j])
                    Aq_local[i, j] += real(weight * inner_product)
                    Sq_local[i, j] += real(weight * z * inner_product)
                end
            end

            # Accumulate filtered subspace contribution
            Q_proj_local .+= real.(weight .* workc_local)

        catch err
            @warn "MPI rank $(MPI.Comm_rank(MPI.COMM_WORLD)): Sparse solve failed: $err"
        end
    end

    return Aq_local, Sq_local, Q_proj_local
end

# Sparse residual computation for MPI
function mpi_compute_sparse_residuals!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                      lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                                      M::Int, comm::MPI.Comm) where T<:Real

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Distribute eigenvalue computations
    eigs_per_rank = div(M, nprocs)
    remainder = M % nprocs

    start_idx = rank * eigs_per_rank + min(rank, remainder) + 1
    local_count = eigs_per_rank + (rank < remainder ? 1 : 0)
    end_idx = start_idx + local_count - 1

    # `res` is the full M0 workspace vector. MPI reductions require send and
    # receive buffers with matching lengths, even though only the first M
    # entries are active for the current iteration.
    local_res = zeros(T, length(res))
    for j in start_idx:min(end_idx, M)
        Aq = A * q[:, j]
        Bq = B * q[:, j]
        residual = Aq - lambda[j] * Bq
        local_res[j] = norm(residual) / max(abs(lambda[j]), one(eltype(lambda)))
    end

    # Reduce across ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end

function _mpi_solver_choice(solver::Symbol)
    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice in (:direct, :gmres) ||
        throw(ArgumentError("Unsupported MPI FEAST solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_choice == :gmres && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative MPI FEAST solves."))
    return solver_choice
end

function _mpi_success_count(local_success::Bool, comm::MPI.Comm)
    flag = [local_success ? 1 : 0]
    return MPI.Allreduce(flag, MPI.SUM, comm)[1]
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
    comm::MPI.Comm,
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
        weight = 2 * local_Wne[e]
        try
            if solver_choice == :direct
                F = lu(z * B - A)
                ldiv!(solutions, F, rhs)
            else
                local_success = solve_shifted_iterative!(solutions, rhs, A, B, z,
                                                         tol, maxiter, restart)
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
    comm::MPI.Comm,
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
            weight = 2 * local_Wne[e]
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
            weight = 2 * local_Wne[e]
            try
                current_shift[] = z
                copyto!(rhs_copy, rhs)
                local_success = solve_dense_shifted!(solutions, rhs_copy,
                                                     shifted_mul!, solver_choice,
                                                     tol, maxiter, restart)
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
    comm::MPI.Comm,
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
                                                         tol, maxiter, restart)
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
    comm::MPI.Comm,
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
                                                     tol, maxiter, restart)
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
    for j in start_idx:min(end_idx, M)
        qj = view(q, :, j)
        mul!(residual, A, qj)
        mul!(Bq, B, qj)
        @. residual = residual - lambda[j] * Bq
        local_res[j] = norm(residual) / max(abs(lambda[j]), one(T))
    end
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end

function _mpi_feast_complex_hermitian!(A::AbstractMatrix{Complex{T}},
                                       B::AbstractMatrix{Complex{T}},
                                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                       comm::MPI.Comm = MPI.COMM_WORLD,
                                       root::Int = 0,
                                       solver::Symbol = :direct,
                                       solver_tol::Real = 0.0,
                                       solver_maxiter::Int = 500,
                                       solver_restart::Int = 30) where T<:Real
    rank = MPI.Comm_rank(comm)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    feastdefault!(fpm)
    solver_choice = _mpi_solver_choice(solver)
    tol = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    contour = rank == root ? feast_contour(Emin, Emax, fpm) : nothing
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)

    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    MPI.Bcast!(Q_basis, root, comm)

    Aq_herm = Matrix{Complex{T}}(undef, M0, M0)
    Sq_herm = Matrix{Complex{T}}(undef, M0, M0)
    AQc = similar(Q_basis)
    BQc = similar(Q_basis)
    Q_proj = similar(Q_basis)
    solutions = similar(Q_basis)
    solutions_tmp = similar(Q_basis)
    lambda_vec = zeros(T, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    res_vec = zeros(T, M0)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    M_found = 0
    loop_count = 0

    for loop_idx in 0:fpm[4]
        loop_count = loop_idx
        # Each rank solves its local contour points; only the partial filtered
        # subspace (complex) is needed. The reduced pencil is rebuilt from an
        # ORTHONORMAL basis below — what the old rank-deficient moment path lacked.
        _, _, local_Q_proj, local_success =
            mpi_compute_complex_hermitian_moments(A, B, Q_basis,
                                                  mpi_state.local_Zne,
                                                  mpi_state.local_Wne, M0,
                                                  solver_choice, tol,
                                                  solver_maxiter, solver_restart,
                                                  comm)
        if _mpi_success_count(local_success, comm) != MPI.Comm_size(comm)
            info_code = solver_choice == :direct ? Int(Feast_ERROR_LAPACK) : Int(Feast_ERROR_NO_CONVERGENCE)
            break
        end

        Q_proj .= MPI.Allreduce(local_Q_proj, MPI.SUM, comm)

        try
            # Orthonormalize the filtered subspace, then Hermitian Rayleigh-Ritz:
            # Sq = Qᴴ A Q, Aq = Qᴴ B Q. Orthonormality keeps the reduced pencil
            # well-conditioned even when the filtered subspace is rank deficient.
            Qo = Matrix(qr!(copyto!(AQc, Q_proj)).Q)
            mul!(AQc, A, Qo)
            mul!(Sq_herm, adjoint(Qo), AQc)
            mul!(BQc, B, Qo)
            mul!(Aq_herm, adjoint(Qo), BQc)
            F = try
                eigen(Hermitian(Sq_herm), Hermitian(Aq_herm))
            catch err
                eigen(Sq_herm, Aq_herm)
            end
            lambda_red = real.(F.values)
            v_red = F.vectors
            for idx in 1:M0
                mul!(view(solutions, :, idx), Qo, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda_vec, solutions, perm,
                                             lambda_tmp, solutions_tmp,
                                             Emin, Emax, M0)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end
            for j in 1:M
                qj = view(solutions, :, j)
                nrm = norm(qj)
                nrm > 0 && (qj ./= nrm)
            end
            mpi_compute_complex_residuals!(A, B, lambda_vec, solutions, res_vec, M, comm)
            epsout_val = maximum(res_vec[1:M])
            M_found = M
            if epsout_val <= feast_tolerance(fpm, T)
                info_code = Int(Feast_SUCCESS)
                break
            elseif loop_idx == fpm[4]
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end
            copyto!(Q_basis, solutions)
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            break
        end
    end

    M_found > 1 && feast_sort!(lambda_vec, solutions, res_vec, M_found)
    return FeastResult{T, Complex{T}}(lambda_vec[1:M_found],
                                      solutions[:, 1:M_found], M_found,
                                      res_vec[1:M_found], info_code,
                                      epsout_val, loop_count)
end

function mpi_feast_hcsrgv!(A::SparseMatrixCSC{Complex{T},Int},
                           B::SparseMatrixCSC{Complex{T},Int},
                           Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    return _mpi_feast_complex_hermitian!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_hcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                           Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast_hcsrgv!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_hegv!(A::Matrix{Complex{T}},
                         B::Matrix{Complex{T}},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    ishermitian(A) || throw(ArgumentError("Matrix A must be Hermitian"))
    ishermitian(B) || throw(ArgumentError("Matrix B must be Hermitian positive definite"))
    return _mpi_feast_complex_hermitian!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_heev!(A::Matrix{Complex{T}},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast_hegv!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function _mpi_feast_complex_general!(A::AbstractMatrix{Complex{T}},
                                     B::AbstractMatrix{Complex{T}},
                                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                                     comm::MPI.Comm = MPI.COMM_WORLD,
                                     root::Int = 0,
                                     solver::Symbol = :direct,
                                     solver_tol::Real = 0.0,
                                     solver_maxiter::Int = 500,
                                     solver_restart::Int = 30) where T<:Real
    rank = MPI.Comm_rank(comm)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_grci_input(N, M0, Emid, r, fpm)
    feastdefault!(fpm)
    solver_choice = _mpi_solver_choice(solver)
    tol = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    contour = rank == root ? feast_gcontour(Emid, r, fpm) : nothing
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)

    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    MPI.Bcast!(Q_basis, root, comm)

    Q_proj = similar(Q_basis)
    solutions = similar(Q_basis)
    solutions_tmp = similar(Q_basis)
    AQ = similar(Q_basis)
    BQ = similar(Q_basis)
    Ared = Matrix{Complex{T}}(undef, M0, M0)
    Bred = Matrix{Complex{T}}(undef, M0, M0)
    lambda_vec = Vector{Complex{T}}(undef, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    res_vec = zeros(T, M0)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    M_found = 0
    loop_count = 0

    for loop_idx in 0:fpm[4]
        loop_count = loop_idx
        local_Q_proj, local_success =
            mpi_compute_complex_general_projection(A, B, Q_basis,
                                                   mpi_state.local_Zne,
                                                   mpi_state.local_Wne, M0,
                                                   solver_choice, tol,
                                                   solver_maxiter, solver_restart,
                                                   comm)
        if _mpi_success_count(local_success, comm) != MPI.Comm_size(comm)
            info_code = solver_choice == :direct ? Int(Feast_ERROR_LAPACK) : Int(Feast_ERROR_NO_CONVERGENCE)
            break
        end

        Q_proj .= MPI.Allreduce(local_Q_proj, MPI.SUM, comm)
        try
            # Orthonormalize the filtered subspace before the (non-Hermitian)
            # Rayleigh-Ritz: Ared = Qᴴ A Q, Bred = Qᴴ B Q. Fixes the rank-deficient
            # reduced pencil that made the raw-Q_proj path return garbage.
            Qo = Matrix(qr!(copyto!(AQ, Q_proj)).Q)
            mul!(AQ, A, Qo)
            mul!(Ared, adjoint(Qo), AQ)
            mul!(BQ, B, Qo)
            mul!(Bred, adjoint(Qo), BQ)
            F = eigen(Ared, Bred)
            lambda_vec .= F.values
            for idx in 1:M0
                mul!(view(solutions, :, idx), Qo, view(F.vectors, :, idx))
            end
            M = _feast_reorder_by_gcontour!(lambda_vec, solutions, perm,
                                            lambda_tmp, solutions_tmp,
                                            Emid, r, fpm, M0)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end
            for j in 1:M
                qj = view(solutions, :, j)
                nrm = norm(qj)
                nrm > 0 && (qj ./= nrm)
            end
            mpi_compute_complex_residuals!(A, B, lambda_vec, solutions, res_vec, M, comm)
            epsout_val = maximum(res_vec[1:M])
            M_found = M
            if epsout_val <= feast_tolerance(fpm, T)
                info_code = Int(Feast_SUCCESS)
                break
            elseif loop_idx == fpm[4]
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end
            copyto!(Q_basis, solutions)
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            break
        end
    end

    M_found > 1 && feast_sort_general!(lambda_vec, solutions, res_vec, M_found)
    return FeastGeneralResult{T}(lambda_vec[1:M_found],
                                 solutions[:, 1:M_found], M_found,
                                 res_vec[1:M_found], info_code,
                                 epsout_val, loop_count)
end

function mpi_feast_gcsrgv!(A::SparseMatrixCSC{Complex{T},Int},
                           B::SparseMatrixCSC{Complex{T},Int},
                           Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    return _mpi_feast_complex_general!(A, B, Emid, r, M0, fpm; kwargs...)
end

function mpi_feast_gcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                           Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast_gcsrgv!(A, B, Emid, r, M0, fpm; kwargs...)
end

function mpi_feast_gegv!(A::Matrix{Complex{T}},
                         B::Matrix{Complex{T}},
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    return _mpi_feast_complex_general!(A, B, Emid, r, M0, fpm; kwargs...)
end

function mpi_feast_geev!(A::Matrix{Complex{T}},
                         Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast_gegv!(A, B, Emid, r, M0, fpm; kwargs...)
end

# High-level MPI FeastKit interface
function mpi_feast(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0) where T<:Real
    # Unified MPI interface that detects matrix type

    Emin, Emax = interval

    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end

    # Detect matrix type and call appropriate MPI solver
    if isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return mpi_feast_scsrgv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    else
        return mpi_feast_sygv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    end
end

# Standard eigenvalue problem MPI interface
function mpi_feast(A::AbstractMatrix{T}, interval::Tuple{T,T};
                   M0::Int = 10, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0) where T<:Real
    # MPI interface for standard eigenvalue problems

    N = size(A, 1)

    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end

    # Create identity matrix of appropriate type
    if isa(A, SparseMatrixCSC)
        B = sparse(I, N, N)
    else
        B = Matrix{T}(I, N, N)
    end

    return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm, root=root)
end

function mpi_feast(A::SparseMatrixCSC{Complex{T},Int},
                   B::SparseMatrixCSC{Complex{T},Int},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   solver::Symbol = :direct,
                   solver_tol::Real = 0.0,
                   solver_maxiter::Int = 500,
                   solver_restart::Int = 30) where T<:Real
    Emin, Emax = interval
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_hcsrgv!(A, B, Emin, Emax, M0, params;
                             comm=comm, root=root, solver=solver,
                             solver_tol=solver_tol,
                             solver_maxiter=solver_maxiter,
                             solver_restart=solver_restart)
end

function mpi_feast(A::SparseMatrixCSC{Complex{T},Int},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast(A, B, interval; M0=M0, fpm=fpm, comm=comm, root=root, kwargs...)
end

function mpi_feast(A::Matrix{Complex{T}},
                   B::Matrix{Complex{T}},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   solver::Symbol = :direct,
                   solver_tol::Real = 0.0,
                   solver_maxiter::Int = 500,
                   solver_restart::Int = 30) where T<:Real
    Emin, Emax = interval
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_hegv!(A, B, Emin, Emax, M0, params;
                           comm=comm, root=root, solver=solver,
                           solver_tol=solver_tol,
                           solver_maxiter=solver_maxiter,
                           solver_restart=solver_restart)
end

function mpi_feast(A::Matrix{Complex{T}},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast(A, B, interval; M0=M0, fpm=fpm, comm=comm, root=root, kwargs...)
end

function mpi_feast_general(A::SparseMatrixCSC{Complex{T},Int},
                           B::SparseMatrixCSC{Complex{T},Int},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           solver::Symbol = :direct,
                           solver_tol::Real = 0.0,
                           solver_maxiter::Int = 500,
                           solver_restart::Int = 30) where T<:Real
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_gcsrgv!(A, B, center, radius, M0, params;
                             comm=comm, root=root, solver=solver,
                             solver_tol=solver_tol,
                             solver_maxiter=solver_maxiter,
                             solver_restart=solver_restart)
end

function mpi_feast_general(A::SparseMatrixCSC{Complex{T},Int},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                             comm=comm, root=root, kwargs...)
end

function mpi_feast_general(A::Matrix{Complex{T}},
                           B::Matrix{Complex{T}},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           solver::Symbol = :direct,
                           solver_tol::Real = 0.0,
                           solver_maxiter::Int = 500,
                           solver_restart::Int = 30) where T<:Real
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_gegv!(A, B, center, radius, M0, params;
                           comm=comm, root=root, solver=solver,
                           solver_tol=solver_tol,
                           solver_maxiter=solver_maxiter,
                           solver_restart=solver_restart)
end

function mpi_feast_general(A::Matrix{Complex{T}},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                             comm=comm, root=root, kwargs...)
end

# MPI performance benchmarking
function mpi_feast_benchmark(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int;
                            comm::MPI.Comm = MPI.COMM_WORLD)
    # Benchmark MPI FeastKit performance

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit Performance Benchmark")
        println("="^40)
        println("Matrix size: $(Base.size(A, 1))")
        println("Search interval: $interval")
        println("MPI processes: $nprocs")
        println("Subspace size: $M0")
    end

    # MPI timing
    MPI.Barrier(comm)
    start_time = MPI.Wtime()

    result = mpi_feast(A, B, interval, M0=M0, comm=comm)

    MPI.Barrier(comm)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time

    if rank == 0
        println("\nMPI FeastKit Results:")
        println("Time: $(round(elapsed_time, digits=3)) seconds")
        println("Eigenvalues found: $(result.M)")
        println("Convergence loops: $(result.loop)")
        println("Final residual: $(result.epsout)")
        println("Exit status: $(result.info)")

        if result.M > 0
            println("\nEigenvalues:")
            for i in 1:min(result.M, 5)  # Show first 5
                println("  λ[$i] = $(result.lambda[i])")
            end
            if result.M > 5
                println("  ... and $(result.M - 5) more")
            end
        end
    end

    return result
end

# Utility: Check if MPI is available and initialized
function mpi_feast_available()
    try
        return MPI.Initialized()
    catch
        return false
    end
end

# Utility: Initialize MPI for FeastKit if needed
function mpi_feast_init()
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit initialized with $nprocs processes")
    end

    return comm, rank, nprocs
end

# Utility: Clean up MPI
function mpi_feast_finalize()
    if MPI.Initialized()
        MPI.Finalize()
    end
end
