# Main MPI FeastKit interface
function mpi_feast_sygv!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         comm::MPI.Comm = MPI.COMM_WORLD,
                         root::Int = 0,
                         use_threads::Bool = false) where T<:Real
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
    contour, _ = _mpi_contour(T,fpm,Emin,Emax,root,comm)
    ne = length(contour.Zne)
    Zne_global, Wne_global = contour.Zne, contour.Wne

    # Create MPI state
    mpi_state = MPIFeastState{T}(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm),
                                 N, M0, ne, root)

    # Distribute contour points
    for (i, global_idx) in enumerate(mpi_state.local_points)
        mpi_state.local_Zne[i] = Zne_global[global_idx]
        mpi_state.local_Wne[i] = Wne_global[global_idx]
    end

    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))
    eps_tolerance = feast_tolerance(fpm, T)
    max_loops = fpm[4]
    keep = Vector{Bool}(undef, M0)

    # Half-contour symmetry requires a real trial basis. Its full projector is
    # the real part of the doubled upper-half contribution.
    Q_real = Matrix{T}(undef,N,M0)
    _feast_seeded_subspace!(Q_real)
    Q = Complex{T}.(Q_real)
    MPI.Bcast!(Q, root, comm)
    active_dim = M0

    # Each rank caches its local shifts, or factors them lazily when storage
    # is disabled. Contour work remains distributed in both modes.
    local_factors = _mpi_factorize_contour(A,B,mpi_state.local_Zne,comm;store=fpm[10]==1)
    local_factors === nothing && return FeastResult{T,T}(T[],zeros(T,N,0),0,T[],
        Int(Feast_ERROR_LAPACK),T(Inf),0)
    B_is_identity = (B == I)   # standard problem: skip the per-loop identity matmuls
    # Every rank measures on the same broadcast basis; root's value is used.
    res_scale = _mpi_residual_floor(A, B_is_identity ? nothing : B, Q,
                                    _feast_residual_scale(Emin, Emax), root, comm)
    res_scale === nothing && return FeastResult{T,T}(T[],zeros(T,N,0),0,T[],
        Int(Feast_ERROR_LAPACK),T(Inf),0)

    # Scratch reused across loops. The reduced Rayleigh-Ritz problem mirrors the
    # serial dense Hermitian path (complex QR rank-compression + Hermitian RR),
    # which the old moment-based MPI path lacked — that is why it returned wrong
    # eigenpairs for M0 larger than the number of eigenvalues in the interval.
    Q_proj_local = Matrix{Complex{T}}(undef, N, M0)
    BQ_loop = Matrix{Complex{T}}(undef, N, M0)
    Y_loop = A isa AbstractSparseMatrix ?
        Matrix{Complex{promote_type(T,Float64)}}(undef,N,M0) : Matrix{Complex{T}}(undef,N,M0)
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
    converged = false

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
        qpl = view(Q_proj_local, :, 1:active_dim)
        fill!(qpl, zero(Complex{T}))
        Yv = view(Y_loop, :, 1:active_dim)
        success = try
            B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
            _mpi_local_projection!(qpl,local_factors,bq,mpi_state.local_Wne,Yv;
                                   scale=2,use_threads=use_threads)
        catch err
            @debug "MPI local projection preparation failed" exception=err
            false
        end
        if _mpi_success_count(success,comm) != nprocs
            info_code = Int(Feast_ERROR_LAPACK)
            M_found = 0
            break
        end
        # In-place Allreduce: every rank's partial sum is replaced by the global
        # sum, no per-loop slice copy or fresh receive buffer.
        MPI.Allreduce!(qpl, MPI.SUM, comm)
        @. qpl = real(qpl)

        # qpl is the filtered image of the previous loop's Ritz vectors; the
        # root classifies them by filter response and every rank follows.
        if loop - 1 >= _FEAST_SPURIOUS_MIN_LOOPS
            kept = rank == root ?
                _feast_screen_spurious!(keep, qpl, view(Q, :, 1:active_dim), res,
                                        M_found, eps_tolerance) : nothing
            kept = MPI.bcast(kept, root, comm)
            if kept !== nothing
                MPI.Bcast!(keep, root, comm)
                M_found = _feast_compact_pairs!(lambda, q, res, keep, M_found)
                epsout = M_found > 0 ? maximum(view(res, 1:M_found)) : zero(T)
                converged = true
                break
            end
        end

        rank_r = 0
        M = 0
        success = _mpi_collective_try(comm) do
            rank_r = _feast_qr_compress!(q_basis, qpl, active_dim;
                                         rank_tol=sqrt(eps(T)))
            if rank_r == 0
                return
            end

            q_rank = view(q_basis, :, 1:rank_r)
            AQ_r = view(AQ, :, 1:rank_r)
            BQ_r = view(BQm, :, 1:rank_r)
            Sq_r = view(Sq, 1:rank_r, 1:rank_r)
            Aq_r = view(Aq, 1:rank_r, 1:rank_r)

            mul!(AQ_r, A, q_rank)
            mul!(Sq_r, adjoint(q_rank), AQ_r)
            if B_is_identity
                # Qo orthonormal ⇒ Qᴴ B Q = I; skip the dense identity matmul.
                fill!(Aq_r, zero(Complex{T}))
                @inbounds for i in 1:rank_r
                    Aq_r[i, i] = one(Complex{T})
                end
            else
                mul!(BQ_r, B, q_rank)
                mul!(Aq_r, adjoint(q_rank), BQ_r)
            end

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
                _feast_real_column!(view(q, :, idx), qcol)
                lambda[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                            Emin, Emax, rank_r)
            if M == 0
                return
            end

            for j in 1:M
                nrm = norm(view(q, :, j))
                nrm > 0 && (view(q, :, j) ./= nrm)
            end

            feast_residual!(A, B, lambda, q, res, M,
                            residual_Aq, residual_Bq, residual; scale=res_scale)
            epsout = maximum(view(res, 1:M))
        end
        if !success
            info_code = Int(Feast_ERROR_LAPACK)
            M_found = 0
            break
        end
        # Every rank uses the root's Ritz basis and termination decision.
        rank_r,M,epsout,converged = MPI.bcast((rank_r,M,epsout,epsout<=eps_tolerance),root,comm)
        if M == 0
            # No Ritz value in the interval: it holds no eigenvalues.
            M_found = 0
            epsout = zero(T)
            converged = true
            break
        end
        MPI.Bcast!(lambda,root,comm)
        MPI.Bcast!(q,root,comm)
        MPI.Bcast!(res,root,comm)
        M_found = M
        if converged
            info_code = _feast_exit_info(true,M,M0,N)
            break
        end
        active_dim = rank_r
        copyto!(view(Q,:,1:active_dim),view(q,:,1:active_dim))
    end

    if info_code == Int(Feast_SUCCESS)
        info_code = _feast_exit_info(converged, M_found, M0, N)
    end
    mpi_state.info = info_code
    mpi_state.converged = converged
    mpi_state.epsout = epsout
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
    work_block = view(work, :, 1:M0)
    rhs_real = Matrix{T}(undef, N, M0)
    workc_local = Matrix{Complex{T}}(undef, N, M0)

    # Process each local contour point
    for e in 1:local_ne
        z = local_Zne[e]
        w = local_Wne[e]

        try
            # Form and factorize (z*B - A)
            system_matrix = z * B - A
            F = lu(system_matrix)

            # Right-hand side B * Q0 stays a real product (BLAS for dense B),
            # is widened into the reused complex buffer, and solved in place.
            mul!(rhs_real, B, work_block)
            copyto!(workc_local, rhs_real)
            ldiv!(F, workc_local)

            # Factor of 2 for half-contour symmetry
            weight = 2 * w

            # Accumulate moment contribution (views: no per-(i,j) column copies)
            for j in 1:M0
                for i in 1:M0
                    inner_product = dot(view(work, :, i), view(workc_local, :, j))
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
                               M::Int, comm::MPI.Comm;
                               scale = _feast_default_residual_scale(lambda, M)) where T<:Real

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
    N = size(A, 1)
    Aq_buf = Vector{T}(undef, N)
    Bq_buf = Vector{T}(undef, N)
    for j in start_idx:min(end_idx, M)
        # Normalize by B*q as well as λ, reusing the multiplication buffers.
        qj = view(q, :, j)
        mul!(Aq_buf, A, qj)
        mul!(Bq_buf, B, qj)
        @. Aq_buf -= lambda[j] * Bq_buf
        local_res[j] = _feast_scaled_residual(Aq_buf, Bq_buf, lambda[j], scale)
    end

    # Reduce residuals across all ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end
