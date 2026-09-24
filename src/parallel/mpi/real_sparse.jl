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

    contour, _ = _mpi_contour(T,fpm,Emin,Emax,root,comm)
    ne = length(contour.Zne)
    Zne_global, Wne_global = contour.Zne, contour.Wne

    mpi_state = MPIFeastState{T}(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm),
                                 N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))
    eps_tolerance = feast_tolerance(fpm, T)
    max_loops = fpm[4]
    keep = Vector{Bool}(undef, M0)

    Q_real = Matrix{T}(undef,N,M0)
    _feast_seeded_subspace!(Q_real)
    Q = Complex{T}.(Q_real)
    MPI.Bcast!(Q, root, comm)
    active_dim = M0

    # Cache local shifts only when requested; otherwise factor them on demand.
    local_factors = _mpi_factorize_contour(A,B,mpi_state.local_Zne,comm;store=fpm[10]==1)
    local_factors === nothing && return FeastResult{T,T}(T[],zeros(T,N,0),0,T[],
        Int(Feast_ERROR_LAPACK),T(Inf),0)
    B_is_identity = (B == I)   # standard problem: skip the per-loop identity matmuls
    # Every rank measures on the same broadcast basis; root's value is used.
    res_scale = _mpi_residual_floor(A, B_is_identity ? nothing : B, Q,
                                    _feast_residual_scale(Emin, Emax), root, comm)
    res_scale === nothing && return FeastResult{T,T}(T[],zeros(T,N,0),0,T[],
        Int(Feast_ERROR_LAPACK),T(Inf),0)

    Q_proj_local = Matrix{Complex{T}}(undef, N, M0)
    BQ_loop = Matrix{Complex{T}}(undef, N, M0)
    # UMFPACK factors are always ComplexF64 (it promotes Float32 inputs), so the
    # in-place solve buffer must match that eltype, not Complex{T}.
    Y_loop = Matrix{Complex{promote_type(T, Float64)}}(undef, N, M0)
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
        qpl = view(Q_proj_local, :, 1:active_dim)
        fill!(qpl, zero(Complex{T}))
        Yv = view(Y_loop, :, 1:active_dim)
        success = try
            B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
            _mpi_local_projection!(qpl,local_factors,bq,mpi_state.local_Wne,Yv; scale=2)
        catch err
            @debug "MPI local projection preparation failed" exception=err
            false
        end
        if _mpi_success_count(success,comm) != nprocs
            info_code = Int(Feast_ERROR_LAPACK)
            M_found = 0
            break
        end

        # Sum the partial filtered subspaces to ROOT ONLY. The reduced
        # Rayleigh-Ritz (QR compress + dense N×M0 work) then runs once on root
        # instead of being replicated on every rank (which contended for memory
        # bandwidth and capped scaling). Results are broadcast back. The reduce
        # is in place: root's qpl becomes the global sum, no fresh buffer.
        MPI.Reduce!(qpl, MPI.SUM, root, comm)
        @. qpl = real(qpl)

        rank_r = active_dim
        M = 0
        status = 0
        if rank == root
            try
                # qpl is the filtered image of the previous loop's Ritz vectors:
                # classify them by filter response before a new Rayleigh-Ritz.
                kept = loop - 1 >= _FEAST_SPURIOUS_MIN_LOOPS ?
                    _feast_screen_spurious!(keep, qpl, view(Q, :, 1:active_dim), res,
                                            M_found, eps_tolerance) : nothing
                if kept !== nothing
                    M = _feast_compact_pairs!(lambda, q, res, keep, M_found)
                    epsout = M > 0 ? maximum(view(res, 1:M)) : zero(T)
                    status = 1
                else
                    rank_r = _feast_qr_compress!(q_basis, qpl, active_dim;
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
                        if B_is_identity
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
                            @inbounds for i in 1:N
                                q[i, idx] = real(qcol[i])
                            end
                            lambda[idx] = lambda_red[idx]
                        end
                        M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                                        Emin, Emax, rank_r)
                        if M == 0
                            # No Ritz value in the interval: it holds no eigenvalues.
                            epsout = zero(T)
                            status = 1
                        else
                            for j in 1:M
                                nrm = norm(view(q, :, j))
                                nrm > 0 && (view(q, :, j) ./= nrm)
                            end
                            feast_residual!(A, B, lambda, q, res, M,
                                            residual_Aq, residual_Bq, residual; scale=res_scale)
                            epsout = maximum(view(res, 1:M))
                            status = epsout <= eps_tolerance ? 1 : 0
                        end
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
            mpi_state.info = _feast_exit_info(true, M, M0, N)
            feast_sort!(lambda, q, res, M)
            return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                     mpi_state.info, epsout, loop)
        end

        active_dim = rank_r
        copyto!(view(Q, :, 1:active_dim), view(q, :, 1:active_dim))
    end

    if info_code == Int(Feast_SUCCESS)
        info_code = _feast_exit_info(false, M_found, M0, N)
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
                                      M::Int, comm::MPI.Comm;
                                      scale = _feast_default_residual_scale(lambda, M)) where T<:Real

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
        local_res[j] = _feast_scaled_residual(residual, Bq, lambda[j], scale)
    end

    # Reduce across ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end
