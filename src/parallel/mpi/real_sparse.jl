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
    # Resolve the defaults on every rank before anything reads fpm: building
    # the contour fills them in on root alone (see mpi_feast_sygv!).
    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))

    contour, custom = _mpi_contour(T,fpm,Emin,Emax,root,comm)
    offset = _feast_filter_offset(T, fpm, custom)
    ne = length(contour.Zne)
    Zne_global, Wne_global = contour.Zne, contour.Wne

    mpi_state = MPIFeastState{T}(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm),
                                 N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    eps_tolerance = feast_tolerance(fpm, T)
    max_loops = fpm[4]
    keep = Vector{Bool}(undef, M0)

    # Real trial basis, filtered subspace and reduced pencil, as in
    # mpi_feast_sygv!; only the shifted solves are complex.
    Q = Matrix{T}(undef, N, M0)
    _feast_seeded_subspace!(Q)
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

    Q_proj_local = Matrix{Complex{T}}(undef, N, M0)   # this rank's resolvent sum
    Q_filtered = Matrix{T}(undef, N, M0)              # its real part, summed on root
    rhs = Matrix{Complex{T}}(undef, N, M0)            # B*Q for the shifted solves
    # UMFPACK factors are always ComplexF64 (it promotes Float32 inputs), so the
    # in-place solve buffer must match that eltype, not Complex{T}.
    Y_loop = Matrix{Complex{promote_type(T, Float64)}}(undef, N, M0)
    q_basis = Matrix{T}(undef, N, M0)
    AQ = Matrix{T}(undef, N, M0)
    BQm = Matrix{T}(undef, N, M0)
    Sq = Matrix{T}(undef, M0, M0)   # Qᵀ A Q
    Aq = Matrix{T}(undef, M0, M0)   # Qᵀ B Q
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

    empty_dist = -one(T)   # root only
    # Loops are counted from 0 and fpm[4] refinements follow the first sweep,
    # as in the serial drivers.
    for loop in 0:max_loops
        loop_done = loop
        mpi_state.loop = loop

        # Distributed contour solves: each rank applies only its local resolvents.
        qblk = view(Q, :, 1:active_dim)
        bq = view(BQm, :, 1:active_dim)
        rhs_blk = view(rhs, :, 1:active_dim)
        qpl = view(Q_proj_local, :, 1:active_dim)
        qf = view(Q_filtered, :, 1:active_dim)
        Yv = view(Y_loop, :, 1:active_dim)
        success = try
            # B*Q in real arithmetic, widened once for the complex solves.
            B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
            copyto!(rhs_blk, bq)
            _mpi_local_projection!(qpl,local_factors,rhs_blk,mpi_state.local_Wne,Yv; scale=2)
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
        # bandwidth and capped scaling). Results are broadcast back. The
        # filtered subspace is the real part of the sum, so the ranks send real
        # parts, half the data. The reduce is in place: root's qf becomes the
        # global sum, no fresh buffer.
        @. qf = real(qpl)
        MPI.Reduce!(qf, MPI.SUM, root, comm)
        offset == zero(T) || (qf .+= offset .* qblk)

        rank_r = active_dim
        M = 0
        status = 0
        if rank == root
            try
                # qf is the filtered image of the previous loop's Ritz vectors:
                # classify them by filter response before a new Rayleigh-Ritz.
                kept = loop >= _FEAST_SPURIOUS_MIN_LOOPS ?
                    _feast_screen_spurious!(keep, qf, qblk, res,
                                            M_found, eps_tolerance, lambda, Zne_global,
                                            Wne_global; half=true, offset=offset) : nothing
                if kept !== nothing
                    M = _feast_compact_pairs!(lambda, q, res, keep, M_found)
                    epsout = M > 0 ? maximum(view(res, 1:M)) : zero(T)
                    status = 1
                else
                    rank_r = _feast_qr_compress!(q_basis, Q_filtered, active_dim;
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
                        mul!(Sq_r, transpose(q_rank), AQ_r)
                        _feast_symmetric_part!(Sq_r)
                        if B_is_identity
                            fill!(Aq_r, zero(T))
                            @inbounds for i in 1:rank_r
                                Aq_r[i, i] = one(T)
                            end
                        else
                            mul!(BQ_r, B, q_rank)
                            mul!(Aq_r, transpose(q_rank), BQ_r)
                            _feast_symmetric_part!(Aq_r)
                        end
                        local lambda_red, v_red
                        try
                            Fr = eigen(Symmetric(Sq_r), Symmetric(Aq_r))
                            lambda_red = Fr.values
                            v_red = Fr.vectors
                        catch err
                            (err isa PosDefException || err isa LinearAlgebra.LAPACKException ||
                             err isa SingularException) || rethrow(err)
                            Fr = eigen(Sq_r, Aq_r)
                            lambda_red = real.(Fr.values)
                            v_red = real.(Fr.vectors)
                        end
                        # Ritz vectors on the orthonormal basis.
                        mul!(view(q, :, 1:rank_r), q_rank, v_red)
                        copyto!(view(lambda, 1:rank_r), lambda_red)
                        M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                                        Emin, Emax, rank_r)
                        if M == 0
                            # No Ritz value in the interval. It is empty only
                            # once the subspace has settled outside it (see
                            # _feast_region_looks_empty); otherwise filter again.
                            empty, empty_dist = _feast_region_looks_empty(
                                view(lambda, 1:rank_r), (Emin + Emax) / 2,
                                (Emax - Emin) / 2, Zne_global, Wne_global, true,
                                empty_dist; offset=offset)
                            epsout = zero(T)
                            if empty
                                status = 1
                            elseif loop == max_loops
                                status = 2
                                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                            else
                                status = 0
                            end
                        else
                            empty_dist = -one(T)
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

        # Broadcast the control word and residual before any rank can stop.
        MPI.Bcast!(flags, root, comm)
        rank_r = flags[1]; M = flags[2]; status = flags[3]; info_code = flags[4]
        epsbuf[1] = epsout
        MPI.Bcast!(epsbuf, root, comm)
        epsout = epsbuf[1]
        mpi_state.epsout = epsout
        if status == 2
            M_found = 0
            break
        end
        MPI.Bcast!(lambda, root, comm)
        MPI.Bcast!(q, root, comm)
        MPI.Bcast!(res, root, comm)
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
