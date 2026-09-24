# Parallel sparse Feast
# Sparse analogue of _pfeast_factorize_contour: factorize each shifted system
# (z*B - A) once with a sparse LU and reuse across refinement loops. The factor
# type is taken from a sample so this works for both Float64 and Float32 inputs.
function _pfeast_factorize_contour_sparse(A::SparseMatrixCSC{T,Int},
                                          B::SparseMatrixCSC{T,Int},
                                          Zne::Vector{Complex{T}},
                                          use_threads::Bool) where T<:Real
    ne = length(Zne)
    f1 = lu(Zne[1] * B - A)
    factors = Vector{typeof(f1)}(undef, ne)
    factors[1] = f1
    if use_threads && Threads.nthreads() > 1 && ne > 2
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for e in 2:ne
                factors[e] = lu(Zne[e] * B - A)
            end
        finally
            BLAS.set_num_threads(blas_threads)
        end
    else
        for e in 2:ne
            factors[e] = lu(Zne[e] * B - A)
        end
    end
    return factors
end

# Keep UMFPACK objects on their owning workers: they contain process-local
# pointers and must never be fetched/serialized back to the coordinator.
struct _PFeastDistributedFactors
    pids::Vector{Int}
    chunks::Vector{Vector{Int}}
    futures::Vector{Future}
end

function _pfeast_factorize_contour_distributed(A, B, nodes)
    pids = workers()[1:min(length(workers()), length(nodes))]
    # Functions and sparse factor types cannot be deserialized until the
    # package is loaded. Support the documented addprocs + feast workflow.
    Distributed.remotecall_eval(Main, pids, :(using FeastKit))
    chunks = distribute_contour_points(length(nodes), length(pids))
    futures = Future[]
    try
        for (pid, chunk) in zip(pids, chunks)
            push!(futures, remotecall(_pfeast_factorize_contour_sparse, pid,
                                      A, B, nodes[chunk], false))
        end
    catch
        foreach(finalize, futures)
        rethrow()
    end
    return _PFeastDistributedFactors(pids, chunks, futures)
end

function _pfeast_project_sparse_chunk(factors::Future, rhs::AbstractMatrix,
                                      weights::AbstractVector)
    partial = zeros(eltype(rhs), size(rhs))
    # This fetch occurs on the worker that owns the factorizations.
    _pfeast_accumulate_qproj_sparse!(partial, fetch(factors), rhs, weights, false)
    return partial
end

function _pfeast_accumulate_qproj_sparse!(Q_proj::AbstractMatrix,
                                          factors::_PFeastDistributedFactors,
                                          BQ::AbstractMatrix, weights::AbstractVector,
                                          use_threads::Bool)
    # Materialize only active columns before serialization, then send the same
    # RHS to each worker. Each returns one accumulated block per sweep.
    rhs = Matrix(BQ)
    pending = Future[]
    try
        for i in eachindex(factors.pids)
            push!(pending, remotecall(_pfeast_project_sparse_chunk, factors.pids[i],
                                      factors.futures[i], rhs, weights[factors.chunks[i]]))
        end
        for future in pending
            Q_proj .+= fetch(future)
        end
    finally
        foreach(finalize, pending)
    end
    return Q_proj
end

# Sparse analogue of _pfeast_accumulate_qproj!: Q_proj = sum 2*Wne[e] *
# (z*B - A)^{-1} (B*Q) using cached sparse factorizations and a precomputed BQ.
# Each per-point sparse solve runs on its own thread; results are converted to
# Complex{T} so the Float32 path (where UMFPACK promotes to ComplexF64) stays
# type-correct.
function _pfeast_accumulate_qproj_sparse!(Q_proj::AbstractMatrix{Complex{T}},
                                          factors::Vector{FAC},
                                          BQ::AbstractMatrix{Complex{T}},
                                          Wne::Vector{Complex{T}},
                                          use_threads::Bool) where {T<:Real, FAC}
    ne = length(factors)
    if use_threads && Threads.nthreads() > 1 && ne > 1
        contribs = Vector{Matrix{Complex{T}}}(undef, ne)
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for e in 1:ne
                # Scale the solve result in place; assignment converts to
                # Complex{T} only when UMFPACK promoted the eltype (Float32).
                local Yt = factors[e] \ BQ
                @. Yt *= 2 * Wne[e]
                contribs[e] = Yt
            end
        finally
            BLAS.set_num_threads(blas_threads)
        end
        for c in contribs
            Q_proj .+= c
        end
    else
        for e in 1:ne
            Ys = factors[e] \ BQ
            @. Q_proj += (2 * Wne[e]) * Ys
        end
    end
    Q_proj .= real.(Q_proj)
    return Q_proj
end

function pfeast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                        use_threads::Bool = true, verbose::Bool = false) where T<:Real
    # Parallel sparse real-symmetric generalized FEAST. Mirrors the corrected
    # dense path: cached sparse factorizations + threaded contour solves, then a
    # rank-compressed Hermitian Rayleigh-Ritz reduced problem (pivoted QR keeps
    # an oversized trial subspace from making the projected pencil rank deficient
    # — the bug that made the old moment-based path return wrong eigenpairs).
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))

    contour = feast_contour(Emin, Emax, fpm)
    Zne = contour.Zne
    Wne = contour.Wne

    eps_tol = feast_tolerance(fpm, T)
    max_loops = fpm[4]
    keep = Vector{Bool}(undef, M0)

    Q = Matrix{Complex{T}}(undef, N, M0)
    _feast_seeded_subspace_complex!(Q)
    active_dim = M0

    Q_proj = zeros(Complex{T}, N, M0)
    q_basis = Matrix{Complex{T}}(undef, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQ = Matrix{Complex{T}}(undef, N, M0)
    Sq = Matrix{Complex{T}}(undef, M0, M0)
    Aq = Matrix{Complex{T}}(undef, M0, M0)
    BQ_loop = Matrix{Complex{T}}(undef, N, M0)
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

    # With workers and threading disabled, keep each contour chunk's factors
    # on its worker and reuse them across refinement loops.
    factors = !use_threads && _distributed_backend_ready() ?
        _pfeast_factorize_contour_distributed(A, B, Zne) :
        _pfeast_factorize_contour_sparse(A, B, Zne, use_threads)
    B_is_identity = (B == I)   # standard problem: skip per-loop identity matmuls
    res_scale = _feast_residual_floor(
        _feast_spectral_scale(A, B_is_identity ? nothing : B, Q),
        _feast_residual_scale(Emin, Emax))
    verbose && println("pfeast_scsrgv!: $(length(Zne)) contour points, threads=$(Threads.nthreads())")

    try
        for loop in 1:max_loops
            loop_done = loop
            fill!(Q_proj, zero(Complex{T}))

            qblk = view(Q, :, 1:active_dim)
            bq = view(BQ_loop, :, 1:active_dim)
            B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
            _pfeast_accumulate_qproj_sparse!(view(Q_proj, :, 1:active_dim), factors,
                                             bq, Wne, use_threads)

            # The sweep has filtered the previous loop's Ritz vectors (Q); their
            # filter response separates genuine pairs from spurious ones.
            if loop - 1 >= _FEAST_SPURIOUS_MIN_LOOPS
                kept = _feast_screen_spurious!(keep, Q_proj, Q, res, M_found, eps_tol)
                if kept !== nothing
                    M = _feast_compact_pairs!(lambda, q, res, keep, M_found)
                    M > 1 && feast_sort!(lambda, q, res, M)
                    epsout = M > 0 ? maximum(view(res, 1:M)) : zero(T)
                    info_screened = M == 0 ? Int(Feast_SUCCESS) : _feast_exit_info(true, M, M0, N)
                    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                             info_screened, epsout, loop)
                end
            end

            try
                rank = _feast_qr_compress!(q_basis, Q_proj, active_dim;
                                           rank_tol=sqrt(eps(T)))
                if rank == 0
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end

                q_rank = view(q_basis, :, 1:rank)
                AQ_r = view(AQ, :, 1:rank)
                BQ_r = view(BQ, :, 1:rank)
                Sq_r = view(Sq, 1:rank, 1:rank)
                Aq_r = view(Aq, 1:rank, 1:rank)

                mul!(AQ_r, A, q_rank)
                mul!(Sq_r, adjoint(q_rank), AQ_r)
                if B_is_identity
                    fill!(Aq_r, zero(Complex{T}))
                    @inbounds for i in 1:rank
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

                for idx in 1:rank
                    mul!(qcol, q_rank, view(v_red, :, idx))
                    @inbounds for i in 1:N
                        q[i, idx] = real(qcol[i])
                    end
                    lambda[idx] = lambda_red[idx]
                end

                M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                                Emin, Emax, rank)
                if M == 0
                    # No Ritz value in the interval: it holds no eigenvalues.
                    return FeastResult{T, T}(T[], zeros(T, N, 0), 0, T[],
                                             Int(Feast_SUCCESS), zero(T), loop)
                end

                for j in 1:M
                    nrm = norm(view(q, :, j))
                    nrm > 0 && (view(q, :, j) ./= nrm)
                end

                feast_residual!(A, B, lambda, q, res, M,
                                residual_Aq, residual_Bq, residual; scale=res_scale)
                epsout = maximum(view(res, 1:M))
                M_found = M

                if epsout <= eps_tol
                    feast_sort!(lambda, q, res, M)
                    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                             _feast_exit_info(true, M, M0, N), epsout, loop)
                end

                active_dim = rank
                copyto!(view(Q, :, 1:active_dim), view(q, :, 1:active_dim))
            catch err
                info_code = Int(Feast_ERROR_LAPACK)
                @warn "pfeast_scsrgv! reduced eigenproblem failed" exception=err
                break
            end
        end

        if info_code == Int(Feast_SUCCESS)
            info_code = _feast_exit_info(false, M_found, M0, N)
        end
        M = M_found
        M > 1 && feast_sort!(lambda, q, res, M)
        return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                                 info_code, epsout, loop_done)
    finally
        # Release remote LU storage on success, non-convergence, and exceptions.
        factors isa _PFeastDistributedFactors && foreach(finalize, factors.futures)
    end
end
