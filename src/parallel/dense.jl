# The contour sweeps thread over contour points and keep BLAS single-threaded
# meanwhile. Saving and restoring the BLAS thread count per call is not safe
# when solves overlap: the second caller would save the "1" set by the first
# and restore it last. Count the active regions and restore once.
const _PFEAST_BLAS_LOCK = ReentrantLock()
const _PFEAST_BLAS_DEPTH = Ref(0)
const _PFEAST_BLAS_SAVED = Ref(1)

function _with_single_blas_thread(f)
    lock(_PFEAST_BLAS_LOCK) do
        if _PFEAST_BLAS_DEPTH[] == 0
            _PFEAST_BLAS_SAVED[] = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
        end
        _PFEAST_BLAS_DEPTH[] += 1
    end
    try
        return f()
    finally
        lock(_PFEAST_BLAS_LOCK) do
            _PFEAST_BLAS_DEPTH[] -= 1
            _PFEAST_BLAS_DEPTH[] == 0 && BLAS.set_num_threads(_PFEAST_BLAS_SAVED[])
        end
    end
end

# Parallel FeastKit for real symmetric problems
const _PFeastLU{T} = LinearAlgebra.LU{Complex{T}, Matrix{Complex{T}}, Vector{Int}}

# Factorize every shifted system (z*B - A) once. The points are independent, so
# the factorizations are optionally threaded; BLAS is pinned to one thread inside
# the region to avoid nthreads × BLAS oversubscription. These factorizations are
# reused across all refinement loops (mirrors the serial dense factor cache),
# which is the dominant cost — recomputing them per loop made the old threaded
# path slower than serial.
function _pfeast_factorize_contour(A::Matrix{T}, B::Matrix{T},
                                   Zne::Vector{Complex{T}},
                                   use_threads::Bool) where T<:Real
    ne = length(Zne)
    factors = Vector{_PFeastLU{T}}(undef, ne)
    if use_threads && Threads.nthreads() > 1 && ne > 1
        _with_single_blas_thread() do
            # lu! factorizes the broadcast temporary in place — no second copy.
            Threads.@threads for e in 1:ne
                factors[e] = lu!(Zne[e] .* B .- A)
            end
        end
    else
        for e in 1:ne
            factors[e] = lu!(Zne[e] .* B .- A)
        end
    end
    return factors
end

# Accumulate the filtered subspace Q_proj = sum 2*Wne[e] * (z*B - A)^{-1} (B*Q)
# using the cached factorizations and a single precomputed BQ = B*Q. Contour
# points are independent → threaded; each per-point triangular solve is cheap
# relative to the one-time factorization above.
function _pfeast_accumulate_qproj!(Q_proj::AbstractMatrix{Complex{T}},
                                   factors::Vector{_PFeastLU{T}},
                                   BQ::AbstractMatrix{Complex{T}},
                                   Wne::Vector{Complex{T}},
                                   use_threads::Bool) where T<:Real
    ne = length(factors)
    N, M = size(BQ)
    if use_threads && Threads.nthreads() > 1 && ne > 1
        # Chunk the points so each task owns one solve buffer and one local
        # accumulator: 2 allocations per chunk instead of one retained N×M
        # matrix per contour point. Chunks are indexed by position (never by
        # threadid()), so task migration cannot alias buffers.
        nchunks = min(Threads.nthreads(), ne)
        chunks = collect(Iterators.partition(1:ne, cld(ne, nchunks)))
        partials = Vector{Matrix{Complex{T}}}(undef, length(chunks))
        _with_single_blas_thread() do
            Threads.@threads for ci in eachindex(chunks)
                local Yt = Matrix{Complex{T}}(undef, N, M)
                local acc = zeros(Complex{T}, N, M)
                for e in chunks[ci]
                    copyto!(Yt, BQ)
                    ldiv!(factors[e], Yt)
                    @. acc += (2 * Wne[e]) * Yt
                end
                partials[ci] = acc
            end
        end
        for c in partials
            Q_proj .+= c
        end
    else
        Yserial = Matrix{Complex{T}}(undef, N, M)
        for e in 1:ne
            copyto!(Yserial, BQ)
            ldiv!(factors[e], Yserial)
            @. Q_proj += (2 * Wne[e]) * Yserial
        end
    end
    # The lower half contributes the conjugate for a real trial subspace.
    # Complete the projector before QR; retaining its imaginary part changes
    # the filtered subspace and can discard enclosed eigendirections.
    Q_proj .= real.(Q_proj)
    return Q_proj
end

function pfeast_sygv!(A::Matrix{T}, B::Matrix{T},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                      use_threads::Bool = true, verbose::Bool = false) where T<:Real
    # Parallel dense real-symmetric generalized FEAST: the contour-point solves
    # (the expensive, embarrassingly parallel part) are threaded, or spread over
    # the Distributed workers with use_threads=false.
    return _pfeast_real_symmetric!(A, B, Emin, Emax, M0, fpm, use_threads, verbose,
                                   "pfeast_sygv!"; warn_serial=true)
end

# The real symmetric driver behind pfeast_sygv! and pfeast_scsrgv!; only the
# shifted factorizations and solves depend on the storage. The rest mirrors the
# serial kernel feast_srci!, including pivoted-QR rank compression, so an
# oversized trial subspace (M0 greater than the number of eigenvalues in the
# interval) does not make the projected pencil rank deficient — that was the
# bug that produced wrong eigenpairs in the old moment-based parallel paths.
#
# As in feast_srci!, the trial subspace, its filtered image and the reduced
# pencil are real; only the shifted solves are complex. These drivers used to
# keep all of it complex, and a product of a real dense A or B with a complex
# block does not use BLAS: at N = 2000 the three such products of a
# refinement loop took longer than all of its contour solves.
function _pfeast_real_symmetric!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                 Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                                 use_threads::Bool, verbose::Bool, name::String;
                                 warn_serial::Bool = false) where T<:Real
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))

    distributed = !use_threads && _distributed_backend_ready()
    # Preserve the historical contract: with threading off and no worker
    # processes, computation proceeds serially over contour points.
    if warn_serial && !use_threads && !distributed
        @warn "No worker processes available, falling back to serial computation"
    end

    # A registered custom contour replaces the built-in one, as in the serial
    # drivers and the RCI kernel.
    contour = feast_get_custom_contour(T, fpm)
    offset = _feast_filter_offset(T, fpm, contour !== nothing)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne

    eps_tol = feast_tolerance(fpm, T)
    max_loops = fpm[4]
    keep = Vector{Bool}(undef, M0)

    # Trial subspace: the serial kernel's deterministic seed.
    Q = Matrix{T}(undef, N, M0)
    _feast_seeded_subspace!(Q)
    active_dim = M0

    # Complex storage only for the shifted solves: their right-hand side B*Q
    # and the resolvent sum over the upper half of the contour. For a real
    # trial subspace the lower half contributes the conjugate of that sum, so
    # the filtered subspace is its real part.
    rhs = Matrix{Complex{T}}(undef, N, M0)
    Q_proj = Matrix{Complex{T}}(undef, N, M0)
    Q_filtered = Matrix{T}(undef, N, M0)
    q_basis = Matrix{T}(undef, N, M0)
    AQ = Matrix{T}(undef, N, M0)
    BQ = Matrix{T}(undef, N, M0)
    Aq = Matrix{T}(undef, M0, M0)   # Qᵀ A Q
    Bq = Matrix{T}(undef, M0, M0)   # Qᵀ B Q
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

    # Factorize each shifted system once and reuse across refinement loops.
    # With workers and threading disabled, each contour chunk is factorized on
    # its worker and the factors stay there; a sweep ships only B*Q.
    factors = distributed ? _pfeast_factorize_contour_distributed(A, B, Zne) :
                            _pfeast_factorize_contour(A, B, Zne, use_threads)
    B_is_identity = (B == I)   # standard problem: skip per-loop identity matmuls
    res_scale = _feast_residual_floor(
        _feast_spectral_scale(A, B_is_identity ? nothing : B, Q),
        _feast_residual_scale(Emin, Emax))
    verbose && println("$name: $(length(Zne)) contour points, threads=$(Threads.nthreads())")

    try
        empty_dist = -one(T)
        # Loops are counted from 0 and fpm[4] refinements follow the first sweep,
        # as in the serial drivers.
        for loop in 0:max_loops
            loop_done = loop
            qblk = view(Q, :, 1:active_dim)

            # Parallel contour sweep with cached factorizations:
            # Q_proj = sum 2*Wne * (z*B - A)^{-1} (B*Q). B*Q is formed once, in
            # real arithmetic, and widened for the complex solves.
            bq = view(BQ, :, 1:active_dim)
            B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
            rhs_blk = view(rhs, :, 1:active_dim)
            copyto!(rhs_blk, bq)
            proj = view(Q_proj, :, 1:active_dim)
            fill!(proj, zero(Complex{T}))
            _pfeast_accumulate_qproj!(proj, factors, rhs_blk, Wne, use_threads)
            qf = view(Q_filtered, :, 1:active_dim)
            @. qf = real(proj)
            offset == zero(T) || (qf .+= offset .* qblk)

            # The sweep has filtered the previous loop's Ritz vectors (Q); their
            # filter response separates genuine pairs from spurious ones.
            if loop >= _FEAST_SPURIOUS_MIN_LOOPS
                kept = _feast_screen_spurious!(keep, qf, qblk, res, M_found, eps_tol,
                                               lambda, Zne, Wne; half=true, offset=offset)
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
                # Orthonormalize / rank-compress the filtered subspace.
                rank = _feast_qr_compress!(q_basis, Q_filtered, active_dim;
                                           rank_tol=sqrt(eps(T)))
                if rank == 0
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end

                q_rank = view(q_basis, :, 1:rank)
                AQ_r = view(AQ, :, 1:rank)
                BQ_r = view(BQ, :, 1:rank)
                Aq_r = view(Aq, 1:rank, 1:rank)
                Bq_r = view(Bq, 1:rank, 1:rank)

                # Reduced symmetric-definite pencil: Aq = Qᵀ A Q, Bq = Qᵀ B Q.
                mul!(AQ_r, A, q_rank)
                mul!(Aq_r, transpose(q_rank), AQ_r)
                _feast_symmetric_part!(Aq_r)
                if B_is_identity
                    fill!(Bq_r, zero(T))
                    @inbounds for i in 1:rank
                        Bq_r[i, i] = one(T)
                    end
                else
                    mul!(BQ_r, B, q_rank)
                    mul!(Bq_r, transpose(q_rank), BQ_r)
                    _feast_symmetric_part!(Bq_r)
                end

                # Solve Aq*v = lambda*Bq*v; the fallback covers a reduced B that
                # is not numerically positive definite.
                local lambda_red, v_red
                try
                    Fr = eigen(Symmetric(Aq_r), Symmetric(Bq_r))
                    lambda_red = Fr.values
                    v_red = Fr.vectors
                catch err
                    (err isa PosDefException || err isa LinearAlgebra.LAPACKException ||
                     err isa SingularException) || rethrow(err)
                    Fr = eigen(Aq_r, Bq_r)
                    lambda_red = real.(Fr.values)
                    v_red = real.(Fr.vectors)
                end

                # Ritz vectors on the orthonormal basis.
                mul!(view(q, :, 1:rank), q_rank, v_red)
                copyto!(view(lambda, 1:rank), lambda_red)

                M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                                Emin, Emax, rank)
                if M == 0
                    # No Ritz value in the interval. It is empty only once the
                    # subspace has settled outside it; until then, filter again
                    # (see _feast_region_looks_empty).
                    empty, empty_dist = _feast_region_looks_empty(
                        view(lambda, 1:rank), (Emin + Emax) / 2, (Emax - Emin) / 2,
                        Zne, Wne, true, empty_dist; offset=offset)
                    if empty || loop == max_loops
                        return FeastResult{T, T}(T[], zeros(T, N, 0), 0, T[],
                                                 empty ? Int(Feast_SUCCESS) :
                                                         Int(Feast_ERROR_NO_CONVERGENCE),
                                                 zero(T), loop)
                    end
                    M_found = 0
                    active_dim = rank
                    copyto!(view(Q, :, 1:active_dim), view(q, :, 1:active_dim))
                    continue
                end
                empty_dist = -one(T)

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
                @warn "$name reduced eigenproblem failed" exception=err
                break
            end
        end

        # Did not converge within max_loops (or broke early).
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
