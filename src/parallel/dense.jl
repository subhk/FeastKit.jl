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
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            # lu! factorizes the broadcast temporary in place — no second copy.
            Threads.@threads for e in 1:ne
                factors[e] = lu!(Zne[e] .* B .- A)
            end
        finally
            BLAS.set_num_threads(blas_threads)
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
        blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
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
        finally
            BLAS.set_num_threads(blas_threads)
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
    # Parallel dense real-symmetric generalized FEAST. The contour-point solves
    # (the expensive, embarrassingly parallel part) are threaded; the reduced
    # Rayleigh-Ritz problem mirrors the serial dense Hermitian path, including
    # pivoted-QR rank compression so an oversized trial subspace (M0 greater than
    # the number of eigenvalues in the interval) does not make the projected
    # pencil rank deficient — that was the bug that produced wrong eigenpairs.
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))

    # Preserve the historical contract: with threading off and no worker
    # processes, computation proceeds serially over contour points.
    if !use_threads && !_distributed_backend_ready()
        @warn "No worker processes available, falling back to serial computation"
    end

    contour = feast_contour(Emin, Emax, fpm)
    Zne = contour.Zne
    Wne = contour.Wne

    eps_tol = feast_tolerance(fpm, T)
    max_loops = fpm[4]

    # Trial subspace (deterministic complex seed, matches serial Hermitian path).
    Q = Matrix{Complex{T}}(undef, N, M0)
    _feast_seeded_subspace_complex!(Q)
    active_dim = M0

    # Complex storage accommodates shifted solves. The accumulated projector
    # is made real before forming the reduced Rayleigh-Ritz problem.
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

    # Factorize each shifted system once and reuse across refinement loops.
    factors = _pfeast_factorize_contour(A, B, Zne, use_threads)
    B_is_identity = (B == I)   # standard problem: skip per-loop identity matmuls
    verbose && println("pfeast_sygv!: $(length(Zne)) contour points, threads=$(Threads.nthreads())")

    for loop in 1:max_loops
        loop_done = loop
        fill!(Q_proj, zero(Complex{T}))

        # Parallel contour sweep with cached factorizations:
        # Q_proj = sum 2*Wne * (z*B - A)^{-1} (B*Q), with BQ = B*Q formed once.
        qblk = view(Q, :, 1:active_dim)
        bq = view(BQ_loop, :, 1:active_dim)
        B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
        _pfeast_accumulate_qproj!(view(Q_proj, :, 1:active_dim), factors, bq,
                                  Wne, use_threads)

        try
            # Orthonormalize / rank-compress the completed real projector.
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

            # Reduced Hermitian-definite pencil: Sq = Qᴴ A Q, Aq = Qᴴ B Q.
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

            # Solve Sq*v = lambda*Aq*v. Eigenvalues real (Hermitian-definite);
            # the fallback covers a non-positive-definite reduced B.
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

            # Project to complex eigenvectors; store the phase-corrected real
            # part for output. real(qcol) alone collapses a column whose global
            # phase sits near +-i.
            for idx in 1:rank
                mul!(qcol, q_rank, view(v_red, :, idx))
                _feast_real_column!(view(q, :, idx), qcol)
                lambda[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda, q, perm, lambda_tmp, q_tmp,
                                            Emin, Emax, rank)
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
            @warn "pfeast_sygv! reduced eigenproblem failed" exception=err
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
end
