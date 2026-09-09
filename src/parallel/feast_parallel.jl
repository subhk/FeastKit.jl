# Parallel FeastKit implementation
# Each contour point is solved independently using distributed computing

using Distributed
using SharedArrays
using LinearAlgebra

function _pfeast_dense_shifted_system!(dest::AbstractMatrix{Complex{T}},
                                       z::Complex{T},
                                       A::AbstractMatrix{T},
                                       B::AbstractMatrix{T}) where T<:Real
    @boundscheck size(dest) == size(A) == size(B) || throw(DimensionMismatch("matrix sizes must match"))
    @inbounds @simd for i in eachindex(dest, A, B)
        dest[i] = z * B[i] - A[i]
    end
    return dest
end

function _pfeast_store_complex_moments!(Aq::AbstractMatrix{Complex{T}},
                                        Sq::AbstractMatrix{Complex{T}},
                                        Q_proj::AbstractMatrix{Complex{T}},
                                        temp::AbstractMatrix{Complex{T}},
                                        workc::AbstractMatrix{Complex{T}},
                                        weight::Complex{T},
                                        z::Complex{T}) where T<:Real
    weighted_z = weight * z
    @inbounds for j in axes(temp, 2), i in axes(temp, 1)
        val = temp[i, j]
        Aq[i, j] = weight * val
        Sq[i, j] = weighted_z * val
    end
    @inbounds for j in axes(workc, 2), i in axes(workc, 1)
        Q_proj[i, j] = weight * workc[i, j]
    end
    return Aq, Sq, Q_proj
end

function _pfeast_store_real_moments!(Aq::AbstractMatrix{T},
                                     Sq::AbstractMatrix{T},
                                     Q_proj::AbstractMatrix{T},
                                     temp::AbstractMatrix{Complex{T}},
                                     workc::AbstractMatrix{Complex{T}},
                                     weight::Complex{T},
                                     z::Complex{T}) where T<:Real
    weighted_z = weight * z
    @inbounds for j in axes(temp, 2), i in axes(temp, 1)
        val = temp[i, j]
        Aq[i, j] = real(weight * val)
        Sq[i, j] = real(weighted_z * val)
    end
    @inbounds for j in axes(workc, 2), i in axes(workc, 1)
        Q_proj[i, j] = real(weight * workc[i, j])
    end
    return Aq, Sq, Q_proj
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

    # Preserve the historical contract: with threading off and no worker
    # processes, computation proceeds serially over contour points.
    if !use_threads && nworkers() == 1
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

    # Scratch reused across refinement loops. Kept complex through the reduced
    # Rayleigh-Ritz problem (mirrors the serial dense Hermitian path); only the
    # final eigenvectors are projected to real for output.
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
            # Orthonormalize / rank-compress the (complex) filtered subspace.
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
                                         Int(Feast_SUCCESS), epsout, loop)
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
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    M = M_found
    M > 1 && feast_sort!(lambda, q, res, M)
    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                             info_code, epsout, loop_done)
end

# Threaded computation of moments (returns complex moments and Q_proj contributions)
# Each contour point is assigned to a different thread for parallel execution
function pfeast_compute_moments_threaded(A::Matrix{T}, B::Matrix{T},
                                        work::Matrix{T}, contour::FeastContour{T},
                                        M0::Int; verbose::Bool=false) where T<:Real
    ne = length(contour.Zne)
    N = size(A, 1)
    nthreads = Threads.nthreads()

    # Pre-allocate thread-local storage for complex moments AND Q_proj contributions
    # Now returns tuple of (Aq, Sq, Q_proj_contribution)
    moments = Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, ne)

    # Track which thread processes each contour point (for verification)
    thread_assignments = verbose ? zeros(Int, ne) : Int[]
    work_block = view(work, :, 1:M0)

    # Parallel loop over integration points - each contour point goes to a thread
    Threads.@threads for e in 1:ne
        # Record which thread is handling this contour point
        verbose && (thread_assignments[e] = Threads.threadid())

        z = contour.Zne[e]
        w = contour.Wne[e]

        # Local complex moment matrices for this thread
        Aq_local = zeros(Complex{T}, M0, M0)
        Sq_local = zeros(Complex{T}, M0, M0)
        Q_proj_local = zeros(Complex{T}, N, M0)
        system_matrix = Matrix{Complex{T}}(undef, N, N)
        rhs = Matrix{Complex{T}}(undef, N, M0)
        workc_local = Matrix{Complex{T}}(undef, N, M0)
        temp = Matrix{Complex{T}}(undef, M0, M0)

        try
            # Form and factorize (z*B - A)
            _pfeast_dense_shifted_system!(system_matrix, z, A, B)

            # LU factorization
            F = lu!(system_matrix)

            # Right-hand side: B * Q0
            mul!(rhs, B, work_block)

            # Solve all linear systems at once: Y = (z*B - A) \ (B*Q0)
            copyto!(workc_local, rhs)
            ldiv!(F, workc_local)

            # Compute complex moment contribution with factor of 2 for half-contour symmetry
            # Keep as complex - symmetrization happens when accumulating
            mul!(temp, adjoint(work_block), workc_local)
            weight = 2 * w  # Factor of 2 for conjugate half-contour
            _pfeast_store_complex_moments!(Aq_local, Sq_local, Q_proj_local,
                                           temp, workc_local, weight, z)

            moments[e] = (Aq_local, Sq_local, Q_proj_local)

        catch err
            # Handle factorization failure
            @warn "Factorization failed for contour point $e: $err"
            fill!(Aq_local, zero(Complex{T}))
            fill!(Sq_local, zero(Complex{T}))
            fill!(Q_proj_local, zero(Complex{T}))
            moments[e] = (Aq_local, Sq_local, Q_proj_local)
        end
    end

    # Print distribution info if verbose
    if verbose
        println("Contour point distribution across $nthreads threads:")
        for tid in 1:nthreads
            points = findall(==(tid), thread_assignments)
            if !isempty(points)
                println("  Thread $tid: contour points $points")
            end
        end
    end

    return moments
end

"""
    pfeast_show_distribution(ne::Int; use_threads::Bool=true)

Display how contour points would be distributed across processors/threads.

# Arguments
- `ne::Int`: Number of contour points (integration nodes)
- `use_threads::Bool`: If true, show thread distribution; if false, show worker distribution
"""
function pfeast_show_distribution(ne::Int; use_threads::Bool=true)
    if use_threads
        nthreads = Threads.nthreads()
        println("Thread-based distribution for $ne contour points across $nthreads threads:")

        # Simulate the distribution that Threads.@threads would do
        # Julia uses a static schedule by default
        points_per_thread = cld(ne, nthreads)
        for tid in 1:nthreads
            start_idx = (tid - 1) * points_per_thread + 1
            end_idx = min(tid * points_per_thread, ne)
            if start_idx <= ne
                println("  Thread $tid: contour points $start_idx:$end_idx")
            end
        end
    else
        nw = nworkers()
        println("Distributed computation for $ne contour points across $nw workers:")
        chunks = distribute_contour_points(ne, nw)
        for (i, chunk) in enumerate(chunks)
            if !isempty(chunk)
                println("  Worker $(workers()[i]): contour points $chunk")
            end
        end
    end
end

# Distributed computation of moments (returns complex moments and Q_proj)
function pfeast_compute_moments_distributed(A::Matrix{T}, B::Matrix{T},
                                           work::Matrix{T}, contour::FeastContour{T},
                                           M0::Int) where T<:Real
    ne = length(contour.Zne)

    # Distribute work across available workers
    if nworkers() == 1
        @warn "No worker processes available, falling back to serial computation"
        return pfeast_compute_moments_serial(A, B, work, contour, M0)
    end

    # Split contour points among workers
    work_chunks = distribute_contour_points(ne, nworkers())

    # Compute moments in parallel using @distributed
    moment_futures = Vector{Future}(undef, length(work_chunks))

    for (i, chunk) in enumerate(work_chunks)
        moment_futures[i] = @spawnat workers()[i] begin
            pfeast_solve_contour_chunk(A, B, work, contour.Zne, contour.Wne, chunk, M0)
        end
    end

    # Collect results (complex moments and Q_proj)
    moments = Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, ne)
    for (i, future) in enumerate(moment_futures)
        chunk_moments = fetch(future)
        chunk = work_chunks[i]
        for (j, e) in enumerate(chunk)
            moments[e] = chunk_moments[j]
        end
    end

    return moments
end

# Serial fallback computation (returns complex moments and Q_proj)
function pfeast_compute_moments_serial(A::Matrix{T}, B::Matrix{T},
                                      work::Matrix{T}, contour::FeastContour{T},
                                      M0::Int) where T<:Real
    ne = length(contour.Zne)
    moments = Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, ne)

    for e in 1:ne
        moments[e] = pfeast_solve_single_point(A, B, work, contour.Zne[e],
                                             contour.Wne[e], M0)
    end

    return moments
end

# Solve a chunk of contour points on a worker (returns complex moments and Q_proj)
function pfeast_solve_contour_chunk(A::Matrix{T}, B::Matrix{T},
                                   work::Matrix{T}, contour_nodes::Vector{Complex{T}},
                                   contour_weights::Vector{Complex{T}},
                                   chunk_indices::Vector{Int}, M0::Int) where T<:Real
    chunk_moments = Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, length(chunk_indices))

    for (i, e) in enumerate(chunk_indices)
        z = contour_nodes[e]
        w = contour_weights[e]
        chunk_moments[i] = pfeast_solve_single_point(A, B, work, z, w, M0)
    end

    return chunk_moments
end

# Solve for a single contour point (returns complex moments and Q_proj contribution)
function pfeast_solve_single_point(A::Matrix{T}, B::Matrix{T}, work::Matrix{T},
                                  z::Complex{T}, w::Complex{T}, M0::Int) where T<:Real
    N = size(A, 1)

    # Local complex moment matrices and Q_proj contribution
    Aq_local = zeros(Complex{T}, M0, M0)
    Sq_local = zeros(Complex{T}, M0, M0)
    Q_proj_local = zeros(Complex{T}, N, M0)

    work_block = view(work, :, 1:M0)
    try
        # Factorize (z*B - A) in place on the broadcast temporary — one copy.
        F = lu!(z .* B .- A)

        # Right-hand side: B * Q0 (real BLAS product), promoted to complex once
        # in workc_local, then solved in place.
        workc_local = Complex{T}.(B * work_block)
        ldiv!(F, workc_local)

        # Compute complex moment contribution with factor of 2 for half-contour symmetry
        temp = work_block' * workc_local
        weight = 2 * w  # Factor of 2 for conjugate half-contour
        @. Aq_local = weight * temp
        @. Sq_local = (weight * z) * temp

        # Accumulate filtered subspace contribution
        @. Q_proj_local = weight * workc_local

    catch err
        @warn "Linear solve failed for contour point z=$z: $err"
        # Return zero contribution
    end

    return (Aq_local, Sq_local, Q_proj_local)
end

# Distribute contour points among workers
function distribute_contour_points(ne::Int, nw::Int)
    points_per_worker = div(ne, nw)
    remainder = ne % nw
    
    chunks = Vector{Vector{Int}}(undef, nw)
    start_idx = 1
    
    for i in 1:nw
        chunk_size = points_per_worker + (i <= remainder ? 1 : 0)
        chunks[i] = collect(start_idx:(start_idx + chunk_size - 1))
        start_idx += chunk_size
    end
    
    return chunks
end

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

    contour = feast_contour(Emin, Emax, fpm)
    Zne = contour.Zne
    Wne = contour.Wne

    eps_tol = feast_tolerance(fpm, T)
    max_loops = fpm[4]

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

    # Factorize each shifted system once; reuse across refinement loops.
    factors = _pfeast_factorize_contour_sparse(A, B, Zne, use_threads)
    B_is_identity = (B == I)   # standard problem: skip per-loop identity matmuls
    verbose && println("pfeast_scsrgv!: $(length(Zne)) contour points, threads=$(Threads.nthreads())")

    for loop in 1:max_loops
        loop_done = loop
        fill!(Q_proj, zero(Complex{T}))

        qblk = view(Q, :, 1:active_dim)
        bq = view(BQ_loop, :, 1:active_dim)
        B_is_identity ? copyto!(bq, qblk) : mul!(bq, B, qblk)
        _pfeast_accumulate_qproj_sparse!(view(Q_proj, :, 1:active_dim), factors,
                                         bq, Wne, use_threads)

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
                                         Int(Feast_SUCCESS), epsout, loop)
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
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    M = M_found
    M > 1 && feast_sort!(lambda, q, res, M)
    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M],
                             info_code, epsout, loop_done)
end

# Threaded sparse moment computation (returns moments and Q_proj)
function pfeast_compute_sparse_moments_threaded(A::SparseMatrixCSC{T,Int},
                                               B::SparseMatrixCSC{T,Int},
                                               work::Matrix{T}, contour::FeastContour{T},
                                               M0::Int; verbose::Bool=false) where T<:Real
    ne = length(contour.Zne)
    N = size(A, 1)
    nthreads = Threads.nthreads()
    moments = Vector{Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}(undef, ne)
    thread_assignments = verbose ? zeros(Int, ne) : Int[]
    work_block = view(work, :, 1:M0)

    Threads.@threads for e in 1:ne
        # Record which thread is handling this contour point
        verbose && (thread_assignments[e] = Threads.threadid())

        z = contour.Zne[e]
        w = contour.Wne[e]

        # Local moment matrices and Q_proj contribution
        Aq_local = zeros(T, M0, M0)
        Sq_local = zeros(T, M0, M0)
        Q_proj_local = zeros(T, N, M0)
        workc_local = Matrix{Complex{T}}(undef, N, M0)
        temp = Matrix{Complex{T}}(undef, M0, M0)

        try
            # Form sparse system matrix
            system_matrix = z * B - A

            # Sparse LU factorization
            F = lu(system_matrix)

            # Right-hand side B * Q0 lands directly in the complex solve buffer,
            # then the sparse solve runs in place — no separate rhs copy.
            mul!(workc_local, B, work_block)
            ldiv!(F, workc_local)

            # Compute moment contribution with factor of 2 for half-contour symmetry
            mul!(temp, adjoint(work_block), workc_local)
            weight = 2 * w  # Factor of 2 for conjugate half-contour
            _pfeast_store_real_moments!(Aq_local, Sq_local, Q_proj_local,
                                        temp, workc_local, weight, z)

            moments[e] = (Aq_local, Sq_local, Q_proj_local)

        catch err
            @warn "Sparse solve failed for contour point $e: $err"
            fill!(Aq_local, zero(T))
            fill!(Sq_local, zero(T))
            fill!(Q_proj_local, zero(T))
            moments[e] = (Aq_local, Sq_local, Q_proj_local)
        end
    end

    # Print distribution info if verbose
    if verbose
        println("Contour point distribution across $nthreads threads (sparse):")
        for tid in 1:nthreads
            points = findall(==(tid), thread_assignments)
            if !isempty(points)
                println("  Thread $tid: contour points $points")
            end
        end
    end

    return moments
end

# Distributed sparse moment computation (returns moments and Q_proj)
function pfeast_compute_sparse_moments_distributed(A::SparseMatrixCSC{T,Int},
                                                  B::SparseMatrixCSC{T,Int},
                                                  work::Matrix{T}, contour::FeastContour{T},
                                                  M0::Int) where T<:Real
    ne = length(contour.Zne)

    if nworkers() == 1
        return pfeast_compute_sparse_moments_serial(A, B, work, contour, M0)
    end

    # Distribute work
    work_chunks = distribute_contour_points(ne, nworkers())

    # Parallel computation
    moment_futures = Vector{Future}(undef, length(work_chunks))

    for (i, chunk) in enumerate(work_chunks)
        moment_futures[i] = @spawnat workers()[i] begin
            pfeast_solve_sparse_chunk(A, B, work, contour.Zne, contour.Wne, chunk, M0)
        end
    end

    # Collect results (moments and Q_proj)
    moments = Vector{Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}(undef, ne)
    for (i, future) in enumerate(moment_futures)
        chunk_moments = fetch(future)
        chunk = work_chunks[i]
        for (j, e) in enumerate(chunk)
            moments[e] = chunk_moments[j]
        end
    end

    return moments
end

# Serial sparse computation (returns moments and Q_proj)
function pfeast_compute_sparse_moments_serial(A::SparseMatrixCSC{T,Int},
                                             B::SparseMatrixCSC{T,Int},
                                             work::Matrix{T}, contour::FeastContour{T},
                                             M0::Int) where T<:Real
    ne = length(contour.Zne)
    moments = Vector{Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}(undef, ne)

    for e in 1:ne
        z = contour.Zne[e]
        w = contour.Wne[e]
        moments[e] = pfeast_solve_sparse_single_point(A, B, work, z, w, M0)
    end

    return moments
end

# Solve sparse chunk on worker (returns moments and Q_proj)
function pfeast_solve_sparse_chunk(A::SparseMatrixCSC{T,Int},
                                  B::SparseMatrixCSC{T,Int},
                                  work::Matrix{T}, contour_nodes::Vector{Complex{T}},
                                  contour_weights::Vector{Complex{T}},
                                  chunk_indices::Vector{Int}, M0::Int) where T<:Real
    chunk_moments = Vector{Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}(undef, length(chunk_indices))

    for (i, e) in enumerate(chunk_indices)
        z = contour_nodes[e]
        w = contour_weights[e]
        chunk_moments[i] = pfeast_solve_sparse_single_point(A, B, work, z, w, M0)
    end

    return chunk_moments
end

# Solve single sparse point (returns moments and Q_proj contribution)
function pfeast_solve_sparse_single_point(A::SparseMatrixCSC{T,Int},
                                         B::SparseMatrixCSC{T,Int},
                                         work::Matrix{T}, z::Complex{T}, w::Complex{T},
                                         M0::Int) where T<:Real
    N = size(A, 1)
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    Q_proj_local = zeros(T, N, M0)

    try
        # Form and solve sparse system
        system_matrix = z * B - A
        F = lu(system_matrix)

        # Right-hand side: B * Q0
        rhs = B * work[:, 1:M0]

        # Solve: Y = (z*B - A) \ (B*Q0)
        workc_local = F \ rhs

        # Compute moment contribution with factor of 2 for half-contour symmetry
        temp = work[:, 1:M0]' * workc_local
        weight = 2 * w  # Factor of 2 for conjugate half-contour
        Aq_local .= real.(weight .* temp)
        Sq_local .= real.(weight * z .* temp)

        # Accumulate filtered subspace contribution
        Q_proj_local .= real.(weight .* workc_local)

    catch err
        @warn "Sparse linear solve failed: $err"
    end

    return (Aq_local, Sq_local, Q_proj_local)
end

# Parallel performance monitoring
function pfeast_benchmark(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int;
                         max_workers::Int = nworkers(), use_threads::Bool = true)
    # Benchmark parallel vs serial performance
    
    println("FeastKit Parallel Performance Benchmark")
    println("="^50)
    println("Matrix size: $(size(A, 1))")
    println("Search interval: $interval")
    println("Subspace size: $M0")
    println("Available threads: $(Threads.nthreads())")
    println("Available workers: $(nworkers())")
    
    # Serial timing
    println("\nSerial execution:")
    serial_time = @elapsed begin
        result_serial = feast(A, B, interval, M0=M0)
    end
    println("Time: $(round(serial_time, digits=3)) seconds")
    println("Eigenvalues found: $(result_serial.M)")

    # Parallel timing (threaded)
    if use_threads && Threads.nthreads() > 1
        println("\nParallel execution (threads):")
        parallel_time = @elapsed begin
            if isa(A, SparseMatrixCSC)
                result_parallel = pfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2], M0, zeros(Int, 64), use_threads=true)
            else
                result_parallel = pfeast_sygv!(copy(A), copy(B), interval[1], interval[2], M0, zeros(Int, 64), use_threads=true)
            end
        end
        println("Time: $(round(parallel_time, digits=3)) seconds")
        println("Eigenvalues found: $(result_parallel.M)")
        println("Speedup: $(round(serial_time/parallel_time, digits=2))x")
    end

    # Parallel timing (distributed)
    if nworkers() > 1
        println("\nParallel execution (distributed):")
        distributed_time = @elapsed begin
            if isa(A, SparseMatrixCSC)
                result_distributed = pfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2], M0, zeros(Int, 64), use_threads=false)
            else
                result_distributed = pfeast_sygv!(copy(A), copy(B), interval[1], interval[2], M0, zeros(Int, 64), use_threads=false)
            end
        end
        println("Time: $(round(distributed_time, digits=3)) seconds")
        println("Eigenvalues found: $(result_distributed.M)")
        println("Speedup: $(round(serial_time/distributed_time, digits=2))x")
    end
    
    return nothing
end
