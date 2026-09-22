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
# Distributed computation of moments (returns complex moments and Q_proj)
function pfeast_compute_moments_distributed(A::Matrix{T}, B::Matrix{T},
                                           work::Matrix{T}, contour::FeastContour{T},
                                           M0::Int) where T<:Real
    ne = length(contour.Zne)

    # Distribute work across available workers
    if !_distributed_backend_ready()
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
