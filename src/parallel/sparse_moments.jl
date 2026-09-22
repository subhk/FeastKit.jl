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

    if !_distributed_backend_ready()
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
