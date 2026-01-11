# Parallel FeastKit implementation
# Each contour point is solved independently using distributed computing

using Distributed
using SharedArrays
using LinearAlgebra

# Parallel FeastKit for real symmetric problems
function pfeast_sygv!(A::Matrix{T}, B::Matrix{T},
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                      use_threads::Bool = true, verbose::Bool = false) where T<:Real
    # Parallel FeastKit for dense real symmetric generalized eigenvalue problem

    N = size(A, 1)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    # Initialize Feast parameters first (needed for contour generation)
    feastdefault!(fpm)

    # Generate integration contour
    contour = feast_contour(Emin, Emax, fpm)
    ne = length(contour.Zne)

    # Initialize moment matrices (shared across workers)
    Aq = zeros(T, M0, M0)
    Sq = zeros(T, M0, M0)

    # Initialize workspace for subspace
    workspace = FeastWorkspaceReal{T}(N, M0)

    # Generate random initial subspace and normalize columns (matches serial FEAST)
    randn!(workspace.work)
    for j in 1:M0
        col_norm = norm(view(workspace.work, :, j))
        if col_norm > 0
            workspace.work[:, j] ./= col_norm
        end
    end
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Complex moment matrices for accumulation
    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)

    # Filtered subspace accumulator (spectral projector applied to Q)
    Q_proj = zeros(Complex{T}, N, M0)

    # Main Feast refinement loop
    for loop in 1:max_loops
        # Reset complex moment matrices and Q_proj
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(Q_proj, zero(Complex{T}))

        # Parallel computation of moments for each contour point
        if use_threads && Threads.nthreads() > 1
            # Use threading for shared memory parallelism
            # Only print verbose output on first iteration to avoid spam
            moments = pfeast_compute_moments_threaded(A, B, workspace.work, contour, M0; verbose=(verbose && loop == 1))
        else
            # Use distributed computing for multiple processes
            moments = pfeast_compute_moments_distributed(A, B, workspace.work, contour, M0)
        end

        # Accumulate complex moments AND Q_proj
        for (aq_contrib, sq_contrib, qproj_contrib) in moments
            zAq .+= aq_contrib
            zSq .+= sq_contrib
            Q_proj .+= qproj_contrib
        end

        # Solve reduced eigenvalue problem
        try
            # For half-contour integration, take real part directly (no 0.5 symmetrization needed)
            Aq .= real.(zAq)
            Sq .= real.(zSq)

            # Symmetrize reduced matrices using wrapper
            Aq_sym = Symmetric(Aq)
            Sq_sym = Symmetric(Sq)

            # IMPORTANT: Solve Sq*x = lambda*Aq*x (not Aq*x = lambda*Sq*x)
            # This is because zAq ≈ Q'*P*Q and zSq ≈ Q'*A*P*Q where P is the spectral projector
            F = try
                eigen(Sq_sym, Aq_sym)
            catch e
                # Fall back to general solver if not positive definite
                eigen(Sq, Aq)
            end

            lambda_red = real.(F.values)
            v_red = real.(F.vectors)

            # Get real part of filtered subspace for projection
            Q_proj_real = real.(Q_proj)

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q
            # This is the core FEAST algorithm - use spectral projector output
            for idx in 1:M0
                workspace.q[:, idx] = Q_proj_real * view(v_red, :, idx)
                workspace.lambda[idx] = lambda_red[idx]
            end

            # Reorder: put eigenvalues inside interval first while maintaining pairing
            inside_mask = [Emin <= workspace.lambda[i] <= Emax for i in 1:M0]
            inside_indices = findall(inside_mask)
            outside_indices = findall(.!inside_mask)
            perm = vcat(inside_indices, outside_indices)

            workspace.lambda[1:M0] = workspace.lambda[perm]
            workspace.q[:, 1:M0] = workspace.q[:, perm]

            M = length(inside_indices)

            if M == 0
                return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[],
                                       Int(Feast_ERROR_NO_CONVERGENCE), zero(T), loop)
            end

            # Normalize eigenvectors (matches dense solver)
            for j in 1:M
                q_norm = norm(view(workspace.q, :, j))
                if q_norm > 0
                    workspace.q[:, j] ./= q_norm
                end
            end

            # Compute residuals
            feast_residual!(A, B, workspace.lambda, workspace.q, workspace.res, M)
            epsout = maximum(workspace.res[1:M])

            # Check convergence
            if epsout <= eps_tolerance
                feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
                return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                                       workspace.res[1:M], Int(Feast_SUCCESS), epsout, loop)
            end

            # Prepare for next iteration - update ALL M0 columns to maintain subspace
            workspace.work[:, 1:M0] = workspace.q[:, 1:M0]
            
        catch e
            return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                   Int(Feast_ERROR_LAPACK), zero(T), loop)
        end
    end
    
    # Maximum iterations reached
    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    # Handle case where no eigenvalues found (avoid maximum on empty array)
    final_epsout = M > 0 ? maximum(workspace.res[1:M]) : zero(T)
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                           workspace.res[1:M], Int(Feast_ERROR_NO_CONVERGENCE),
                           final_epsout, max_loops)
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
    thread_assignments = zeros(Int, ne)

    # Parallel loop over integration points - each contour point goes to a thread
    Threads.@threads for e in 1:ne
        # Record which thread is handling this contour point
        thread_assignments[e] = Threads.threadid()

        z = contour.Zne[e]
        w = contour.Wne[e]

        # Local complex moment matrices for this thread
        Aq_local = zeros(Complex{T}, M0, M0)
        Sq_local = zeros(Complex{T}, M0, M0)
        Q_proj_local = zeros(Complex{T}, N, M0)

        # Form and factorize (z*B - A)
        system_matrix = z * B - A

        try
            # LU factorization
            F = lu(system_matrix)

            # Right-hand side: B * Q0
            rhs = Complex{T}.(B * work[:, 1:M0])

            # Solve all linear systems at once: Y = (z*B - A) \ (B*Q0)
            workc_local = F \ rhs

            # Compute complex moment contribution with factor of 2 for half-contour symmetry
            # Keep as complex - symmetrization happens when accumulating
            temp = work[:, 1:M0]' * workc_local
            weight = 2 * w  # Factor of 2 for conjugate half-contour
            Aq_local .= weight .* temp
            Sq_local .= weight * z .* temp

            # Accumulate filtered subspace contribution
            Q_proj_local .= weight .* workc_local

            moments[e] = (Aq_local, Sq_local, Q_proj_local)

        catch err
            # Handle factorization failure
            @warn "Factorization failed for contour point $e: $err"
            moments[e] = (zeros(Complex{T}, M0, M0), zeros(Complex{T}, M0, M0), zeros(Complex{T}, N, M0))
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

    try
        # Form and factorize (z*B - A)
        system_matrix = z * B - A
        F = lu(system_matrix)

        # Right-hand side: B * Q0 (complex)
        rhs = Complex{T}.(B * work[:, 1:M0])

        # Solve linear systems: Y = (z*B - A) \ (B*Q0)
        workc_local = F \ rhs

        # Compute complex moment contribution with factor of 2 for half-contour symmetry
        temp = work[:, 1:M0]' * workc_local
        weight = 2 * w  # Factor of 2 for conjugate half-contour
        Aq_local .= weight .* temp
        Sq_local .= weight * z .* temp

        # Accumulate filtered subspace contribution
        Q_proj_local .= weight .* workc_local

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
function pfeast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                        use_threads::Bool = true, verbose::Bool = false) where T<:Real
    # Parallel FeastKit for sparse matrices

    N = size(A, 1)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    # Initialize Feast parameters first (needed for contour generation)
    feastdefault!(fpm)

    # Generate integration contour
    contour = feast_contour(Emin, Emax, fpm)
    ne = length(contour.Zne)

    # Initialize workspace with normalized random vectors (matches serial FEAST)
    workspace = FeastWorkspaceReal{T}(N, M0)
    randn!(workspace.work)
    for j in 1:M0
        col_norm = norm(view(workspace.work, :, j))
        if col_norm > 0
            workspace.work[:, j] ./= col_norm
        end
    end
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]

    # Filtered subspace accumulator (spectral projector applied to Q)
    Q_proj = zeros(T, N, M0)

    # Main Feast refinement loop
    for loop in 1:max_loops
        # Reset Q_proj for this iteration
        fill!(Q_proj, zero(T))

        # Compute moments in parallel
        if use_threads && Threads.nthreads() > 1
            # Only print verbose output on first iteration to avoid spam
            moments = pfeast_compute_sparse_moments_threaded(A, B, workspace.work, contour, M0; verbose=(verbose && loop == 1))
        else
            moments = pfeast_compute_sparse_moments_distributed(A, B, workspace.work, contour, M0)
        end

        # Accumulate moments AND Q_proj
        Aq = zeros(T, M0, M0)
        Sq = zeros(T, M0, M0)
        for (aq_contrib, sq_contrib, qproj_contrib) in moments
            Aq .+= aq_contrib
            Sq .+= sq_contrib
            Q_proj .+= qproj_contrib
        end

        # Solve reduced eigenvalue problem and check convergence
        try
            # Use Symmetric wrapper directly - moments already computed correctly
            Aq_sym = Symmetric(Aq)
            Sq_sym = Symmetric(Sq)

            # IMPORTANT: Solve Sq*x = lambda*Aq*x (not Aq*x = lambda*Sq*x)
            F = try
                eigen(Sq_sym, Aq_sym)
            catch e
                # Fall back to general solver if not positive definite
                eigen(Sq, Aq)
            end

            lambda_red = real.(F.values)
            v_red = real.(F.vectors)

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q
            for idx in 1:M0
                workspace.q[:, idx] = Q_proj * view(v_red, :, idx)
                workspace.lambda[idx] = lambda_red[idx]
            end

            # Reorder: put eigenvalues inside interval first while maintaining pairing
            inside_mask = [Emin <= workspace.lambda[i] <= Emax for i in 1:M0]
            inside_indices = findall(inside_mask)
            outside_indices = findall(.!inside_mask)
            perm = vcat(inside_indices, outside_indices)

            workspace.lambda[1:M0] = workspace.lambda[perm]
            workspace.q[:, 1:M0] = workspace.q[:, perm]

            M = length(inside_indices)

            if M == 0
                return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[],
                                       Int(Feast_ERROR_NO_CONVERGENCE), zero(T), loop)
            end

            # Normalize eigenvectors (matches dense solver)
            for j in 1:M
                q_norm = norm(view(workspace.q, :, j))
                if q_norm > 0
                    workspace.q[:, j] ./= q_norm
                end
            end

            # Check convergence
            feast_residual!(A, B, workspace.lambda, workspace.q, workspace.res, M)
            epsout = maximum(workspace.res[1:M])

            if epsout <= eps_tolerance
                feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
                return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                                       workspace.res[1:M], Int(Feast_SUCCESS), epsout, loop)
            end

            # Prepare for next iteration - update ALL M0 columns to maintain subspace
            workspace.work[:, 1:M0] = workspace.q[:, 1:M0]
            
        catch e
            return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                   Int(Feast_ERROR_LAPACK), zero(T), loop)
        end
    end
    
    # Did not converge
    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    # Handle case where no eigenvalues found (avoid maximum on empty array)
    final_epsout = M > 0 ? maximum(workspace.res[1:M]) : zero(T)
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                           workspace.res[1:M], Int(Feast_ERROR_NO_CONVERGENCE),
                           final_epsout, max_loops)
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
    thread_assignments = zeros(Int, ne)

    Threads.@threads for e in 1:ne
        # Record which thread is handling this contour point
        thread_assignments[e] = Threads.threadid()

        z = contour.Zne[e]
        w = contour.Wne[e]

        # Local moment matrices and Q_proj contribution
        Aq_local = zeros(T, M0, M0)
        Sq_local = zeros(T, M0, M0)
        Q_proj_local = zeros(T, N, M0)

        try
            # Form sparse system matrix
            system_matrix = z * B - A

            # Sparse LU factorization
            F = lu(system_matrix)

            # Right-hand side: B * Q0
            rhs = B * work[:, 1:M0]

            # Solve sparse linear systems: Y = (z*B - A) \ (B*Q0)
            workc_local = F \ rhs

            # Compute moment contribution with factor of 2 for half-contour symmetry
            temp = work[:, 1:M0]' * workc_local
            weight = 2 * w  # Factor of 2 for conjugate half-contour
            Aq_local .= real.(weight .* temp)
            Sq_local .= real.(weight * z .* temp)

            # Accumulate filtered subspace contribution
            Q_proj_local .= real.(weight .* workc_local)

            moments[e] = (Aq_local, Sq_local, Q_proj_local)

        catch err
            @warn "Sparse solve failed for contour point $e: $err"
            moments[e] = (zeros(T, M0, M0), zeros(T, M0, M0), zeros(T, N, M0))
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
