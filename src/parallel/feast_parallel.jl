# Parallel FeastKit implementation
# Each contour point is solved independently using distributed computing

using Distributed
using SharedArrays
using LinearAlgebra

# Parallel FeastKit for real symmetric problems
function pfeast_sygv!(A::Matrix{T}, B::Matrix{T}, 
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                      use_threads::Bool = true) where T<:Real
    # Parallel FeastKit for dense real symmetric generalized eigenvalue problem
    
    N = size(A, 1)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Generate integration contour
    contour = feast_contour(Emin, Emax, fpm)
    ne = length(contour.Zne)
    
    # Initialize moment matrices (shared across workers)
    Aq = zeros(T, M0, M0)
    Sq = zeros(T, M0, M0)
    
    # Initialize workspace for subspace
    workspace = FeastWorkspaceReal{T}(N, M0)
    
    # Generate random initial subspace
    randn!(workspace.work)
    
    # Initialize Feast parameters
    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Main Feast refinement loop
    for loop in 1:max_loops
        # Reset moment matrices
        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        
        # Parallel computation of moments for each contour point
        if use_threads && Threads.nthreads() > 1
            # Use threading for shared memory parallelism
            moments = pfeast_compute_moments_threaded(A, B, workspace.work, contour, M0)
        else
            # Use distributed computing for multiple processes
            moments = pfeast_compute_moments_distributed(A, B, workspace.work, contour, M0)
        end
        
        # Accumulate moments
        for (aq_contrib, sq_contrib) in moments
            Aq .+= aq_contrib
            Sq .+= sq_contrib
        end
        
        # Solve reduced eigenvalue problem
        try
            F = eigen(Aq, Sq)
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)
            
            # Filter eigenvalues in the search interval
            M = 0
            valid_indices = Int[]
            for i in 1:M0
                if feast_inside_contour(lambda_red[i], Emin, Emax)
                    M += 1
                    push!(valid_indices, i)
                end
            end
            
            if M == 0
                return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                       Int(Feast_ERROR_NO_CONVERGENCE), zero(T), loop)
            end
            
            # Update eigenvectors
            workspace.lambda[1:M] = lambda_red[valid_indices]
            workspace.q[:, 1:M] = workspace.work * v_red[:, valid_indices]
            
            # Compute residuals
            feast_residual!(A, B, workspace.lambda, workspace.q, workspace.res, M)
            epsout = maximum(workspace.res[1:M])
            
            # Check convergence
            if epsout <= eps_tolerance
                feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
                return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M, 
                                       workspace.res[1:M], Int(Feast_SUCCESS), epsout, loop)
            end
            
            # Prepare for next iteration
            workspace.work[:, 1:M] = workspace.q[:, 1:M]
            
        catch e
            return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                   Int(Feast_ERROR_LAPACK), zero(T), loop)
        end
    end
    
    # Maximum iterations reached
    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M, 
                           workspace.res[1:M], Int(Feast_ERROR_NO_CONVERGENCE), 
                           maximum(workspace.res[1:M]), max_loops)
end

# Threaded computation of moments
function pfeast_compute_moments_threaded(A::Matrix{T}, B::Matrix{T}, 
                                        work::Matrix{T}, contour::FeastContour{T}, 
                                        M0::Int) where T<:Real
    ne = length(contour.Zne)
    N = size(A, 1)
    
    # Pre-allocate thread-local storage
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    
    # Parallel loop over integration points
    Threads.@threads for e in 1:ne
        z = contour.Zne[e]
        w = contour.Wne[e]
        
        # Local moment matrices for this thread
        Aq_local = zeros(T, M0, M0)
        Sq_local = zeros(T, M0, M0)
        
        # Form and factorize (z*B - A)
        system_matrix = z * B - A
        
        try
            # LU factorization
            F = lu(system_matrix)
            
            # Solve linear systems for each column of work
            workc_local = zeros(Complex{T}, N, M0)
            rhs = B * work[:, 1:M0]
            
            for j in 1:M0
                workc_local[:, j] = F \ rhs[:, j]
            end
            
            # Compute moment contribution
            for j in 1:M0
                for i in 1:M0
                    moment_val = real(w * dot(work[:, i], workc_local[:, j]))
                    Aq_local[i, j] = moment_val
                    Sq_local[i, j] = moment_val * real(z)
                end
            end
            
            moments[e] = (Aq_local, Sq_local)
            
        catch err
            # Handle factorization failure
            @warn "Factorization failed for contour point $e: $err"
            moments[e] = (zeros(T, M0, M0), zeros(T, M0, M0))
        end
    end
    
    return moments
end

# Distributed computation of moments
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
    
    # Collect results
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    for (i, future) in enumerate(moment_futures)
        chunk_moments = fetch(future)
        chunk = work_chunks[i]
        for (j, e) in enumerate(chunk)
            moments[e] = chunk_moments[j]
        end
    end
    
    return moments
end

# Serial fallback computation
function pfeast_compute_moments_serial(A::Matrix{T}, B::Matrix{T}, 
                                      work::Matrix{T}, contour::FeastContour{T}, 
                                      M0::Int) where T<:Real
    ne = length(contour.Zne)
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    
    for e in 1:ne
        moments[e] = pfeast_solve_single_point(A, B, work, contour.Zne[e], 
                                             contour.Wne[e], M0)
    end
    
    return moments
end

# Solve a chunk of contour points on a worker
function pfeast_solve_contour_chunk(A::Matrix{T}, B::Matrix{T}, 
                                   work::Matrix{T}, contour_nodes::Vector{Complex{T}}, 
                                   contour_weights::Vector{Complex{T}},
                                   chunk_indices::Vector{Int}, M0::Int) where T<:Real
    chunk_moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, length(chunk_indices))
    
    for (i, e) in enumerate(chunk_indices)
        z = contour_nodes[e]
        w = contour_weights[e]
        chunk_moments[i] = pfeast_solve_single_point(A, B, work, z, w, M0)
    end
    
    return chunk_moments
end

# Solve for a single contour point
function pfeast_solve_single_point(A::Matrix{T}, B::Matrix{T}, work::Matrix{T}, 
                                  z::Complex{T}, w::Complex{T}, M0::Int) where T<:Real
    N = size(A, 1)
    
    # Local moment matrices
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    
    try
        # Form and factorize (z*B - A)
        system_matrix = z * B - A
        F = lu(system_matrix)
        
        # Right-hand side
        rhs = B * work[:, 1:M0]
        
        # Solve linear systems
        workc_local = F \ rhs
        
        # Compute moment contribution
        for j in 1:M0
            for i in 1:M0
                moment_val = real(w * dot(work[:, i], workc_local[:, j]))
                Aq_local[i, j] = moment_val
                Sq_local[i, j] = moment_val * real(z)
            end
        end
        
    catch err
        @warn "Linear solve failed for contour point z=$z: $err"
        # Return zero contribution
    end
    
    return (Aq_local, Sq_local)
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
                        use_threads::Bool = true) where T<:Real
    # Parallel FeastKit for sparse matrices
    
    N = size(A, 1)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Generate integration contour
    contour = feast_contour(Emin, Emax, fpm)
    ne = length(contour.Zne)
    
    # Initialize workspace
    workspace = FeastWorkspaceReal{T}(N, M0)
    randn!(workspace.work)
    
    # Initialize Feast parameters
    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Main Feast refinement loop
    for loop in 1:max_loops
        # Compute moments in parallel
        if use_threads && Threads.nthreads() > 1
            moments = pfeast_compute_sparse_moments_threaded(A, B, workspace.work, contour, M0)
        else
            moments = pfeast_compute_sparse_moments_distributed(A, B, workspace.work, contour, M0)
        end
        
        # Accumulate moments
        Aq = zeros(T, M0, M0)
        Sq = zeros(T, M0, M0)
        for (aq_contrib, sq_contrib) in moments
            Aq .+= aq_contrib
            Sq .+= sq_contrib
        end
        
        # Solve reduced eigenvalue problem and check convergence
        try
            F = eigen(Aq, Sq)
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)
            
            # Filter and extract eigenvalues
            M = 0
            valid_indices = Int[]
            for i in 1:M0
                if feast_inside_contour(lambda_red[i], Emin, Emax)
                    M += 1
                    push!(valid_indices, i)
                end
            end
            
            if M == 0
                return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                       Int(Feast_ERROR_NO_CONVERGENCE), zero(T), loop)
            end
            
            # Update solution
            workspace.lambda[1:M] = lambda_red[valid_indices]
            workspace.q[:, 1:M] = workspace.work * v_red[:, valid_indices]
            
            # Check convergence
            feast_residual!(A, B, workspace.lambda, workspace.q, workspace.res, M)
            epsout = maximum(workspace.res[1:M])
            
            if epsout <= eps_tolerance
                feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
                return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M, 
                                       workspace.res[1:M], Int(Feast_SUCCESS), epsout, loop)
            end
            
            workspace.work[:, 1:M] = workspace.q[:, 1:M]
            
        catch e
            return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                   Int(Feast_ERROR_LAPACK), zero(T), loop)
        end
    end
    
    # Did not converge
    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M, 
                           workspace.res[1:M], Int(Feast_ERROR_NO_CONVERGENCE), 
                           maximum(workspace.res[1:M]), max_loops)
end

# Threaded sparse moment computation
function pfeast_compute_sparse_moments_threaded(A::SparseMatrixCSC{T,Int}, 
                                               B::SparseMatrixCSC{T,Int},
                                               work::Matrix{T}, contour::FeastContour{T}, 
                                               M0::Int) where T<:Real
    ne = length(contour.Zne)
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    
    Threads.@threads for e in 1:ne
        z = contour.Zne[e]
        w = contour.Wne[e]
        
        # Local moment matrices
        Aq_local = zeros(T, M0, M0)
        Sq_local = zeros(T, M0, M0)
        
        try
            # Form sparse system matrix
            system_matrix = z * B - A
            
            # Sparse LU factorization
            F = lu(system_matrix)
            
            # Right-hand side
            rhs = B * work[:, 1:M0]
            
            # Solve sparse linear systems
            workc_local = F \ rhs
            
            # Compute moment contribution
            for j in 1:M0
                for i in 1:M0
                    moment_val = real(w * dot(work[:, i], workc_local[:, j]))
                    Aq_local[i, j] = moment_val
                    Sq_local[i, j] = moment_val * real(z)
                end
            end
            
            moments[e] = (Aq_local, Sq_local)
            
        catch err
            @warn "Sparse solve failed for contour point $e: $err"
            moments[e] = (zeros(T, M0, M0), zeros(T, M0, M0))
        end
    end
    
    return moments
end

# Distributed sparse moment computation
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
    
    # Collect results
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    for (i, future) in enumerate(moment_futures)
        chunk_moments = fetch(future)
        chunk = work_chunks[i]
        for (j, e) in enumerate(chunk)
            moments[e] = chunk_moments[j]
        end
    end
    
    return moments
end

# Serial sparse computation
function pfeast_compute_sparse_moments_serial(A::SparseMatrixCSC{T,Int}, 
                                             B::SparseMatrixCSC{T,Int},
                                             work::Matrix{T}, contour::FeastContour{T}, 
                                             M0::Int) where T<:Real
    ne = length(contour.Zne)
    moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, ne)
    
    for e in 1:ne
        z = contour.Zne[e]
        w = contour.Wne[e]
        moments[e] = pfeast_solve_sparse_single_point(A, B, work, z, w, M0)
    end
    
    return moments
end

# Solve sparse chunk on worker
function pfeast_solve_sparse_chunk(A::SparseMatrixCSC{T,Int}, 
                                  B::SparseMatrixCSC{T,Int},
                                  work::Matrix{T}, contour_nodes::Vector{Complex{T}},
                                  contour_weights::Vector{Complex{T}}, 
                                  chunk_indices::Vector{Int}, M0::Int) where T<:Real
    chunk_moments = Vector{Tuple{Matrix{T}, Matrix{T}}}(undef, length(chunk_indices))
    
    for (i, e) in enumerate(chunk_indices)
        z = contour_nodes[e]
        w = contour_weights[e]
        chunk_moments[i] = pfeast_solve_sparse_single_point(A, B, work, z, w, M0)
    end
    
    return chunk_moments
end

# Solve single sparse point
function pfeast_solve_sparse_single_point(A::SparseMatrixCSC{T,Int}, 
                                         B::SparseMatrixCSC{T,Int},
                                         work::Matrix{T}, z::Complex{T}, w::Complex{T}, 
                                         M0::Int) where T<:Real
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    
    try
        # Form and solve sparse system
        system_matrix = z * B - A
        F = lu(system_matrix)
        rhs = B * work[:, 1:M0]
        workc_local = F \ rhs
        
        # Compute moments
        for j in 1:M0
            for i in 1:M0
                moment_val = real(w * dot(work[:, i], workc_local[:, j]))
                Aq_local[i, j] = moment_val
                Sq_local[i, j] = moment_val * real(z)
            end
        end
        
    catch err
        @warn "Sparse linear solve failed: $err"
    end
    
    return (Aq_local, Sq_local)
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
    println("Time: $(serial_time:.3f) seconds")
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
        println("Time: $(parallel_time:.3f) seconds")
        println("Eigenvalues found: $(result_parallel.M)")
        println("Speedup: $(serial_time/parallel_time:.2f)x")
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
        println("Time: $(distributed_time:.3f) seconds")
        println("Eigenvalues found: $(result_distributed.M)")
        println("Speedup: $(serial_time/distributed_time:.2f)x")
    end
    
    return nothing
end
