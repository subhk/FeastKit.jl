# High-level MPI interfaces for FEAST
# Provides seamless integration between threaded, distributed, and MPI parallelism

# Note: Main feast() interfaces are defined in feast_interfaces.jl
# The MPI features are accessed through the existing interface with comm parameter

# Note: determine_parallel_backend is defined in feast_backend_utils.jl

# Hybrid parallel approach: MPI + threads within each MPI process
function feast_hybrid(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                     interval::Tuple{T,T}; M0::Int = 10,
                     fpm::Union{Vector{Int}, Nothing} = nothing,
                     comm::MPI.Comm = MPI.COMM_WORLD,
                     use_threads_per_rank::Bool = true) where T<:Real
    # Hybrid MPI + threading approach
    # Each MPI rank uses multiple threads for its local contour points
    
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Generate contour on root and broadcast
    contour = nothing
    if rank == 0
        contour = feast_contour(interval[1], interval[2], fpm)
    end
    
    ne = MPI.bcast(rank == 0 ? length(contour.Zne) : 0, 0, comm)
    Zne_global = MPI.bcast(rank == 0 ? contour.Zne : Vector{Complex{T}}(undef, ne), 0, comm)
    Wne_global = MPI.bcast(rank == 0 ? contour.Wne : Vector{Complex{T}}(undef, ne), 0, comm)
    
    # Distribute contour points among MPI ranks
    points_per_rank = div(ne, size)
    remainder = ne % size
    
    start_idx = rank * points_per_rank + min(rank, remainder) + 1
    local_count = points_per_rank + (rank < remainder ? 1 : 0)
    local_points = start_idx:(start_idx + local_count - 1)
    
    local_Zne = Zne_global[local_points]
    local_Wne = Wne_global[local_points]
    
    # Initialize workspace
    N = size(A, 1)
    workspace = FeastWorkspaceReal{T}(N, M0)
    randn!(workspace.work)
    MPI.Bcast!(workspace.work, 0, comm)
    
    # FEAST parameters
    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Main refinement loop
    for loop in 1:max_loops
        # Compute local moments using threads within this MPI rank
        if use_threads_per_rank && Threads.nthreads() > 1
            local_Aq, local_Sq = hybrid_compute_threaded_moments(A, B, workspace.work,
                                                                local_Zne, local_Wne, M0)
        else
            local_Aq, local_Sq = mpi_compute_local_moments(A, B, workspace.work,
                                                          local_Zne, local_Wne, M0)
        end
        
        # MPI reduction of moments
        global_Aq = MPI.Allreduce(local_Aq, MPI.SUM, comm)
        global_Sq = MPI.Allreduce(local_Sq, MPI.SUM, comm)
        
        # Solve reduced eigenvalue problem
        try
            F = eigen(global_Aq, global_Sq)
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)
            
            M = 0
            valid_indices = Int[]
            for i in 1:M0
                if feast_inside_contour(lambda_red[i], interval[1], interval[2])
                    M += 1
                    push!(valid_indices, i)
                end
            end
            
            if M == 0
                # No eigenvalues found
                return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                       Int(FEAST_ERROR_NO_CONVERGENCE.value), zero(T), loop)
            end
            
            workspace.lambda[1:M] = lambda_red[valid_indices]
            workspace.q[:, 1:M] = workspace.work * v_red[:, valid_indices]
            
            # Compute residuals
            mpi_compute_residuals!(A, B, workspace.lambda, workspace.q,
                                 workspace.res, M, comm)
            
            epsout = maximum(workspace.res[1:M])
            
            if epsout <= eps_tolerance
                # Converged
                feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
                return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                                       workspace.res[1:M], Int(FEAST_SUCCESS), epsout, loop)
            end
            
            # Prepare for next iteration
            workspace.work[:, 1:M] = workspace.q[:, 1:M]
            
        catch e
            return FeastResult{T, T}(T[], Matrix{T}(undef, N, 0), 0, T[], 
                                   Int(FEAST_ERROR_LAPACK.value), zero(T), loop)
        end
    end
    
    # Did not converge
    M = count(i -> feast_inside_contour(workspace.lambda[i], interval[1], interval[2]), 1:M0)
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                           workspace.res[1:M], Int(FEAST_ERROR_NO_CONVERGENCE.value),
                           maximum(workspace.res[1:M]), max_loops)
end

# Threaded moment computation within an MPI rank
function hybrid_compute_threaded_moments(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                        work::Matrix{T}, local_Zne::Vector{Complex{T}},
                                        local_Wne::Vector{Complex{T}}, M0::Int) where T<:Real
    
    local_ne = length(local_Zne)
    local_Aq = zeros(T, M0, M0)
    local_Sq = zeros(T, M0, M0)
    
    # Thread-local storage for moment contributions
    thread_contributions = [zeros(T, M0, M0) for _ in 1:Threads.nthreads()]
    thread_sq_contributions = [zeros(T, M0, M0) for _ in 1:Threads.nthreads()]
    
    # Parallel loop over local contour points
    Threads.@threads for e in 1:local_ne
        tid = Threads.threadid()
        z = local_Zne[e]
        w = local_Wne[e]
        
        try
            # Solve linear system for this contour point
            system_matrix = z * B - A
            F = lu(system_matrix)
            rhs = B * work[:, 1:M0]
            workc_local = F \ rhs
            
            # Accumulate to thread-local storage
            for j in 1:M0
                for i in 1:M0
                    moment_val = real(w * dot(work[:, i], workc_local[:, j]))
                    thread_contributions[tid][i, j] += moment_val
                    thread_sq_contributions[tid][i, j] += moment_val * real(z)
                end
            end
            
        catch err
            @warn "Thread $tid: Linear solve failed for contour point $e: $err"
        end
    end
    
    # Reduce thread contributions
    for tid in 1:Threads.nthreads()
        local_Aq .+= thread_contributions[tid]
        local_Sq .+= thread_sq_contributions[tid]
    end
    
    return local_Aq, local_Sq
end

# Performance comparison across all parallel backends
function feast_parallel_comparison(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int)
    # Compare performance across all available parallel backends
    
    println("FEAST Parallel Backend Comparison")
    println("="^50)
    println("Matrix size: $(size(A, 1))")
    println("Search interval: $interval")
    println("Subspace size: $M0")
    println("Available backends:")
    println("  Threads: $(Threads.nthreads())")
    println("  Workers: $(nworkers())")
    println("  MPI: $(mpi_available() ? "Yes" : "No")")
    
    results = Dict{Symbol, Any}()
    
    # Serial benchmark
    println("\n1. Serial execution:")
    serial_time = @elapsed begin
        results[:serial] = feast(A, B, interval, M0=M0, parallel=:serial)
    end
    println("   Time: $(serial_time:.3f) seconds")
    println("   Eigenvalues: $(results[:serial].M)")
    
    # Threading benchmark
    if Threads.nthreads() > 1
        println("\n2. Threading execution:")
        thread_time = @elapsed begin
            results[:threads] = feast(A, B, interval, M0=M0, parallel=:threads)
        end
        println("   Time: $(thread_time:.3f) seconds")
        println("   Eigenvalues: $(results[:threads].M)")
        println("   Speedup: $(serial_time/thread_time:.2f)x")
    end
    
    # Distributed benchmark
    if nworkers() > 1
        println("\n3. Distributed execution:")
        dist_time = @elapsed begin
            results[:distributed] = feast(A, B, interval, M0=M0, parallel=:distributed)
        end
        println("   Time: $(dist_time:.3f) seconds") 
        println("   Eigenvalues: $(results[:distributed].M)")
        println("   Speedup: $(serial_time/dist_time:.2f)x")
    end
    
    # MPI benchmark
    if mpi_available()
        println("\n4. MPI execution:")
        mpi_time = @elapsed begin
            results[:mpi] = feast(A, B, interval, M0=M0, parallel=:mpi)
        end
        println("   Time: $(mpi_time:.3f) seconds")
        println("   Eigenvalues: $(results[:mpi].M)")
        println("   Speedup: $(serial_time/mpi_time:.2f)x")
        
        # Hybrid MPI+threads if both available
        if Threads.nthreads() > 1
            println("\n5. Hybrid MPI+Threading:")
            hybrid_time = @elapsed begin
                results[:hybrid] = feast_hybrid(A, B, interval, M0=M0, use_threads_per_rank=true)
            end
            println("   Time: $(hybrid_time:.3f) seconds")
            println("   Eigenvalues: $(results[:hybrid].M)")
            println("   Speedup: $(serial_time/hybrid_time:.2f)x")
        end
    end
    
    return results
end

# Note: feast_parallel_info() is defined in feast_backend_utils.jl