# Parallel RCI (Reverse Communication Interface) for FeastKit
# Maintains the RCI interface while enabling parallel contour point computation

# Parallel RCI state management
mutable struct ParallelFeastState{T<:Real}
    # Standard RCI state
    ijob::Int
    Ze::Complex{T}
    loop::Int
    epsout::T
    mode::Int
    info::Int

    # Parallel-specific state
    contour_points::Vector{Complex{T}}
    contour_weights::Vector{Complex{T}}
    current_point::Int
    total_points::Int

    # Parallel computation results storage (complex to match pfeast_solve_single_point return type)
    moment_contributions::Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}}}

    # Worker management
    use_parallel::Bool
    use_threads::Bool

    function ParallelFeastState{T}(ne::Int, M0::Int, use_parallel::Bool=true, use_threads::Bool=true) where T<:Real
        new(
            -1,                                    # ijob (initialize)
            zero(Complex{T}),                      # Ze
            0,                                     # loop
            zero(T),                               # epsout
            0,                                     # mode
            0,                                     # info
            Vector{Complex{T}}(undef, ne),         # contour_points
            Vector{Complex{T}}(undef, ne),         # contour_weights
            1,                                     # current_point
            ne,                                    # total_points
            Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, ne),  # moment_contributions (complex)
            use_parallel,                          # use_parallel
            use_threads                            # use_threads
        )
    end
end

# Parallel FeastKit RCI for real symmetric problems
function pfeast_srci!(state::ParallelFeastState{T}, N::Int,
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                      Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{T}, 
                      res::Vector{T}) where T<:Real
    
    if state.ijob == -1  # Initialization
        # Initialize Feast parameters
        feastdefault!(fpm)
        state.info = Int(Feast_SUCCESS)
        
        # Input validation
        if N <= 0
            state.info = Int(Feast_ERROR_N)
            state.ijob = Int(Feast_RCI_DONE)
            return
        end
        
        if M0 <= 0 || M0 > N
            state.info = Int(Feast_ERROR_M0)
            state.ijob = Int(Feast_RCI_DONE)
            return
        end
        
        if Emin >= Emax
            state.info = Int(Feast_ERROR_EMIN_EMAX)
            state.ijob = Int(Feast_RCI_DONE)
            return
        end
        
        # Generate integration contour
        contour = feast_contour(Emin, Emax, fpm)
        state.contour_points = contour.Zne
        state.contour_weights = contour.Wne
        state.total_points = length(contour.Zne)
        
        # Initialize workspace
        fill!(work, zero(T))
        fill!(workc, zero(Complex{T}))
        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        fill!(lambda, zero(T))
        fill!(q, zero(T))
        fill!(res, zero(T))
        
        # Initialize moment contributions storage (complex)
        for i in 1:state.total_points
            state.moment_contributions[i] = (zeros(Complex{T}, M0, M0), zeros(Complex{T}, M0, M0))
        end
        
        state.loop = 0
        state.current_point = 1
        
        if state.use_parallel
            # Start parallel computation of all contour points
            state.ijob = Int(Feast_RCI_PARALLEL_SOLVE)
        else
            # Traditional RCI: process one point at a time
            state.Ze = state.contour_points[1]
            state.ijob = Int(Feast_RCI_FACTORIZE)
        end
        
        return
    end
    
    if state.ijob == Int(Feast_RCI_PARALLEL_SOLVE)
        # User should solve all contour points in parallel
        # This is a new RCI job type for parallel execution
        state.ijob = Int(Feast_RCI_PARALLEL_ACCUMULATE)
        return
    end
    
    if state.ijob == Int(Feast_RCI_PARALLEL_ACCUMULATE)
        # Accumulate results from parallel computation
        # Moments are stored as complex; extract real symmetric part

        # Accumulate complex moments first
        zAq = zeros(Complex{T}, size(Aq))
        zSq = zeros(Complex{T}, size(Sq))
        for i in 1:state.total_points
            Aq_contrib, Sq_contrib = state.moment_contributions[i]
            zAq .+= Aq_contrib
            zSq .+= Sq_contrib
        end

        # Extract real symmetric part (matches dense solver approach in feast_parallel.jl)
        Aq .= real.(0.5 .* (zAq .+ adjoint(zAq)))
        Sq .= real.(0.5 .* (zSq .+ adjoint(zSq)))

        # Proceed to eigenvalue computation
        state.ijob = Int(Feast_RCI_EIGEN_SOLVE)
        return
    end
    
    if state.ijob == Int(Feast_RCI_EIGEN_SOLVE)
        # Solve reduced eigenvalue problem
        # IMPORTANT: Solve Sq*v = lambda*Aq*v (not Aq*v = lambda*Sq*v)
        # Aq = sum(w * Q' * Y), Sq = sum(w * z * Q' * Y)
        try
            F = eigen(Sq[1:M0, 1:M0], Aq[1:M0, 1:M0])
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)
            
            # Filter eigenvalues in search interval
            M = 0
            for i in 1:M0
                if feast_inside_contour(lambda_red[i], Emin, Emax)
                    M += 1
                    lambda[M] = lambda_red[i]
                    q[:, M] = work[:, 1:M0] * v_red[:, i]
                end
            end
            
            state.mode = M
            
            if M == 0
                state.info = Int(Feast_ERROR_NO_CONVERGENCE)
                state.ijob = Int(Feast_RCI_DONE)
                return
            end
            
            # Compute residuals
            state.ijob = Int(Feast_RCI_MULT_A)
            return
            
        catch e
            state.info = Int(Feast_ERROR_LAPACK)
            state.ijob = Int(Feast_RCI_DONE)
            return
        end
    end
    
    if state.ijob == Int(Feast_RCI_MULT_A)
        # User has computed A*q in work, now compute residuals
        # For generalized eigenvalue problem A*x = lambda*B*x:
        # Residual r_j = ||A*q_j - lambda_j*B*q_j|| / ||A*q_j||
        # work[:, j] contains A*q[:, j]
        M = state.mode

        for j in 1:M
            # Compute relative residual: ||A*q - lambda*B*q|| / ||A*q||
            # Note: For standard eigenvalue (B=I), this reduces to ||A*q - lambda*q|| / ||A*q||
            Aq_norm = norm(work[:, j])
            if Aq_norm > 0
                # work[:, j] = A*q[:, j], so residual = ||work - lambda*B*q||
                # For now, approximate with B=I assumption (user should provide B*q if needed)
                res[j] = norm(work[:, j] - lambda[j] * q[:, j]) / Aq_norm
            else
                res[j] = zero(eltype(res))
            end
        end

        # Check convergence
        state.epsout = maximum(res[1:M])
        eps_tolerance = feast_tolerance(fpm)
        
        if state.epsout <= eps_tolerance || state.loop >= fpm[4]
            # Converged or maximum iterations reached
            feast_sort!(lambda, q, res, M)
            state.ijob = Int(Feast_RCI_DONE)
        else
            # Start new refinement loop
            state.loop += 1
            
            # Reset moment matrices
            fill!(Aq, zero(T))
            fill!(Sq, zero(T))
            
            # Use current eigenvectors as initial guess
            work[:, 1:M] = q[:, 1:M]
            
            if state.use_parallel
                state.ijob = Int(Feast_RCI_PARALLEL_SOLVE)
            else
                state.current_point = 1
                state.Ze = state.contour_points[1]
                state.ijob = Int(Feast_RCI_FACTORIZE)
            end
        end
    end
    
    # Traditional RCI paths for serial execution
    if state.ijob == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*B - A)
        state.ijob = Int(Feast_RCI_SOLVE)
        return
    end
    
    if state.ijob == Int(Feast_RCI_SOLVE)
        # User has solved linear systems for current point
        # workc contains Y = (z*B - A)^{-1} * (B*Q) where Q = work[:, 1:M0]
        e = state.current_point
        w = state.contour_weights[e]
        z = state.contour_points[e]

        # Compute moment contribution: Q' * Y (M0 × M0 matrix)
        # Factor of 2 for half-contour symmetry (conjugate half handled implicitly)
        temp = work[:, 1:M0]' * workc[:, 1:M0]
        weight = 2 * w

        # Update reduced matrices
        # IMPORTANT: Sq uses real(w * z * ...), NOT real(w * ...) * real(z)
        Aq .+= real.(weight .* temp)
        Sq .+= real.((weight * z) .* temp)

        # Move to next integration point
        state.current_point += 1
        
        if state.current_point <= state.total_points
            # More points to process
            state.Ze = state.contour_points[state.current_point]
            state.ijob = Int(Feast_RCI_FACTORIZE)
        else
            # All points processed
            state.current_point = 1
            state.ijob = Int(Feast_RCI_EIGEN_SOLVE)
        end
    end
end

# Helper function for parallel contour computation
function pfeast_compute_all_contour_points!(state::ParallelFeastState{T}, 
                                           A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                           work::Matrix{T}, M0::Int) where T<:Real
    # This function implements the parallel computation of all contour points
    # It's called by the user when ijob == Feast_RCI_PARALLEL_SOLVE
    
    ne = state.total_points
    
    if state.use_threads && Threads.nthreads() > 1
        # Use threading
        Threads.@threads for e in 1:ne
            z = state.contour_points[e]
            w = state.contour_weights[e]
            
            Aq_local, Sq_local = pfeast_solve_single_point(A, B, work, z, w, M0)
            state.moment_contributions[e] = (Aq_local, Sq_local)
        end
    elseif nworkers() > 1
        # Use distributed computing
        work_chunks = distribute_contour_points(ne, nworkers())

        # Parallel computation
        moment_futures = Vector{Future}(undef, length(work_chunks))

        for (i, chunk) in enumerate(work_chunks)
            moment_futures[i] = @spawnat workers()[i] begin
                # Use Complex{T} to match pfeast_solve_single_point return type
                chunk_results = Vector{Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}}}(undef, length(chunk))
                for (j, e) in enumerate(chunk)
                    z = state.contour_points[e]
                    w = state.contour_weights[e]
                    chunk_results[j] = pfeast_solve_single_point(A, B, work, z, w, M0)
                end
                chunk_results
            end
        end

        # Collect results
        for (i, future) in enumerate(moment_futures)
            chunk_results = fetch(future)
            chunk = work_chunks[i]
            for (j, e) in enumerate(chunk)
                state.moment_contributions[e] = chunk_results[j]
            end
        end
    else
        # Serial fallback
        for e in 1:ne
            z = state.contour_points[e]
            w = state.contour_weights[e]
            state.moment_contributions[e] = pfeast_solve_single_point(A, B, work, z, w, M0)
        end
    end
end

# Convenience wrapper for parallel Feast with automatic RCI handling
function feast_parallel(A::AbstractMatrix{T}, B::AbstractMatrix{T}, 
                        interval::Tuple{T,T}; M0::Int = 10, 
                        fpm::Union{Vector{Int}, Nothing} = nothing,
                        use_threads::Bool = true,
                        auto_rci::Bool = true) where T<:Real
    # Parallel FeastKit with automatic RCI management
    
    Emin, Emax = interval
    N = size(A, 1)
    
    # Initialize Feast parameters
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Create parallel state
    state = ParallelFeastState{T}(fpm[2], M0, true, use_threads)
    
    # Initialize workspace
    work = Matrix{T}(undef, N, M0)
    workc = Matrix{Complex{T}}(undef, N, M0)
    Aq = Matrix{T}(undef, M0, M0)
    Sq = Matrix{T}(undef, M0, M0)
    lambda = Vector{T}(undef, M0)
    q = Matrix{T}(undef, N, M0)
    res = Vector{T}(undef, M0)
    
    # Generate initial random subspace
    randn!(work)
    
    while true
        # Call parallel RCI
        pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, 
                    Emin, Emax, M0, lambda, q, res)
        
        if state.ijob == Int(Feast_RCI_PARALLEL_SOLVE) && auto_rci
            # Automatically handle parallel computation
            pfeast_compute_all_contour_points!(state, A, B, work, M0)
            
        elseif state.ijob == Int(Feast_RCI_MULT_A) && auto_rci
            # Automatically compute A*q for residual calculation
            M = state.mode
            work[:, 1:M] .= A * q[:, 1:M]
            
        elseif state.ijob == Int(Feast_RCI_DONE)
            break
            
        elseif !auto_rci
            # User must handle RCI manually
            break
        end
    end
    
    # Extract results
    M = state.mode
    return FeastResult{T, T}(lambda[1:M], q[:, 1:M], M, res[1:M], 
                           state.info, state.epsout, state.loop)
end

# Extended RCI job types for parallel operation
@enum ParallelFeastRCIJob begin
    Feast_RCI_PARALLEL_SOLVE = 50      # Solve all contour points in parallel
    Feast_RCI_PARALLEL_ACCUMULATE = 51 # Accumulate parallel results
    Feast_RCI_EIGEN_SOLVE = 52         # Solve reduced eigenvalue problem
end

# Performance monitoring for parallel RCI
function pfeast_rci_benchmark(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int;
                             compare_serial::Bool = true)
    # Compare parallel RCI performance with serial RCI
    
    println("Parallel RCI Performance Comparison")
    println("="^45)
    println("Matrix size: $(size(A, 1))")
    println("Integration points: $(feast_integration_points(zeros(Int, 64)))")
    println("Threads available: $(Threads.nthreads())")
    println("Workers available: $(nworkers())")
    
    # Parallel with threads
    if Threads.nthreads() > 1
        println("\nParallel FeastKit (threaded):")
        thread_time = @elapsed begin
            result_thread = feast_parallel(A, B, interval, M0=M0, use_threads=true)
        end
        println("Time: $(round(thread_time, digits=3)) seconds")
        println("Eigenvalues found: $(result_thread.M)")
        println("Convergence loops: $(result_thread.loop)")
    end

    # Parallel with processes
    if nworkers() > 1
        println("\nParallel FeastKit (distributed):")
        dist_time = @elapsed begin
            result_dist = feast_parallel(A, B, interval, M0=M0, use_threads=false)
        end
        println("Time: $(round(dist_time, digits=3)) seconds")
        println("Eigenvalues found: $(result_dist.M)")
        println("Convergence loops: $(result_dist.loop)")
    end

    # Serial comparison
    if compare_serial
        println("\nSerial FeastKit:")
        serial_time = @elapsed begin
            result_serial = feast(A, B, interval, M0=M0)
        end
        println("Time: $(round(serial_time, digits=3)) seconds")
        println("Eigenvalues found: $(result_serial.M)")

        if Threads.nthreads() > 1 && @isdefined(thread_time)
            println("Thread speedup: $(round(serial_time/thread_time, digits=2))x")
        end
        if nworkers() > 1 && @isdefined(dist_time)
            println("Distributed speedup: $(round(serial_time/dist_time, digits=2))x")
        end
    end
    
    return nothing
end
