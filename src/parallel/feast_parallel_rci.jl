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
    kernel::FeastSRCIState{T}
    kernel_job::Int
    contour_solutions::Vector{Matrix{Complex{T}}}
    # Shifted factorizations for the contour points, made on the first sweep of
    # a solve and kept until it is done: a vector of factors, or a
    # `_PFeastDistributedFactors` whose factors live on the workers.
    factors::Any
    factor_key::Tuple{UInt,UInt}
    # Whether `work` holds the caller's starting subspace on initialization
    # (the default). The kernel checks such a subspace with independent probes
    # before accepting it, as it does an `initial_subspace`; `feast_parallel`
    # clears this and starts from the kernel's own subspace, which needs no
    # check (it cost one or two extra refinement loops per solve).
    user_subspace::Bool

    function ParallelFeastState{T}(ne::Int, M0::Int, use_parallel::Bool=true, use_threads::Bool=true) where T<:Real
        # These buffers are resized to the kernel's contour at initialization,
        # so `ne` only preallocates. `feastinit().fpm[2]` is an unset sentinel
        # until defaults are applied; accept it rather than fail to allocate.
        ne = max(ne, 0)
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
            use_threads,                           # use_threads
            FeastSRCIState{T}(),
            -1,
            Matrix{Complex{T}}[],
            nothing,
            (UInt(0), UInt(0)),
            true                                   # user_subspace
        )
    end
end

# Advance the shared numerical kernel; the adapter only batches contour solves.
function _pfeast_kernel_step!(state::ParallelFeastState{T}, N, work, workc,
                              Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res) where T
    job = Ref(state.kernel_job)
    z = Ref(state.Ze)
    epsout = Ref(state.epsout)
    loop = Ref(state.loop)
    mode = Ref(state.mode)
    info = Ref(state.info)
    feast_srci!(job, N, z, work, workc, Aq, Sq, fpm, epsout, loop,
                Emin, Emax, M0, lambda, q, mode, res, info; state=state.kernel)
    state.kernel_job = job[]
    state.Ze = z[]
    state.epsout = epsout[]
    state.loop = loop[]
    state.mode = mode[]
    state.info = info[]
    return nothing
end

"""
    pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)

Parallel adapter for the real symmetric RCI kernel. On initialization, `work`
holds the caller's trial subspace, which is checked with independent probes
before it is accepted, as an `initial_subspace` is. Set
`state.user_subspace = false` before the first call to start from the default
subspace instead, as `feast_parallel` does.

Handle `MULT_A` and `MULT_B` by writing
`A*q[:,1:state.mode]` or `B*q[:,1:state.mode]` into `work`, respectively.
For `PARALLEL_SOLVE`, call `pfeast_compute_all_contour_points!`; the adapter
then feeds those solves to the shared rank-compression/Rayleigh–Ritz kernel.
With `use_parallel=false`, handle ordinary FACTORIZE/SOLVE jobs instead.
"""
function pfeast_srci!(state::ParallelFeastState{T}, N::Int,
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                      Emin::T, Emax::T, M0::Int, lambda::Vector{T},
                      q::Matrix{T}, res::Vector{T}) where T<:Real
    if state.ijob == Int(Feast_RCI_DONE)
        return nothing
    elseif state.ijob == -1
        _pfeast_release_factors!(state)
        state.kernel = FeastSRCIState{T}()
        state.kernel_job = -1
        # Parallel RCI has always accepted the seed in work. Tell the shared
        # kernel whether to use it, without changing the caller's parameter
        # setting; without a seed the kernel writes its default one to work.
        old_initial = fpm[5]
        fpm[5] = state.user_subspace ? 1 : 0
        try
            _pfeast_kernel_step!(state,N,work,workc,Aq,Sq,fpm,Emin,Emax,M0,lambda,q,res)
        finally
            fpm[5] = old_initial
        end
        if state.kernel_job != Int(Feast_RCI_DONE)
            state.contour_points = copy(state.kernel.Zne)
            state.contour_weights = copy(state.kernel.Wne)
            state.total_points = length(state.contour_points)
            resize!(state.moment_contributions, state.total_points)
            state.contour_solutions = [zeros(Complex{T},N,M0) for _ in 1:state.total_points]
        end
    elseif state.ijob == Int(Feast_RCI_PARALLEL_SOLVE)
        state.ijob = Int(Feast_RCI_PARALLEL_ACCUMULATE)
        return nothing
    elseif state.ijob == Int(Feast_RCI_PARALLEL_ACCUMULATE)
        for e in 1:state.total_points
            state.current_point = e
            # FACTORIZE -> SOLVE supplies the trial basis in work.
            _pfeast_kernel_step!(state,N,work,workc,Aq,Sq,fpm,Emin,Emax,M0,lambda,q,res)
            copyto!(workc, state.contour_solutions[e])
            # SOLVE accumulates the projector; the last node starts Ritz work.
            _pfeast_kernel_step!(state,N,work,workc,Aq,Sq,fpm,Emin,Emax,M0,lambda,q,res)
            state.kernel_job == Int(Feast_RCI_DONE) && break
        end
    else
        state.kernel_job = state.ijob
        _pfeast_kernel_step!(state,N,work,workc,Aq,Sq,fpm,Emin,Emax,M0,lambda,q,res)
    end

    state.ijob = state.kernel_job
    state.ijob == Int(Feast_RCI_DONE) && _pfeast_release_factors!(state)
    if state.use_parallel && state.kernel_job == Int(Feast_RCI_FACTORIZE)
        # Refinement changes Q0 inside the shared kernel. All workers must see
        # that new basis, not the previous residual matrix left in work.
        active = state.kernel.active
        fill!(work, zero(T))
        copyto!(view(work,:,1:active), view(state.kernel.Q0,:,1:active))
        state.current_point = 1
        state.ijob = Int(Feast_RCI_PARALLEL_SOLVE)
    end
    return nothing
end

# Factorize (z*B - A) for each node, optionally threaded over the nodes.
function _pfeast_rci_factorize(A, B, nodes, threaded::Bool)
    f1 = lu(nodes[1] * B - A)
    factors = Vector{typeof(f1)}(undef, length(nodes))
    factors[1] = f1
    if threaded && Threads.nthreads() > 1 && length(nodes) > 1
        _with_single_blas_thread() do
            Threads.@threads for e in 2:length(nodes)
                factors[e] = lu(nodes[e] * B - A)
            end
        end
    else
        for e in 2:length(nodes)
            factors[e] = lu(nodes[e] * B - A)
        end
    end
    return factors
end

# On a worker: solve with each of its stored factorizations.
function _pfeast_rci_solve_chunk(factors::Future, rhs::AbstractMatrix)
    owned = fetch(factors)
    owned isa Exception && throw(owned)
    return [Matrix{eltype(rhs)}(f \ rhs) for f in owned]
end

# The factorizations of one solve, made once on its first sweep. They depend on
# A and B, so a state reused for another pencil (or a new solve, see
# `pfeast_srci!`) factorizes again.
function _pfeast_rci_factors!(state::ParallelFeastState, A, B)
    key = (objectid(A), objectid(B))
    state.factors !== nothing && state.factor_key == key && return state.factors
    _pfeast_release_factors!(state)
    nodes = state.contour_points
    state.factors = if !(state.use_threads && Threads.nthreads() > 1) &&
                       _distributed_backend_ready()
        pids = workers()[1:min(length(workers()), length(nodes))]
        Distributed.remotecall_eval(Main, pids, :(using FeastKit))
        chunks = distribute_contour_points(length(nodes), length(pids))
        futures = Future[]
        try
            for (pid, chunk) in zip(pids, chunks)
                push!(futures, remotecall(_pfeast_rci_factorize, pid, A, B, nodes[chunk], false))
            end
            for (pid, future) in zip(pids, futures)
                failure = remotecall_fetch(_pfeast_remote_failure, pid, future)
                failure === nothing ||
                    error("Shifted factorization failed on worker $pid: $failure")
            end
        catch
            foreach(finalize, futures)
            rethrow()
        end
        _PFeastDistributedFactors(pids, chunks, futures)
    else
        _pfeast_rci_factorize(A, B, nodes, state.use_threads)
    end
    state.factor_key = key
    return state.factors
end

function _pfeast_release_factors!(state::ParallelFeastState)
    state.factors isa _PFeastDistributedFactors && foreach(finalize, state.factors.futures)
    state.factors = nothing
    return state
end

# Store both legacy reduced moments and the full solved subspace. The latter
# is essential: moments alone cannot reconstruct eigenvectors in the original
# space when the initial trial basis is not already an invariant subspace.
function _pfeast_store_point!(state::ParallelFeastState, e::Int, work, Y)
    z = state.contour_points[e]
    w = state.contour_weights[e]
    moment = transpose(work) * Y
    state.moment_contributions[e] = (2 * w .* moment, 2 * w * z .* moment)
    copyto!(state.contour_solutions[e], Y)
    return nothing
end

"""
    pfeast_compute_all_contour_points!(state, A, B, work, M0)

Solve `(z*B - A) Y = B*work` at every contour point of `state` and store the
solutions for [`pfeast_srci!`](@ref). The shifted systems are factorized on the
first call of a solve and reused for the rest of it, so `A` and `B` must not
change in between. With `state.use_threads` the points are threaded; otherwise,
when Distributed workers are available, each worker factorizes and keeps its
share of the points, and a sweep sends it only `B*work`.
"""
function pfeast_compute_all_contour_points!(state::ParallelFeastState{T},
                                           A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                           work::Matrix{T}, M0::Int) where T<:Real
    ne = state.total_points
    factors = _pfeast_rci_factors!(state, A, B)
    rhs = Matrix{Complex{T}}(B * work)
    if factors isa _PFeastDistributedFactors
        pending = Future[]
        try
            for i in eachindex(factors.pids)
                push!(pending, remotecall(_pfeast_rci_solve_chunk, factors.pids[i],
                                          factors.futures[i], rhs))
            end
            for (chunk, future) in zip(factors.chunks, pending)
                for (e, Y) in zip(chunk, fetch(future))
                    _pfeast_store_point!(state, e, work, Y)
                end
            end
        finally
            foreach(finalize, pending)
        end
    elseif state.use_threads && Threads.nthreads() > 1
        _with_single_blas_thread() do
            Threads.@threads for e in 1:ne
                _pfeast_store_point!(state, e, work, Matrix{Complex{T}}(factors[e] \ rhs))
            end
        end
    else
        for e in 1:ne
            _pfeast_store_point!(state, e, work, Matrix{Complex{T}}(factors[e] \ rhs))
        end
    end
    return nothing
end

# Convenience wrapper for parallel Feast with automatic RCI handling
"""
    feast_parallel(A, B, interval; M0=10, fpm=nothing, use_threads=true, auto_rci=true)

Solve a real symmetric pencil with automatic parallel reverse communication.
`auto_rci=false` is unsupported: manual callers must retain a
`ParallelFeastState` and drive `pfeast_srci!` directly.
"""
function feast_parallel(A::AbstractMatrix{T}, B::AbstractMatrix{T}, 
                        interval::Tuple{T,T}; M0::Int = 10, 
                        fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                        use_threads::Bool = true,
                        auto_rci::Bool = true) where T<:Real
    # Parallel FeastKit with automatic RCI management
    auto_rci || throw(ArgumentError(
        "feast_parallel requires auto_rci=true; use pfeast_srci! with a ParallelFeastState for manual reverse communication"))
    
    Emin, Emax = interval
    N = size(A, 1)
    
    # Initialize Feast parameters
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    fpm = fpm isa FeastParameters ? fpm.fpm : fpm
    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Create parallel state. It starts from the kernel's default subspace, as
    # the serial solvers do, which the kernel writes to work.
    state = ParallelFeastState{T}(fpm[2], M0, true, use_threads)
    state.user_subspace = false

    # Initialize workspace
    work = Matrix{T}(undef, N, M0)
    workc = Matrix{Complex{T}}(undef, N, M0)
    Aq = Matrix{T}(undef, M0, M0)
    Sq = Matrix{T}(undef, M0, M0)
    lambda = Vector{T}(undef, M0)
    q = Matrix{T}(undef, N, M0)
    res = Vector{T}(undef, M0)

    while true
        # Call parallel RCI
        pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, 
                    Emin, Emax, M0, lambda, q, res)
        
        if state.ijob == Int(Feast_RCI_PARALLEL_SOLVE)
            # Automatically handle parallel computation
            pfeast_compute_all_contour_points!(state, A, B, work, M0)
            
        elseif state.ijob == Int(Feast_RCI_MULT_A)
            # Automatically compute A*q for residual calculation
            M = state.mode
            work[:, 1:M] .= A * q[:, 1:M]
            
        elseif state.ijob == Int(Feast_RCI_MULT_B)
            M = state.mode
            work[:, 1:M] .= B * q[:, 1:M]

        elseif state.ijob == Int(Feast_RCI_DONE)
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
    # The runs below use the default parameters, so report their contour size.
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    feastdefault!(fpm)
    println("Integration points: $(feast_integration_points(fpm))")
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
    if _distributed_backend_ready()
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
        if _distributed_backend_ready() && @isdefined(dist_time)
            println("Distributed speedup: $(round(serial_time/dist_time, digits=2))x")
        end
    end
    
    return nothing
end
