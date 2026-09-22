
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
    if _distributed_backend_ready()
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
