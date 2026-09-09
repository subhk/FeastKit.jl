# Backend timing comparison. Lives in the parent module rather than the MPI
# extension: every MPI call it makes is behind an `mpi_available()` guard.

"""
    feast_parallel_comparison(A, B, interval, M0)

Run the same problem through every parallel backend available in this session
and print a timing comparison. Threads and workers are used when present; the
MPI rows appear only when MPI is loaded and initialized.
"""
function feast_parallel_comparison(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int)
    # Compare performance across all available parallel backends

    println("FeastKit Parallel Backend Comparison")
    println("="^50)
    println("Matrix size: $(size(A, 1))")
    println("Search interval: $interval")
    println("Subspace size: $M0")
    println("Available backends:")
    println("  Threads: $(Threads.nthreads())")
    println("  Workers: $(Distributed.nworkers())")
    println("  MPI: $(mpi_available() ? "Yes" : "No")")

    results = Dict{Symbol, Any}()

    # Serial benchmark
    println("\n1. Serial execution:")
    serial_time = @elapsed begin
        results[:serial] = feast(A, B, interval, M0=M0, parallel=:serial)
    end
    println("   Time: $(round(serial_time, digits=3)) seconds")
    println("   Eigenvalues: $(results[:serial].M)")

    # Threading benchmark
    if Threads.nthreads() > 1
        println("\n2. Threading execution:")
        thread_time = @elapsed begin
            results[:threads] = feast(A, B, interval, M0=M0, parallel=:threads)
        end
        println("   Time: $(round(thread_time, digits=3)) seconds")
        println("   Eigenvalues: $(results[:threads].M)")
        println("   Speedup: $(round(serial_time/thread_time, digits=2))x")
    end

    # Distributed benchmark
    if Distributed.nworkers() > 1
        println("\n3. Distributed execution:")
        dist_time = @elapsed begin
            results[:distributed] = feast(A, B, interval, M0=M0, parallel=:distributed)
        end
        println("   Time: $(round(dist_time, digits=3)) seconds")
        println("   Eigenvalues: $(results[:distributed].M)")
        println("   Speedup: $(round(serial_time/dist_time, digits=2))x")
    end

    # MPI benchmark
    if mpi_available()
        println("\n4. MPI execution:")
        mpi_time = @elapsed begin
            results[:mpi] = feast(A, B, interval, M0=M0, parallel=:mpi)
        end
        println("   Time: $(round(mpi_time, digits=3)) seconds")
        println("   Eigenvalues: $(results[:mpi].M)")
        println("   Speedup: $(round(serial_time/mpi_time, digits=2))x")

        # Hybrid MPI+threads if both available
        if Threads.nthreads() > 1
            println("\n5. Hybrid MPI+Threading:")
            hybrid_time = @elapsed begin
                results[:hybrid] = feast_hybrid(A, B, interval, M0=M0, use_threads_per_rank=true)
            end
            println("   Time: $(round(hybrid_time, digits=3)) seconds")
            println("   Eigenvalues: $(results[:hybrid].M)")
            println("   Speedup: $(round(serial_time/hybrid_time, digits=2))x")
        end
    end

    return results
end
