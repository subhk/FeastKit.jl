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

    # A backend that cannot run this problem (complex or non-symmetric input
    # on the threaded and distributed drivers, say) is reported and skipped.
    function run_backend(label, key, solve, serial_time)
        println("\n", label, ":")
        local result
        elapsed = try
            @elapsed (result = solve())
        catch err
            err isa ArgumentError || rethrow()
            println("   Skipped: ", sprint(showerror, err))
            return nothing
        end
        results[key] = result
        println("   Time: $(round(elapsed, digits=3)) seconds")
        println("   Eigenvalues: $(result.M)")
        serial_time === nothing || println("   Speedup: $(round(serial_time/elapsed, digits=2))x")
        return elapsed
    end

    serial_time = run_backend("1. Serial execution", :serial,
                              () -> feast(A, B, interval, M0=M0, parallel=:serial), nothing)
    if Threads.nthreads() > 1
        run_backend("2. Threading execution", :threads,
                    () -> feast(A, B, interval, M0=M0, parallel=:threads), serial_time)
    end
    if _distributed_backend_ready()
        run_backend("3. Distributed execution", :distributed,
                    () -> feast(A, B, interval, M0=M0, parallel=:distributed), serial_time)
    end
    if mpi_available()
        run_backend("4. MPI execution", :mpi,
                    () -> feast(A, B, interval, M0=M0, parallel=:mpi), serial_time)
        # Hybrid MPI+threads if both available
        if Threads.nthreads() > 1
            run_backend("5. Hybrid MPI+Threading", :hybrid,
                        () -> feast_hybrid(A, B, interval, M0=M0, use_threads_per_rank=true),
                        serial_time)
        end
    end

    return results
end
