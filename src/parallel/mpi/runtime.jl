# MPI performance benchmarking
function mpi_feast_benchmark(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int;
                            comm::MPI.Comm = MPI.COMM_WORLD)
    # Benchmark MPI FeastKit performance

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit Performance Benchmark")
        println("="^40)
        println("Matrix size: $(Base.size(A, 1))")
        println("Search interval: $interval")
        println("MPI processes: $nprocs")
        println("Subspace size: $M0")
    end

    # MPI timing
    MPI.Barrier(comm)
    start_time = MPI.Wtime()

    result = mpi_feast(A, B, interval, M0=M0, comm=comm)

    MPI.Barrier(comm)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time

    if rank == 0
        println("\nMPI FeastKit Results:")
        println("Time: $(round(elapsed_time, digits=3)) seconds")
        println("Eigenvalues found: $(result.M)")
        println("Convergence loops: $(result.loop)")
        println("Final residual: $(result.epsout)")
        println("Exit status: $(result.info)")

        if result.M > 0
            println("\nEigenvalues:")
            for i in 1:min(result.M, 5)  # Show first 5
                println("  λ[$i] = $(result.lambda[i])")
            end
            if result.M > 5
                println("  ... and $(result.M - 5) more")
            end
        end
    end

    return result
end

# Utility: Check if MPI is available and initialized
function mpi_feast_available()
    try
        return MPI.Initialized()
    catch
        return false
    end
end

# Utility: Initialize MPI for FeastKit if needed
function mpi_feast_init()
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit initialized with $nprocs processes")
    end

    return comm, rank, nprocs
end

# Utility: Clean up MPI
function mpi_feast_finalize()
    if MPI.Initialized()
        MPI.Finalize()
    end
end
