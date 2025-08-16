# Utility functions for parallel backend management and consistency

# Check if MPI is available
function mpi_available()
    return isdefined(FEAST, :MPI_AVAILABLE) && FEAST.MPI_AVAILABLE[]
end

# Determine optimal parallel backend
function determine_parallel_backend(parallel::Symbol, comm=nothing)
    if parallel == :mpi
        # Explicit MPI request
        if !mpi_available()
            @warn "MPI requested but not available, falling back to distributed"
            return nworkers() > 1 ? :distributed : (Threads.nthreads() > 1 ? :threads : :serial)
        end
        return :mpi
        
    elseif parallel == :distributed
        return nworkers() > 1 ? :distributed : :serial
        
    elseif parallel == :threads
        return Threads.nthreads() > 1 ? :threads : :serial
        
    elseif parallel == :serial
        return :serial
        
    elseif parallel == :auto
        # Automatic backend selection based on available resources
        if comm !== nothing || mpi_available()
            return :mpi
        elseif nworkers() > 1
            return :distributed
        elseif Threads.nthreads() > 1
            return :threads
        else
            return :serial
        end
        
    else
        throw(ArgumentError("Unknown parallel backend: $parallel. Use :auto, :mpi, :distributed, :threads, or :serial"))
    end
end

# Execute FEAST with the appropriate backend
function feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads)
    if backend == :mpi && mpi_available()
        return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm)
    elseif backend in [:threads, :distributed]
        return feast_parallel(A, B, interval, M0=M0, fpm=fpm, use_threads=(backend==:threads))
    else
        # Fall back to serial execution
        return feast_serial(A, B, interval, M0, fpm)
    end
end

# Serial FEAST execution
function feast_serial(A::AbstractMatrix{T}, B::AbstractMatrix{T}, interval::Tuple{T,T}, M0::Int, fpm::Vector{Int}) where T<:Real
    Emin, Emax = interval
    
    if isa(A, Matrix) && isa(B, Matrix)
        # Dense matrices
        if T <: Real
            return feast_sygv!(copy(A), copy(B), Emin, Emax, M0, fpm)
        else
            return feast_gegv!(A, B, Complex(Emin), Emax - Emin, M0, fpm)
        end
    elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        # Sparse matrices
        return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
    else
        throw(ArgumentError("Unsupported matrix types: $(typeof(A)), $(typeof(B))"))
    end
end

# Check parallel computing capabilities
function feast_parallel_capabilities()
    capabilities = Dict{Symbol, Bool}()
    
    # Check threading
    capabilities[:threads] = Threads.nthreads() > 1
    
    # Check distributed computing
    capabilities[:distributed] = nworkers() > 1
    
    # Check MPI
    capabilities[:mpi] = mpi_available()
    
    return capabilities
end

# Print parallel backend information
function feast_parallel_info()
    println("FEAST Parallel Computing Capabilities")
    println("="^40)
    
    # Threading info
    println("Threading:")
    println("  Available threads: $(Threads.nthreads())")
    println("  Status: $(Threads.nthreads() > 1 ? "Enabled" : "Disabled")")
    
    # Distributed info
    println("\nDistributed Computing:")
    println("  Available workers: $(nworkers())")
    println("  Worker processes: $(workers())")
    println("  Status: $(nworkers() > 1 ? "Enabled" : "Disabled")")
    
    # MPI info
    println("\nMPI:")
    if mpi_available()
        println("  MPI initialized: Yes")
        println("  Status: Enabled")
    else
        println("  MPI initialized: No")
        println("  Status: Disabled")
    end
    
    # Recommendations
    println("\nRecommendations:")
    if mpi_available()
        println("  Best for HPC clusters: Use parallel=:mpi")
        if Threads.nthreads() > 1
            println("  Best for hybrid: Use feast_hybrid() for MPI+threading")
        end
    elseif nworkers() > 1
        println("  Best for multi-core: Use parallel=:distributed")
    elseif Threads.nthreads() > 1
        println("  Best for multi-core: Use parallel=:threads")
    else
        println("  Consider starting Julia with multiple threads or adding workers")
        println("  Use: julia --threads=auto or addprocs(4)")
    end
end