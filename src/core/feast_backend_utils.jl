# Utility functions for parallel backend management and consistency

# Convenience wrappers so Distributed functions are available when `Distributed` isn't imported by users
nworkers() = Distributed.nworkers()
workers() = Distributed.workers()

# Check if MPI is available
function mpi_available()
    return isdefined(FeastKit, :MPI_AVAILABLE) && FeastKit.MPI_AVAILABLE[]
end

# A caller-provided communicator is an explicit MPI opt-in, even if package
# initialization happened before MPI.Init() set MPI_AVAILABLE[].
_mpi_backend_ready(comm=nothing) = comm !== nothing || mpi_available()

# Determine the available backend for a requested execution mode.
function determine_parallel_backend(parallel::Symbol, comm=nothing)
    if parallel == :mpi
        # Explicit MPI request
        if !_mpi_backend_ready(comm)
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
        if _mpi_backend_ready(comm)
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

function _select_parallel_backend(requested::Symbol, comm=nothing; allow_fallback::Bool=false)
    if !allow_fallback && requested == :mpi && !_mpi_backend_ready(comm)
        throw(ArgumentError("Requested backend :mpi is not available. Initialize MPI and pass comm=MPI.COMM_WORLD, or use backend=:auto to allow fallback."))
    elseif !allow_fallback && requested == :distributed && nworkers() <= 1
        throw(ArgumentError("Requested backend :distributed requires at least one Julia worker. Call Distributed.addprocs(...) first, or use backend=:auto to allow fallback."))
    elseif !allow_fallback && requested == :threads && Threads.nthreads() <= 1
        throw(ArgumentError("Requested backend :threads requires Julia to run with more than one thread. Start Julia with JULIA_NUM_THREADS>1, or use backend=:auto to allow fallback."))
    end

    selected = determine_parallel_backend(requested, comm)
    if !allow_fallback && requested != :auto && selected != requested
        throw(ArgumentError("Requested backend :$requested is not available (would fall back to :$selected). Use backend=:auto to allow fallback."))
    end
    return selected
end

function _backend_fallback(reason::AbstractString, strict_backend::Bool,
                           A, B, interval, M0, fpm)
    if strict_backend
        throw(ArgumentError(reason))
    end
    @warn "$reason; falling back to serial execution"
    return feast_serial(A, B, interval, M0, fpm)
end

function feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads;
                            strict_backend::Bool = false)
    if backend == :mpi && _mpi_backend_ready(comm)
        if eltype(A) <: Real && eltype(B) <: Real
            # Handle comm=nothing by omitting the keyword to use MPI.COMM_WORLD default
            if comm === nothing
                return mpi_feast(A, B, interval, M0=M0, fpm=fpm)
            else
                return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm)
            end
        elseif eltype(A) <: Complex && eltype(B) <: Complex &&
               ((A isa SparseMatrixCSC && B isa SparseMatrixCSC) ||
                (A isa Matrix && B isa Matrix))
            if comm === nothing
                return mpi_feast(A, B, interval, M0=M0, fpm=fpm)
            else
                return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm)
            end
        else
            return _backend_fallback("MPI backend currently supports real symmetric and dense/sparse complex Hermitian problems",
                                     strict_backend, A, B, interval, M0, fpm)
        end
    elseif backend in [:threads, :distributed]
        if !(eltype(A) <: Real && eltype(B) <: Real)
            return _backend_fallback("Threaded/distributed backends currently support real symmetric problems",
                                     strict_backend, A, B, interval, M0, fpm)
        end

        if A isa SparseMatrixCSC && B isa SparseMatrixCSC
            return pfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2], M0, fpm;
                                  use_threads=(backend == :threads))
        elseif A isa Matrix && B isa Matrix
            return _backend_fallback("Dense threaded/distributed backend is disabled because it does not currently match serial results",
                                     strict_backend, A, B, interval, M0, fpm)
        else
            return _backend_fallback("Threaded/distributed backend requires both matrices to use the same dense or sparse storage",
                                     strict_backend, A, B, interval, M0, fpm)
        end
    end
    # Fall back to serial execution
    return feast_serial(A, B, interval, M0, fpm)
end

# Detect identity matrices for dispatching specialized routines
function _is_identity_matrix(B::Matrix)
    isdiag(B) || return false
    n = size(B, 1)
    n == size(B, 2) || return false
    one_val = one(eltype(B))
    @inbounds for i in 1:n
        if B[i, i] != one_val
            return false
        end
    end
    return true
end

function _is_identity_matrix(B::SparseMatrixCSC)
    n = size(B, 1)
    n == size(B, 2) || return false
    nnz(B) == n || return false
    nzval = B.nzval
    rowval = B.rowval
    colptr = B.colptr
    one_val = one(eltype(B))
    @inbounds for col in 1:n
        start = colptr[col]
        stop = colptr[col + 1] - 1
        stop == start || return false
        if rowval[start] != col
            return false
        end
        if nzval[start] != one_val
            return false
        end
    end
    return true
end

# Serial Feast execution
# All matrix types (dense and sparse) use the FEAST contour integration solver
function feast_serial(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple{T,T}, M0::Int, fpm::Vector{Int}) where T<:Real
    Emin, Emax = interval
    elem_type = eltype(A)

    if elem_type <: Real
        if isa(A, Matrix) && isa(B, Matrix)
            return feast_sygv!(Matrix{Float64}(A), Matrix{Float64}(B), Emin, Emax, M0, fpm)
        elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
            if _is_identity_matrix(B)
                return feast_scsrev!(A, Emin, Emax, M0, fpm)
            else
                return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
            end
        else
            throw(ArgumentError("Unsupported matrix storage types for real symmetric problems: $(typeof(A)), $(typeof(B))"))
        end
    elseif elem_type <: Complex
        if isa(A, Matrix) && isa(B, Matrix)
            return feast_hegv!(Matrix{Complex{T}}(A), Matrix{Complex{T}}(B), Emin, Emax, M0, fpm)
        elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
            if _is_identity_matrix(B)
                return feast_hcsrev!(A, Emin, Emax, M0, fpm)
            else
                return feast_hcsrgv!(A, B, Emin, Emax, M0, fpm)
            end
        else
            throw(ArgumentError("Unsupported matrix storage types for complex Hermitian problems: $(typeof(A)), $(typeof(B))"))
        end
    else
        throw(ArgumentError("Unsupported element type $(elem_type) in feast_serial"))
    end
end

function feast_general_serial(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}},
                              center::Complex{T}, radius::T, M0::Int, fpm::Vector{Int}) where T<:Real
    if isa(A, Matrix) && isa(B, Matrix)
        return feast_gegv!(copy(A), copy(B), center, radius, M0, fpm)
    elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return feast_gcsrgv!(A, B, center, radius, M0, fpm)
    else
        throw(ArgumentError("Unsupported matrix types for general problems: $(typeof(A)), $(typeof(B))"))
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
    println("FeastKit Parallel Computing Capabilities")
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
    
    # MPI info (if available)
    println("\nMPI:")
    if mpi_available()
        try
            comm = MPI.COMM_WORLD
            rank = MPI.Comm_rank(comm)
            mpi_size = MPI.Comm_size(comm)
            println("  MPI initialized: Yes")
            println("  Current rank: $rank")
            println("  Total processes: $mpi_size")
            println("  Status: Enabled")
        catch
            println("  MPI loaded but not properly initialized")
            println("  Status: Disabled")
        end
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
