# Utility functions for parallel backend management and consistency

# Convenience wrapper so `nworkers()` is available when `Distributed` isn't imported by users
nworkers() = Distributed.nworkers()

# Check if MPI is available
function mpi_available()
    return isdefined(FeastKit, :MPI_AVAILABLE) && FeastKit.MPI_AVAILABLE[]
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

function _direct_real_dense_feast(A::AbstractMatrix{T}, B::AbstractMatrix{T}, Emin::T, Emax::T, M0::Int) where T<:Real
    F = eigen(Symmetric(Matrix(A)), Symmetric(Matrix(B)))
    eigenvalues = F.values
    vectors = F.vectors
    idx = [i for i in eachindex(eigenvalues) if Emin <= eigenvalues[i] <= Emax]
    take = min(length(idx), M0)
    if take == 0
        return FeastResult{T, T}(T[], Matrix{T}(undef, size(A,1), 0), 0, T[], Int(Feast_ERROR_NO_CONVERGENCE), zero(T), 0)
    end
    selected = idx[1:take]
    lambda = eigenvalues[selected]
    q = vectors[:, selected]
    res = similar(lambda)
    for (j, val) in enumerate(lambda)
        vec = q[:, j]
        res[j] = norm(Matrix(A)*vec - val*(Matrix(B)*vec))
    end
    epsout = maximum(res)
    return FeastResult{T, T}(lambda, q, take, res, 0, epsout, 1)
end

function _direct_complex_dense_feast(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}}, Emin::T, Emax::T, M0::Int) where T<:Real
    F = eigen(Hermitian(Matrix(A)), Hermitian(Matrix(B)))
    eigenvalues = real.(F.values)
    vectors = F.vectors
    idx = [i for i in eachindex(eigenvalues) if Emin <= eigenvalues[i] <= Emax]
    take = min(length(idx), M0)
    if take == 0
        return FeastResult{T, Complex{T}}(T[], Matrix{Complex{T}}(undef, size(A,1), 0), 0, T[], Int(Feast_ERROR_NO_CONVERGENCE), zero(T), 0)
    end
    selected = idx[1:take]
    lambda = eigenvalues[selected]
    q = vectors[:, selected]
    res = similar(lambda)
    for (j, val) in enumerate(lambda)
        vec = q[:, j]
        res[j] = norm(Matrix(A)*vec - val*(Matrix(B)*vec))
    end
    epsout = maximum(res)
    return FeastResult{T, Complex{T}}(lambda, q, take, res, 0, epsout, 1)
end

function feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads)
    if backend == :mpi && mpi_available()
        if eltype(A) <: Real && eltype(B) <: Real
            return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm)
        else
            @warn "MPI backend currently supports real symmetric problems; falling back to serial execution"
        end
    elseif backend in [:threads, :distributed]
        if eltype(A) <: Real && eltype(B) <: Real
            return feast_parallel(A, B, interval, M0=M0, fpm=fpm, use_threads=(backend==:threads))
        else
            @warn "Threaded/distributed backends currently support real symmetric problems; falling back to serial execution"
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
function feast_serial(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple{T,T}, M0::Int, fpm::Vector{Int}) where T<:Real
    Emin, Emax = interval
    elem_type = eltype(A)

    if elem_type <: Real
        if isa(A, Matrix) && isa(B, Matrix)
            return feast_sygv!(copy(A), copy(B), Emin, Emax, M0, fpm)
        elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
            return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
        else
            throw(ArgumentError("Unsupported matrix storage types for real symmetric problems: $(typeof(A)), $(typeof(B))"))
        end
    elseif elem_type <: Complex
        if isa(A, Matrix) && isa(B, Matrix)
            if _is_identity_matrix(B)
                return feast_heev!(copy(A), Emin, Emax, M0, fpm)
            else
                return feast_hegv!(copy(A), copy(B), Emin, Emax, M0, fpm)
            end
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
            comm = @eval MPI.COMM_WORLD
            rank = @eval MPI.Comm_rank(comm)
            size = @eval MPI.Comm_size(comm)
            println("  MPI initialized: Yes")
            println("  Current rank: $rank")
            println("  Total processes: $size")
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
