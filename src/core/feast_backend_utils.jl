# Serial storage dispatch and backend capability reporting.

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
        # A sparse identity has exactly one stored entry in each column and that
        # entry must sit on the diagonal with value one.
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
function feast_serial(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple{T,T}, M0::Int, fpm::Vector{Int}; solver_options...) where T<:Real
    Emin, Emax = interval
    elem_type = eltype(A)

    if elem_type <: Real
        if isa(A, Matrix) && isa(B, Matrix)
            return feast_sygv!(A, B, convert(elem_type, Emin),
                                convert(elem_type, Emax), M0, fpm; solver_options...)
        elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
            if _is_identity_matrix(B)
                return feast_scsrev!(A, Emin, Emax, M0, fpm; solver_options...)
            else
                return feast_scsrgv!(A, B, Emin, Emax, M0, fpm; solver_options...)
            end
        else
            throw(ArgumentError("Unsupported matrix storage types for real symmetric problems: $(typeof(A)), $(typeof(B))"))
        end
    elseif elem_type <: Complex
        if isa(A, Matrix) && isa(B, Matrix)
            return feast_hegv!(A, B, Emin, Emax, M0, fpm; solver_options...)
        elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
            if _is_identity_matrix(B)
                return feast_hcsrev!(A, Emin, Emax, M0, fpm; solver_options...)
            else
                return feast_hcsrgv!(A, B, Emin, Emax, M0, fpm; solver_options...)
            end
        else
            throw(ArgumentError("Unsupported matrix storage types for complex Hermitian problems: $(typeof(A)), $(typeof(B))"))
        end
    else
        throw(ArgumentError("Unsupported element type $(elem_type) in feast_serial"))
    end
end

function feast_general_serial(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}},
                              center::Complex{T}, radius::T, M0::Int, fpm::Vector{Int}; solver_options...) where T<:Real
    if isa(A, Matrix) && isa(B, Matrix)
        return feast_gegv!(A, B, center, radius, M0, fpm; solver_options...)
    elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return feast_gcsrgv!(A, B, center, radius, M0, fpm; solver_options...)
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
    capabilities[:distributed] = _distributed_backend_ready()
    
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
    println("  Status: $(_distributed_backend_ready() ? "Enabled" : "Disabled")")
    
    # MPI info (if available)
    println("\nMPI:")
    if mpi_available()
        try
            comm = _mpi_world_comm()
            rank = _mpi_comm_rank(comm)
            mpi_size = _mpi_comm_size(comm)
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
    elseif _distributed_backend_ready()
        println("  Best for multi-core: Use parallel=:distributed")
    elseif Threads.nthreads() > 1
        println("  Best for multi-core: Use parallel=:threads")
    else
        println("  Consider starting Julia with multiple threads or adding workers")
        println("  Use: julia --threads=auto or addprocs(4)")
    end
end
