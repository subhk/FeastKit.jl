# Utility functions for parallel backend management and consistency

# Convenience wrappers so Distributed functions are available when `Distributed` isn't imported by users
nworkers() = Distributed.nworkers()
workers() = Distributed.workers()

# With no added workers Julia reports nworkers()==1 for the main process.
# nprocs distinguishes that case from one actual worker plus the main process.
_distributed_backend_ready() = Distributed.nprocs() > 1

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
            return _distributed_backend_ready() ? :distributed : (Threads.nthreads() > 1 ? :threads : :serial)
        end
        return :mpi

    elseif parallel == :distributed
        return _distributed_backend_ready() ? :distributed : :serial

    elseif parallel == :threads
        return Threads.nthreads() > 1 ? :threads : :serial

    elseif parallel == :serial
        return :serial

    elseif parallel == :auto
        # Automatic backend selection based on available resources
        if _mpi_backend_ready(comm)
            return :mpi
        elseif _distributed_backend_ready()
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
    # Strict requests fail early with actionable setup guidance. Auto/fallback
    # requests can still degrade to a backend that is actually available.
    if !allow_fallback && requested == :mpi && !_mpi_backend_ready(comm)
        throw(ArgumentError("Requested backend :mpi is not available. Initialize MPI and pass comm=MPI.COMM_WORLD, or use backend=:auto to allow fallback."))
    elseif !allow_fallback && requested == :distributed && !_distributed_backend_ready()
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

@inline function _normalize_parallel(parallel::Union{Bool,Symbol})
    parallel === true && return :auto
    parallel === false && return :serial
    parallel isa Symbol && return parallel
    throw(ArgumentError("Invalid parallel option: $parallel"))
end

function _normalize_backend(parallel::Union{Bool,Symbol,Nothing},
                            backend::Union{Symbol,Nothing})
    # `parallel` is the legacy keyword and `backend` is the explicit replacement.
    # Accept both only when they describe the same execution mode.
    if backend !== nothing
        requested = backend
        if parallel !== nothing
            legacy_requested = _normalize_parallel(parallel)
            legacy_requested == requested ||
                throw(ArgumentError("Conflicting backend requests: backend=$requested and parallel=$legacy_requested"))
        end
    elseif parallel !== nothing
        requested = _normalize_parallel(parallel)
    else
        requested = :serial
    end

    requested in (:serial, :auto, :threads, :distributed, :mpi) ||
        throw(ArgumentError("Unknown backend: $requested. Use :serial, :auto, :threads, :distributed, or :mpi"))
    return requested
end

function _allow_backend_fallback(parallel::Union{Bool,Symbol,Nothing},
                                 backend::Union{Symbol,Nothing},
                                 strict_backend::Bool)
    strict_backend && return false
    backend === :auto && return true
    backend !== nothing && return false
    parallel === true && return true
    parallel === :auto && return true
    return false
end

function _feast_backend_solver_issue(backend, solver, A; general::Bool=false)
    supported = solver === :direct || backend === :serial ||
                (backend === :mpi && (general || eltype(A) <: Complex))
    return supported ? nothing :
        "solver=:gmres is not supported by backend=:$backend for this problem; use backend=:serial or solver=:direct"
end

# Central compatibility policy for problem family, storage, and inner solver.
# Availability is handled above; execution and runtime failures are handled by
# parallel/feast_backend_execution.jl. Nothing means the combination is supported.
function _feast_backend_issue(backend, A, B; general::Bool=false, solver=:direct)
    issue = _feast_backend_solver_issue(backend, solver, A; general=general)
    issue === nothing || return issue
    backend === :serial && return nothing
    dense = A isa Matrix && B isa Matrix
    sparse = A isa SparseMatrixCSC && B isa SparseMatrixCSC
    real = eltype(A) <: Real && eltype(B) <: Real
    complex = eltype(A) <: Complex && eltype(B) <: Complex
    if general
        backend === :mpi || return "Threaded/distributed execution for general problems is not yet available"
        return (dense || sparse) && complex ? nothing :
            "MPI backend for general problems requires dense or sparse complex matrices"
    elseif backend === :mpi
        return real || ((dense || sparse) && complex) ? nothing :
            "MPI backend currently supports real symmetric and dense/sparse complex Hermitian problems"
    elseif backend in (:threads, :distributed)
        real || return "Threaded/distributed backends currently support real symmetric problems"
        dense || sparse || return "Threaded/distributed backend requires both matrices to use the same dense or sparse storage"
        backend === :distributed && dense && return "Dense distributed backend is not implemented; contour points are distributed across workers only for sparse storage"
    end
    return nothing
end

function _feast_backend_or_serial(backend, message, allow_fallback)
    message === nothing && return backend
    allow_fallback || throw(ArgumentError(message))
    @warn "$message; falling back to serial execution"
    return :serial
end

function _feast_solver_backend(backend, solver_options, A, allow_fallback; general::Bool=false)
    issue = _feast_backend_solver_issue(backend, solver_options.solver, A; general=general)
    return _feast_backend_or_serial(backend, issue, allow_fallback)
end

function _feast_compatible_backend(backend, A, B, allow_fallback; kwargs...)
    issue = _feast_backend_issue(backend, A, B; kwargs...)
    return _feast_backend_or_serial(backend, issue, allow_fallback)
end
