# Execution routing; backend compatibility decisions live in core/feast_backend_policy.jl.

function feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads;
                            strict_backend::Bool = false)
    # The low-level helper retains its serial fallback when MPI is unavailable.
    backend === :mpi && !_mpi_backend_ready(comm) &&
        return feast_serial(A, B, interval, M0, fpm)
    backend = _feast_compatible_backend(backend, A, B, !strict_backend)
    if backend === :mpi
        comm_options = comm === nothing ? (;) : (; comm=comm)
        return mpi_feast(A, B, interval; M0=M0, fpm=fpm, comm_options...)
    elseif backend in (:threads, :distributed)
        if A isa SparseMatrixCSC
            return pfeast_scsrgv!(copy(A), copy(B), interval[1], interval[2], M0, fpm;
                                  use_threads=(backend === :threads))
        end
        return pfeast_sygv!(copy(A), copy(B), interval[1], interval[2], M0, fpm;
                            use_threads=true)
    end
    return feast_serial(A, B, interval, M0, fpm)
end

@inline function _execute_feast(A, B, interval, backend, M0, fpm, comm, use_threads, allow_backend_fallback;
                                solver_options=(;))
    # Centralize backend execution so all high-level Hermitian/symmetric methods
    # share the same fallback semantics.
    if backend != :serial
        try
            if backend === :mpi && get(solver_options, :solver, :direct) !== :direct
                comm_options = comm === nothing ? (;) : (; comm=comm)
                return mpi_feast(A, B, interval; M0=M0, fpm=fpm,
                                 comm_options..., solver_options...)
            end
            return feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads;
                                      strict_backend=!allow_backend_fallback)
        catch e
            allow_backend_fallback || rethrow(e)
            @warn "Backend $backend failed; falling back to serial execution" exception=e
        end
    end
    return _feast_run_serial(A, B, interval, M0, fpm; solver_options...)
end

function _feast_run_serial(A, B, interval, M0, fpm; solver_options...)
    return feast_serial(A, B, interval, M0, fpm; solver_options...)
end

@inline function _execute_feast_general(A, B, center, radius, backend, M0, fpm, comm, use_threads, allow_backend_fallback;
                                        solver_options=(;))
    backend = _feast_compatible_backend(backend, A, B, allow_backend_fallback;
                                        general=true, solver=get(solver_options, :solver, :direct))
    if backend === :mpi && _mpi_backend_ready(comm)
        try
            comm_options = comm === nothing ? (;) : (; comm=comm)
            return mpi_feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                                     comm_options..., solver_options...)
        catch e
            allow_backend_fallback || rethrow(e)
            @warn "MPI backend for general problems failed; falling back to serial execution" exception=e
        end
    end
    return _feast_run_general_serial(A, B, center, radius, M0, fpm; solver_options...)
end

function _feast_run_general_serial(A, B, center, radius, M0, fpm; solver_options...)
    return feast_general_serial(A, B, center, radius, M0, fpm; solver_options...)
end
