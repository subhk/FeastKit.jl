# Parallel sparse Feast
# Sparse analogue of _pfeast_factorize_contour: factorize each shifted system
# (z*B - A) once with a sparse LU and reuse across refinement loops. The factor
# type is taken from a sample so this works for both Float64 and Float32 inputs.
function _pfeast_factorize_contour_sparse(A::SparseMatrixCSC{T,Int},
                                          B::SparseMatrixCSC{T,Int},
                                          Zne::Vector{Complex{T}},
                                          use_threads::Bool) where T<:Real
    ne = length(Zne)
    f1 = lu(Zne[1] * B - A)
    factors = Vector{typeof(f1)}(undef, ne)
    factors[1] = f1
    if use_threads && Threads.nthreads() > 1 && ne > 2
        _with_single_blas_thread() do
            Threads.@threads for e in 2:ne
                factors[e] = lu(Zne[e] * B - A)
            end
        end
    else
        for e in 2:ne
            factors[e] = lu(Zne[e] * B - A)
        end
    end
    return factors
end

# Keep UMFPACK objects on their owning workers: they contain process-local
# pointers and must never be fetched/serialized back to the coordinator.
struct _PFeastDistributedFactors
    pids::Vector{Int}
    chunks::Vector{Vector{Int}}
    futures::Vector{Future}
end

# Factorize one worker's share of the contour points, in its own storage.
_pfeast_factorize_chunk(A::SparseMatrixCSC, B::SparseMatrixCSC, nodes) =
    _pfeast_factorize_contour_sparse(A, B, nodes, false)
_pfeast_factorize_chunk(A::Matrix, B::Matrix, nodes) =
    _pfeast_factorize_contour(A, B, nodes, false)

# On the owning worker, `fetch` of a failed remotecall returns the stored
# RemoteException instead of throwing it.
function _pfeast_remote_failure(factors::Future)
    result = fetch(factors)
    result isa Exception || return nothing
    cause = result isa RemoteException ? result.captured.ex : result
    return sprint(showerror, cause)
end

function _pfeast_factorize_contour_distributed(A, B, nodes)
    pids = workers()[1:min(length(workers()), length(nodes))]
    # Functions and sparse factor types cannot be deserialized until the
    # package is loaded. Support the documented addprocs + feast workflow.
    Distributed.remotecall_eval(Main, pids, :(using FeastKit))
    chunks = distribute_contour_points(length(nodes), length(pids))
    futures = Future[]
    try
        for (pid, chunk) in zip(pids, chunks)
            push!(futures, remotecall(_pfeast_factorize_chunk, pid,
                                      A, B, nodes[chunk]))
        end
        # Report a failed factorization here, with its cause, rather than as
        # a MethodError when the stored exception reaches the first solve.
        for (pid, future) in zip(pids, futures)
            failure = remotecall_fetch(_pfeast_remote_failure, pid, future)
            failure === nothing ||
                error("Shifted factorization failed on worker $pid: $failure")
        end
    catch
        foreach(finalize, futures)
        rethrow()
    end
    return _PFeastDistributedFactors(pids, chunks, futures)
end

function _pfeast_project_sparse_chunk(factors::Future, rhs::AbstractMatrix,
                                      weights::AbstractVector)
    partial = zeros(eltype(rhs), size(rhs))
    # This fetch occurs on the worker that owns the factorizations.
    owned = fetch(factors)
    owned isa Exception && throw(owned)
    _pfeast_accumulate_qproj_sparse!(partial, owned, rhs, weights, false)
    return partial
end

# Dense drivers reach the same worker-side accumulation.
_pfeast_accumulate_qproj!(Q_proj::AbstractMatrix, factors::_PFeastDistributedFactors,
                          BQ::AbstractMatrix, Wne::AbstractVector, use_threads::Bool) =
    _pfeast_accumulate_qproj_sparse!(Q_proj, factors, BQ, Wne, use_threads)

function _pfeast_accumulate_qproj_sparse!(Q_proj::AbstractMatrix,
                                          factors::_PFeastDistributedFactors,
                                          BQ::AbstractMatrix, weights::AbstractVector,
                                          use_threads::Bool)
    # Materialize only active columns before serialization, then send the same
    # RHS to each worker. Each returns one accumulated block per sweep.
    rhs = Matrix(BQ)
    pending = Future[]
    try
        for i in eachindex(factors.pids)
            push!(pending, remotecall(_pfeast_project_sparse_chunk, factors.pids[i],
                                      factors.futures[i], rhs, weights[factors.chunks[i]]))
        end
        for future in pending
            Q_proj .+= fetch(future)
        end
    finally
        foreach(finalize, pending)
    end
    return Q_proj
end

# Sparse analogue of _pfeast_accumulate_qproj!: Q_proj = sum 2*Wne[e] *
# (z*B - A)^{-1} (B*Q) using cached sparse factorizations and a precomputed BQ.
# Each per-point sparse solve runs on its own thread; results are converted to
# Complex{T} so the Float32 path (where UMFPACK promotes to ComplexF64) stays
# type-correct.
function _pfeast_accumulate_qproj_sparse!(Q_proj::AbstractMatrix{Complex{T}},
                                          factors::Vector{FAC},
                                          BQ::AbstractMatrix{Complex{T}},
                                          Wne::Vector{Complex{T}},
                                          use_threads::Bool) where {T<:Real, FAC}
    ne = length(factors)
    if use_threads && Threads.nthreads() > 1 && ne > 1
        contribs = Vector{Matrix{Complex{T}}}(undef, ne)
        _with_single_blas_thread() do
            Threads.@threads for e in 1:ne
                # Scale the solve result in place; assignment converts to
                # Complex{T} only when UMFPACK promoted the eltype (Float32).
                local Yt = factors[e] \ BQ
                @. Yt *= 2 * Wne[e]
                contribs[e] = Yt
            end
        end
        for c in contribs
            Q_proj .+= c
        end
    else
        for e in 1:ne
            Ys = factors[e] \ BQ
            @. Q_proj += (2 * Wne[e]) * Ys
        end
    end
    Q_proj .= real.(Q_proj)
    return Q_proj
end

# The sparse halves of the storage dispatch in _pfeast_real_symmetric! (the
# shared driver behind pfeast_sygv! and pfeast_scsrgv!, in dense.jl).
_pfeast_factorize_contour(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                          Zne::Vector{Complex{T}}, use_threads::Bool) where T<:Real =
    _pfeast_factorize_contour_sparse(A, B, Zne, use_threads)

_pfeast_accumulate_qproj!(Q_proj::AbstractMatrix, factors::AbstractVector,
                          BQ::AbstractMatrix, Wne::AbstractVector, use_threads::Bool) =
    _pfeast_accumulate_qproj_sparse!(Q_proj, factors, BQ, Wne, use_threads)

function pfeast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                        Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                        use_threads::Bool = true, verbose::Bool = false) where T<:Real
    # Parallel sparse real-symmetric generalized FEAST: cached sparse
    # factorizations with threaded or distributed contour solves, and the same
    # real Rayleigh-Ritz as pfeast_sygv! (see _pfeast_real_symmetric!).
    return _pfeast_real_symmetric!(A, B, Emin, Emax, M0, fpm, use_threads, verbose,
                                   "pfeast_scsrgv!")
end
