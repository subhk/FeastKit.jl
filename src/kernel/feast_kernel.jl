# FEAST reverse-communication kernels.
#
# These functions implement the solver-neutral state machines. Callers provide
# storage-specific work for each ijob request: factorize a shifted system, solve
# it, multiply by A/B, and then re-enter the kernel with the same state object.

# FEAST's contract is that the trial subspace is wider than the number of
# eigenvalues in the search region. When every column of the converged subspace
# turns out to be an in-region Ritz pair, the region held at least M0 of them
# and there is no way to tell whether more were missed -- so the answer cannot
# be certified, however small its residual. Report that as `Feast_ERROR_M0`
# ("subspace too small") rather than as success or plain non-convergence, which
# is what Fortran FEAST does and what lets a caller retry with a larger M0.
@inline function _feast_exit_info(converged::Bool, M::Int, M0::Int, N::Int)
    # M0 == N is the exception: the trial subspace already spans the whole
    # space, so a full count is complete by construction and cannot be hiding
    # anything, however many eigenvalues the region holds.
    M >= M0 && M0 < N && return Int(Feast_ERROR_M0)
    return converged ? Int(Feast_SUCCESS) : Int(Feast_ERROR_NO_CONVERGENCE)
end

# Real symmetric / Hermitian FEAST reverse-communication kernel.
#
# One refinement loop issues, in order:
#   FACTORIZE/SOLVE  ne times  -- apply the contour resolvent to the trial subspace
#   MULT_A, MULT_B             -- build the reduced pencil on the compressed basis
#   MULT_A, MULT_B             -- evaluate residuals of the Ritz pairs
# `state` carries the contour, the trial subspace, and which of the two
# MULT_A/MULT_B pairs is outstanding, so it must be the same object for every
# call in one loop.
@views function feast_srci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                            work::Matrix{T}, workc::Matrix{Complex{T}},
                            Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                            epsout::Ref{T}, loop::Ref{Int},
                            Emin::T, Emax::T, M0::Int,
                            lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                            res::Vector{T}, info::Ref{Int};
                            state::FeastSRCIState{T} = FeastSRCIState{T}()) where T<:Real

    # mode is an output: publish a count only when issuing a multiply or
    # returning Ritz pairs. Early exits must not leak the previous job's rank.
    mode[] = 0

    if ijob[] == -1  # Initialization
        feastdefault!(fpm)

        info[] = Int(Feast_SUCCESS)

        if N <= 0
            info[] = Int(Feast_ERROR_N)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if Emin >= Emax
            info[] = Int(Feast_ERROR_EMIN_EMAX)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        contour = feast_get_custom_contour(T, fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end

        # Cache the contour and per-call work arrays in explicit state. Older
        # implementations relied on global dictionaries keyed by objectid, which
        # made nested and threaded RCI calls fragile.
        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)
        state.ne = length(contour.Zne)
        state.e = 1
        state.phase = FEAST_PHASE_IDLE
        state.rank = 0
        state.M = 0
        state.active = M0
        state.initialized = true

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(contour.Zne)
        fpm[52] = 0
        fpm[53] = 1

        loop[] = 0

        # The RCI contract expects callers to pass reusable buffers; reset all
        # observable outputs on init so repeated solves start from a clean slate.
        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        fill!(lambda, zero(T))
        fill!(q, zero(T))
        fill!(res, zero(T))
        fill!(workc, zero(Complex{T}))

        if fpm[5] == 1
            # User-provided initial subspace: normalize columns
            fallback_rng = MersenneTwister(hash((N, M0, :fallback)))
            for j in 1:M0
                if norm(work[:, j]) > 0
                    work[:, j] ./= norm(work[:, j])
                else
                    for i in 1:N
                        work[i, j] = randn(fallback_rng, T)
                    end
                    work[:, j] ./= norm(work[:, j])
                end
            end
        else
            # Deterministic seeded random subspace for reproducibility
            _feast_seeded_subspace!(view(work, :, 1:M0))
        end

        state.Q0 = copy(work[:, 1:M0])
        state.Q_proj = zeros(Complex{T}, N, M0)
        state.Qb = Matrix{T}(undef, N, M0)
        state.AQ = Matrix{T}(undef, N, M0)
        state.perm = Vector{Int}(undef, M0)
        state.q_tmp = Matrix{T}(undef, N, M0)
        state.lambda_tmp = Vector{T}(undef, M0)
        state.residual = Vector{T}(undef, N)

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state.initialized = false
        state.phase = FEAST_PHASE_IDLE
        return
    end

    # Every job other than init continues a loop that init started. A fresh
    # state object here means the caller forgot to thread the same `state`
    # through the loop, which used to fail silently with M = 0.
    if !state.initialized
        throw(ArgumentError(
            "feast_srci! called with ijob=$(ijob[]) on an uninitialized state. " *
            "Pass the same `state=FeastSRCIState{T}()` object to every call in " *
            "one RCI loop, and start the loop with ijob[] = -1."))
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # After the caller factorizes Ze*B - A, feed the current trial subspace
        # into work and request the shifted solve.
        ijob[] = Int(Feast_RCI_SOLVE)
        active = state.active
        copyto!(view(work, :, 1:active), view(state.Q0, :, 1:active))
        # Columns past the active block are stale from an earlier loop; zero
        # them so a caller that solves all M0 right-hand sides cannot trip over
        # leftover Inf/NaN.
        active < M0 && fill!(view(work, :, (active + 1):M0), zero(T))
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        Zne = state.Zne
        Wne = state.Wne
        e = state.e
        ne = state.ne
        active = state.active

        Q_proj = state.Q_proj

        # Reset the accumulator at the start of each contour sweep. Every point
        # adds a weighted resolvent contribution to the spectral projector.
        if e == 1
            fill!(Q_proj, zero(Complex{T}))
        end

        weight = 2 * Wne[e]  # Account for conjugate half-contour
        Q_proj[:, 1:active] .+= weight .* workc[:, 1:active]

        fpm[50] = e + 1  # Store incremented counter in fpm
        state.e = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        fpm[50] = 1  # Reset for next refinement loop
        state.e = 1

        # The omitted conjugate half of the contour contributes the conjugate of
        # what was accumulated, so the full projector is the real part.
        Q_proj_real = view(state.q_tmp, :, 1:active)
        @inbounds for j in 1:active, i in 1:N
            Q_proj_real[i, j] = real(Q_proj[i, j])
        end

        # Rank-compress before Rayleigh-Ritz. Without this an M0 larger than the
        # number of eigenvalues in the interval yields a rank-deficient pencil
        # and spurious Ritz pairs.
        rank = _feast_qr_compress!(state.Qb, state.q_tmp, active;
                                   rank_tol=sqrt(eps(T)))
        if rank == 0
            info[] = Int(Feast_ERROR_NO_CONVERGENCE)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0
            state.initialized = false
            return
        end
        state.rank = rank

        # Hand the orthonormal basis to the caller for A*Qb, then B*Qb.
        copyto!(view(q, :, 1:rank), view(state.Qb, :, 1:rank))
        state.phase = FEAST_PHASE_PROJECT_A
        mode[] = rank
        ijob[] = Int(Feast_RCI_MULT_A)
        return
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        rank = state.rank

        if state.phase == FEAST_PHASE_PROJECT_A
            # work holds A*Qb: form the reduced stiffness block Qb' A Qb.
            Aq_block = view(Aq, 1:rank, 1:rank)
            mul!(Aq_block, transpose(view(state.Qb, :, 1:rank)),
                 view(work, :, 1:rank))
            _feast_symmetric_part!(Aq_block)

            copyto!(view(q, :, 1:rank), view(state.Qb, :, 1:rank))
            state.phase = FEAST_PHASE_PROJECT_B
            mode[] = rank
            ijob[] = Int(Feast_RCI_MULT_B)
            return
        end

        if state.phase == FEAST_PHASE_RESIDUAL_A
            # work holds A*q for the current Ritz vectors; stash it while the
            # caller computes B*q.
            M = state.M
            copyto!(view(state.AQ, :, 1:M), view(work, :, 1:M))
            state.phase = FEAST_PHASE_RESIDUAL_B
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_B)
            return
        end

        throw(ArgumentError("feast_srci!: MULT_A received in unexpected phase $(state.phase)"))
    end

    if ijob[] == Int(Feast_RCI_MULT_B)
        if state.phase == FEAST_PHASE_PROJECT_B
            rank = state.rank
            Aq_block = view(Aq, 1:rank, 1:rank)
            Sq_block = view(Sq, 1:rank, 1:rank)

            # work holds B*Qb: form the reduced mass block Qb' B Qb.
            mul!(Sq_block, transpose(view(state.Qb, :, 1:rank)),
                 view(work, :, 1:rank))
            _feast_symmetric_part!(Sq_block)

            local lambda_red, v_red
            try
                # Symmetric-definite reduced pencil Aq*v = lambda*Sq*v.
                F = eigen(Symmetric(Aq_block), Symmetric(Sq_block))
                lambda_red = F.values
                v_red = F.vectors
            catch err
                if err isa PosDefException || err isa LinearAlgebra.LAPACKException ||
                   err isa SingularException
                    @debug "Symmetric-definite reduced pencil failed; using the general solver" exception=err
                    try
                        F = eigen(Aq_block, Sq_block)
                        lambda_red = real.(F.values)
                        v_red = real.(F.vectors)
                    catch err2
                        @debug "Reduced eigenproblem failed" exception=err2
                        info[] = Int(Feast_ERROR_LAPACK)
                        ijob[] = Int(Feast_RCI_DONE)
                        fpm[53] = 0
                        state.initialized = false
                        return
                    end
                else
                    @debug "Reduced eigenproblem failed" exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0
                    state.initialized = false
                    return
                end
            end

            # Ritz vectors on the orthonormal basis, normalized so the residual
            # below is a genuine relative quantity.
            mul!(view(q, :, 1:rank), view(state.Qb, :, 1:rank), v_red)
            copyto!(view(lambda, 1:rank), view(lambda_red, 1:rank))
            for j in 1:rank
                nrm = norm(view(q, :, j))
                nrm > 0 && (view(q, :, j) ./= nrm)
            end

            M = _feast_reorder_by_interval!(lambda, q, state.perm,
                                            state.lambda_tmp, state.q_tmp,
                                            Emin, Emax, rank)
            fpm[52] = M
            state.M = M

            if M == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end

            state.phase = FEAST_PHASE_RESIDUAL_A
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_A)
            return
        end

        if state.phase == FEAST_PHASE_RESIDUAL_B
            # state.AQ holds A*q and work holds B*q for the same Ritz vectors,
            # so the residual is the true generalized one.
            M = state.M
            residual = state.residual
            AQ = state.AQ
            for j in 1:M
                @inbounds for i in 1:N
                    residual[i] = AQ[i, j] - lambda[j] * work[i, j]
                end
                qnorm = norm(view(q, :, j))
                denom = max(abs(lambda[j]), one(T)) * max(qnorm, eps(T))
                res[j] = norm(residual) / denom
            end
            epsout[] = maximum(res[1:M])
            state.phase = FEAST_PHASE_IDLE

            eps_tolerance = feast_tolerance(fpm, T)
            maxloop = fpm[4]
            converged = epsout[] <= eps_tolerance

            if converged || loop[] >= maxloop
                feast_sort!(lambda, q, res, M)
                mode[] = M
                # Running out of refinement loops is not success: report it so
                # callers can distinguish a converged answer from a truncated one.
                info[] = _feast_exit_info(converged, M, M0, N)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0  # Clear initialization flag
                state.initialized = false
                return
            end

            loop[] += 1
            fill!(Aq, zero(T))
            fill!(Sq, zero(T))

            # Restart from the Ritz vectors spanning the compressed subspace.
            state.active = state.rank
            copyto!(view(state.Q0, :, 1:state.rank), view(q, :, 1:state.rank))

            state.e = 1
            fpm[50] = 1  # Reset integration point counter
            Ze[] = state.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        throw(ArgumentError("feast_srci!: MULT_B received in unexpected phase $(state.phase)"))
    end

    state.initialized = false
    error("FEAST RCI kernel: Invalid job code ijob=$(ijob[]). " *
          "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
          "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_A)) (mult_a), " *
          "$(Int(Feast_RCI_MULT_B)) (mult_b), or $(Int(Feast_RCI_DONE)) (done)")
end


function feast_srcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      state::FeastSRCIState{T} = FeastSRCIState{T}()) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_srci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                    Emin, Emax, M0, lambda, q, mode, res, info; state=state)
    end
end

function feast_hrcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{Complex{T}}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      state::FeastHRCIState{T} = FeastHRCIState{T}()) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_hrci!(ijob, N, Ze, work, workc, zAq, zSq, fpm, epsout, loop,
                    Emin, Emax, M0, lambda, q, mode, res, info; state=state)
    end
end

function feast_grcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{Complex{T}}, Sq::Matrix{Complex{T}}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emid::Complex{T}, r::T, M0::Int,
                      lambda::Vector{Complex{T}}, q::Matrix{Complex{T}}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      state::FeastGRCIState{T} = FeastGRCIState{T}()) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_grci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info; state=state)
    end
end

"""
    ifeast_srci!(...)

Iterative-FEAST-compatible RCI entry point for real symmetric problems. The RCI
kernel is solver-neutral: it issues the same factorize, solve, multiply, and
done jobs as `feast_srci!`, while the caller decides whether each shifted solve
uses a direct or iterative method.
"""
function ifeast_srci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int},
                      Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int};
                      state::FeastSRCIState{T} = FeastSRCIState{T}()) where T<:Real
    return feast_srci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                       Emin, Emax, M0, lambda, q, mode, res, info; state=state)
end

"""
    ifeast_hrci!(...)

Iterative-FEAST-compatible RCI entry point for complex Hermitian problems. The
caller owns the shifted linear solve requested by the RCI job code, so this
wrapper preserves the existing `feast_hrci!` state machine and exposes the
iterative FEAST API name.
"""
function ifeast_hrci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}},
                      fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                      Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{Complex{T}},
                      mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                      state::FeastHRCIState{T} = FeastHRCIState{T}()) where T<:Real
    return feast_hrci!(ijob, N, Ze, work, workc, zAq, zSq, fpm, epsout, loop,
                       Emin, Emax, M0, lambda, q, mode, res, info; state=state)
end

"""
    ifeast_grci!(...)

Iterative-FEAST-compatible RCI entry point for general non-Hermitian problems.
It is a solver-neutral wrapper around `feast_grci!`; callers provide the
direct, GMRES, or other shifted-system solve whenever `ijob` requests it.
"""
function ifeast_grci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{Complex{T}}, Sq::Matrix{Complex{T}},
                      fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                      Emid::Complex{T}, r::T, M0::Int,
                      lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                      mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                      state::FeastGRCIState{T} = FeastGRCIState{T}()) where T<:Real
    return feast_grci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                       Emid, r, M0, lambda, q, mode, res, info; state=state)
end

# Complex Hermitian FEAST reverse-communication kernel. Same protocol as
# `feast_srci!`: the contour sweep is followed by MULT_A/MULT_B to build the
# reduced pencil and a second MULT_A/MULT_B to evaluate residuals. Results of
# the multiply requests go into `workc`, which is complex.
@views function feast_hrci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                            work::Matrix{T}, workc::Matrix{Complex{T}},
                            zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}},
                            fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                            Emin::T, Emax::T, M0::Int,
                            lambda::Vector{T}, q::Matrix{Complex{T}},
                            mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                            state::FeastHRCIState{T} = FeastHRCIState{T}()) where T<:Real

    mode[] = 0

    if ijob[] == -1
        feastdefault!(fpm)

        info[] = Int(Feast_SUCCESS)
        if N <= 0
            info[] = Int(Feast_ERROR_N)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end
        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end
        if Emin >= Emax
            info[] = Int(Feast_ERROR_EMIN_EMAX)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        contour = feast_get_custom_contour(T, fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end

        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)
        state.ne = length(contour.Zne)
        state.eps = feast_tolerance(fpm, T)
        state.maxloop = fpm[4]
        state.e = 1
        state.M = 0
        state.rank = 0
        state.active = M0
        state.phase = FEAST_PHASE_IDLE
        state.initialized = true

        fpm[50] = 1
        fpm[51] = length(contour.Zne)
        fpm[52] = 0
        fpm[53] = 1

        loop[] = 0

        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(lambda, zero(T))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))
        fill!(work, zero(T))

        if fpm[5] == 1
            # User-provided initial subspace: normalize columns
            fallback_rng = MersenneTwister(hash((N, M0, :fallback_hrci)))
            for j in 1:M0
                if norm(workc[:, j]) > 0
                    workc[:, j] ./= norm(workc[:, j])
                else
                    for i in 1:N
                        workc[i, j] = Complex{T}(randn(fallback_rng, T), zero(T))
                    end
                    workc[:, j] ./= norm(workc[:, j])
                end
            end
        else
            # Deterministic seeded random subspace for reproducibility
            _feast_seeded_subspace_complex!(view(workc, :, 1:M0))
        end

        # Save initial subspace for the contour sweep
        state.Q0 = copy(workc[:, 1:M0])
        state.Q_proj = zeros(Complex{T}, N, M0)
        state.Qb = Matrix{Complex{T}}(undef, N, M0)
        state.AQ = Matrix{Complex{T}}(undef, N, M0)
        state.perm = Vector{Int}(undef, M0)
        state.q_tmp = Matrix{Complex{T}}(undef, N, M0)
        state.lambda_tmp = Vector{T}(undef, M0)
        state.residual = Vector{Complex{T}}(undef, N)

        Ze[] = state.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state.initialized = false
        state.phase = FEAST_PHASE_IDLE
        return
    end

    if !state.initialized
        throw(ArgumentError(
            "feast_hrci! called with ijob=$(ijob[]) on an uninitialized state. " *
            "Pass the same `state=FeastHRCIState{T}()` object to every call in " *
            "one RCI loop, and start the loop with ijob[] = -1."))
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        ijob[] = Int(Feast_RCI_SOLVE)
        active = state.active
        copyto!(view(workc, :, 1:active), view(state.Q0, :, 1:active))
        active < M0 && fill!(view(workc, :, (active + 1):M0), zero(Complex{T}))
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        e = state.e
        ne = state.ne
        Zne = state.Zne
        Wne = state.Wne
        active = state.active

        Q_proj = state.Q_proj
        if e == 1
            fill!(Q_proj, zero(Complex{T}))
        end

        weight = 2 * Wne[e]
        Q_proj[:, 1:active] .+= weight .* workc[:, 1:active]

        fpm[50] = e + 1
        state.e = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        fpm[50] = 1
        state.e = 1

        # Hermitian eigenvectors are genuinely complex, so unlike the real
        # kernel the filtered subspace stays complex here.
        rank = _feast_qr_compress!(state.Qb, state.Q_proj, active;
                                   rank_tol=sqrt(eps(T)))
        if rank == 0
            info[] = Int(Feast_ERROR_NO_CONVERGENCE)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0
            state.initialized = false
            return
        end
        state.rank = rank

        copyto!(view(q, :, 1:rank), view(state.Qb, :, 1:rank))
        state.phase = FEAST_PHASE_PROJECT_A
        mode[] = rank
        ijob[] = Int(Feast_RCI_MULT_A)
        return
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        rank = state.rank

        if state.phase == FEAST_PHASE_PROJECT_A
            zAq_block = view(zAq, 1:rank, 1:rank)
            mul!(zAq_block, adjoint(view(state.Qb, :, 1:rank)),
                 view(workc, :, 1:rank))
            _feast_hermitian_part!(zAq_block)

            copyto!(view(q, :, 1:rank), view(state.Qb, :, 1:rank))
            state.phase = FEAST_PHASE_PROJECT_B
            mode[] = rank
            ijob[] = Int(Feast_RCI_MULT_B)
            return
        end

        if state.phase == FEAST_PHASE_RESIDUAL_A
            M = state.M
            copyto!(view(state.AQ, :, 1:M), view(workc, :, 1:M))
            state.phase = FEAST_PHASE_RESIDUAL_B
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_B)
            return
        end

        throw(ArgumentError("feast_hrci!: MULT_A received in unexpected phase $(state.phase)"))
    end

    if ijob[] == Int(Feast_RCI_MULT_B)
        if state.phase == FEAST_PHASE_PROJECT_B
            rank = state.rank
            zAq_block = view(zAq, 1:rank, 1:rank)
            zSq_block = view(zSq, 1:rank, 1:rank)

            mul!(zSq_block, adjoint(view(state.Qb, :, 1:rank)),
                 view(workc, :, 1:rank))
            _feast_hermitian_part!(zSq_block)

            local lambda_red, v_red
            try
                F = eigen(Hermitian(zAq_block), Hermitian(zSq_block))
                lambda_red = F.values
                v_red = F.vectors
            catch err
                if err isa PosDefException || err isa LinearAlgebra.LAPACKException ||
                   err isa SingularException
                    @debug "Hermitian-definite reduced pencil failed; using the general solver" exception=err
                    try
                        F = eigen(zAq_block, zSq_block)
                        lambda_red = real.(F.values)
                        v_red = F.vectors
                    catch err2
                        @debug "Reduced eigenproblem failed" exception=err2
                        info[] = Int(Feast_ERROR_LAPACK)
                        ijob[] = Int(Feast_RCI_DONE)
                        fpm[53] = 0
                        state.initialized = false
                        return
                    end
                else
                    @debug "Reduced eigenproblem failed" exception=err
                    info[] = Int(Feast_ERROR_LAPACK)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0
                    state.initialized = false
                    return
                end
            end

            mul!(view(q, :, 1:rank), view(state.Qb, :, 1:rank), v_red)
            copyto!(view(lambda, 1:rank), view(lambda_red, 1:rank))
            for j in 1:rank
                nrm = norm(view(q, :, j))
                nrm > 0 && (view(q, :, j) ./= nrm)
            end

            M = _feast_reorder_by_interval!(lambda, q, state.perm,
                                            state.lambda_tmp, state.q_tmp,
                                            Emin, Emax, rank)
            fpm[52] = M
            state.M = M

            if M == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end

            state.phase = FEAST_PHASE_RESIDUAL_A
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_A)
            return
        end

        if state.phase == FEAST_PHASE_RESIDUAL_B
            M = state.M
            residual = state.residual
            AQ = state.AQ
            for j in 1:M
                @inbounds for i in 1:N
                    residual[i] = AQ[i, j] - lambda[j] * workc[i, j]
                end
                qnorm = norm(view(q, :, j))
                denom = max(abs(lambda[j]), one(T)) * max(qnorm, eps(T))
                res[j] = norm(residual) / denom
            end
            epsout[] = maximum(res[1:M])
            state.phase = FEAST_PHASE_IDLE

            converged = epsout[] <= state.eps

            if converged || loop[] >= state.maxloop
                feast_sort!(lambda, q, res, M)
                mode[] = M
                info[] = _feast_exit_info(converged, M, M0, N)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end

            loop[] += 1
            fill!(zAq, zero(Complex{T}))
            fill!(zSq, zero(Complex{T}))

            state.active = state.rank
            copyto!(view(state.Q0, :, 1:state.rank), view(q, :, 1:state.rank))

            state.e = 1
            fpm[50] = 1
            Ze[] = state.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        throw(ArgumentError("feast_hrci!: MULT_B received in unexpected phase $(state.phase)"))
    end

    state.initialized = false
    error("FEAST RCI kernel (Hermitian): Invalid job code ijob=$(ijob[]). " *
          "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
          "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_A)) (mult_a), " *
          "$(Int(Feast_RCI_MULT_B)) (mult_b), or $(Int(Feast_RCI_DONE)) (done)")
end

@views function feast_grci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                            work::Matrix{T}, workc::Matrix{Complex{T}},
                            Aq::Matrix{Complex{T}}, Sq::Matrix{Complex{T}},
                            fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                            Emid::Complex{T}, r::T, M0::Int,
                            lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                            mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                            state::FeastGRCIState{T} = FeastGRCIState{T}()) where T<:Real

    mode[] = 0

    # Feast RCI for general (non-Hermitian) eigenvalue problems
    # Uses circular contour in complex plane

    # Use fpm slots 50-64 for internal state storage
    # fpm[50] = current integration point e
    # fpm[51] = total integration points ne
    # fpm[52] = stored M value
    # fpm[53] = initialization flag (1 = initialized, 0 = not initialized)

    if ijob[] == -1  # Initialization
        feastdefault!(fpm)

        info[] = Int(Feast_SUCCESS)

        if N <= 0
            info[] = Int(Feast_ERROR_N)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if r <= 0
            info[] = Int(Feast_ERROR_EMID_R)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        contour = feast_get_custom_contour(T, fpm)
        if contour === nothing
            contour = feast_gcontour(Emid, r, fpm)
        end

        # Cache contour in state to avoid regeneration on every SOLVE call
        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(contour.Zne)
        fpm[52] = 0
        fpm[53] = 1

        loop[] = 0

        # Initialize workspace arrays
        fill!(Aq, zero(Complex{T}))
        fill!(Sq, zero(Complex{T}))

        fill!(lambda, zero(Complex{T}))

        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))

        # Initialize workc with initial subspace
        if fpm[5] == 1
            # User-provided initial subspace: normalize columns
            fallback_rng = MersenneTwister(hash((N, M0, :fallback_grci)))
            for j in 1:M0
                if norm(workc[:, j]) > 0
                    workc[:, j] ./= norm(workc[:, j])
                else
                    for i in 1:N
                        workc[i, j] = Complex{T}(randn(fallback_rng, T), randn(fallback_rng, T))
                    end
                    workc[:, j] ./= norm(workc[:, j])
                end
            end
        else
            # Deterministic seeded random subspace for reproducibility
            _feast_seeded_subspace_complex!(view(workc, :, 1:M0))
        end

        # work is used for real intermediate results
        fill!(work, zero(T))
        state.Q0 = copy(workc[:, 1:M0])
        state.perm = Vector{Int}(undef, M0)
        state.workc_tmp = Matrix{Complex{T}}(undef, N, M0)
        state.Qb = Matrix{Complex{T}}(undef, N, M0)
        state.AQ = Matrix{Complex{T}}(undef, N, M0)
        state.rank = M0
        state.active = M0
        state.residual = Vector{Complex{T}}(undef, N)
        state.phase = FEAST_PHASE_IDLE
        state.initialized = true

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state.initialized = false
        state.phase = FEAST_PHASE_IDLE
        return
    end

    if !state.initialized
        throw(ArgumentError(
            "feast_grci! called with ijob=$(ijob[]) on an uninitialized state. " *
            "Pass the same `state=FeastGRCIState{T}()` object to every call in " *
            "one RCI loop, and start the loop with ijob[] = -1."))
    end

    # Main Feast iteration loop for general (non-Hermitian) eigenvalue problems
    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*B - A) for general matrices
        ijob[] = Int(Feast_RCI_SOLVE)
        active = state.active
        copyto!(view(workc, :, 1:active), view(state.Q0, :, 1:active))
        active < M0 && fill!(view(workc, :, (active + 1):M0), zero(Complex{T}))
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        # User has solved linear systems (Ze*B - A)*workc = rhs
        e = fpm[50]  # Get current integration point from fpm
        ne = fpm[51]  # Get total integration points from fpm

        # Use cached contour from state (set during init and refinement loop reset)
        Zne = state.Zne
        Wne = state.Wne
        active = state.active

        # Accumulate subspace vectors Q
        for j in 1:active
            for i in 1:N
                q[i, j] += Wne[e] * workc[i, j]
            end
        end

        # Move to next integration point
        fpm[50] = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            # All integration points processed
            fpm[50] = 1  # Reset for next refinement loop

            # Rank-compress the filtered subspace. Without this an M0 larger
            # than the number of eigenvalues inside the contour leaves the
            # reduced pencil rank deficient, which produces spurious Ritz pairs
            # and stalls refinement -- the symmetric and Hermitian kernels have
            # always compressed here.
            rank = _feast_qr_compress!(state.Qb, q, state.active; rank_tol=sqrt(eps(T)))
            if rank == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end
            state.rank = rank
            copyto!(view(q, :, 1:rank), view(state.Qb, :, 1:rank))
            rank < state.active && fill!(view(q, :, (rank + 1):state.active), zero(Complex{T}))

            # Ask user to compute work = B*Q
            fill!(work, zero(T))
            ijob[] = Int(Feast_RCI_MULT_B)
            mode[] = rank
            return
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_B) && state.phase == FEAST_PHASE_IDLE
        # User has computed workc = B*Q
        # Form zBq = Q^H * (B*Q) = Q^H * workc
        rank = state.rank
        mul!(view(Sq, 1:rank, 1:rank),
             adjoint(view(q, :, 1:rank)),
             view(workc, :, 1:rank))

        # Now ask user to compute work = A*Q
        fill!(workc, zero(Complex{T}))
        ijob[] = Int(Feast_RCI_MULT_A)
        mode[] = rank
        state.phase = FEAST_PHASE_PROJECT_A  # Next MULT_A is for forming zAq
        return
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        if state.phase == FEAST_PHASE_PROJECT_A
            # Computing zAq = Q^H * A * Q
            rank = state.rank
            mul!(view(Aq, 1:rank, 1:rank),
                 adjoint(view(q, :, 1:rank)),
                 view(workc, :, 1:rank))

            # Now solve reduced eigenvalue problem: zAq*v = lambda*zBq*v
            try
                F = eigen(view(Aq, 1:rank, 1:rank), view(Sq, 1:rank, 1:rank))
                lambda_red = F.values
                v_red = F.vectors

                # Partition eigenvalues by contour membership in a single pass:
                # in-contour pairs to the front of perm, outside pairs to the
                # back. One ellipse membership test per eigenvalue (was two).
                M = 0
                perm = state.perm
                tail = rank
                for i in 1:rank
                    if feast_inside_gcontour(lambda_red[i], Emid, r; fpm=fpm)
                        M += 1
                        perm[M] = i
                        lambda[M] = lambda_red[i]
                    else
                        perm[tail] = i
                        tail -= 1
                    end
                end

                fpm[52] = M

                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0
                    state.initialized = false
                    return
                end

                # Project the rank compressed basis onto the Ritz vectors:
                # workc = q * v_red via BLAS (replaces an O(N·rank²) scalar loop).
                mul!(view(workc, :, 1:rank), view(q, :, 1:rank), v_red)

                copyto!(view(state.workc_tmp, :, 1:rank), view(workc, :, 1:rank))
                for new_idx in 1:rank
                    old_idx = perm[new_idx]
                    lambda[new_idx] = lambda_red[old_idx]
                    copyto!(view(workc, :, new_idx),
                            view(state.workc_tmp, :, old_idx))
                end

                # Normalize eigenvectors
                for idx in 1:rank
                    q_norm_sq = zero(T)
                    @inbounds for k in 1:N
                        q_norm_sq += abs2(workc[k, idx])
                    end
                    q_norm = sqrt(q_norm_sq)
                    if q_norm > zero(T)
                        inv_norm = inv(q_norm)
                        @inbounds for k in 1:N
                            workc[k, idx] *= inv_norm
                        end
                    end
                end

                # Copy the Ritz vectors back to q for the next iteration
                copyto!(view(q, :, 1:rank), view(workc, :, 1:rank))

                # Now compute residuals: need A*q_new, then B*q_new
                fill!(workc, zero(Complex{T}))
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                state.phase = FEAST_PHASE_RESIDUAL_A
                return

            catch err
                @debug "Reduced eigenproblem failed in feast_grci!" exception=err
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end
        elseif state.phase == FEAST_PHASE_RESIDUAL_A
            # workc holds A*q; stash it while the caller computes B*q so the
            # residual can be the generalized one rather than ||Aq - lambda q||.
            M = fpm[52]
            copyto!(view(state.AQ, :, 1:M), view(workc, :, 1:M))
            fill!(workc, zero(Complex{T}))
            state.phase = FEAST_PHASE_RESIDUAL_B
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_B)
            return
        else
            throw(ArgumentError("feast_grci!: MULT_A received in unexpected phase $(state.phase)"))
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_B) && state.phase == FEAST_PHASE_RESIDUAL_B
        # state.AQ holds A*q and workc holds B*q for the same Ritz vectors.
        M = fpm[52]
        state.phase = FEAST_PHASE_IDLE

        residual = state.residual
        AQ = state.AQ
        for j in 1:M
            @inbounds for i in 1:N
                residual[i] = AQ[i, j] - lambda[j] * workc[i, j]
            end
            qnorm = norm(view(q, :, j))
            denom = max(abs(lambda[j]), one(T)) * max(qnorm, eps(T))
            res[j] = norm(residual) / denom
        end

        max_res = zero(T)
        @inbounds for j in 1:M
            max_res = max(max_res, res[j])
        end
        epsout[] = max_res

        eps_tolerance = feast_tolerance(fpm, T)
        maxloop = fpm[4]
        converged = epsout[] <= eps_tolerance

        if converged || loop[] >= maxloop
            feast_sort_general!(lambda, q, res, M)
            mode[] = M
            info[] = _feast_exit_info(converged, M, M0, N)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0
            state.initialized = false
            return
        end

        # Start new refinement loop from the compressed Ritz basis.
        loop[] += 1

        rank = state.rank
        copyto!(view(state.Q0, :, 1:rank), view(q, :, 1:rank))
        state.active = rank

        fill!(Aq, zero(Complex{T}))
        fill!(Sq, zero(Complex{T}))
        fill!(q, zero(Complex{T}))

        copyto!(view(workc, :, 1:rank), view(state.Q0, :, 1:rank))

        # Re-cache contour for next refinement loop
        contour = feast_get_custom_contour(T, fpm)
        if contour === nothing
            contour = feast_gcontour(Emid, r, fpm)
        end
        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)
        fpm[50] = 1

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    # Safety check: if we reach here, ijob has an invalid value
    if ijob[] != -1 && ijob[] != Int(Feast_RCI_FACTORIZE) &&
       ijob[] != Int(Feast_RCI_SOLVE) && ijob[] != Int(Feast_RCI_MULT_A) &&
       ijob[] != Int(Feast_RCI_MULT_B) && ijob[] != Int(Feast_RCI_DONE)
        state.initialized = false
        error("FEAST RCI kernel (General): Invalid job code ijob=$(ijob[]). " *
              "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
              "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_B)) (mult_b), " *
              "$(Int(Feast_RCI_MULT_A)) (mult_a), or $(Int(Feast_RCI_DONE)) (done)")
    end
end

############################
# Polynomial RCI interfaces
############################

@inline function _feast_poly_contour_vector(::Type{T},
                                           nodes::Vector{Complex{T}}) where T<:Real
    return nodes
end

function _feast_poly_contour_vector(::Type{T},
                                    nodes::AbstractVector{<:Complex}) where T<:Real
    return Complex{T}.(nodes)
end

function _ensure_poly_rci_state!(state::FeastPolyRCIState{T}, N::Int,
                                 M0::Int) where T<:Real
    if size(state.S0) != (N, M0)
        state.S0 = zeros(Complex{T}, N, M0)
        state.S1 = zeros(Complex{T}, N, M0)
        state.basis = Matrix{Complex{T}}(undef, N, M0)
    end
    if length(state.residual) != N
        state.residual = Vector{Complex{T}}(undef, N)
    end
    state.rank = 0
    state.initialized = true
    return state
end

"""
    _feast_beyn_reduce!(state, rank_tol)

Beyn's reduction of the contour moments held in `state`. Truncates
`S0 = U Σ Wᴴ` at `rank_tol` (relative to the leading singular value) and returns
`(k, B)` where `k` is the detected rank and `B = Uₖᴴ S1 Wₖ Σₖ⁻¹` is the `k x k`
matrix whose eigenvalues are the eigenvalues of `P` inside the contour. The
leading `k` columns of `state.basis` hold `Uₖ`, so an eigenvector `s` of `B`
lifts to the eigenvector `Uₖ s` of `P`.

Returns `(0, nothing)` when the moments are numerically zero, which is what a
contour enclosing a root together with its negative produces: the two residues
cancel and there is nothing to recover.
"""
function _feast_beyn_reduce!(state::FeastPolyRCIState{T}, rank_tol::T) where T<:Real
    S0 = state.S0
    S1 = state.S1
    F = svd(S0)
    sigma = F.S
    isempty(sigma) && return 0, nothing

    scale = sigma[1]
    scale > zero(T) || return 0, nothing
    threshold = max(rank_tol, eps(T) * maximum(size(S0))) * scale
    cand = 0
    @inbounds for value in sigma
        value > threshold || break
        cand += 1
    end
    cand == 0 && return 0, nothing

    k = cand

    Uk = view(F.U, :, 1:k)
    Wk = view(F.V, :, 1:k)
    copyto!(view(state.basis, :, 1:k), Uk)
    state.rank = k

    # B = Ukᴴ S1 Wk Σk⁻¹
    B = (Uk' * S1) * Wk
    @inbounds for j in 1:k
        inv_sigma = inv(sigma[j])
        for i in 1:k
            B[i, j] *= inv_sigma
        end
    end
    return k, B
end

function feast_grcipevx!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                         work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                         Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                         fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                         Emid::Complex{T}, r::T, M0::Int,
                         lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                         mode::Ref{Int}, res::Vector{T}, info::Ref{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}};
                         state::FeastPolyRCIState{T}=FeastPolyRCIState{T}()) where {T<:Real, TZ<:Real, TW<:Real}
    contour_nodes = _feast_poly_contour_vector(T, Zne)
    contour_weights = _feast_poly_contour_vector(T, Wne)
    _feast_poly_grci!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                      Emid, r, M0, lambda, q, mode, res, info,
                      contour_nodes, contour_weights; state=state)
end

function feast_grcipev!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                        work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                        Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                        fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                        Emid::Complex{T}, r::T, M0::Int,
                        lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                        mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                        state::FeastPolyRCIState{T}=FeastPolyRCIState{T}()) where T<:Real
    contour = feast_gcontour(Emid, r, fpm)
    feast_grcipevx!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info,
                    contour.Zne, contour.Wne; state=state)
end

function feast_srcipevx!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                         work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                         Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                         fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                         Emid::Complex{T}, r::Real, M0::Int,
                         lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                         mode::Ref{Int}, res::Vector{T}, info::Ref{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}};
                         state::FeastPolyRCIState{T}=FeastPolyRCIState{T}()) where {T<:Real, TZ<:Real, TW<:Real}
    contour_nodes = _feast_poly_contour_vector(T, Zne)
    contour_weights = _feast_poly_contour_vector(T, Wne)
    _feast_poly_grci!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                      Emid, T(r), M0, lambda, q, mode, res, info,
                      contour_nodes, contour_weights; state=state)
end

function feast_srcipev!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                        work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                        Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                        fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                        Emid::Complex{T}, r::Real, M0::Int,
                        lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                        mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                        state::FeastPolyRCIState{T}=FeastPolyRCIState{T}()) where T<:Real
    contour = feast_gcontour(Emid, T(r), fpm)
    feast_srcipevx!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info,
                    contour.Zne, contour.Wne; state=state)
end

@views function _feast_poly_grci!(ijob::Ref{Int}, dmax::Int, N::Int,
                                  Ze::Ref{Complex{T}},
                                  work::Matrix{Complex{T}},
                                  workc::Matrix{Complex{T}},
                                  Aq::Matrix{Complex{T}},
                                  Bq::Matrix{Complex{T}},
                                  fpm::Vector{Int}, epsout::Ref{T},
                                  loop::Ref{Int}, Emid::Complex{T}, r::T,
                                  M0::Int, lambda::Vector{Complex{T}},
                                  q::Matrix{Complex{T}}, mode::Ref{Int},
                                  res::Vector{T}, info::Ref{Int},
                                  Zne::Vector{Complex{T}},
                                  Wne::Vector{Complex{T}};
                                  state::FeastPolyRCIState{T}=FeastPolyRCIState{T}()) where T<:Real

    # Use fpm slots 50-64 for internal state storage
    # fpm[50] = current integration point e
    # fpm[51] = total integration points ne
    # fpm[52] = stored M value
    # fpm[53] = initialization flag (1 = initialized, 0 = not initialized)

    if ijob[] == -1
        feastdefault!(fpm)

        info[] = Int(Feast_SUCCESS)

        if dmax < 1
            info[] = Int(Feast_ERROR_INTERNAL)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if N <= 0
            info[] = Int(Feast_ERROR_N)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if M0 <= 0
            info[] = Int(Feast_ERROR_M0)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        if r <= zero(T)
            info[] = Int(Feast_ERROR_EMID_R)
            mode[] = 0
            ijob[] = Int(Feast_RCI_DONE)
            return
        end

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(Zne)
        fpm[52] = 0
        fpm[53] = 1

        _ensure_poly_rci_state!(state, N, M0)

        fill!(Aq, zero(Complex{T}))
        fill!(Bq, zero(Complex{T}))
        fill!(lambda, zero(Complex{T}))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))

        if fpm[5] == 1
            # User-provided initial subspace: normalize columns, replace zero columns
            fallback_rng = MersenneTwister(hash((N, M0, :fallback_poly)))
            for j in 1:M0
                normval = norm(work[:, j])
                if normval > 0
                    work[:, j] ./= normval
                else
                    for i in 1:N
                        work[i, j] = Complex{T}(randn(fallback_rng, T), randn(fallback_rng, T))
                    end
                    work[:, j] ./= norm(work[:, j])
                end
            end
        else
            # Deterministic seeded random subspace for reproducibility
            _feast_seeded_subspace_complex!(view(work, :, 1:M0))
        end

        loop[] = 0

        Ze[] = Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        ijob[] = Int(Feast_RCI_SOLVE)
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        e = fpm[50]  # Get current integration point from fpm
        ne = fpm[51]  # Get total integration points from fpm

        if !state.initialized
            _ensure_poly_rci_state!(state, N, M0)
        end

        # Accumulate Beyn's moments at full width. workc holds P(z_e)⁻¹ Q.
        S0 = state.S0
        S1 = state.S1
        if e == 1
            fill!(S0, zero(Complex{T}))
            fill!(S1, zero(Complex{T}))
        end
        weight = Wne[e]
        zweight = weight * Zne[e]
        @inbounds for col in 1:M0
            for row in 1:N
                val = workc[row, col]
                S0[row, col] += weight * val
                S1[row, col] += zweight * val
            end
        end

        fpm[50] = e + 1  # Store incremented counter in fpm
        if e < ne
            Ze[] = Zne[e + 1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        fpm[50] = 1  # Reset for next refinement loop
        try
            # Truncate S0 at its numerical rank before forming the reduced
            # matrix. The projected M0 x M0 form this replaced was singular
            # whenever M0 exceeded the number of eigenvalues inside the
            # contour, which put the residual on a floor that extra contour
            # points made worse rather than better.
            k, B_red = _feast_beyn_reduce!(state, eps(T))
            if k == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                mode[] = 0
                ijob[] = Int(Feast_RCI_DONE)
                fpm[52] = 0
                fpm[53] = 0
                state.initialized = false
                return
            end

            # Keep the reduced matrix visible to callers that inspect the RCI
            # workspace. Bq is unused by this kernel now that the moments live
            # in the state object at full width.
            copyto!(view(Aq, 1:k, 1:k), B_red)

            F = eigen(B_red)
            lambda_red = F.values
            v_red = F.vectors

            M = 0
            basis = view(state.basis, :, 1:k)
            for i in 1:k
                if feast_inside_gcontour(lambda_red[i], Emid, r; fpm=fpm)
                    M += 1
                    lambda[M] = lambda_red[i]
                    q_col = view(q, :, M)
                    # Beyn lifts an eigenvector of the reduced matrix back with
                    # the truncated left singular basis, not with the raw trial
                    # block.
                    mul!(q_col, basis, view(v_red, :, i))
                    q_norm_sq = zero(T)
                    @inbounds for row in 1:N
                        q_norm_sq += abs2(q[row, M])
                    end
                    q_norm = sqrt(q_norm_sq)
                    if q_norm > 0
                        inv_norm = inv(q_norm)
                        @inbounds for row in 1:N
                            q[row, M] *= inv_norm
                        end
                    end
                end
            end

            if M == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                mode[] = 0
                ijob[] = Int(Feast_RCI_DONE)
                fpm[52] = 0
                fpm[53] = 0  # Clear initialization flag
                state.initialized = false
                return
            end

            fpm[52] = M  # Store M in fpm
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_A)
            return
        catch err
            @debug "Polynomial reduced eigenproblem failed" exception=err
            info[] = Int(Feast_ERROR_LAPACK)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            state.initialized = false
            return
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        M = fpm[52]  # Get M from fpm
        max_res = zero(T)
        if !state.initialized
            _ensure_poly_rci_state!(state, N, M0)
        end
        for j in 1:M
            # The caller writes P(lambda_j) * q_j into workc, so the polynomial
            # residual is that column's norm -- there is no separate lambda*q
            # term to subtract, unlike the linear kernels.
            qnorm = norm(view(q, :, j))
            denom = max(abs(lambda[j]), one(T)) * max(qnorm, eps(T))
            res[j] = norm(view(workc, :, j)) / denom
            max_res = max(max_res, res[j])
        end
        epsout[] = max_res

        eps_tolerance = feast_tolerance(fpm, T)
        maxloop = max(1, fpm[4])
        converged = epsout[] <= eps_tolerance

        if converged || loop[] >= maxloop
            feast_sort_general!(lambda, q, res, M)
            mode[] = M
            # A full-rank moment may hide additional roots. Only a linear
            # problem spanning all N dimensions is complete by construction;
            # higher-degree problems can contain up to dmax*N roots.
            saturated = M == min(N, M0) && M < dmax * N
            info[] = saturated ? Int(Feast_ERROR_M0) :
                     converged ? Int(Feast_SUCCESS) : Int(Feast_ERROR_NO_CONVERGENCE)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            state.initialized = false
            return
        else
            loop[] += 1
            fill!(Aq, zero(Complex{T}))
            fill!(Bq, zero(Complex{T}))
            copyto!(view(work, :, 1:M), view(q, :, 1:M))
            fpm[50] = 1  # Reset integration point counter
            Ze[] = Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    if ijob[] != Int(Feast_RCI_DONE)
        error("FEAST polynomial RCI kernel: unexpected ijob=$(ijob[]).")
    end
end
