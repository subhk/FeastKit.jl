function feast_srci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                     epsout::Ref{T}, loop::Ref{Int},
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                     res::Vector{T}, info::Ref{Int};
                     state::FeastSRCIState{T} = FeastSRCIState{T}()) where T<:Real

    if ijob[] == -1  # Initialization
        feastdefault!(fpm)

        info[] = Int(Feast_SUCCESS)

        if N <= 0
            info[] = Int(Feast_ERROR_N)
            return
        end

        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            return
        end

        if Emin >= Emax
            info[] = Int(Feast_ERROR_EMIN_EMAX)
            return
        end

        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end

        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)
        state.ne = length(contour.Zne)
        state.e = 1
        state.initialized = true

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(contour.Zne)
        fpm[52] = 0
        fpm[53] = 1

        loop[] = 0

        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        fill!(lambda, zero(T))
        fill!(q, zero(T))
        fill!(res, zero(T))
        fill!(workc, zero(Complex{T}))

        if fpm[5] == 1
            # User-provided initial subspace: normalize columns
            for j in 1:M0
                if norm(work[:, j]) > 0
                    work[:, j] ./= norm(work[:, j])
                else
                    for i in 1:N
                        work[i, j] = randn(T)
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
        state.zAq = zeros(Complex{T}, M0, M0)
        state.zSq = zeros(Complex{T}, M0, M0)

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # After factorization, request linear solve
        ijob[] = Int(Feast_RCI_SOLVE)
        Q0 = state.Q0
        work[:, 1:size(Q0, 2)] = Q0
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        if !state.initialized
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_contour(Emin, Emax, fpm)
            end
            state.Zne = copy(contour.Zne)
            state.Wne = copy(contour.Wne)
            state.ne = length(contour.Zne)
            state.e = 1
            state.initialized = true
        end
        Zne = state.Zne
        Wne = state.Wne
        e = state.e
        ne = state.ne

        Q0 = state.Q0
        M_current = size(Q0, 2)

        Q_proj = state.Q_proj
        zAq = state.zAq
        zSq = state.zSq

        # Reset accumulators at start of new contour loop
        if e == 1
            fill!(Q_proj, zero(Complex{T}))
            fill!(zAq, zero(Complex{T}))
            fill!(zSq, zero(Complex{T}))
        end

        weight = 2 * Wne[e]  # Account for conjugate half-contour

        # Accumulate filtered subspace (spectral projector applied to Q0)
        Q_proj[:, 1:M_current] .+= weight .* workc[:, 1:M_current]

        # Accumulate moment matrices in complex (take real() only after full contour)
        temp = Q0' * workc[:, 1:M_current]
        zAq[1:M_current, 1:M_current] .+= weight * temp
        zSq[1:M_current, 1:M_current] .+= weight * Zne[e] * temp

        fpm[50] = e + 1  # Store incremented counter in fpm
        state.e = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            fpm[50] = 1  # Reset for next refinement loop
            state.e = 1

            # Extract real part after full contour summation
            # (imaginary parts cancel by conjugate symmetry of the half-contour)
            Aq[1:M_current, 1:M_current] .= real.(zAq[1:M_current, 1:M_current])
            Sq[1:M_current, 1:M_current] .= real.(zSq[1:M_current, 1:M_current])

            try
                # Solve generalized eigenvalue problem: Sq*v = lambda*Aq*v
                F = eigen(Sq[1:M_current, 1:M_current], Aq[1:M_current, 1:M_current])
                lambda_red = real.(F.values)
                v_red = real.(F.vectors)

                # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q0
                Q_proj_real = real.(Q_proj[:, 1:M_current])
                q[:, 1:M_current] = Q_proj_real * v_red[:, 1:M_current]
                lambda[1:M_current] = lambda_red[1:M_current]

                # Reorder: put eigenvalues inside the interval first
                inside_mask = [feast_inside_contour(lambda[i], Emin, Emax) for i in 1:M_current]
                inside_indices = findall(inside_mask)
                outside_indices = findall(.!inside_mask)
                perm = vcat(inside_indices, outside_indices)

                lambda[1:M_current] = lambda[perm]
                q[:, 1:M_current] = q[:, perm]

                M = length(inside_indices)

                fpm[52] = M  # Store M in fpm
                state.M = M

                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0  # Clear initialization flag
                    state.initialized = false
                    return
                end

                # Request caller to compute A * q[:, 1:M] and store result in work[:, 1:M].
                # The residual will be computed as ||work[:,j] - lambda[j]*q[:,j]||.
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                return
            catch err
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0  # Clear initialization flag
                state.initialized = false
                return
            end
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        # Caller must have computed A * q[:, 1:M] into work[:, 1:M]
        M = fpm[52]  # Get M from fpm

        for j in 1:M
            # Relative residual: ||Ax - λx|| / max(|λ|, 1) to avoid scale dependence
            res[j] = norm(work[:, j] - lambda[j] * q[:, j]) / max(abs(lambda[j]), one(T))
        end
        epsout[] = maximum(res[1:M])

        eps_tolerance = feast_tolerance(fpm)
        maxloop = fpm[4]

        if epsout[] <= eps_tolerance || loop[] >= maxloop
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            state.initialized = false
            return
        else
            loop[] += 1
            fill!(Aq, zero(T))
            fill!(Sq, zero(T))
            # Use full subspace (M0 columns) for next iteration
            work[:, 1:M0] = q[:, 1:M0]

            # Get contour for next refinement loop
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_contour(Emin, Emax, fpm)
            end
            state.Zne = copy(contour.Zne)
            state.Wne = copy(contour.Wne)
            state.ne = length(contour.Zne)
            state.e = 1
            fpm[50] = 1  # Reset integration point counter

            state.Q0 = copy(work[:, 1:M0])
            Ze[] = contour.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state.initialized = false
        return
    end

    state.initialized = false
    error("FEAST RCI kernel: Invalid job code ijob=$(ijob[]). " *
          "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
          "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_A)) (mult_a), " *
          "or $(Int(Feast_RCI_DONE)) (done)")
end


function feast_srcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_srci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                    Emin, Emax, M0, lambda, q, mode, res, info)
    end
end

function feast_hrcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emin::T, Emax::T, M0::Int,
                      lambda::Vector{T}, q::Matrix{Complex{T}}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_hrci!(ijob, N, Ze, work, workc, zAq, zSq, fpm, epsout, loop,
                    Emin, Emax, M0, lambda, q, mode, res, info)
    end
end

function feast_grcix!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                      work::Matrix{T}, workc::Matrix{Complex{T}},
                      Aq::Matrix{Complex{T}}, Sq::Matrix{Complex{T}}, fpm::Vector{Int},
                      epsout::Ref{T}, loop::Ref{Int}, Emid::Complex{T}, r::T, M0::Int,
                      lambda::Vector{Complex{T}}, q::Matrix{Complex{T}}, mode::Ref{Int},
                      res::Vector{T}, info::Ref{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    with_custom_contour(fpm, Zne, Wne) do
        feast_grci!(ijob, N, Ze, work, workc, Aq, Sq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info)
    end
end

function feast_hrci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}},
                     fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{Complex{T}},
                     mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                     state::FeastHRCIState{T} = FeastHRCIState{T}()) where T<:Real

    if ijob[] == -1
        feastdefault!(fpm)
        state.initialized = true

        info[] = Int(Feast_SUCCESS)
        if N <= 0
            info[] = Int(Feast_ERROR_N)
            state.initialized = false
            return
        end
        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            state.initialized = false
            return
        end
        if Emin >= Emax
            info[] = Int(Feast_ERROR_EMIN_EMAX)
            state.initialized = false
            return
        end

        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end

        state.Zne = copy(contour.Zne)
        state.Wne = copy(contour.Wne)
        state.ne = length(contour.Zne)
        state.eps = feast_tolerance(fpm)
        state.maxloop = fpm[4]
        state.e = 1
        state.M = 0

        loop[] = 0

        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(lambda, zero(T))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))
        fill!(work, zero(T))

        if fpm[5] == 1
            # User-provided initial subspace: normalize columns
            for j in 1:M0
                if norm(workc[:, j]) > 0
                    workc[:, j] ./= norm(workc[:, j])
                else
                    for i in 1:N
                        workc[i, j] = Complex{T}(randn(T), randn(T))
                    end
                    workc[:, j] ./= norm(workc[:, j])
                end
            end
        else
            # Deterministic seeded random subspace for reproducibility
            _feast_seeded_subspace_complex!(view(workc, :, 1:M0))
        end

        # Save initial subspace for moment accumulation
        state.Q0 = copy(workc[:, 1:M0])
        state.Q_proj = zeros(Complex{T}, N, M0)

        Ze[] = state.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        ijob[] = Int(Feast_RCI_SOLVE)
        Q0 = state.Q0
        M_current = size(Q0, 2)
        workc[:, 1:M_current] = Q0
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        if !state.initialized
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_contour(Emin, Emax, fpm)
            end
            state.Zne = copy(contour.Zne)
            state.Wne = copy(contour.Wne)
            state.ne = length(contour.Zne)
            state.e = 1
            state.initialized = true
        end

        e = state.e
        ne = state.ne
        Zne = state.Zne
        Wne = state.Wne

        # Use saved initial subspace Q0 and current solution in workc
        Q0 = state.Q0
        M_current = size(Q0, 2)

        Q_proj = state.Q_proj

        # Reset Q_proj at start of new contour loop
        if e == 1
            fill!(Q_proj, zero(Complex{T}))
        end

        weight = 2 * Wne[e]

        # Accumulate filtered subspace (spectral projector applied to Q0)
        Q_proj[:, 1:M_current] .+= weight .* workc[:, 1:M_current]

        # Accumulate moments: Q0' * Y where Y is the solution
        temp = Q0' * workc[:, 1:M_current]
        zAq[1:M_current, 1:M_current] .+= weight * temp
        zSq[1:M_current, 1:M_current] .+= weight * Zne[e] * temp

        state.e = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            state.e = 1
            try
                Q0 = state.Q0
                M_current = size(Q0, 2)

                # Solve generalized eigenvalue problem: Sq*v = lambda*Aq*v
                F = eigen(zSq[1:M_current, 1:M_current], zAq[1:M_current, 1:M_current])
                lambda_red = real.(F.values)
                v_red = F.vectors

                # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q0
                # For complex Hermitian problems, Q_proj is complex — do NOT take real()
                # since eigenvectors of complex Hermitian matrices are genuinely complex.
                Q_proj = state.Q_proj
                q[:, 1:M_current] = Q_proj[:, 1:M_current] * v_red[:, 1:M_current]
                lambda[1:M_current] = lambda_red[1:M_current]

                # Reorder: put eigenvalues inside the interval first
                inside_mask = [feast_inside_contour(lambda[i], Emin, Emax) for i in 1:M_current]
                inside_indices = findall(inside_mask)
                outside_indices = findall(.!inside_mask)
                perm = vcat(inside_indices, outside_indices)

                lambda[1:M_current] = lambda[perm]
                q[:, 1:M_current] = q[:, perm]

                M = length(inside_indices)

                state.M = M

                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    state.initialized = false
                    return
                end

                # Request caller to compute A * q[:, 1:M] and store result in workc[:, 1:M].
                # The residual will be computed as ||workc[:,j] - lambda[j]*q[:,j]||.
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                return
            catch err
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                state.initialized = false
                return
            end
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        # Caller must have computed A * q[:, 1:M] into workc[:, 1:M]
        M = state.M
        for j in 1:M
            # Relative residual: ||Ax - λx|| / max(|λ|, 1)
            res[j] = norm(workc[:, j] - lambda[j] * q[:, j]) / max(abs(lambda[j]), one(T))
        end
        epsout[] = maximum(res[1:M])
        eps = state.eps
        maxloop = state.maxloop

        if epsout[] <= eps || loop[] >= maxloop
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            state.initialized = false
            return
        else
            loop[] += 1
            fill!(zAq, zero(Complex{T}))
            fill!(zSq, zero(Complex{T}))
            # Use full subspace (M0 columns) for next iteration
            workc[:, 1:M0] = q[:, 1:M0]
            # Update Q0 for next refinement iteration with full subspace
            state.Q0 = copy(q[:, 1:M0])
            Ze[] = state.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state.initialized = false
        return
    end

    state.initialized = false
    error("FEAST RCI kernel (Hermitian): Invalid job code ijob=$(ijob[]). " *
          "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
          "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_A)) (mult_a), " *
          "or $(Int(Feast_RCI_DONE)) (done)")
end

function feast_grci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     Aq::Matrix{Complex{T}}, Sq::Matrix{Complex{T}},
                     fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                     Emid::Complex{T}, r::T, M0::Int,
                     lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                     mode::Ref{Int}, res::Vector{T}, info::Ref{Int};
                     state::FeastGRCIState{T} = FeastGRCIState{T}()) where T<:Real

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
            return
        end

        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            return
        end

        if r <= 0
            info[] = Int(Feast_ERROR_EMID_R)
            return
        end

        contour = feast_get_custom_contour(fpm)
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
            for j in 1:M0
                if norm(workc[:, j]) > 0
                    workc[:, j] ./= norm(workc[:, j])
                else
                    for i in 1:N
                        workc[i, j] = Complex{T}(randn(T), randn(T))
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
        state.initialized = true

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    # Main Feast iteration loop for general (non-Hermitian) eigenvalue problems
    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*B - A) for general matrices
        ijob[] = Int(Feast_RCI_SOLVE)
        Q0 = state.Q0
        M_current = size(Q0, 2)
        workc[:, 1:M_current] = Q0
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        # User has solved linear systems (Ze*B - A)*workc = rhs
        e = fpm[50]  # Get current integration point from fpm
        ne = fpm[51]  # Get total integration points from fpm

        # Use cached contour from state (set during init and refinement loop reset)
        Zne = state.Zne
        Wne = state.Wne

        # Accumulate subspace vectors Q
        for j in 1:M0
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

            # Ask user to compute work = B*Q
            fill!(work, zero(T))
            ijob[] = Int(Feast_RCI_MULT_B)
            mode[] = M0
            return
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_B)
        # User has computed workc = B*Q
        # Form zBq = Q^H * (B*Q) = Q^H * workc
        Sq[1:M0, 1:M0] = q[:, 1:M0]' * workc[:, 1:M0]

        # Now ask user to compute work = A*Q
        fill!(workc, zero(Complex{T}))
        ijob[] = Int(Feast_RCI_MULT_A)
        mode[] = M0
        state.mult_a_for_projection = true  # Next MULT_A is for forming zAq
        return
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        if state.mult_a_for_projection
            # Computing zAq = Q^H * A * Q
            Aq[1:M0, 1:M0] = q[:, 1:M0]' * workc[:, 1:M0]
            state.mult_a_for_projection = false

            # Now solve reduced eigenvalue problem: zAq*v = lambda*zBq*v
            try
                F = eigen(Aq[1:M0, 1:M0], Sq[1:M0, 1:M0])
                lambda_red = F.values
                v_red = F.vectors

                # Count eigenvalues inside circular region
                M = 0
                indices = Int[]
                for i in 1:M0
                    if feast_inside_gcontour(lambda_red[i], Emid, r)
                        M += 1
                        push!(indices, i)
                        lambda[M] = lambda_red[i]
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

                # Project ALL M0 eigenvectors to maintain full subspace
                fill!(workc, zero(Complex{T}))
                for idx in 1:M0
                    for k in 1:N
                        for j in 1:M0
                            workc[k, idx] += q[k, j] * v_red[j, idx]
                        end
                    end
                end

                # Reorder: put eigenvalues inside contour first
                outside_indices = setdiff(1:M0, indices)
                perm = vcat(indices, outside_indices)

                lambda_temp = copy(lambda_red)
                workc_temp = copy(workc[:, 1:M0])
                for (new_idx, old_idx) in enumerate(perm)
                    lambda[new_idx] = lambda_temp[old_idx]
                    workc[:, new_idx] = workc_temp[:, old_idx]
                end

                # Normalize eigenvectors
                for idx in 1:M0
                    q_norm = norm(view(workc, :, idx))
                    if q_norm > 0
                        workc[:, idx] ./= q_norm
                    end
                end

                # Copy all M0 back to q for next iteration
                q[:, 1:M0] = workc[:, 1:M0]

                # Now compute residuals: need A*q_new
                fill!(workc, zero(Complex{T}))
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                state.mult_a_for_projection = false  # Next MULT_A is for residual
                return

            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            end
        else
            # Computing residuals
            M = fpm[52]

            residual = zeros(Complex{T}, N)
            for j in 1:M
                @inbounds for i in 1:N
                    residual[i] = workc[i, j] - lambda[j] * q[i, j]
                end
                # Relative residual: ||Ax - λx|| / max(|λ|, 1)
                res[j] = norm(residual) / max(abs(lambda[j]), one(T))
            end

            epsout[] = maximum(res[1:M])

            eps_tolerance = feast_tolerance(fpm)
            maxloop = fpm[4]

            if epsout[] <= eps_tolerance || loop[] >= maxloop
                feast_sort_general!(lambda, q, res, M)
                mode[] = M
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0
                state.initialized = false
                return
            else
                # Start new refinement loop
                loop[] += 1

                q_saved = copy(q[:, 1:M0])

                fill!(Aq, zero(Complex{T}))
                fill!(Sq, zero(Complex{T}))
                fill!(q, zero(Complex{T}))

                workc[:, 1:M0] = q_saved

                # Re-cache contour for next refinement loop
                contour = feast_get_custom_contour(fpm)
                if contour === nothing
                    contour = feast_gcontour(Emid, r, fpm)
                end
                state.Zne = copy(contour.Zne)
                state.Wne = copy(contour.Wne)
                fpm[50] = 1

                Ze[] = contour.Zne[1]
                state.Q0 = copy(workc[:, 1:M0])
                ijob[] = Int(Feast_RCI_FACTORIZE)
                return
            end
        end
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

function feast_grcipevx!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                         work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                         Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                         fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                         Emid::Complex{T}, r::T, M0::Int,
                         lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                         mode::Ref{Int}, res::Vector{T}, info::Ref{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    contour_nodes = Complex{T}.(Zne)
    contour_weights = Complex{T}.(Wne)
    _feast_poly_grci!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                      Emid, r, M0, lambda, q, mode, res, info,
                      contour_nodes, contour_weights)
end

function feast_grcipev!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                        work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                        Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                        fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                        Emid::Complex{T}, r::T, M0::Int,
                        lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                        mode::Ref{Int}, res::Vector{T}, info::Ref{Int}) where T<:Real
    contour = feast_gcontour(Emid, r, fpm)
    feast_grcipevx!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info,
                    contour.Zne, contour.Wne)
end

function feast_srcipevx!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                         work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                         Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                         fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                         Emid::Complex{T}, r::Real, M0::Int,
                         lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                         mode::Ref{Int}, res::Vector{T}, info::Ref{Int},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    contour_nodes = Complex{T}.(Zne)
    contour_weights = Complex{T}.(Wne)
    _feast_poly_grci!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                      Emid, T(r), M0, lambda, q, mode, res, info,
                      contour_nodes, contour_weights)
end

function feast_srcipev!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                        work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                        Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                        fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                        Emid::Complex{T}, r::Real, M0::Int,
                        lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                        mode::Ref{Int}, res::Vector{T}, info::Ref{Int}) where T<:Real
    contour = feast_gcontour(Emid, T(r), fpm)
    feast_srcipevx!(ijob, dmax, N, Ze, work, workc, Aq, Bq, fpm, epsout, loop,
                    Emid, r, M0, lambda, q, mode, res, info,
                    contour.Zne, contour.Wne)
end

function _feast_poly_grci!(ijob::Ref{Int}, dmax::Int, N::Int, Ze::Ref{Complex{T}},
                           work::Matrix{Complex{T}}, workc::Matrix{Complex{T}},
                           Aq::Matrix{Complex{T}}, Bq::Matrix{Complex{T}},
                           fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                           Emid::Complex{T}, r::T, M0::Int,
                           lambda::Vector{Complex{T}}, q::Matrix{Complex{T}},
                           mode::Ref{Int}, res::Vector{T}, info::Ref{Int},
                           Zne::Vector{Complex{T}}, Wne::Vector{Complex{T}}) where T<:Real

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
            return
        end

        if N <= 0
            info[] = Int(Feast_ERROR_N)
            return
        end

        if M0 <= 0
            info[] = Int(Feast_ERROR_M0)
            return
        end

        if r <= zero(T)
            info[] = Int(Feast_ERROR_EMID_R)
            return
        end

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(Zne)
        fpm[52] = 0
        fpm[53] = 1

        fill!(Aq, zero(Complex{T}))
        fill!(Bq, zero(Complex{T}))
        fill!(lambda, zero(Complex{T}))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))

        if fpm[5] == 1
            for j in 1:M0
                normval = norm(work[:, j])
                if normval > 0
                    work[:, j] ./= normval
                else
                    for i in 1:N
                        work[i, j] = Complex{T}(randn(T), randn(T))
                    end
                    work[:, j] ./= norm(work[:, j])
                end
            end
        else
            for j in 1:M0
                for i in 1:N
                    work[i, j] = Complex{T}(randn(T), randn(T))
                end
                work[:, j] ./= norm(work[:, j])
            end
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

        temp = work[:, 1:M0]' * workc[:, 1:M0]
        Aq .+= Wne[e] * temp
        Bq .+= Wne[e] * Zne[e] * temp

        fpm[50] = e + 1  # Store incremented counter in fpm
        if e < ne
            Ze[] = Zne[e + 1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end

        fpm[50] = 1  # Reset for next refinement loop
        try
            F = eigen(Aq[1:M0, 1:M0], Bq[1:M0, 1:M0])
            lambda_red = F.values
            v_red = F.vectors

            M = 0
            for i in 1:M0
                if feast_inside_gcontour(lambda_red[i], Emid, r)
                    M += 1
                    lambda[M] = lambda_red[i]
                    q[:, M] .= work[:, 1:M0] * v_red[:, i]
                    q_norm = norm(q[:, M])
                    if q_norm > 0
                        q[:, M] ./= q_norm
                    end
                end
            end

            if M == 0
                info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0  # Clear initialization flag
                return
            end

            fpm[52] = M  # Store M in fpm
            mode[] = M
            ijob[] = Int(Feast_RCI_MULT_A)
            return
        catch err
            info[] = Int(Feast_ERROR_LAPACK)
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            return
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        M = fpm[52]  # Get M from fpm
        max_res = zero(T)
        for j in 1:M
            residual = view(workc, :, j) .- lambda[j] .* view(q, :, j)
            # Relative residual: normalize by max(|λ|, 1)
            res[j] = norm(residual) / max(abs(lambda[j]), one(T))
            max_res = max(max_res, res[j])
        end
        epsout[] = max_res

        eps_tolerance = feast_tolerance(fpm)
        maxloop = max(1, fpm[4])

        if epsout[] <= eps_tolerance || loop[] >= maxloop
            feast_sort_general!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            return
        else
            loop[] += 1
            fill!(Aq, zero(Complex{T}))
            fill!(Bq, zero(Complex{T}))
            work[:, 1:M] .= q[:, 1:M]
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
