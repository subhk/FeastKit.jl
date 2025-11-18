@static if !@isdefined(_feast_srci_state)
    global _feast_srci_state = Dict{UInt64, Dict{Symbol, Any}}()
end

function feast_srci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                     epsout::Ref{T}, loop::Ref{Int},
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                     res::Vector{T}, info::Ref{Int}) where T<:Real

    # Use fpm slots 50-64 for internal state storage
    # fpm[50] = current integration point e
    # fpm[51] = total integration points ne
    # fpm[52] = stored M value
    # fpm[53] = initialization flag (1 = initialized, 0 = not initialized)

    @static if !@isdefined(_feast_srci_state)
        global _feast_srci_state = Dict{UInt64, Dict{Symbol, Any}}()
    end
    state_key = UInt64(objectid(Aq))
    cleanup_state! = () -> pop!(_feast_srci_state, state_key, nothing)

    if ijob[] == -1  # Initialization
        fpm[1] > 0 && println("[DEBUG feast_srci!] Starting initialization")
        # NOTE: feastdefault! should have already been called by the caller (e.g., feast_sygv!)
        # We should NOT call it again here as it may interfere with user-set parameters
        fpm[1] > 0 && println("[DEBUG feast_srci!] Skipping redundant feastdefault! call")

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

        fpm[1] > 0 && println("[DEBUG feast_srci!] Getting contour")
        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            fpm[1] > 0 && println("[DEBUG feast_srci!] Creating default contour for Emin=$Emin, Emax=$Emax, fpm[2]=$(fpm[2])")
            contour = feast_contour(Emin, Emax, fpm)
            fpm[1] > 0 && println("[DEBUG feast_srci!] Contour created with $(length(contour.Zne)) points")
        end

        # Store state in fpm array
        fpm[50] = 1
        fpm[51] = length(contour.Zne)
        fpm[52] = 0
        fpm[53] = 1

        loop[] = 0

        fpm[1] > 0 && println("[DEBUG feast_srci!] Filling workspace arrays")
        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        fill!(lambda, zero(T))
        fill!(q, zero(T))
        fill!(res, zero(T))
        fill!(workc, zero(Complex{T}))
        fpm[1] > 0 && println("[DEBUG feast_srci!] Workspace arrays filled")

        if fpm[5] == 1
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
            for j in 1:M0
                for i in 1:N
                    work[i, j] = randn(T)
                end
                work[:, j] ./= norm(work[:, j])
            end
        end

        Ze[] = contour.Zne[1]
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

        # Get contour points
        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end
        Zne = contour.Zne
        Wne = contour.Wne

        temp = work[:, 1:M0]' * workc[:, 1:M0]
        Aq .+= real(Wne[e] * temp)
        Sq .+= real(Wne[e] * Zne[e] * temp)

        fpm[50] = e + 1  # Store incremented counter in fpm

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            fpm[50] = 1  # Reset for next refinement loop

            try
                F = eigen(Aq[1:M0, 1:M0], Sq[1:M0, 1:M0])
                lambda_red = real.(F.values)
                v_red = real.(F.vectors)

                M = 0
                for i in 1:M0
                    if feast_inside_contour(lambda_red[i], Emin, Emax)
                        M += 1
                        lambda[M] = lambda_red[i]
                        q[:, M] = work[:, 1:M0] * v_red[:, i]
                    end
                end

                fpm[52] = M  # Store M in fpm

                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0  # Clear initialization flag
                    cleanup_state!()
                    return
                end

                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                return
            catch err
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0  # Clear initialization flag
                cleanup_state!()
                return
            end
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        M = fpm[52]  # Get M from fpm

        for j in 1:M
            res[j] = norm(work[:, j] - lambda[j] * q[:, j])
        end
        epsout[] = maximum(res[1:M])

        eps_tolerance = feast_tolerance(fpm)
        maxloop = fpm[4]

        if epsout[] <= eps_tolerance || loop[] >= maxloop
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            cleanup_state!()
            return
        else
            loop[] += 1
            fill!(Aq, zero(T))
            fill!(Sq, zero(T))
            work[:, 1:M] = q[:, 1:M]

            # Get contour for next refinement loop
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_contour(Emin, Emax, fpm)
            end
            fpm[50] = 1  # Reset integration point counter

            Ze[] = contour.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    if ijob[] == Int(Feast_RCI_DONE)
        cleanup_state!()
        return
    end

    cleanup_state!()
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
@static if !@isdefined(_feast_hrci_state)
    global _feast_hrci_state = Dict{UInt64, Dict{Symbol, Any}}()
end
@static if !@isdefined(_feast_hrci_state)
    global _feast_hrci_state = Dict{UInt64, Dict{Symbol, Any}}()
end

function feast_hrci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}},
                     fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{Complex{T}},
                     mode::Ref{Int}, res::Vector{T}, info::Ref{Int}) where T<:Real
    state_key = UInt64(objectid(workc))
    state = get!(() -> Dict{Symbol, Any}(), _feast_hrci_state, state_key)
    cleanup_state! = () -> pop!(_feast_hrci_state, state_key, nothing)

    if ijob[] == -1
        # NOTE: feastdefault! should have already been called by the caller
        # We should NOT call it again here as it may interfere with user-set parameters
        empty!(state)

        info[] = Int(Feast_SUCCESS)
        if N <= 0
            info[] = Int(Feast_ERROR_N)
            state[:cleanup_state]()
            return
        end
        if M0 <= 0 || M0 > N
            info[] = Int(Feast_ERROR_M0)
            state[:cleanup_state]()
            return
        end
        if Emin >= Emax
            info[] = Int(Feast_ERROR_EMIN_EMAX)
            state[:cleanup_state]()
            return
        end

        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            contour = feast_contour(Emin, Emax, fpm)
        end

        state[:Zne] = copy(contour.Zne)
        state[:Wne] = copy(contour.Wne)
        state[:ne] = length(contour.Zne)
        state[:eps] = feast_tolerance(fpm)
        state[:maxloop] = fpm[4]
        state[:e] = 1
        state[:M] = 0

        loop[] = 0

        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(lambda, zero(T))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))
        fill!(work, zero(T))

        if fpm[5] == 1
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
            for j in 1:M0
                for i in 1:N
                    workc[i, j] = Complex{T}(randn(T), randn(T))
                end
                workc[:, j] ./= norm(workc[:, j])
            end
        end

        Ze[] = state[:Zne][1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end

    if ijob[] == Int(Feast_RCI_FACTORIZE)
        ijob[] = Int(Feast_RCI_SOLVE)
        return
    end

    if ijob[] == Int(Feast_RCI_SOLVE)
        if !haskey(state, :Zne)
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_contour(Emin, Emax, fpm)
            end
            state[:Zne] = copy(contour.Zne)
            state[:Wne] = copy(contour.Wne)
            state[:ne] = length(contour.Zne)
            state[:e] = 1
        end

        e = get(state, :e, 1)
        ne = state[:ne]
        Zne = state[:Zne]
        Wne = state[:Wne]

        temp = workc[:, 1:M0]' * workc[:, 1:M0]
        zAq .+= Wne[e] * temp
        zSq .+= Wne[e] * Zne[e] * temp

        state[:e] = e + 1

        if e < ne
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            state[:e] = 1
            try
                F = eigen(zAq[1:M0, 1:M0], zSq[1:M0, 1:M0])
                lambda_red = real.(F.values)
                v_red = F.vectors

                M = 0
                for i in 1:M0
                    if feast_inside_contour(lambda_red[i], Emin, Emax)
                        M += 1
                        lambda[M] = lambda_red[i]
                        q[:, M] = workc[:, 1:M0] * v_red[:, i]
                    end
                end

                state[:M] = M

                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    state[:cleanup_state]()
                    return
                end

                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M
                return
            catch err
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                state[:cleanup_state]()
                return
            end
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        M = get(state, :M, 0)
        for j in 1:M
            res[j] = norm(workc[:, j] - lambda[j] * q[:, j])
        end
        epsout[] = maximum(res[1:M])
        eps = state[:eps]
        maxloop = state[:maxloop]

        if epsout[] <= eps || loop[] >= maxloop
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            state[:cleanup_state]()
            return
        else
            loop[] += 1
            fill!(zAq, zero(Complex{T}))
            fill!(zSq, zero(Complex{T}))
            workc[:, 1:M] = q[:, 1:M]
            Ze[] = state[:Zne][1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    if ijob[] == Int(Feast_RCI_DONE)
        state[:cleanup_state]()
        return
    end

    state[:cleanup_state]()
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
                     mode::Ref{Int}, res::Vector{T}, info::Ref{Int}) where T<:Real
    
    # Feast RCI for general (non-Hermitian) eigenvalue problems
    # Uses circular contour in complex plane

    # Use fpm slots 50-64 for internal state storage
    # fpm[50] = current integration point e
    # fpm[51] = total integration points ne
    # fpm[52] = stored M value
    # fpm[53] = initialization flag (1 = initialized, 0 = not initialized)

    @static if !@isdefined(_feast_grci_state)
        global _feast_grci_state = Dict{UInt64, Dict{Symbol, Any}}()
    end
    state_key = UInt64(objectid(Aq))
    cleanup_state! = () -> pop!(_feast_grci_state, state_key, nothing)

    if ijob[] == -1  # Initialization
        # NOTE: feastdefault! should have already been called by the caller
        # We should NOT call it again here as it may interfere with user-set parameters

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
        # Check if user provides initial guess (fpm[5] = 1) or use random vectors
        if fpm[5] == 1
            # User should have provided initial guess in workc
            # Normalize the columns to ensure numerical stability
            for j in 1:M0
                if norm(workc[:, j]) > 0
                    workc[:, j] ./= norm(workc[:, j])
                else
                    # If zero vector provided, use random
                    for i in 1:N
                        workc[i, j] = Complex{T}(randn(T), randn(T))
                    end
                    workc[:, j] ./= norm(workc[:, j])
                end
            end
        else
            # Generate random initial subspace
            for j in 1:M0
                for i in 1:N
                    workc[i, j] = Complex{T}(randn(T), randn(T))
                end
                # Normalize each column
                workc[:, j] ./= norm(workc[:, j])
            end
        end
        
        # work is used for real intermediate results
        fill!(work, zero(T))

        Ze[] = contour.Zne[1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end
    
    # Main Feast iteration loop for general (non-Hermitian) eigenvalue problems
    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*B - A) for general matrices
        ijob[] = Int(Feast_RCI_SOLVE)
        return
    end
    
    if ijob[] == Int(Feast_RCI_SOLVE)
        # User has solved linear systems
        e = fpm[50]  # Get current integration point from fpm
        ne = fpm[51]  # Get total integration points from fpm

        # Get contour points
        contour = feast_get_custom_contour(fpm)
        if contour === nothing
            contour = feast_gcontour(Emid, r, fpm)
        end
        Zne = contour.Zne
        Wne = contour.Wne

        # Update reduced matrices Aq and Sq (complex accumulation)
        # Compute: Aq += w_e * workc^H * workc, Sq += w_e * z_e * workc^H * workc
        temp = workc[:, 1:M0]' * workc[:, 1:M0]  # M0 x M0 matrix
        Aq .+= Wne[e] * temp
        Sq .+= Wne[e] * Zne[e] * temp

        # Move to next integration point
        fpm[50] = e + 1  # Store incremented counter in fpm

        if e < ne
            # More integration points to process
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            # All integration points processed, solve reduced eigenvalue problem
            fpm[50] = 1  # Reset for next refinement loop
            
            # Solve generalized eigenvalue problem: Aq*v = lambda*Sq*v (complex case)
            try
                F = eigen(Aq[1:M0, 1:M0], Sq[1:M0, 1:M0])
                lambda_red = F.values  # Keep complex eigenvalues
                v_red = F.vectors
                
                # Count eigenvalues in circular region |lambda - Emid| <= r
                M = 0
                for i in 1:M0
                    if feast_inside_gcontour(lambda_red[i], Emid, r)
                        M += 1
                        lambda[M] = lambda_red[i]
                        # Compute eigenvectors: q = workc * v_red (complex matrix-vector product)
                        for k in 1:N
                            q[k, M] = zero(Complex{T})
                            for j in 1:M0
                                q[k, M] += workc[k, j] * v_red[j, i]
                            end
                        end
                        # Normalize the computed eigenvector
                        q_norm = norm(q[:, M])
                        if q_norm > 0
                            q[:, M] ./= q_norm
                        end
                    end
                end

                fpm[52] = M  # Store M in fpm

                # Check if any eigenvalues found
                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    fpm[53] = 0  # Clear initialization flag
                    cleanup_state!()
                    return
                end

                # Compute residuals - need A*q
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M  # Number of eigenvectors to multiply
                return

            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                fpm[53] = 0  # Clear initialization flag
                cleanup_state!()
                return
            end
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        # User has computed A*q, now compute residuals
        M = fpm[52]  # Get M from fpm

        for j in 1:M
            # Residual: r = A*q - lambda*B*q
            # For general eigenvalue problems, we assume B = I for simplicity
            # workc contains A*q (complex result from user computation)
            residual = zeros(Complex{T}, N)
            for i in 1:N
                # workc[i,j] contains (A*q)[i] for j-th eigenvector
                residual[i] = workc[i, j] - lambda[j] * q[i, j]
            end
            res[j] = norm(residual)
        end

        # Check convergence
        epsout[] = maximum(res[1:M])

        eps_tolerance = feast_tolerance(fpm)
        maxloop = fpm[4]

        if epsout[] <= eps_tolerance || loop[] >= maxloop
            # Converged or maximum iterations reached
            # Sort by eigenvalue magnitude (for complex eigenvalues)
            feast_sort_general!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
            fpm[53] = 0  # Clear initialization flag
            cleanup_state!()
            return
        else
            # Start new refinement loop
            loop[] += 1

            # Reset for next iteration
            fill!(Aq, zero(Complex{T}))
            fill!(Sq, zero(Complex{T}))

            # Use current eigenvectors as initial guess
            workc[:, 1:M] = q[:, 1:M]

            # Get contour for next refinement loop
            contour = feast_get_custom_contour(fpm)
            if contour === nothing
                contour = feast_gcontour(Emid, r, fpm)
            end
            fpm[50] = 1  # Reset integration point counter

            Ze[] = contour.Zne[1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        end
    end

    # Safety check: if we reach here, ijob has an invalid value
    if ijob[] != -1 && ijob[] != Int(Feast_RCI_FACTORIZE) &&
       ijob[] != Int(Feast_RCI_SOLVE) && ijob[] != Int(Feast_RCI_MULT_A) &&
       ijob[] != Int(Feast_RCI_DONE)
        cleanup_state!()
       error("FEAST RCI kernel (General): Invalid job code ijob=$(ijob[]). " *
             "Expected: -1 (init), $(Int(Feast_RCI_FACTORIZE)) (factorize), " *
             "$(Int(Feast_RCI_SOLVE)) (solve), $(Int(Feast_RCI_MULT_A)) (mult_a), " *
             "or $(Int(Feast_RCI_DONE)) (done)")
    end
end

############################
# Polynomial RCI interfaces
############################

@static if !@isdefined(_feast_poly_rci_state)
    global _feast_poly_rci_state = Dict{UInt64, Dict{Symbol, Any}}()
end

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

    @static if !@isdefined(_feast_poly_rci_state)
        global _feast_poly_rci_state = Dict{UInt64, Dict{Symbol, Any}}()
    end
    state_key = UInt64(objectid(Aq))
    cleanup_state! = () -> pop!(_feast_poly_rci_state, state_key, nothing)

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
                cleanup_state!()
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
            cleanup_state!()
            return
        end
    end

    if ijob[] == Int(Feast_RCI_MULT_A)
        M = fpm[52]  # Get M from fpm
        max_res = zero(T)
        for j in 1:M
            residual = view(workc, :, j) .- lambda[j] .* view(q, :, j)
            res[j] = norm(residual)
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
            cleanup_state!()
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
        cleanup_state!()
        error("FEAST polynomial RCI kernel: unexpected ijob=$(ijob[]).")
    end
end
