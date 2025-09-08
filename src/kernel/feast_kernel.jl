# Feast kernel routines - Reverse Communication Interface (RCI)
# Translated from dzfeast.f90

using Random

function feast_srci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}}, 
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     Aq::Matrix{T}, Sq::Matrix{T}, fpm::Vector{Int},
                     epsout::Ref{T}, loop::Ref{Int}, 
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{T}, mode::Ref{Int},
                     res::Vector{T}, info::Ref{Int}) where T<:Real
    
    # Feast RCI for real symmetric eigenvalue problems
    # Solves: A*q = lambda*B*q where A is symmetric, B is symmetric positive definite
    
    # Static variables (persistent across calls)
    @static if !@isdefined(_feast_srci_state)
        global _feast_srci_state = Dict{Symbol, Any}()
    end
    
    state = _feast_srci_state
    
    if ijob[] == -1  # Initialization
        # Initialize Feast parameters
        feastdefault!(fpm)
        
        # Check input parameters
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
        
        # Initialize state variables
        state[:ne] = fpm[2]  # Number of integration points
        state[:eps] = feast_tolerance(fpm)
        state[:maxloop] = fpm[4]
        
        # Generate integration contour
        contour = feast_contour(Emin, Emax, fpm)
        state[:Zne] = contour.Zne
        state[:Wne] = contour.Wne
        
        # Initialize counters
        state[:e] = 1  # Current integration point
        loop[] = 0
        state[:M] = 0  # Number of eigenvalues found
        
        # Initialize workspace
        fill!(work, zero(T))
        fill!(workc, zero(Complex{T}))
        fill!(Aq, zero(T))
        fill!(Sq, zero(T))
        fill!(lambda, zero(T))
        fill!(q, zero(T))
        fill!(res, zero(T))
        
        # Set first integration point
        Ze[] = state[:Zne][1]
        
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end
    
    # Main Feast iteration loop
    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*B - A)
        ijob[] = Int(Feast_RCI_SOLVE)
        return
    end
    
    if ijob[] == Int(Feast_RCI_SOLVE)
        # User has solved linear systems
        e = state[:e]
        ne = state[:ne]
        
        # Accumulate moment matrices
        Zne = state[:Zne]
        Wne = state[:Wne]
        
        # Update reduced matrices Aq and Sq
        for j in 1:M0
            for i in 1:M0
                Aq[i,j] += real(Wne[e] * workc[i,j])
                Sq[i,j] += real(Wne[e] * workc[i,j] * Zne[e])
            end
        end
        
        # Move to next integration point
        state[:e] = e + 1
        
        if e < ne
            # More integration points to process
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            # All integration points processed, solve reduced eigenvalue problem
            state[:e] = 1  # Reset for next refinement loop
            
            # Solve generalized eigenvalue problem: Aq*v = lambda*Sq*v
            try
                F = eigen(Aq[1:M0, 1:M0], Sq[1:M0, 1:M0])
                lambda_red = real.(F.values)
                v_red = real.(F.vectors)
                
                # Count eigenvalues in interval [Emin, Emax]
                M = 0
                for i in 1:M0
                    if feast_inside_contour(lambda_red[i], Emin, Emax)
                        M += 1
                        lambda[M] = lambda_red[i]
                        # Compute eigenvectors: q = work * v_red
                        q[:, M] = work[:, 1:M0] * v_red[:, i]
                    end
                end
                
                state[:M] = M
                
                # Check convergence
                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    return
                end
                
                # Compute residuals
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = 1  # Compute A*q
                return
                
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                return
            end
        end
    end
    
    if ijob[] == Int(Feast_RCI_MULT_A)
        # User has computed A*q, now compute residuals
        M = state[:M]
        
        for j in 1:M
            # Residual: r = A*q - lambda*q (assuming B = I)
            res[j] = norm(work[:, j] - lambda[j] * q[:, j])
        end
        
        # Check convergence
        epsout[] = maximum(res[1:M])
        
        if epsout[] <= state[:eps] || loop[] >= state[:maxloop]
            # Converged or maximum iterations reached
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
        else
            # Start new refinement loop
            loop[] += 1
            
            # Reset for next iteration
            fill!(Aq, zero(T))
            fill!(Sq, zero(T))
            
            # Use current eigenvectors as initial guess
            work[:, 1:M] = q[:, 1:M]
            
            Ze[] = state[:Zne][1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
        end
    end
end

function feast_hrci!(ijob::Ref{Int}, N::Int, Ze::Ref{Complex{T}},
                     work::Matrix{T}, workc::Matrix{Complex{T}},
                     zAq::Matrix{Complex{T}}, zSq::Matrix{Complex{T}}, 
                     fpm::Vector{Int}, epsout::Ref{T}, loop::Ref{Int},
                     Emin::T, Emax::T, M0::Int,
                     lambda::Vector{T}, q::Matrix{Complex{T}}, 
                     mode::Ref{Int}, res::Vector{T}, info::Ref{Int}) where T<:Real
    
    # Feast RCI for complex Hermitian eigenvalue problems
    # Similar structure to feast_srci! but for complex matrices
    
    @static if !@isdefined(_feast_hrci_state)
        global _feast_hrci_state = Dict{Symbol, Any}()
    end
    
    state = _feast_hrci_state
    
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
        
        state[:ne] = fpm[2]
        state[:eps] = feast_tolerance(fpm)
        state[:maxloop] = fpm[4]
        
        contour = feast_contour(Emin, Emax, fpm)
        state[:Zne] = contour.Zne
        state[:Wne] = contour.Wne
        
        state[:e] = 1
        loop[] = 0
        state[:M] = 0
        
        fill!(work, zero(T))
        fill!(workc, zero(Complex{T}))
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(lambda, zero(T))
        fill!(q, zero(Complex{T}))
        fill!(res, zero(T))
        
        Ze[] = state[:Zne][1]
        ijob[] = Int(Feast_RCI_FACTORIZE)
        return
    end
    
    # Main Feast iteration loop for complex Hermitian matrices
    if ijob[] == Int(Feast_RCI_FACTORIZE)
        # User should factorize (Ze*I - A) where A is Hermitian
        ijob[] = Int(Feast_RCI_SOLVE)
        return
    end
    
    if ijob[] == Int(Feast_RCI_SOLVE)
        # User has solved linear systems
        e = state[:e]
        ne = state[:ne]
        
        # Accumulate moment matrices (complex version)
        Zne = state[:Zne]
        Wne = state[:Wne]
        
        # Update reduced matrices zAq and zSq (complex accumulation)
        for j in 1:M0
            for i in 1:M0
                zAq[i,j] += Wne[e] * workc[i,j]
                zSq[i,j] += Wne[e] * workc[i,j] * Zne[e]
            end
        end
        
        # Move to next integration point
        state[:e] = e + 1
        
        if e < ne
            # More integration points to process
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            # All integration points processed, solve reduced eigenvalue problem
            state[:e] = 1  # Reset for next refinement loop
            
            # Solve generalized eigenvalue problem: zAq*v = lambda*zSq*v
            try
                F = eigen(zAq[1:M0, 1:M0], zSq[1:M0, 1:M0])
                lambda_red = real.(F.values)
                v_red = F.vectors
                
                # Count eigenvalues in interval [Emin, Emax]
                M = 0
                for i in 1:M0
                    if feast_inside_contour(lambda_red[i], Emin, Emax)
                        M += 1
                        lambda[M] = lambda_red[i]
                        # Compute eigenvectors: q = workc * v_red (complex)
                        for k in 1:N
                            q[k, M] = zero(Complex{T})
                            for j in 1:M0
                                q[k, M] += workc[k, j] * v_red[j, i]
                            end
                        end
                    end
                end
                
                state[:M] = M
                
                # Check if any eigenvalues found
                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    return
                end
                
                # Compute residuals - need A*q
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M  # Number of eigenvectors to multiply
                return
                
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                return
            end
        end
    end
    
    if ijob[] == Int(Feast_RCI_MULT_A)
        # User has computed A*q, now compute residuals
        M = state[:M]
        
        for j in 1:M
            # Residual: r = A*q - lambda*q (for Hermitian A)
            # For Hermitian matrices, the user should store A*q in workc
            # since A is complex Hermitian
            residual = zeros(Complex{T}, N)
            for i in 1:N
                # workc contains A*q (complex result from Hermitian A)
                residual[i] = workc[i, j] - lambda[j] * q[i, j]
            end
            res[j] = norm(residual)
        end
        
        # Check convergence
        epsout[] = maximum(res[1:M])
        
        if epsout[] <= state[:eps] || loop[] >= state[:maxloop]
            # Converged or maximum iterations reached
            feast_sort!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
        else
            # Start new refinement loop
            loop[] += 1
            
            # Reset for next iteration
            fill!(zAq, zero(Complex{T}))
            fill!(zSq, zero(Complex{T}))
            
            # Use current eigenvectors as initial guess
            workc[:, 1:M] = q[:, 1:M]
            
            Ze[] = state[:Zne][1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
        end
    end
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
    @static if !@isdefined(_feast_grci_state)
        global _feast_grci_state = Dict{Symbol, Any}()
    end
    
    state = _feast_grci_state
    
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
        
        state[:ne] = fpm[2]
        state[:eps] = feast_tolerance(fpm)
        state[:maxloop] = fpm[4]
        
        contour = feast_gcontour(Emid, r, fpm)
        state[:Zne] = contour.Zne
        state[:Wne] = contour.Wne
        
        state[:e] = 1
        loop[] = 0
        state[:M] = 0
        
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
        
        Ze[] = state[:Zne][1]
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
        e = state[:e]
        ne = state[:ne]
        
        # Accumulate moment matrices (complex version for general case)
        Zne = state[:Zne]
        Wne = state[:Wne]
        
        # Update reduced matrices Aq and Sq (complex accumulation)
        for j in 1:M0
            for i in 1:M0
                Aq[i,j] += Wne[e] * workc[i,j]
                Sq[i,j] += Wne[e] * workc[i,j] * Zne[e]
            end
        end
        
        # Move to next integration point
        state[:e] = e + 1
        
        if e < ne
            # More integration points to process
            Ze[] = Zne[e+1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
            return
        else
            # All integration points processed, solve reduced eigenvalue problem
            state[:e] = 1  # Reset for next refinement loop
            
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
                
                state[:M] = M
                
                # Check if any eigenvalues found
                if M == 0
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    ijob[] = Int(Feast_RCI_DONE)
                    return
                end
                
                # Compute residuals - need A*q
                ijob[] = Int(Feast_RCI_MULT_A)
                mode[] = M  # Number of eigenvectors to multiply
                return
                
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                ijob[] = Int(Feast_RCI_DONE)
                return
            end
        end
    end
    
    if ijob[] == Int(Feast_RCI_MULT_A)
        # User has computed A*q, now compute residuals
        M = state[:M]
        
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
        
        if epsout[] <= state[:eps] || loop[] >= state[:maxloop]
            # Converged or maximum iterations reached
            # Sort by eigenvalue magnitude (for complex eigenvalues)
            feast_sort_general!(lambda, q, res, M)
            mode[] = M
            ijob[] = Int(Feast_RCI_DONE)
        else
            # Start new refinement loop
            loop[] += 1
            
            # Reset for next iteration
            fill!(Aq, zero(Complex{T}))
            fill!(Sq, zero(Complex{T}))
            
            # Use current eigenvectors as initial guess
            workc[:, 1:M] = q[:, 1:M]
            
            Ze[] = state[:Zne][1]
            ijob[] = Int(Feast_RCI_FACTORIZE)
        end
    end
end