# Feast auxiliary routines
# Translated from feast_aux.f90

using Printf

function check_feast_srci_input(N::Int, M0::Int, Emin::T, Emax::T, 
                               fpm::Vector{Int}) where T<:Real
    # Check validity of RCI Feast inputs for symmetric/Hermitian problems
    
    if N <= 0
        throw(ArgumentError("Matrix size N must be positive"))
    end
    
    if M0 <= 0 || M0 > N
        throw(ArgumentError("Number of eigenvalues M0 must be between 1 and N"))
    end
    
    if Emin >= Emax
        throw(ArgumentError("Search interval [Emin, Emax] must be valid"))
    end
    
    if length(fpm) < 64
        throw(ArgumentError("fpm array must have at least 64 elements"))
    end
    
    # If user provided a positive number of integration points, validate it.
    # Allow 0 to defer to default handling elsewhere.
    if length(fpm) >= 2
        if fpm[2] > 0 && fpm[2] < 3
            throw(ArgumentError("Number of integration points must be at least 3"))
        end
    end
    
    return true
end

function check_feast_grci_input(N::Int, M0::Int, Emid::Complex{T}, r::T,
                               fpm::Vector{Int}) where T<:Real
    # Check validity of RCI Feast inputs for general problems
    
    if N <= 0
        throw(ArgumentError("Matrix size N must be positive"))
    end
    
    if M0 <= 0 || M0 > N
        throw(ArgumentError("Number of eigenvalues M0 must be between 1 and N"))
    end
    
    if r <= 0
        throw(ArgumentError("Contour radius must be positive"))
    end
    
    if length(fpm) < 64
        throw(ArgumentError("fpm array must have at least 64 elements"))
    end
    
    if fpm[2] < 3
        throw(ArgumentError("Number of integration points must be at least 3"))
    end
    
    return true
end

function feast_inside_contour_old(lambda::T, Emin::T, Emax::T, 
                                 fpm::Vector{Int}) where T<:Real
    # Legacy version of eigenvalue location check
    return feast_inside_contour(lambda, Emin, Emax)
end

function feast_inside_contourx(lambda::Complex{T}, Zne::Vector{Complex{T}}, 
                              Wne::Vector{Complex{T}}) where T<:Real
    # Check if eigenvalue is inside custom contour using winding number
    ne = length(Zne)
    
    # Compute winding number
    winding = zero(Complex{T})
    for i in 1:ne
        winding += Wne[i] / (Zne[i] - lambda)
    end
    
    # Eigenvalue is inside if winding number ≈ 2πi
    return abs(real(winding / (2π * im)) - 1.0) < 0.1
end

function feast_bary_coef(lambda::Vector{T}, Zne::Vector{Complex{T}}) where T<:Real
    # Compute barycentric coordinates (legacy function)
    M = length(lambda)
    ne = length(Zne)
    
    coef = Matrix{Complex{T}}(undef, M, ne)
    
    for j in 1:M
        for i in 1:ne
            coef[j, i] = 1.0 / (Zne[i] - lambda[j])
        end
        # Normalize
        coef[j, :] ./= sum(coef[j, :])
    end
    
    return coef
end

function feast_info_symmetric(fpm::Vector{Int}, N::Int, M0::Int, M::Int,
                             Emin::T, Emax::T, loop::Int, epsout::T,
                             info::Int) where T<:Real
    # Print Feast information for symmetric/Hermitian problems
    
    if fpm[1] == 0  # No output
        return
    end
    
    println("Feast Eigenvalue Solver - Symmetric/Hermitian")
    println("="^50)
    println("Matrix size (N): ", N)
    println("Search subspace size (M0): ", M0)
    println("Eigenvalues found (M): ", M)
    println("Search interval: [", Emin, ", ", Emax, "]")
    println("Integration points: ", fpm[2])
    println("Refinement loops: ", loop)
    println("Final residual: ", epsout)
    println("Exit code (info): ", info)
    
    if info == 0
        println("✓ Feast converged successfully")
    elseif info == 1
        println("⚠ Invalid matrix size")
    elseif info == 2
        println("⚠ Invalid subspace size")
    elseif info == 3
        println("⚠ Invalid search interval")
    elseif info == 5
        println("⚠ Feast did not converge")
    else
        println("⚠ Feast terminated with error code: ", info)
    end
    
    println("="^50)
end

function feast_info_general(fpm::Vector{Int}, N::Int, M0::Int, M::Int,
                           Emid::Complex{T}, r::T, loop::Int, epsout::T,
                           info::Int) where T<:Real
    # Print Feast information for general problems
    
    if fpm[1] == 0  # No output
        return
    end
    
    println("Feast Eigenvalue Solver - General")
    println("="^50)
    println("Matrix size (N): ", N)
    println("Search subspace size (M0): ", M0)
    println("Eigenvalues found (M): ", M)
    println("Search contour center: ", Emid)
    println("Search contour radius: ", r)
    println("Integration points: ", fpm[2])
    println("Refinement loops: ", loop)
    println("Final residual: ", epsout)
    println("Exit code (info): ", info)
    
    if info == 0
        println("✓ Feast converged successfully")
    else
        println("⚠ Feast terminated with error code: ", info)
    end
    
    println("="^50)
end

# Additional helper functions for contour generation and validation

function validate_contour(Zne::Vector{Complex{T}}, Wne::Vector{Complex{T}}) where T<:Real
    # Validate integration contour
    
    ne = length(Zne)
    if length(Wne) != ne
        throw(ArgumentError("Number of nodes and weights must match"))
    end
    
    if ne < 3
        throw(ArgumentError("Contour must have at least 3 points"))
    end
    
    # Check for repeated nodes
    for i in 1:ne
        for j in i+1:ne
            if abs(Zne[i] - Zne[j]) < 1e-14
                @warn "Contour has nearly identical nodes at positions $i and $j"
            end
        end
    end
    
    return true
end

function feast_trace_eigenvalues(lambda::Vector{T}, q::Matrix{VT}, 
                                res::Vector{T}, M::Int, 
                                Emin::T, Emax::T) where {T<:Real, VT}
    # Trace eigenvalue convergence information
    
    println("Eigenvalues in search interval [", Emin, ", ", Emax, "]:")
    println("-"^60)
    println(@sprintf("%-5s %-15s %-15s", "No.", "Eigenvalue", "Residual"))
    println("-"^60)
    
    for i in 1:M
        if VT <: Real
            println(@sprintf("%-5d %-15.8e %-15.8e", i, lambda[i], res[i]))
        else
            println(@sprintf("%-5d %-15.8e %-15.8e", i, real(lambda[i]), res[i]))
        end
    end
    
    println("-"^60)
end

function feast_memory_estimate(N::Int, M0::Int, precision::Type{T}) where T<:Real
    # Estimate memory requirements for Feast
    
    # Main workspace arrays
    work_size = N * M0 * sizeof(T)
    workc_size = N * M0 * sizeof(Complex{T})
    reduced_size = 2 * M0 * M0 * sizeof(T)  # Aq and Sq
    eigen_size = (N * M0 + 2 * M0) * sizeof(T)  # eigenvectors and eigenvalues
    
    total_size = work_size + workc_size + reduced_size + eigen_size
    
    println("Feast Memory Estimate:")
    println("  Workspace (real): ", @sprintf("%.2f MB", work_size / 1024^2))
    println("  Workspace (complex): ", @sprintf("%.2f MB", workc_size / 1024^2))
    println("  Reduced matrices: ", @sprintf("%.2f MB", reduced_size / 1024^2))
    println("  Eigendata: ", @sprintf("%.2f MB", eigen_size / 1024^2))
    println("  Total estimate: ", @sprintf("%.2f MB", total_size / 1024^2))
    
    return total_size
end
