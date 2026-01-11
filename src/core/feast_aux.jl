# Feast auxiliary routines
# Translated from feast_aux.f90

using Printf

const FEAST_CUSTOM_CONTOURS = IdDict{Vector{Int}, FeastContour}()

function _copy_contour(contour::FeastContour{T}) where T<:Real
    return FeastContour{T}(copy(contour.Zne), copy(contour.Wne))
end

function feast_distribution_type(N::Int,
                                 isa::AbstractVector{<:Integer},
                                 jsa::AbstractVector{<:Integer};
                                 comm::Any = nothing)
    if length(isa) == N + 1 && !isempty(jsa)
        return :csr
    else
        return :unknown
    end
end

function feast_set_custom_contour!(fpm::Vector{Int}, contour::FeastContour{T}) where T<:Real
    validate_contour(contour.Zne, contour.Wne)
    FEAST_CUSTOM_CONTOURS[fpm] = _copy_contour(contour)
    fpm[15] = 1
    fpm[2] = length(contour.Zne)
    return FEAST_CUSTOM_CONTOURS[fpm]
end

function feast_set_custom_contour!(fpm::Vector{Int},
                                   Zne::AbstractVector{Complex{T1}},
                                   Wne::AbstractVector{Complex{T2}}) where {T1<:Real, T2<:Real}
    base = promote_type(T1, T2)
    Zvec = Vector{Complex{base}}(Zne)
    Wvec = Vector{Complex{base}}(Wne)
    contour = FeastContour{base}(Zvec, Wvec)
    return feast_set_custom_contour!(fpm, contour)
end

function feast_get_custom_contour(fpm::Vector{Int})
    return get(FEAST_CUSTOM_CONTOURS, fpm, nothing)
end

function feast_clear_custom_contour!(fpm::Vector{Int})
    pop!(FEAST_CUSTOM_CONTOURS, fpm, nothing)
    return nothing
end

function with_custom_contour(solver::Function, fpm::Vector{Int}, contour::FeastContour{T}) where T<:Real
    old_flag = fpm[15]
    old_ne = fpm[2]
    old_contour = feast_get_custom_contour(fpm)
    feast_set_custom_contour!(fpm, contour)
    try
        return solver()
    finally
        if old_contour === nothing
            feast_clear_custom_contour!(fpm)
        else
            feast_set_custom_contour!(fpm, old_contour)
        end
        fpm[15] = old_flag
        fpm[2] = old_ne
    end
end

function with_custom_contour(solver::Function,
                             fpm::Vector{Int},
                             Zne::AbstractVector{Complex{T1}},
                             Wne::AbstractVector{Complex{T2}}) where {T1<:Real, T2<:Real}
    base = promote_type(T1, T2)
    contour = FeastContour{base}(Vector{Complex{base}}(Zne),
                                 Vector{Complex{base}}(Wne))
    return with_custom_contour(solver, fpm, contour)
end

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
    # Matches Fortran dcheck_feast_grci_input (validates r, M0, N only)

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

    # Note: Fortran dcheck_feast_grci_input does not validate fpm[2] or fpm[8]
    # Integration point validation is handled by feastdefault!

    return true
end

function feast_inside_contour_old(lambda::T, Emin::T, Emax::T, 
                                 fpm::Vector{Int}) where T<:Real
    # Legacy version of eigenvalue location check
    return feast_inside_contour(lambda, Emin, Emax)
end

function feast_inside_contourx(lambda::Complex{T}, Zne::Vector{Complex{T}},
                              Wne::Vector{Complex{T}}) where T<:Real
    # Check if eigenvalue is inside custom contour using triangulation
    # Matches Fortran zfeast_inside_contourx algorithm
    # The contour points define a polygon; we check if the eigenvalue
    # is inside any triangle formed by the first point and pairs of other points.

    ne = length(Zne)

    # Check for NaN
    if isnan(real(lambda)) || isnan(imag(lambda))
        return false
    end

    # First vertex of all triangles
    x1 = real(Zne[1])
    y1 = imag(Zne[1])

    for i in 2:ne
        x2 = real(Zne[i])
        y2 = imag(Zne[i])
        z1i = (Zne[i] - Zne[1]) / abs(Zne[i] - Zne[1])

        for j in (i+1):ne
            x3 = real(Zne[j])
            y3 = imag(Zne[j])
            z1j = (Zne[j] - Zne[1]) / abs(Zne[j] - Zne[1])

            # Skip collinear points (Fortran tolerance: 1e-8)
            dot_product = real(z1i) * real(z1j) + imag(z1i) * imag(z1j)
            if abs(one(T) - abs(dot_product)) > T(1e-8)
                # Barycentric coordinates for the point in triangle (Zne[1], Zne[i], Zne[j])
                xp = real(lambda)
                yp = imag(lambda)

                denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                lambda1 = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / denom
                lambda2 = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / denom
                lambda3 = one(T) - lambda1 - lambda2

                # Point is inside triangle if all barycentric coords >= 0 (with tolerance)
                tol = T(1e-14)
                if lambda1 > -tol && lambda2 > -tol && lambda3 > -tol
                    return true
                end
            end
        end
    end

    return false
end

"""
    _feast_bary_coef_triangle(v1, v2, v3) -> Real

Compute signed area (barycentric-style coefficient) for triangle check.
Matches Fortran zfeast_bary_coef: coef = adbc1 - adbc2 + adbc3
where adbc_i = cross-product term for signed area.
"""
function _feast_bary_coef_triangle(v1::Complex{T}, v2::Complex{T}, v3::Complex{T}) where T<:Real
    adbc1 = real(v2) * imag(v3) - real(v3) * imag(v2)
    adbc2 = real(v1) * imag(v3) - real(v3) * imag(v1)
    adbc3 = real(v1) * imag(v2) - real(v2) * imag(v1)
    return adbc1 - adbc2 + adbc3
end

"""
    feast_cauchy_weights(lambda, Zne) -> Matrix{Complex}

Compute normalized Cauchy kernel weights for eigenvalue filtering.
For each eigenvalue λ_j, computes w_ji = (1/(Zne_i - λ_j)) normalized.

Note: This is NOT the same as the Fortran `zfeast_bary_coef` which computes
triangle signed areas. This function is used for eigenvalue filtering
in the rational approximation context.
"""
function feast_cauchy_weights(lambda::Vector{T}, Zne::Vector{Complex{T}}) where T<:Real
    M = length(lambda)
    ne = length(Zne)

    coef = Matrix{Complex{T}}(undef, M, ne)

    for j in 1:M
        for i in 1:ne
            coef[j, i] = one(T) / (Zne[i] - lambda[j])
        end
        # Normalize
        coef[j, :] ./= sum(coef[j, :])
    end

    return coef
end

# Legacy alias for backward compatibility
const feast_bary_coef = feast_cauchy_weights

function feast_info_symmetric(fpm::Vector{Int}, N::Int, M0::Int, M::Int,
                             Emin::T, Emax::T, loop::Int, epsout::T,
                             info::Int) where T<:Real
    # Print FeastKit information for symmetric/Hermitian problems
    
    if fpm[1] == 0  # No output
        return
    end
    
    println("FeastKit Eigenvalue Solver - Symmetric/Hermitian")
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
        println("FeastKit converged successfully")
    elseif info == 1
        println("Warning: Invalid matrix size")
    elseif info == 2
        println("Warning: Invalid subspace size")
    elseif info == 3
        println("Warning: Invalid search interval")
    elseif info == 5
        println("Warning: FeastKit did not converge")
    else
        println("Warning: FeastKit terminated with error code: ", info)
    end
    
    println("="^50)
end

function feast_info_general(fpm::Vector{Int}, N::Int, M0::Int, M::Int,
                           Emid::Complex{T}, r::T, loop::Int, epsout::T,
                           info::Int) where T<:Real
    # Print FeastKit information for general problems

    if fpm[1] == 0  # No output
        return
    end

    println("FeastKit Eigenvalue Solver - General")
    println("="^50)
    println("Matrix size (N): ", N)
    println("Search subspace size (M0): ", M0)
    println("Eigenvalues found (M): ", M)
    println("Search contour center: ", Emid)
    println("Search contour radius: ", r)
    println("Integration points: ", fpm[8], " (full contour)")  # fpm[8] for general problems
    println("Ellipse ratio (a/b): ", fpm[18] / 100.0)
    println("Rotation angle: ", fpm[19], " degrees")
    println("Refinement loops: ", loop)
    println("Final residual: ", epsout)
    println("Exit code (info): ", info)

    if info == 0
        println("FeastKit converged successfully")
    else
        println("Warning: FeastKit terminated with error code: ", info)
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
    # Estimate memory requirements for FeastKit
    
    # Main workspace arrays
    work_size = N * M0 * sizeof(T)
    workc_size = N * M0 * sizeof(Complex{T})
    reduced_size = 2 * M0 * M0 * sizeof(T)  # Aq and Sq
    eigen_size = (N * M0 + 2 * M0) * sizeof(T)  # eigenvectors and eigenvalues
    
    total_size = work_size + workc_size + reduced_size + eigen_size
    
    println("FeastKit Memory Estimate:")
    println("  Workspace (real): ", @sprintf("%.2f MB", work_size / 1024^2))
    println("  Workspace (complex): ", @sprintf("%.2f MB", workc_size / 1024^2))
    println("  Reduced matrices: ", @sprintf("%.2f MB", reduced_size / 1024^2))
    println("  Eigendata: ", @sprintf("%.2f MB", eigen_size / 1024^2))
    println("  Total estimate: ", @sprintf("%.2f MB", total_size / 1024^2))
    
    return total_size
end
@inline function check_complex_symmetric(A::AbstractMatrix{Complex{T}}) where T<:Real
    issymmetric(A) || throw(ArgumentError("Matrix must be complex-symmetric (equal to its transpose)."))
    return true
end
