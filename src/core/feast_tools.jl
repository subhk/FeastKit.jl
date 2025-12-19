# Feast utility and contour generation tools
# Translated from feast_tools.f90

using Random

# Helper functions for integration nodes
function gauss_legendre_point(n::Int, k::Int)
    # Use FastGaussQuadrature directly for robust Gauss–Legendre nodes/weights
    x, w = FastGaussQuadrature.gausslegendre(n)
    return x[k], w[k]
end

@inline function _feast_seeded_subspace!(work::AbstractMatrix{T}) where T<:Real
    N, M0 = size(work)
    seed = hash((N, M0))
    rng = MersenneTwister(seed)
    for j in 1:M0
        randn!(rng, view(work, :, j))
        col_norm = norm(view(work, :, j))
        if col_norm == 0
            work[1, j] = one(T)
            col_norm = one(T)
        end
        work[:, j] ./= col_norm
    end
    return work
end

@inline function _feast_seeded_subspace_complex!(work::AbstractMatrix{Complex{T}}) where T<:Real
    N, M0 = size(work)
    M0 == 0 && return work
    seed = hash((N, M0, :complex))
    rng = MersenneTwister(seed)
    for j in 1:M0
        for i in 1:N
            work[i, j] = Complex{T}(randn(rng, T), randn(rng, T))
        end
        col_norm = norm(view(work, :, j))
        if col_norm == 0
            work[1, j] = Complex{T}(one(T), zero(T))
            col_norm = one(T)
        end
        work[:, j] ./= col_norm
    end
    return work
end

function zolotarev_point(n::Int, k::Int)
    # Zolotarev rational approximation points and weights
    # Optimal for elliptical domains - simplified implementation
    # For full implementation, see original Feast Fortran code
    
    if k == 0  # Special case for initialization
        return 0.0 + 0.0im, 1.0 + 0.0im
    end
    
    # Approximate Zolotarev points using Chebyshev mapping
    theta = π * (2*k - 1) / (2*n)
    r_zol = 0.5  # Simplified radius
    
    zxe = r_zol * (cos(theta) + im * sin(theta))
    zwe = (π / n) * im * zxe  # Simplified weight
    
    return zxe, zwe
end

function feast_contour(Emin::T, Emax::T, fpm::Vector{Int}) where T<:Real
    # Ensure fpm parameters are initialized
    # If fpm[2] is still -111 (uninitialized), apply defaults first
    # NOTE: This should only happen if feastdefault! wasn't called by the caller
    if fpm[2] == FEAST_UNINITIALIZED || fpm[2] <= 0
        @debug "feast_contour calling feastdefault! because fpm[2]=$(fpm[2])"
        feastdefault!(fpm)
    end

    ne = fpm[2]   # Number of integration points (half-contour)
    fpm16 = fpm[16]  # Integration type: 0=Gauss, 1=Trapezoidal, 2=Zolotarev
    fpm18 = fpm[18]  # Ellipse ratio a/b * 100 (default: 100 = circle)

    # Parameters from Fortran implementation (matches zfeast_contour)
    r = (Emax - Emin) / T(2)  # Semi-major axis (horizontal)
    Emid = Emin + r           # Center point
    aspect_ratio = fpm18 * T(0.01)  # a/b ratio (vertical/horizontal)

    # Integration limits for parametrization (matches Fortran ba, ab)
    ba = -T(π) / 2
    ab = T(π) / 2

    # Generate half-contour (symmetric about real axis)
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)

    # Precompute Gauss–Legendre nodes/weights if needed
    x_gl = nothing
    w_gl = nothing
    if fpm16 == 0
        x_gl, w_gl = FastGaussQuadrature.gausslegendre(ne)
    end

    for e in 1:ne
        if fpm16 == 0  # Gauss-Legendre integration (matches Fortran exactly)
            xe = x_gl[e]
            we = w_gl[e]

            # Map x ∈ [-1, 1] to θ: theta = ba*xe + ab = -π/2*xe + π/2
            # This gives θ ∈ [π, 0] as xe goes from -1 to 1 (Fortran convention)
            theta = ba * xe + ab

            # Elliptical contour point: z(θ) = Emid + r*cos(θ) + i*r*aspect*sin(θ)
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta)

            # Jacobian (Fortran formula): jac = r*i*sin(θ) + r*aspect*cos(θ)
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)

            # Weight: Wne = (1/4) * we * jac (matches Fortran exactly)
            Wne[e] = T(0.25) * we * jac

        elseif fpm16 == 2  # Zolotarev integration (optimal for ellipses)
            zxe, zwe = zolotarev_point(ne, e)
            Zne[e] = zxe * r + Emid
            Wne[e] = zwe * r

        else  # Trapezoidal integration (fpm16 == 1)
            # Fortran: theta = π - (π/ne)/2 - (π/ne)*(e-1)
            theta = T(π) - (T(π)/ne)/2 - (T(π)/ne) * (e-1)

            # Elliptical contour point
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta)

            # Jacobian (Fortran formula)
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)

            # Weight: Wne = (1/(2*ne)) * jac (matches Fortran)
            Wne[e] = (one(T) / (2 * ne)) * jac
        end
    end

    return FeastContour{T}(Zne, Wne)
end

function feast_gcontour(Emid::Complex{T}, r::T, fpm::Vector{Int}) where T<:Real
    # Ensure fpm parameters are initialized
    if fpm[8] == FEAST_UNINITIALIZED || fpm[8] <= 0
        feastdefault!(fpm)
    end

    # For non-Hermitian problems, use fpm[8] for full-contour point count
    ne = fpm[8]      # Number of integration points (full contour)
    fpm16 = fpm[16]  # Integration type: 0=Gauss, 1=Trapezoidal
    fpm18 = fpm[18]  # Ellipse ratio a/b * 100 (100 = circle)
    fpm19 = fpm[19]  # Rotation angle in degrees [-180, 180]

    aspect_ratio = fpm18 * T(0.01)  # a/b ratio

    # Ellipse axis rotation (convert degrees to radians)
    rotation_theta = (fpm19 / T(180)) * T(π)
    nr = r * (cos(rotation_theta) + im * sin(rotation_theta))

    # Integration limits (matches Fortran ba, ab for half-contour)
    ba = -T(π) / 2
    ab = T(π) / 2

    # Generate full contour in complex plane (matches zfeast_gcontour)
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)

    if fpm16 == 0  # Gauss-Legendre integration
        # Fortran: uses two half-contours (upper and lower)
        x_gl_upper, w_gl_upper = FastGaussQuadrature.gausslegendre(ne ÷ 2)
        x_gl_lower, w_gl_lower = FastGaussQuadrature.gausslegendre(ne - ne ÷ 2)

        # Upper half (e = 1 to ne/2)
        for e in 1:(ne ÷ 2)
            xe = x_gl_upper[e]
            we = w_gl_upper[e]

            # theta = ba*xe + ab = -π/2*xe + π/2 (upper half: θ ∈ [π, 0])
            theta = ba * xe + ab

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/4) * we * jac
            Wne[e] = T(0.25) * we * jac
        end

        # Lower half (e = ne/2+1 to ne)
        for e in (ne ÷ 2 + 1):ne
            idx = e - ne ÷ 2
            xe = x_gl_lower[idx]
            we = w_gl_lower[idx]

            # theta = -ba*xe - ab = π/2*xe - π/2 (lower half: θ ∈ [0, -π] -> [-π, 0])
            theta = -ba * xe - ab

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian (same formula)
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/4) * we * jac
            Wne[e] = T(0.25) * we * jac
        end

    else  # Trapezoidal integration (fpm16 == 1)
        # Fortran: theta = π - (2π/ne)/2 - (2π/ne)*(e-1)
        for e in 1:ne
            theta = T(π) - (2 * T(π) / ne) / 2 - (2 * T(π) / ne) * (e - 1)

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/ne) * jac (full contour trapezoidal)
            Wne[e] = (one(T) / ne) * jac
        end
    end

    return FeastContour{T}(Zne, Wne)
end

# Overload for real Emid (convenience function)
function feast_gcontour(Emid::T, r::T, fpm::Vector{Int}) where T<:Real
    return feast_gcontour(Complex{T}(Emid, zero(T)), r, fpm)
end

function feast_customcontour(Zne::Vector{Complex{T}}, 
                           fpm::Vector{Int}) where T<:Real
    ne = length(Zne)
    fpm[2] = ne  # Set number of integration points
    fpm[15] = 1  # Mark as custom contour
    
    # Compute weights using trapezoidal rule
    Wne = Vector{Complex{T}}(undef, ne)
    
    for i in 1:ne
        i_prev = i == 1 ? ne : i - 1
        i_next = i == ne ? 1 : i + 1
        
        # Trapezoidal rule weight
        Wne[i] = (Zne[i_next] - Zne[i_prev]) / (2 * ne)
    end
    
    return FeastContour{T}(Zne, Wne)
end

# Advanced contour generation functions following Fortran interface

"""
    feast_contour_expert(Emin, Emax, ne, integration_type, ellipse_ratio)

Generate Feast integration contour with expert-level control matching original Fortran implementation.

# Arguments
- `Emin, Emax`: Search interval bounds
- `ne`: Number of integration points (half-contour)
- `integration_type`: 0=Gauss-Legendre, 1=Trapezoidal, 2=Zolotarev  
- `ellipse_ratio`: Aspect ratio a/b * 100 (100 = circle)

# Returns
- `FeastContour` with integration nodes and weights
"""
function feast_contour_expert(Emin::T, Emax::T, ne::Int, 
                            integration_type::Int=0, 
                            ellipse_ratio::Int=100) where T<:Real
    fpm = zeros(Int, 64)
    fpm[2] = ne
    fpm[16] = integration_type  
    fpm[18] = ellipse_ratio
    
    return feast_contour(Emin, Emax, fpm)
end

"""
    feast_contour_custom_weights!(Zne, Wne)

Custom contour integration with user-provided nodes and weights.
Follows the Fortran expert interface pattern.

# Arguments  
- `Zne`: Complex integration nodes
- `Wne`: Complex integration weights (modified in-place)

# Returns
- `FeastContour` using provided nodes and computed/modified weights
"""  
function feast_contour_custom_weights!(Zne::Vector{Complex{T}}, 
                                     Wne::Vector{Complex{T}}) where T<:Real
    ne = length(Zne)
    
    # Validate input
    if length(Wne) != ne
        throw(ArgumentError("Zne and Wne must have same length"))
    end
    
    # User is responsible for providing correct weights
    # This interface allows maximum flexibility like the Fortran version
    
    return FeastContour{T}(copy(Zne), copy(Wne))
end

"""
    feast_rationalx(Zne, Wne, lambda)

Compute rational function values using custom integration nodes and weights.
Direct translation of dfeast_rationalx from Fortran.

For eigenvalues inside the contour, the rational function returns ≈1.
For eigenvalues outside, it returns ≈0.

# Arguments
- `Zne`: Integration nodes (half-contour)
- `Wne`: Integration weights (half-contour)
- `lambda`: Real eigenvalues to evaluate

# Returns
- Vector of rational function values
"""
function feast_rationalx(Zne::Vector{Complex{T}},
                         Wne::Vector{Complex{T}},
                         lambda::Vector{T}) where T<:Real
    ne = length(Zne)
    M = length(lambda)
    f = zeros(T, M)

    # Compute rational function (matches dfeast_rationalx exactly)
    # f(λ) = 2 * Re(Σ Wne[e] / (Zne[e] - λ))
    # Factor of 2 accounts for symmetric half-contour
    for j in 1:M
        for e in 1:ne
            f[j] += 2 * real(Wne[e] / (Zne[e] - lambda[j]))
        end
    end

    return f
end

"""
    feast_rational(Emin, Emax, fpm, lambda)

Compute rational function values for real eigenvalues using default ellipsoid contour.
Direct translation of dfeast_rational from Fortran.

# Arguments
- `Emin, Emax`: Search interval bounds
- `fpm`: FEAST parameters (fpm[2]=nodes, fpm[16]=integration type, fpm[18]=aspect ratio)
- `lambda`: Real eigenvalues to evaluate

# Returns
- Vector of rational function values
"""
function feast_rational(lambda::Vector{T}, Emin::T, Emax::T,
                        fpm::Vector{Int}) where T<:Real
    # Generate contour (matches dfeast_rational calling zfeast_contour)
    contour = feast_contour(Emin, Emax, fpm)

    # Compute rational using feast_rationalx
    f = feast_rationalx(contour.Zne, contour.Wne, lambda)

    # Add Zolotarev initialization if needed (matches Fortran)
    if fpm[16] == 2  # Zolotarev
        _, zwe = zolotarev_point(fpm[2], 0)
        f .+= real(zwe)
    end

    return f
end

"""
    feast_grationalx(Zne, Wne, lambda)

Compute rational function values for complex eigenvalues using custom contour.
Direct translation of zfeast_grationalx from Fortran.

# Arguments
- `Zne`: Integration nodes (full contour)
- `Wne`: Integration weights (full contour)
- `lambda`: Complex eigenvalues to evaluate

# Returns
- Vector of complex rational function values
"""
function feast_grationalx(Zne::Vector{Complex{T}},
                          Wne::Vector{Complex{T}},
                          lambda::Vector{Complex{T}}) where T<:Real
    ne = length(Zne)
    M = length(lambda)
    f = Vector{Complex{T}}(undef, M)

    # Compute rational function (matches zfeast_grationalx exactly)
    # f(λ) = Σ Wne[e] / (Zne[e] - λ)  (full contour, no factor of 2)
    for j in 1:M
        f[j] = zero(Complex{T})
        for e in 1:ne
            f[j] += Wne[e] / (Zne[e] - lambda[j])
        end
    end

    return f
end

"""
    feast_grational(Emid, r, fpm, lambda)

Compute rational function values for complex eigenvalues using default ellipsoid contour.
Direct translation of zfeast_grational from Fortran.

# Arguments
- `Emid`: Center of search region (complex)
- `r`: Radius of search region
- `fpm`: FEAST parameters (fpm[8]=nodes, fpm[16]=integration type, etc.)
- `lambda`: Complex eigenvalues to evaluate

# Returns
- Vector of complex rational function values
"""
function feast_grational(lambda::Vector{Complex{T}}, Emid::Complex{T},
                         r::T, fpm::Vector{Int}) where T<:Real
    # Generate contour (matches zfeast_grational calling zfeast_gcontour)
    contour = feast_gcontour(Emid, r, fpm)

    # Compute rational using feast_grationalx
    return feast_grationalx(contour.Zne, contour.Wne, lambda)
end

# Convenience overloads for type flexibility
function feast_rationalx(lambda::Vector{T},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    ne = length(Zne)
    ne == length(Wne) || throw(ArgumentError("Zne and Wne must have the same length"))
    base = promote_type(T, TZ, TW)
    return feast_rationalx(
        Vector{Complex{base}}(Zne),
        Vector{Complex{base}}(Wne),
        Vector{base}(lambda))
end

function feast_grationalx(lambda::Vector{Complex{T}},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    ne = length(Zne)
    ne == length(Wne) || throw(ArgumentError("Zne and Wne must have the same length"))
    base = promote_type(T, TZ, TW)
    return feast_grationalx(
        Vector{Complex{base}}(Zne),
        Vector{Complex{base}}(Wne),
        Vector{Complex{base}}(lambda))
end

# Legacy alias for backward compatibility
const feast_rational_expert = feast_rationalx

# Check if eigenvalue is inside contour
function feast_inside_contour(lambda::T, Emin::T, Emax::T) where T<:Real
    return Emin <= lambda <= Emax
end

function feast_inside_gcontour(lambda::Complex{T}, Emid::Complex{T}, r::T) where T<:Real
    return abs(lambda - Emid) <= r
end

# Sort eigenvalues and eigenvectors
function feast_sort!(lambda::Vector{T}, q::Matrix{VT}, 
                    res::Vector{T}, M::Int) where {T<:Real, VT}
    # Sort by eigenvalue magnitude
    perm = sortperm(lambda[1:M])
    lambda[1:M] = lambda[perm]
    q[:, 1:M] = q[:, perm]
    res[1:M] = res[perm]
    return nothing
end

# Sort function for complex eigenvalues (general case)
function feast_sort_general!(lambda::Vector{Complex{T}}, q::Matrix{Complex{T}}, 
                           res::Vector{T}, M::Int) where T<:Real
    # Sort by eigenvalue magnitude |lambda|
    perm = sortperm(abs.(lambda[1:M]))
    lambda[1:M] = lambda[perm]
    q[:, 1:M] = q[:, perm]
    res[1:M] = res[perm]
    return nothing
end

# Compute residual norms
function feast_residual!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                        lambda::Vector{T}, q::Matrix{T}, res::Vector{T}, 
                        M::Int) where T
    N = size(A, 1)
    
    for j in 1:M
        # Compute residual: r = A*q - lambda*B*q
        r = A * q[:, j] - lambda[j] * (B * q[:, j])
        res[j] = norm(r)
    end
    
    return nothing
end

# Extract name from Feast code (similar to feast_name subroutine)
function feast_name(code::Int)
    digits = Vector{Int}(undef, 6)
    rem = code
    
    for i in 1:6
        digits[6-i+1] = rem % 10
        rem = rem ÷ 10
    end
    
    name = ""
    
    # Parallel/serial
    if digits[1] == 2
        name *= "p"
    end
    
    # Precision
    if digits[2] == 1
        name *= "s"
    elseif digits[2] == 2
        name *= "d"
    elseif digits[2] == 3
        name *= "c"
    elseif digits[2] == 4
        name *= "z"
    end
    
    # Direct/iterative
    if digits[3] == 2
        name *= "i"
    end
    
    name *= "feast_"
    
    # Matrix type
    if digits[4] == 1
        name *= "s"
    elseif digits[4] == 2
        name *= "h"
    elseif digits[4] == 3
        name *= "g"
    end
    
    # Interface type
    if digits[5] == 1
        name *= "rci"
    elseif digits[5] == 2
        name *= "y"
    elseif digits[5] == 3
        name *= "b"
    elseif digits[5] == 4
        name *= "csr"
    elseif digits[5] == 5
        name *= "e"
    end
    
    # Variant
    if digits[6] == 1
        name *= "x"
    elseif digits[6] == 2
        name *= "ev"
    elseif digits[6] == 3
        name *= "evx"
    elseif digits[6] == 4
        name *= "gv"
    elseif digits[6] == 5
        name *= "gvx"
    elseif digits[6] == 6
        name *= "pep"
    elseif digits[6] == 7
        name *= "pepx"
    end
    
    return name
end
