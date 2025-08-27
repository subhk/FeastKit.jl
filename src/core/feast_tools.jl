# FEAST utility and contour generation tools
# Translated from feast_tools.f90

# Helper functions for integration nodes
function gauss_legendre_point(n::Int, k::Int)
    # Compute k-th Gauss-Legendre point and weight for n-point quadrature
    # This is a simplified implementation - for production use, consider using
    # a specialized library like FastGaussQuadrature.jl
    
    # Approximate roots using Chebyshev approximation
    x = cos(π * (k - 0.25) / (n + 0.5))
    
    # Newton-Raphson refinement
    # Initialize outside the loop so they are defined for weight calculation
    p1 = 1.0
    p2 = 0.0
    for _ in 1:10
        p1 = 1.0
        p2 = 0.0
        
        # Compute Legendre polynomial P_n(x) using recurrence
        for j in 1:n
            p3 = p2
            p2 = p1
            p1 = ((2*j - 1) * x * p2 - (j - 1) * p3) / j
        end
        
        # Compute derivative
        pp = n * (x * p1 - p2) / (x * x - 1)
        
        # Newton step
        x1 = x
        x = x1 - p1 / pp
        
        if abs(x - x1) < 1e-14
            break
        end
    end
    
    # Compute weight using last-updated Legendre values
    pp = n * (x * p1 - p2) / (x * x - 1)
    w = 2 / ((1 - x * x) * pp * pp)
    
    return x, w
end

function zolotarev_point(n::Int, k::Int)
    # Zolotarev rational approximation points and weights
    # Optimal for elliptical domains - simplified implementation
    # For full implementation, see original FEAST Fortran code
    
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
    ne = fpm[2]  # Number of integration points (half-contour)
    fpm16 = get(fpm, 16, 0)  # Integration type: 0=Gauss, 1=Trapezoidal, 2=Zolotarev
    fpm18 = get(fpm, 18, 100)  # Ellipse ratio a/b * 100 (default: 100 = circle)
    
    # Parameters from Fortran implementation
    r = (Emax - Emin) / 2  # Semi-major axis
    Emid = Emin + r        # Center point
    aspect_ratio = fpm18 * 0.01  # Convert percentage to ratio
    
    # Generate half-contour (symmetric about real axis)
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)
    
    for e in 1:ne
        if fpm16 == 0  # Gauss-Legendre integration
            xe, we = gauss_legendre_point(ne, e)
            theta = -π/2 * xe + π/2  # Map [-1,1] to [-π/2, π/2]
            
            # Elliptical contour point
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta)
            
            # Jacobian and weight
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)
            Wne[e] = 0.25 * we * jac
            
        elseif fpm16 == 2  # Zolotarev integration (optimal for ellipses)
            zxe, zwe = zolotarev_point(ne, e)
            Zne[e] = zxe * r + Emid
            Wne[e] = zwe * r
            
        else  # Trapezoidal integration (fpm16 == 1)
            theta = π - (π/ne)/2 - (π/ne) * (e-1)
            
            # Elliptical contour point
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta) 
            
            # Jacobian and weight
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)
            Wne[e] = (1.0 / (2 * ne)) * jac
        end
    end
    
    return FeastContour{T}(Zne, Wne)
end

function feast_gcontour(Emid::Complex{T}, r::T, fpm::Vector{Int}) where T<:Real
    ne = fpm[2]  # Number of integration points
    
    # Generate circular contour in complex plane
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)
    
    # Generate integration points on circle
    for i in 1:ne
        theta = 2π * (i - 1) / ne
        z = Emid + r * exp(im * theta)
        
        # Integration node
        Zne[i] = z
        
        # Integration weight  
        Wne[i] = r * im * exp(im * theta) * 2π / ne
    end
    
    return FeastContour{T}(Zne, Wne)
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

Generate FEAST integration contour with expert-level control matching original Fortran implementation.

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
    feast_rational_expert(Zne, Wne, lambda)

Compute rational function values using custom integration nodes and weights.
Direct translation of dfeast_rationalx from Fortran.

# Arguments
- `Zne`: Integration nodes  
- `Wne`: Integration weights
- `lambda`: Eigenvalues to evaluate

# Returns  
- Vector of rational function values
"""
function feast_rational_expert(Zne::Vector{Complex{T}}, 
                             Wne::Vector{Complex{T}},
                             lambda::Vector{T}) where T<:Real
    ne = length(Zne)
    M = length(lambda) 
    f = zeros(T, M)
    
    # Compute rational function: f(λ) = (1/2πi) ∫ w(z)/(z-λ) dz
    for j in 1:M
        sum_val = zero(Complex{T})
        for i in 1:ne
            z = Zne[i] 
            w = Wne[i]
            sum_val += w / (z - lambda[j])
        end
        # Factor of 2 accounts for symmetry (half-contour integration) 
        f[j] = 2 * real(sum_val)
    end
    
    return f
end

function feast_rational(lambda::Vector{T}, Emin::T, Emax::T, 
                       fpm::Vector{Int}) where T<:Real
    # Compute rational function values for eigenvalue filtering
    M = length(lambda)
    rational_values = Vector{T}(undef, M)
    
    contour = feast_contour(Emin, Emax, fpm)
    ne = length(contour.Zne)
    
    for j in 1:M
        sum_val = zero(Complex{T})
        for i in 1:ne
            z = contour.Zne[i]
            w = contour.Wne[i]
            sum_val += w / (z - lambda[j])
        end
        rational_values[j] = real(sum_val) / (2π * im)
    end
    
    return rational_values
end

function feast_grational(lambda::Vector{Complex{T}}, Emid::Complex{T}, 
                        r::T, fpm::Vector{Int}) where T<:Real
    # Compute rational function values for complex eigenvalues
    M = length(lambda)
    rational_values = Vector{T}(undef, M)
    
    contour = feast_gcontour(Emid, r, fpm)
    ne = length(contour.Zne)
    
    for j in 1:M
        sum_val = zero(Complex{T})
        for i in 1:ne
            z = contour.Zne[i]
            w = contour.Wne[i]
            sum_val += w / (z - lambda[j])
        end
        rational_values[j] = abs(sum_val) / (2π)
    end
    
    return rational_values
end

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

# Extract name from FEAST code (similar to feast_name subroutine)
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
