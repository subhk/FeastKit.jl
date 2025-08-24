# FEAST utility and contour generation tools
# Translated from feast_tools.f90

function feast_contour(Emin::T, Emax::T, fpm::Vector{Int}) where T<:Real
    ne = fpm[2]  # Number of integration points (half-contour)
    fpm16 = get(fpm, 16, 0)  # Integration type: 0=Gauss, 1=Trapezoidal, 2=Zolotarev
    fmp18 = get(fmp, 18, 100)  # Ellipse ratio a/b * 100 (default: 100 = circle)
    
    # Parameters from Fortran implementation
    r = (Emax - Emin) / 2  # Semi-major axis
    Emid = Emin + r        # Center point
    aspect_ratio = fmp18 * 0.01  # Convert percentage to ratio
    
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