"""
    feast_circle(center, radius; n=64)

Construct a full circular `FeastContour` using `n ≥ 3` midpoint trapezoidal
nodes. `center` may be real or complex; `radius` must be finite and positive.
Weights already include `dz/(2π*im)`. Register with `with_custom_contour` and
use `feast_general` with `fpm[16] = 1`. No parameters are modified here.
See also [`feast_ellipse`](@ref) and [`feast_rectangle`](@ref).
"""
feast_circle(center::Number, radius::Real; n::Int=64) =
    feast_ellipse(center, radius, radius; n=n)

"""
    feast_ellipse(center, a, b; n=64, rotation=0)

Construct a full elliptical `FeastContour` with positive finite semiaxes
`a` and `b`, rotated counterclockwise by `rotation` **radians**. `n ≥ 3`
midpoint trapezoidal nodes follow the boundary counterclockwise. All geometry
must be finite. Weights include `dz/(2π*im)`; do not normalize them again.

Use with `FeastKit.with_custom_contour(fpm, contour)` and `feast_general`,
setting `fpm[16] = 1`. These full contours are not the half-contours returned
by [`feast_contour_expert`](@ref) for symmetric/Hermitian interval problems.
"""
function feast_ellipse(center::Number, a::Real, b::Real;
                       n::Int=64, rotation::Real=0)
    n >= 3 || throw(ArgumentError("n must be at least 3"))
    isfinite(center) && isfinite(a) && isfinite(b) && isfinite(rotation) ||
        throw(ArgumentError("Contour geometry must be finite"))
    a > 0 && b > 0 || throw(ArgumentError("Semiaxes must be positive"))
    T = float(promote_type(typeof(real(center)), typeof(a), typeof(b), typeof(rotation)))
    c, aa, bb = Complex{T}(center), T(a), T(b)
    rot = cis(T(rotation))
    nodes = Vector{Complex{T}}(undef, n)
    weights = similar(nodes)
    for j in 1:n
        θ = 2T(π) * (T(j) - T(0.5)) / n
        nodes[j] = c + rot * complex(aa * cos(θ), bb * sin(θ))
        weights[j] = rot * complex(bb * cos(θ), aa * sin(θ)) / n
    end
    return FeastContour{T}(nodes, weights)
end

"""
    feast_rectangle(xmin, xmax, ymin, ymax; points_per_edge=32)

Construct a full counterclockwise box contour with strictly increasing finite
real and imaginary bounds. Uses midpoint quadrature with `points_per_edge ≥ 1`
nodes on each edge (total `4points_per_edge`). Weights include `dz/(2π*im)`.
Increase the count for eigenvalues close to an edge; avoid eigenvalues on the
boundary. Use with `with_custom_contour` and `feast_general`, with `fpm[16] = 1`.
The rectangle, rather than the nominal circle arguments of `feast_general`,
selects eigenvalues when registered as a custom contour.
"""
function feast_rectangle(xmin::Real, xmax::Real, ymin::Real, ymax::Real;
                         points_per_edge::Int=32)
    all(isfinite, (xmin, xmax, ymin, ymax)) ||
        throw(ArgumentError("Rectangle bounds must be finite"))
    xmin < xmax && ymin < ymax || throw(ArgumentError("Rectangle bounds must increase"))
    points_per_edge >= 1 || throw(ArgumentError("points_per_edge must be positive"))
    T = float(promote_type(typeof(xmin), typeof(xmax), typeof(ymin), typeof(ymax)))
    corners = Complex{T}[complex(xmin, ymin), complex(xmax, ymin),
                          complex(xmax, ymax), complex(xmin, ymax)]
    n = points_per_edge
    nodes = Vector{Complex{T}}(undef, 4n)
    weights = similar(nodes)
    for edge in 1:4
        start = corners[edge]
        step = (corners[mod1(edge+1, 4)] - start) / n
        for j in 1:n
            k = (edge-1)*n+j
            nodes[k] = start + (T(j)-T(0.5))*step
            weights[k] = step / (2T(π)*im)
        end
    end
    return FeastContour{T}(nodes, weights)
end
