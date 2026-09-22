function _feast_contour_setup(contour::FeastContour, fpm, quadrature_points)
    n = length(contour.Zne)
    n >= 3 && length(contour.Wne) == n ||
        throw(ArgumentError("A full contour must have at least three nodes and matching weights"))
    all(isfinite, contour.Zne) && all(isfinite, contour.Wne) ||
        throw(ArgumentError("Contour nodes and weights must be finite"))
    boundary = contour.vertices === nothing ? contour.Zne : contour.vertices
    length(boundary) >= 3 && all(isfinite, boundary) ||
        throw(ArgumentError("Contour boundary must contain at least three finite vertices"))
    quadrature_points === nothing ||
        (quadrature_points isa Integer && !(quadrature_points isa Bool) && quadrature_points == n) ||
        throw(ArgumentError("quadrature_points must match the contour's $n nodes; set the node count when constructing the contour"))
    xmin, xmax = extrema(real, boundary)
    ymin, ymax = extrema(imag, boundary)
    center = complex(xmin / 2 + xmax / 2, ymin / 2 + ymax / 2)
    radius = maximum(z -> abs(z - center), boundary)
    isfinite(radius) && radius > 0 || throw(ArgumentError("Contour must enclose a finite, nonzero region"))
    # Full-contour weights are already provided. The enclosing circle only
    # satisfies the legacy driver's arguments; membership uses the contour.
    params = copy(_ensure_feast_parameters(fpm))
    # This call owns a fresh registration, even when fpm was copied from an
    # active solve. Do not temporarily remove another caller's contour.
    params[29] = 0
    params[8] = n
    params[16] = 1
    return params, center, radius
end

"""
    feast(A, contour::FeastContour; kwargs...)
    feast(A, B, contour::FeastContour; kwargs...)

Find eigenpairs inside a full complex contour, such as `feast_rectangle(...)`,
`feast_ellipse(...)`, or `feast_circle(...)`. Uses the general eigensolver and
returns `FeastGeneralResult`, including for real matrices. Accepts the same
named options as `feast_general`. Supplied `fpm` is left unchanged. Set the
node count in the contour constructor; `quadrature_points`, if supplied, must
match it. Half-contours for symmetric interval problems are not supported here.
"""
function feast(A::Union{AbstractMatrix,MatrixFreeOperator{<:Complex}}, contour::FeastContour;
               fpm::Union{Vector{Int},FeastParameters,Nothing}=nothing,
               quadrature_points=nothing, kwargs...)
    params, center, radius = _feast_contour_setup(contour, fpm, quadrature_points)
    return with_custom_contour(params, contour) do
        feast_general(A, center, radius; fpm=params, kwargs...)
    end
end

function feast(A::Union{AbstractMatrix,MatrixFreeOperator{<:Complex}},
               B::Union{AbstractMatrix,MatrixFreeOperator{<:Complex}}, contour::FeastContour;
               fpm::Union{Vector{Int},FeastParameters,Nothing}=nothing,
               quadrature_points=nothing, kwargs...)
    params, center, radius = _feast_contour_setup(contour, fpm, quadrature_points)
    return with_custom_contour(params, contour) do
        feast_general(A, B, center, radius; fpm=params, kwargs...)
    end
end

function feast_custom_contour(nodes::Vector{Complex{T}},
                             A::AbstractMatrix, B::AbstractMatrix,
                             interval::Tuple{T,T};
                             M0::Int = 10,
                             fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing) where T<:Real
    # Feast with custom integration contour
    # Computes trapezoidal weights from nodes, registers as custom contour, then runs feast

    params = _ensure_feast_parameters(fpm)
    contour = feast_customcontour(nodes, params)

    return with_custom_contour(params, contour) do
        feast(A, B, interval; M0=M0, fpm=params)
    end
end

function feast_custom_contour(nodes::Vector{Complex{T}},
                             A::AbstractMatrix,
                             interval::Tuple{T,T};
                             M0::Int = 10,
                             fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing) where T<:Real
    N = size(A, 1)
    B = isa(A, SparseMatrixCSC) ? spdiagm(0 => fill(one(eltype(A)), N)) :
        Matrix{eltype(A)}(I, N, N)
    return feast_custom_contour(nodes, A, B, interval; M0=M0, fpm=fpm)
end
