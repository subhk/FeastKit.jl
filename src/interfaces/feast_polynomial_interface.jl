# Polynomial eigenvalue problems
function feast_polynomial(coeffs::Vector{<:AbstractMatrix{Complex{T}}},
                         center::Complex{T}, radius::T; M0::Int = 10,
                         fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing) where T<:Real
    # Feast for polynomial eigenvalue problems
    # P(λ) = coeffs[1] + λ*coeffs[2] + λ²*coeffs[3] + ...

    # Initialize Feast parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end

    d = length(coeffs) - 1  # Degree of polynomial
    # The companion linearization is dense; accept the public AbstractMatrix
    # contract by materializing sparse and structured coefficients here.
    dense_coeffs = Matrix{Complex{T}}[Matrix{Complex{T}}(A) for A in coeffs]
    return feast_pep!(dense_coeffs, d, center, radius, M0, _ensure_feast_parameters(fpm))
end
