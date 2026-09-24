# `scales[k]` multiplies coefficient k (the λ^(k-1) term), for a balanced
# polynomial; see `_feast_polynomial_balance`.
function _matrix_free_polynomial_companion_operators(
        coeffs_ops::AbstractVector{<:MatrixFreeOperator{Complex{T}}},
        scales::AbstractVector{T} = ones(T, length(coeffs_ops))) where T<:Real

    d = length(coeffs_ops) - 1
    if d < 1
        throw(ArgumentError("Need at least 2 coefficient operators (degree ≥ 1)"))
    end

    N = size(coeffs_ops[1], 1)
    for op in coeffs_ops
        if size(op) != (N, N)
            throw(DimensionMismatch("All coefficient operators must have size ($N, $N)"))
        end
    end

    companion_tmp = Vector{Complex{T}}(undef, N)
    companion_x = Vector{Complex{T}}(undef, N)

    function A_companion_mul!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})
        fill!(y, zero(Complex{T}))

        for block in 1:d-1
            y_offset = (block - 1) * N
            x_offset = block * N
            @inbounds for i in 1:N
                y[y_offset+i] = x[x_offset+i]
            end
        end

        last_offset = (d - 1) * N
        for block in 0:d-1
            x_offset = block * N
            @inbounds for i in 1:N
                companion_x[i] = x[x_offset+i]
            end

            mul!(companion_tmp, coeffs_ops[block+1], companion_x)
            scale = scales[block+1]

            @inbounds for i in 1:N
                y[last_offset+i] -= scale * companion_tmp[i]
            end
        end

        return y
    end

    function B_companion_mul!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})
        fill!(y, zero(Complex{T}))

        for block in 0:d-2
            offset = block * N
            @inbounds for i in 1:N
                y[offset+i] = x[offset+i]
            end
        end

        last_offset = (d - 1) * N
        @inbounds for i in 1:N
            companion_x[i] = x[last_offset+i]
        end
        mul!(companion_tmp, coeffs_ops[end], companion_x)
        scale = scales[end]
        @inbounds for i in 1:N
            y[last_offset+i] = scale * companion_tmp[i]
        end

        return y
    end

    companion_size = (d * N, d * N)
    A_comp = LinearOperator{Complex{T}}(A_companion_mul!, companion_size)
    B_comp = LinearOperator{Complex{T}}(B_companion_mul!, companion_size)
    return A_comp, B_comp
end

"""
    feast_polynomial(coeffs_ops, center, radius; kwargs...)

Matrix-free Feast for polynomial eigenvalue problems.

Solves the polynomial eigenvalue problem P(λ)x = 0 where:
P(λ) = coeffs_ops[1] + λ*coeffs_ops[2] + λ²*coeffs_ops[3] + ... + λᵈ*coeffs_ops[d+1]

The polynomial is linearized using companion matrices to form a generalized
eigenvalue problem (A - λB)y = 0 of size (d*N × d*N), where y = [x; λx; λ²x; ...; λᵈ⁻¹x].

# Arguments
- `coeffs_ops`: Vector of matrix operators [C₀, C₁, C₂, ..., Cᵈ] where P(λ) = Σᵢ λⁱCᵢ
- `center`: Center of circular search region in complex plane
- `radius`: Radius of circular search region
- `M0`: Maximum number of eigenvalues to find
- `solver`: Linear solver type or custom function
- `kwargs`: Additional options passed to feast_general

# Returns
- `FeastGeneralResult` with eigenvalues λ and corresponding eigenvectors x (first N components of full eigenvector)
"""
function feast_polynomial(coeffs_ops::Vector{<:MatrixFreeOperator{Complex{T}}},
                         center::Complex{T}, radius::T;
                         M0::Union{Int,Nothing} = nothing,
                         solver::Union{Symbol, Function} = :gmres,
                         kwargs...) where T<:Real

    d = length(coeffs_ops) - 1  # Polynomial degree
    N = size(coeffs_ops[1], 1)

    # Validate input
    if d < 1
        throw(ArgumentError("Need at least 2 coefficient operators (degree ≥ 1)"))
    end

    for i in 1:length(coeffs_ops)
        if size(coeffs_ops[i]) != (N, N)
            throw(DimensionMismatch("All coefficient operators must have size ($N, $N)"))
        end
    end

    # Balance the polynomial (see _feast_polynomial_balance) so that neither the
    # size of the coefficients nor the units of λ affect the solve. Operator
    # sizes come from a few random probes. A caller-supplied `solver` solves
    # the companion system as given, so it is left unbalanced.
    γ, δ = one(T), one(T)
    if !(solver isa Function)
        probes = Matrix{Complex{T}}(undef, N, min(N, 4))
        _feast_seeded_subspace_complex!(probes)
        image = Vector{Complex{T}}(undef, N)
        norms = map(coeffs_ops) do op
            total = zero(T)
            for j in axes(probes, 2)
                mul!(image, op, view(probes, :, j))
                total += sum(abs2, image)
            end
            sqrt(total) / norm(probes)
        end
        γ, δ = _feast_polynomial_balance(norms)
    end
    scales = [δ * γ^(k - 1) for k in 1:d+1]
    A_comp, B_comp = _matrix_free_polynomial_companion_operators(coeffs_ops, scales)

    # Solve the linearized problem for μ = λ/γ over the scaled search region.
    options = Dict{Symbol,Any}(kwargs)
    fpm = get(options, :fpm, nothing)
    fpm_vec = fpm isa FeastParameters ? fpm.fpm : fpm
    result = if fpm_vec === nothing
        feast_general(A_comp, B_comp, center / γ, radius / γ; M0=M0, solver=solver, kwargs...)
    else
        _feast_with_scaled_contour(fpm_vec, T, γ) do inner
            options[:fpm] = inner
            feast_general(A_comp, B_comp, center / γ, radius / γ; M0=M0, solver=solver, options...)
        end
    end

    # Extract original eigenvectors (first N components)
    if result.M > 0
        q_original = result.q[1:N, :]
        return FeastGeneralResult{T}(
            γ .* result.lambda,
            q_original,
            result.M,
            result.res,
            result.info,
            result.epsout,
            result.loop
        )
    else
        return result
    end
end

"""
    validate_companion_matrices(A_companion_mul!, B_companion_mul!, coeffs_ops, test_lambda, test_x)

Validate that the companion matrices correctly linearize the polynomial eigenvalue problem.

Tests that if P(λ)x = 0, then (A - λB)y = 0 where y = [x; λx; λ²x; ...; λᵈ⁻¹x].
Each residual is judged relative to the size of the terms that cancel in it,
so the verdicts do not depend on the scale of the coefficients or of `λ`.
"""
function validate_companion_matrices(A_companion_mul!::Function,
                                   B_companion_mul!::Function,
                                   coeffs_ops::Vector{<:MatrixFreeOperator{Complex{T}}},
                                   test_lambda::Complex{T},
                                   test_x::AbstractVector{Complex{T}}) where T<:Real

    d = length(coeffs_ops) - 1
    N = length(test_x)

    # Construct companion eigenvector: y = [x; λx; λ²x; ...; λᵈ⁻¹x]
    y = zeros(Complex{T}, d * N)
    lambda_power = one(Complex{T})
    for i in 0:d-1
        y[i*N+1:(i+1)*N] .= lambda_power .* test_x
        lambda_power *= test_lambda
    end

    # Test (A - λB)y = 0
    Ay = similar(y)
    By = similar(y)

    A_companion_mul!(Ay, y)
    B_companion_mul!(By, y)

    residual = Ay - test_lambda * By
    residual_norm = norm(residual)
    companion_terms = norm(Ay) + abs(test_lambda) * norm(By)

    # Also verify that P(λ)x = 0
    Px = zeros(Complex{T}, N)
    temp = similar(test_x)
    lambda_power = one(Complex{T})
    polynomial_terms = zero(T)

    for i in 0:d
        mul!(temp, coeffs_ops[i+1], test_x)
        Px .+= lambda_power .* temp
        polynomial_terms += abs(lambda_power) * norm(temp)
        lambda_power *= test_lambda
    end

    polynomial_residual = norm(Px)

    return (
        companion_residual = residual_norm,
        polynomial_residual = polynomial_residual,
        companion_valid = residual_norm <= 1e-12 * companion_terms,
        polynomial_valid = polynomial_residual <= 1e-12 * polynomial_terms
    )
end
