# Residual inverse form of the FEAST projector, for dense Float64 pencils:
# R = A*Q - B*Q*Λ,
# (zB-A)^(-1) BQ = (Q + (zB-A)^(-1) R) (zI-Λ)^(-1).
# Only the correction solves use Float32. Q, R, projection, and the outer
# eigenpair residual stay Float64. See Gavin/Miedlar/Polizzi (2018), §4.2:
# https://arxiv.org/abs/1801.09794 (specialized here to a linear pencil).
const _FeastLU32 = LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{Int}}
const _FeastLU64 = LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{Int}}

mutable struct _FeastMixedShift
    low::Union{Nothing,_FeastLU32}
    high::Union{Nothing,_FeastLU64}
    scale::Float64
    z::ComplexF64
end

mutable struct _FeastMixedWorkspace{TA,TB}
    A::TA
    B::TB
    factors::Vector{Union{Nothing,_FeastMixedShift}}
    store::Bool
    q::Matrix{ComplexF64}
    aq::Matrix{ComplexF64}
    bq::Matrix{ComplexF64}
    residual::Matrix{ComplexF64}
    correction::Matrix{ComplexF64}
    check_a::Matrix{ComplexF64}
    check_b::Matrix{ComplexF64}
    low_rhs::Matrix{ComplexF32}
    values::Vector{ComplexF64}
    scales::Vector{Float64}
    fallbacks::Int
end

function _feast_mixed_workspace(A, B, M0, fpm, solver)
    fpm[42] == 1 || return nothing
    solver === :direct || throw(ArgumentError("mixed_precision requires solver=:direct"))
    A isa Matrix && eltype(A) in (Float64, ComplexF64) &&
        (B === nothing || (B isa Matrix && eltype(B) == eltype(A))) ||
        throw(ArgumentError("mixed_precision currently requires dense Float64 or ComplexF64 matrices; sparse UMFPACK does not provide Float32 factors"))
    N = size(A, 1)
    buffers = ntuple(_ -> Matrix{ComplexF64}(undef, N, M0), 7)
    return _FeastMixedWorkspace(A, B, Union{Nothing,_FeastMixedShift}[], fpm[10] == 1,
        buffers..., Matrix{ComplexF32}(undef,N,M0), zeros(ComplexF64,M0), zeros(M0), 0)
end

function _feast_mixed_factorize!(ws, z, point, count)
    if isempty(ws.factors)
        resize!(ws.factors, ws.store ? count : 1)
        fill!(ws.factors, nothing)
    end
    slot = ws.store ? point : 1
    ws.factors[slot] === nothing || return ws.factors[slot]
    shifted = _feast_shifted_complex(ws.A, ws.B, ComplexF64(z))
    scale = maximum(abs, shifted)
    isfinite(scale) && scale > 0 || throw(ArgumentError("Shifted matrix has zero or nonfinite scale"))
    low_matrix = Matrix{ComplexF32}(undef, size(shifted))
    @inbounds for i in eachindex(shifted)
        low_matrix[i] = shifted[i] / scale
    end
    low = try
        lu!(low_matrix)
    catch err
        err isa LinearAlgebra.SingularException || err isa LinearAlgebra.ZeroPivotException || rethrow()
        nothing
    end
    factor = _FeastMixedShift(low, nothing, scale, z)
    ws.factors[slot] = factor
    return factor
end

function _feast_mixed_prepare!(ws, Q)
    copyto!(ws.q, Q)
    mul!(ws.aq, ws.A, ws.q)
    ws.B === nothing ? copyto!(ws.bq, ws.q) : mul!(ws.bq, ws.B, ws.q)
    for j in axes(Q, 2)
        bq = view(ws.bq, :, j)
        aq = view(ws.aq, :, j)
        bn = norm(bq)
        # Least-squares Rayleigh estimate is valid for arbitrary trial vectors,
        # including independent completeness probes added by the RCI kernel.
        value = iszero(bn) ? zero(ComplexF64) : dot(bq / bn, aq / bn)
        ws.values[j] = value
        @views @. ws.residual[:,j] = ws.aq[:,j] - value * ws.bq[:,j]
        ws.scales[j] = norm(view(ws.residual,:,j))
    end
    return nothing
end

function _feast_mixed_fallback!(dest, ws, factor)
    if factor.high === nothing
        factor.high = lu!(_feast_shifted_complex(ws.A, ws.B, factor.z))
        ws.fallbacks += 1
    end
    copyto!(dest, ws.bq)
    ldiv!(factor.high, dest)
    return dest
end

function _feast_mixed_solve!(dest, ws, z, Q, point, count)
    point == 1 && _feast_mixed_prepare!(ws, Q)
    factor = _feast_mixed_factorize!(ws, z, point, count)
    try
        if factor.low === nothing || factor.high !== nothing ||
           any(j -> !isfinite(ws.values[j]) || !isfinite(ws.scales[j]) ||
               abs(z-ws.values[j]) <= sqrt(eps(Float64))*max(abs(z),abs(ws.values[j]),1.0) ||
               (iszero(norm(view(ws.bq,:,j))) && !iszero(norm(view(ws.q,:,j)))), axes(Q,2))
            return _feast_mixed_fallback!(dest, ws, factor)
        end
        for j in axes(Q,2)
            scale = ws.scales[j]
            @inbounds for i in axes(Q,1)
                ws.low_rhs[i,j] = iszero(scale) ? zero(ComplexF32) : ws.residual[i,j] / scale
            end
        end
        ldiv!(factor.low, ws.low_rhs)
        for j in axes(Q,2)
            scale = ws.scales[j] / factor.scale
            @inbounds for i in axes(Q,1)
                ws.correction[i,j] = ComplexF64(ws.low_rhs[i,j]) * scale
            end
        end
        # Reject poor low-precision correction solves using an explicitly
        # recomputed Float64 residual. Full precision is cached per bad shift.
        mul!(ws.check_a, ws.A, ws.correction)
        ws.B === nothing ? copyto!(ws.check_b, ws.correction) : mul!(ws.check_b, ws.B, ws.correction)
        @. ws.check_a = z * ws.check_b - ws.check_a - ws.residual
        for j in axes(Q,2)
            error = norm(view(ws.check_a,:,j))
            if !isfinite(error) || error > 1e-3 * ws.scales[j]
                return _feast_mixed_fallback!(dest, ws, factor)
            end
        end
        for j in axes(Q,2), i in axes(Q,1)
            dest[i,j] = (ws.q[i,j] + ws.correction[i,j]) / (z-ws.values[j])
        end
        return dest
    finally
        ws.store || (ws.factors[1] = nothing)
    end
end
