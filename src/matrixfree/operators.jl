"""
    MatrixFreeOperator{T}

Abstract type for matrix-free operators.
"""
abstract type MatrixFreeOperator{T} end

"""
    MatrixVecFunction{T}

Matrix-free operator defined by a matrix-vector multiplication function.

# Fields
- `mul!`: Function with signature `mul!(y, op, x)` that computes `y = op * x`
- `size`: Size of the operator as `(m, n)`
- `issymmetric`: Whether the operator is symmetric
- `ishermitian`: Whether the operator is Hermitian
- `isposdef`: Whether the operator is positive definite
"""
struct MatrixVecFunction{T,F} <: MatrixFreeOperator{T}
    mul!::F
    size::Tuple{Int, Int}
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
end

function MatrixVecFunction{T}(mul!::F, size::Tuple{Int, Int};
                              issymmetric::Bool = false,
                              ishermitian::Bool = false,
                              isposdef::Bool = false) where {T,F}
    return MatrixVecFunction{T,F}(mul!, size, issymmetric, ishermitian, isposdef)
end

# Convenience constructors
MatrixVecFunction(mul!::F, size::Tuple{Int, Int}; kwargs...) where F =
    MatrixVecFunction{Float64}(mul!, size; kwargs...)

"""
    LinearOperator{T}

Matrix-free operator that supports multiple operations.

# Fields
- `A_mul!`: Function `(y, x) -> y = A*x`
- `At_mul!`: Function `(y, x) -> y = A'*x` (optional)
- `Ac_mul!`: Function `(y, x) -> y = A†*x` (optional)
- `solve!`: Function `(y, z, x) -> y = (z*I - A)\\x` (linear solver)
- `size`: Operator size
- `issymmetric`, `ishermitian`, `isposdef`: Properties
"""
struct LinearOperator{T,FA,FT,FC,FS} <: MatrixFreeOperator{T}
    A_mul!::FA
    At_mul!::FT
    Ac_mul!::FC
    solve!::FS
    size::Tuple{Int, Int}
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
end

function LinearOperator{T}(A_mul!::FA, size::Tuple{Int, Int};
                           At_mul!::FT = nothing,
                           Ac_mul!::FC = nothing,
                           solve!::FS = nothing,
                           issymmetric::Bool = false,
                           ishermitian::Bool = false,
                           isposdef::Bool = false) where {T,FA,FT,FC,FS}
    return LinearOperator{T,FA,FT,FC,FS}(A_mul!, At_mul!, Ac_mul!, solve!, size,
                                        issymmetric, ishermitian, isposdef)
end

LinearOperator(A_mul!::FA, size::Tuple{Int, Int}; kwargs...) where FA =
    LinearOperator{Float64}(A_mul!, size; kwargs...)

"""
    FeastLinearOperator

Alias of FeastKit's [`LinearOperator`](@ref). LinearOperators.jl, commonly
loaded alongside Krylov.jl, also exports a `LinearOperator`; with both packages
loaded the unqualified name becomes ambiguous, and `FeastLinearOperator`
(or `FeastKit.LinearOperator`) refers to FeastKit's type unambiguously.
"""
const FeastLinearOperator = LinearOperator

# Interface functions
Base.size(op::MatrixFreeOperator) = op.size
Base.size(op::MatrixFreeOperator, dim::Int) = op.size[dim]
LinearAlgebra.issymmetric(op::MatrixFreeOperator) = op.issymmetric
LinearAlgebra.ishermitian(op::MatrixFreeOperator) = op.ishermitian
LinearAlgebra.isposdef(op::MatrixFreeOperator) = op.isposdef
Base.eltype(::MatrixFreeOperator{T}) where T = T
Base.eltype(::Type{<:MatrixFreeOperator{T}}) where T = T

# Matrix-vector multiplication
function LinearAlgebra.mul!(y::AbstractVector, op::MatrixVecFunction, x::AbstractVector)
    op.mul!(y, op, x)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::LinearOperator, x::AbstractVector)
    op.A_mul!(y, x)
    return y
end

# Transpose multiplication
function LinearAlgebra.mul!(y::AbstractVector,
                           At::LinearAlgebra.Transpose{T, <:LinearOperator{T}},
                           x::AbstractVector) where T
    op = At.parent
    if op.At_mul! !== nothing
        op.At_mul!(y, x)
    elseif op.issymmetric
        op.A_mul!(y, x)
    else
        throw(ArgumentError("Transpose not available for this operator"))
    end
    return y
end

# Adjoint multiplication
function LinearAlgebra.mul!(y::AbstractVector,
                           Ac::LinearAlgebra.Adjoint{T, <:LinearOperator{T}},
                           x::AbstractVector) where T
    op = Ac.parent
    if op.Ac_mul! !== nothing
        op.Ac_mul!(y, x)
    elseif op.ishermitian
        op.A_mul!(y, x)
    elseif op.issymmetric && T <: Real
        op.A_mul!(y, x)
    else
        throw(ArgumentError("Adjoint not available for this operator"))
    end
    return y
end
