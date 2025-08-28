# Feast type definitions and structures
# Translated from Feast Fortran library

# Feast parameter structure
struct FeastParameters
    fpm::Vector{Int}
    
    function FeastParameters()
        fpm = zeros(Int, 64)
        new(fpm)
    end
    
    function FeastParameters(fpm::Vector{Int})
        length(fpm) >= 64 || error("fpm array must have at least 64 elements")
        new(fpm)
    end
end

# Feast workspace structure for real symmetric problems
mutable struct FeastWorkspaceReal{T<:Real}
    N::Int
    M0::Int
    work::Matrix{T}
    workc::Matrix{Complex{T}}
    Aq::Matrix{T}
    Sq::Matrix{T}
    lambda::Vector{T}
    q::Matrix{T}
    res::Vector{T}
    
    function FeastWorkspaceReal{T}(N::Int, M0::Int) where T<:Real
        work = zeros(T, N, M0)
        workc = zeros(Complex{T}, N, M0)
        Aq = zeros(T, M0, M0)
        Sq = zeros(T, M0, M0)
        lambda = zeros(T, M0)
        q = zeros(T, N, M0)
        res = zeros(T, M0)
        new(N, M0, work, workc, Aq, Sq, lambda, q, res)
    end
end

# Feast workspace structure for complex Hermitian problems
mutable struct FeastWorkspaceComplex{T<:Real}
    N::Int
    M0::Int
    work::Matrix{T}
    workc::Matrix{Complex{T}}
    zAq::Matrix{Complex{T}}
    zSq::Matrix{Complex{T}}
    lambda::Vector{T}
    q::Matrix{Complex{T}}
    res::Vector{T}
    
    function FeastWorkspaceComplex{T}(N::Int, M0::Int) where T<:Real
        work = zeros(T, N, M0)
        workc = zeros(Complex{T}, N, M0)
        zAq = zeros(Complex{T}, M0, M0)
        zSq = zeros(Complex{T}, M0, M0)
        lambda = zeros(T, M0)
        q = zeros(Complex{T}, N, M0)
        res = zeros(T, M0)
        new(N, M0, work, workc, zAq, zSq, lambda, q, res)
    end
end

# Feast result structure
struct FeastResult{T<:Real, VT}
    lambda::Vector{T}
    q::Matrix{VT}
    M::Int
    res::Vector{T}
    info::Int
    epsout::T
    loop::Int
end

# Integration contour structure
struct FeastContour{T<:Real}
    Zne::Vector{Complex{T}}
    Wne::Vector{Complex{T}}
    
    function FeastContour{T}(nodes::Vector{Complex{T}}, weights::Vector{Complex{T}}) where T<:Real
        length(nodes) == length(weights) || error("Number of nodes and weights must match")
        new(nodes, weights)
    end
end

# Feast RCI job identifiers
@enum FeastRCIJob begin
    Feast_RCI_INIT = -1
    Feast_RCI_DONE = 0
    Feast_RCI_FACTORIZE = 10
    Feast_RCI_SOLVE = 20
    Feast_RCI_SOLVE_TRANSPOSE = 21
    Feast_RCI_MULT_A = 30
    Feast_RCI_MULT_B = 40
end

# Error codes
@enum FeastError begin
    Feast_SUCCESS = 0
    Feast_ERROR_N = 1
    Feast_ERROR_M0 = 2
    Feast_ERROR_EMIN_EMAX = 3
    Feast_ERROR_EMID_R = 4
    Feast_ERROR_NO_CONVERGENCE = 5
    Feast_ERROR_MEMORY = 6
    Feast_ERROR_INTERNAL = 7
    Feast_ERROR_LAPACK = 8
    Feast_ERROR_FPM = 9
end

# Provide `.value` property for enum constants as expected by tests
Base.getproperty(x::FeastError, s::Symbol) = s === :value ? Int(x) : getfield(x, s)
