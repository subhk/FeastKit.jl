# High-level Feast interfaces for easy use
# These provide simplified interfaces to the Feast algorithms

const FEAST_PARAMETERS_LENGTH = 64

@inline function _ensure_feast_parameters(fpm::Union{Vector{Int},Nothing})
    if fpm === nothing
        params = zeros(Int, FEAST_PARAMETERS_LENGTH)
        feastinit!(params)
        return params
    end
    length(fpm) >= FEAST_PARAMETERS_LENGTH ||
        throw(ArgumentError("fpm vector must have length ≥ $(FEAST_PARAMETERS_LENGTH)"))
    return fpm
end

@inline function _normalize_parallel(parallel::Union{Bool,Symbol})
    parallel === true && return :auto
    parallel === false && return :serial
    parallel isa Symbol && return parallel
    throw(ArgumentError("Invalid parallel option: $parallel"))
end

function _materialize_matrix(A::AbstractMatrix)
    if A isa Matrix || A isa SparseMatrixCSC
        return A
    elseif A isa Symmetric
        parent(A) isa SparseMatrixCSC && return SparseMatrixCSC(A)
        return Matrix(A)
    elseif A isa Hermitian
        parent(A) isa SparseMatrixCSC && return SparseMatrixCSC(A)
        return Matrix(A)
    else
        return Matrix(A)
    end
end

@inline function _execute_feast(A, B, interval, backend, M0, fpm, comm, use_threads)
    if backend != :serial
        try
            return feast_with_backend(A, B, interval, backend, M0, fpm, comm, use_threads)
        catch e
            @debug "Parallel execution failed, falling back to serial" exception=e
        end
    end
    return _feast_run_serial(A, B, interval, M0, fpm)
end

function _feast_run_serial(A, B, interval, M0, fpm)
    try
        return feast_serial(A, B, interval, M0, fpm)
    catch e
        if e isa ArgumentError || e isa ErrorException || e isa UndefVarError
            rethrow()
        else
            throw(ErrorException(string(e)))
        end
    end
end

@inline function _execute_feast_general(A, B, center, radius, backend, M0, fpm, comm, use_threads)
    if backend != :serial
        @warn "Parallel execution for general problems is not yet available, falling back to serial"
    end
    return _feast_run_general_serial(A, B, center, radius, M0, fpm)
end

function _feast_run_general_serial(A, B, center, radius, M0, fpm)
    try
        return feast_general_serial(A, B, center, radius, M0, fpm)
    catch e
        if e isa ArgumentError || e isa ErrorException || e isa UndefVarError
            rethrow()
        else
            throw(ErrorException(string(e)))
        end
    end
end

@inline _real_component_type(::Type{Complex{T}}) where T<:Real = T
@inline _real_component_type(::Type{T}) where T<:Real = T

function _ensure_complex_matrix(A::AbstractMatrix)
    materialized = _materialize_matrix(A)
    return eltype(materialized) <: Complex ? materialized : Complex.(materialized)
end

# Main Feast interface functions
function feast(A::AbstractMatrix{T}, B::AbstractMatrix{T},
               interval::Tuple{T,T}; M0::Int = 10,
               fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Union{Bool, Symbol} = false,
               use_threads::Bool = true,
               comm = nothing) where T<:Real
    # Main Feast interface for real symmetric generalized eigenvalue problems
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))
    issymmetric(A) || throw(ArgumentError("feast expects a symmetric real matrix A; use feast_general for non-symmetric problems"))
    issymmetric(B) || throw(ArgumentError("B must be symmetric positive definite for real generalized problems"))

    feast_validate_interval(A, interval)

    params = _ensure_feast_parameters(fpm)
    N = size(A, 1)
    M0 = min(M0, N)
    backend = determine_parallel_backend(_normalize_parallel(parallel), comm)

    A_exec = _materialize_matrix(A)
    B_exec = _materialize_matrix(B)

    return _execute_feast(A_exec, B_exec, interval, backend, M0, params, comm, use_threads)
end

function feast(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}},
               interval::Tuple{T,T}; M0::Int = 10,
               fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Union{Bool, Symbol} = false,
               use_threads::Bool = true,
               comm = nothing) where T<:Real
    # Feast interface for complex Hermitian generalized eigenvalue problems
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))
    ishermitian(A) || throw(ArgumentError("feast expects a Hermitian matrix A when using real intervals; call feast_general for non-Hermitian problems"))
    ishermitian(B) || throw(ArgumentError("B must be Hermitian positive definite for complex generalized problems"))

    feast_validate_interval(A, interval)

    params = _ensure_feast_parameters(fpm)
    N = size(A, 1)
    M0 = min(M0, N)
    backend = determine_parallel_backend(_normalize_parallel(parallel), comm)

    A_exec = _materialize_matrix(A)
    B_exec = _materialize_matrix(B)

    return _execute_feast(A_exec, B_exec, interval, backend, M0, params, comm, use_threads)
end

function feast(A::AbstractMatrix{T}, interval::Tuple{T,T};
               M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Union{Bool, Symbol} = false,
               use_threads::Bool = true, comm = nothing) where T<:Real
    # Feast interface for standard real symmetric eigenvalue problems (B = I)
    N = size(A, 1)
    B = isa(A, SparseMatrixCSC) ? spdiagm(0 => fill(one(T), N)) : Matrix{T}(I, N, N)
    return feast(A, B, interval, M0=M0, fpm=fpm, parallel=parallel,
                 use_threads=use_threads, comm=comm)
end

function feast(A::AbstractMatrix{Complex{T}}, interval::Tuple{T,T};
               M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Union{Bool, Symbol} = false,
               use_threads::Bool = true, comm = nothing) where T<:Real
    # Feast interface for standard complex Hermitian eigenvalue problems (B = I)
    N = size(A, 1)
    identity_vals = fill(one(Complex{T}), N)
    B = isa(A, SparseMatrixCSC) ? spdiagm(0 => identity_vals) : Matrix{Complex{T}}(I, N, N)
    return feast(A, B, interval, M0=M0, fpm=fpm, parallel=parallel,
                 use_threads=use_threads, comm=comm)
end

function feast_general(A::AbstractMatrix, B::AbstractMatrix,
                       center::Complex{T}, radius::T; M0::Int = 10,
                       fpm::Union{Vector{Int}, Nothing} = nothing,
                       parallel::Union{Bool, Symbol} = false,
                       use_threads::Bool = true,
                       comm = nothing) where T<:Real
    # Feast interface for general (non-Hermitian) eigenvalue problems
    # Uses circular contour in complex plane

    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    size(B) == size(A) || throw(ArgumentError("B must match the size of A"))

    A_complex = _ensure_complex_matrix(A)
    B_complex = _ensure_complex_matrix(B)
    N = size(A_complex, 1)
    M0 = min(M0, N)

    params = _ensure_feast_parameters(fpm)
    backend = determine_parallel_backend(_normalize_parallel(parallel), comm)

    real_type = promote_type(_real_component_type(eltype(A_complex)),
                             _real_component_type(eltype(B_complex)),
                             _real_component_type(typeof(center)),
                             T)
    complex_type = Complex{real_type}

    A_exec = complex_type.(A_complex)
    B_exec = complex_type.(B_complex)

    center_exec = complex_type(center)
    radius_exec = convert(real_type, radius)
    radius_exec > zero(real_type) ||
        throw(ArgumentError("Radius must be positive, got $radius"))

    return _execute_feast_general(A_exec, B_exec, center_exec, radius_exec,
                                  backend, M0, params, comm, use_threads)
end

function feast_general(A::AbstractMatrix, B::AbstractMatrix,
                       center::Complex{Tc}, radius::Tr; kwargs...) where {Tc<:Real, Tr<:Real}
    T = promote_type(Tc, Tr)
    center_promoted = Complex{T}(center)
    radius_promoted = convert(T, radius)
    return feast_general(A, B, center_promoted, radius_promoted; kwargs...)
end

function feast_general(A::AbstractMatrix, center::Complex{T}, radius::T;
                       M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing,
                       parallel::Union{Bool, Symbol} = false,
                       use_threads::Bool = true,
                       comm = nothing) where T<:Real
    # Feast interface for standard general eigenvalue problems (B = I)

    N = size(A, 1)
    if isa(A, SparseMatrixCSC)
        B = spdiagm(0 => ones(eltype(A), N))
    else
        B = Matrix{eltype(A)}(I, N, N)
    end
    M0 = min(M0, N)
    return feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                         parallel=parallel, use_threads=use_threads, comm=comm)
end

function feast_general(A::AbstractMatrix, center::Complex{Tc}, radius::Tr; kwargs...) where {Tc<:Real, Tr<:Real}
    T = promote_type(Tc, Tr)
    center_promoted = Complex{T}(center)
    radius_promoted = convert(T, radius)
    return feast_general(A, center_promoted, radius_promoted; kwargs...)
end

function feast_banded(A::Matrix{T}, kla::Int, interval::Tuple{T,T};
                     B::Union{Matrix{T}, Nothing} = nothing, klb::Int = 0,
                     M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # Feast interface for real symmetric banded matrices

    Emin, Emax = interval

    # Initialize Feast parameters if not provided
    params = _ensure_feast_parameters(fpm)

    if B === nothing
        # Standard eigenvalue problem - create identity in banded format
        N = size(A, 2)
        B_banded = zeros(T, 1, N)
        B_banded[1, :] .= one(T)
        return feast_sbgv!(copy(A), B_banded, kla, 0, Emin, Emax, M0, params)
    else
        # Generalized eigenvalue problem
        return feast_sbgv!(copy(A), copy(B), kla, klb, Emin, Emax, M0, params)
    end
end

function feast_banded(A::Matrix{Complex{T}}, kla::Int, interval::Tuple{T,T};
                     B::Union{Matrix{Complex{T}}, Nothing} = nothing, klb::Int = 0,
                     M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # Feast interface for complex Hermitian banded matrices

    Emin, Emax = interval

    # Initialize Feast parameters if not provided
    params = _ensure_feast_parameters(fpm)

    if B === nothing
        # Standard eigenvalue problem
        return feast_hbev!(copy(A), kla, Emin, Emax, M0, params)
    else
        # Generalized eigenvalue problem
        return feast_hbgv!(copy(A), copy(B), kla, klb, Emin, Emax, M0, params)
    end
end

# Convenience functions with different interfaces
function eigvals_feast(A::AbstractMatrix, interval::Tuple; kwargs...)
    # Return only eigenvalues
    result = feast(A, interval; kwargs...)
    return result.lambda
end

function eigen_feast(A::AbstractMatrix, interval::Tuple; kwargs...)
    # Return eigenvalues and eigenvectors as Eigen object
    result = feast(A, interval; kwargs...)
    return Eigen(result.lambda, result.q)
end

function eigvals_feast(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple; kwargs...)
    # Return only eigenvalues for generalized problem
    result = feast(A, B, interval; kwargs...)
    return result.lambda
end

function eigen_feast(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple; kwargs...)
    # Return eigenvalues and eigenvectors for generalized problem
    result = feast(A, B, interval; kwargs...)
    return Eigen(result.lambda, result.q)
end

# Polynomial eigenvalue problems
function feast_polynomial(coeffs::Vector{<:AbstractMatrix{Complex{T}}},
                         center::Complex{T}, radius::T; M0::Int = 10,
                         fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # Feast for polynomial eigenvalue problems
    # P(λ) = coeffs[1] + λ*coeffs[2] + λ²*coeffs[3] + ...
    
    # Initialize Feast parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    d = length(coeffs) - 1  # Degree of polynomial
    return feast_pep!(coeffs, d, center, radius, M0, fpm)
end

# Matrix-free interfaces
function feast_matvec(A_mul!::Function, B_mul!::Function, N::Int, 
                     interval::Tuple{T,T}; M0::Int = 10,
                     fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # Feast with matrix-free operations
    # A_mul!(y, x) computes y = A*x
    # B_mul!(y, x) computes y = B*x
    
    Emin, Emax = interval
    
    # Initialize Feast parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    return feast_sparse_matvec!(A_mul!, B_mul!, N, Emin, Emax, M0, fpm)
end

# Advanced configuration functions
function feast_set_defaults!(fpm::Vector{Int}; 
                            print_level::Int = 1,
                            integration_points::Int = 8,
                            tolerance_exp::Int = 12,
                            max_refinement::Int = 20)
    # Set common Feast parameters with user-friendly names
    
    fpm[1] = print_level
    fpm[2] = integration_points  
    fpm[3] = tolerance_exp
    fpm[4] = max_refinement
    
    return fpm
end

function feast_custom_contour(nodes::Vector{Complex{T}}, 
                             A::AbstractMatrix, B::AbstractMatrix;
                             M0::Int = 10,
                             fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # Feast with custom integration contour
    
    # Initialize Feast parameters if not provided  
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Set up custom contour
    contour = feast_customcontour(nodes, fpm)
    
    # This would require modifications to the RCI routines to accept custom contour
    # For now, throw an error indicating it's not fully implemented
    throw(ArgumentError("Custom contour interface not fully implemented yet"))
end

# Utility functions for result analysis
function feast_summary(io::IO, result::FeastResult)
    # Print summary of Feast results to the provided IO
    println(io, "FeastKit Eigenvalue Solution Summary")
    println(io, "="^40)
    println(io, "Eigenvalues found: ", result.M)
    println(io, "Final residual: ", result.epsout)
    println(io, "Refinement loops: ", result.loop)
    println(io, "Exit status: ", result.info == 0 ? "Success" : "Error $(result.info)")
    if result.M > 0
        println(io, "\nEigenvalues:")
        for i in 1:result.M
            println(io, "  λ[$i] = ", result.lambda[i], "  (residual: ", result.res[i], ")")
        end
    end
    return nothing
end

function feast_summary(result::FeastResult)
    feast_summary(stdout, result)
end

# Compatibility shim: allow `redirect_stdout(io::IOBuffer) do ... end` with IOBuffer
# Remove previous IOBuffer redirection shim; tests now use IO-based summary.

function feast_validate_interval(A::AbstractMatrix{T}, interval::Tuple{T,T}) where T<:Real
    # Validate that the search interval makes sense

    Emin, Emax = interval

    if Emin >= Emax
        throw(ArgumentError("Invalid interval: Emin must be less than Emax"))
    end

    # Rough estimate of eigenvalue bounds using Gershgorin circles
    N = size(A, 1)
    min_est = typemax(T)
    max_est = typemin(T)

    for i in 1:N
        center = real(A[i, i])
        radius = sum(abs(A[i, j]) for j in 1:N if j != i)
        min_est = min(min_est, center - radius)
        max_est = max(max_est, center + radius)
    end

    if Emax < min_est || Emin > max_est
        @warn "Search interval [$Emin, $Emax] may not contain eigenvalues. " *
              "Estimated eigenvalue range: [$(min_est), $(max_est)]"
    end

    return (min_est, max_est)
end

function feast_validate_interval(A::AbstractMatrix{Complex{T}}, interval::Tuple{T,T}) where T<:Real
    # Validate that the search interval makes sense for Hermitian matrices
    # Eigenvalues of Hermitian matrices are real, so Gershgorin bounds apply

    Emin, Emax = interval

    if Emin >= Emax
        throw(ArgumentError("Invalid interval: Emin must be less than Emax"))
    end

    # Rough estimate of eigenvalue bounds using Gershgorin circles
    # For Hermitian matrices, diagonal elements are real (or should be)
    N = size(A, 1)
    min_est = typemax(T)
    max_est = typemin(T)

    for i in 1:N
        center = real(A[i, i])  # Diagonal of Hermitian matrix is real
        radius = sum(abs(A[i, j]) for j in 1:N if j != i)
        min_est = min(min_est, center - radius)
        max_est = max(max_est, center + radius)
    end

    if Emax < min_est || Emin > max_est
        @warn "Search interval [$Emin, $Emax] may not contain eigenvalues. " *
              "Estimated eigenvalue range: [$(min_est), $(max_est)]"
    end

    return (min_est, max_est)
end
