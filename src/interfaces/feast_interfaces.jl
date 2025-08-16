# High-level FEAST interfaces for easy use
# These provide simplified interfaces to the FEAST algorithms

# Main FEAST interface functions
function feast(A::AbstractMatrix{T}, B::AbstractMatrix{T}, 
               interval::Tuple{T,T}; M0::Int = 10, 
               fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Bool = false,
               use_threads::Bool = true) where T<:Real
    # Main FEAST interface for generalized eigenvalue problems
    # Automatically detects matrix type and calls appropriate solver
    
    Emin, Emax = interval
    
    # Initialize FEAST parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Use parallel version if requested and available
    if parallel && (Threads.nthreads() > 1 || nworkers() > 1)
        return feast_parallel(A, B, interval, M0=M0, fpm=fpm, use_threads=use_threads)
    end
    
    # Detect matrix structure and call appropriate solver
    if isa(A, Matrix) && isa(B, Matrix)
        # Dense matrices
        if T <: Real
            return feast_sygv!(copy(A), copy(B), Emin, Emax, M0, fpm)
        else
            return feast_gegv!(A, B, Complex(Emin), Emax - Emin, M0, fpm)
        end
    elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        # Sparse matrices
        return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
    else
        throw(ArgumentError("Unsupported matrix types: $(typeof(A)), $(typeof(B))"))
    end
end

function feast(A::AbstractMatrix{T}, interval::Tuple{T,T}; 
               M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing,
               parallel::Bool = false, use_threads::Bool = true) where T<:Real
    # FEAST interface for standard eigenvalue problems (B = I)
    
    Emin, Emax = interval
    N = size(A, 1)
    
    # Initialize FEAST parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Create identity matrix of appropriate type
    if isa(A, Matrix)
        if T <: Real
            B = Matrix{T}(I, N, N)
            if parallel && (Threads.nthreads() > 1 || nworkers() > 1)
                return feast_parallel(A, B, interval, M0=M0, fpm=fpm, use_threads=use_threads)
            else
                return feast_sygv!(copy(A), B, Emin, Emax, M0, fpm)
            end
        else
            return feast_heev!(copy(A), Emin, Emax, M0, fpm)
        end
    elseif isa(A, SparseMatrixCSC)
        B = sparse(I, N, N)
        if parallel && (Threads.nthreads() > 1 || nworkers() > 1)
            return feast_parallel(A, B, interval, M0=M0, fpm=fpm, use_threads=use_threads)
        else
            return feast_scsrgv!(A, B, Emin, Emax, M0, fpm)
        end
    else
        throw(ArgumentError("Unsupported matrix type: $(typeof(A))"))
    end
end

function feast_general(A::AbstractMatrix{Complex{T}}, B::AbstractMatrix{Complex{T}},
                      center::Complex{T}, radius::T; M0::Int = 10,
                      fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # FEAST interface for general (non-Hermitian) eigenvalue problems
    # Uses circular contour in complex plane
    
    # Initialize FEAST parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    if isa(A, Matrix) && isa(B, Matrix)
        return feast_gegv!(copy(A), copy(B), center, radius, M0, fpm)
    elseif isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return feast_gcsrgv!(A, B, center, radius, M0, fpm)
    else
        throw(ArgumentError("Unsupported matrix types: $(typeof(A)), $(typeof(B))"))
    end
end

function feast_banded(A::Matrix{T}, kla::Int, interval::Tuple{T,T};
                     B::Union{Matrix{T}, Nothing} = nothing, klb::Int = 0,
                     M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing) where T<:Real
    # FEAST interface for banded matrices
    
    Emin, Emax = interval
    
    # Initialize FEAST parameters if not provided
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    if B === nothing
        # Standard eigenvalue problem
        if T <: Real
            # Create identity in banded format
            N = size(A, 2)
            B_banded = zeros(T, 1, N)
            B_banded[1, :] .= one(T)
            return feast_sbgv!(copy(A), B_banded, kla, 0, Emin, Emax, M0, fpm)
        else
            return feast_hbev!(copy(A), kla, Emin, Emax, M0, fpm)
        end
    else
        # Generalized eigenvalue problem
        return feast_sbgv!(copy(A), copy(B), kla, klb, Emin, Emax, M0, fpm)
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
    # FEAST for polynomial eigenvalue problems
    # P(λ) = coeffs[1] + λ*coeffs[2] + λ²*coeffs[3] + ...
    
    # Initialize FEAST parameters if not provided
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
    # FEAST with matrix-free operations
    # A_mul!(y, x) computes y = A*x
    # B_mul!(y, x) computes y = B*x
    
    Emin, Emax = interval
    
    # Initialize FEAST parameters if not provided
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
    # Set common FEAST parameters with user-friendly names
    
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
    # FEAST with custom integration contour
    
    # Initialize FEAST parameters if not provided  
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
function feast_summary(result::FeastResult)
    # Print summary of FEAST results
    
    println("FEAST Eigenvalue Solution Summary")
    println("="^40)
    println("Eigenvalues found: ", result.M)
    println("Final residual: ", result.epsout)
    println("Refinement loops: ", result.loop)
    println("Exit status: ", result.info == 0 ? "Success" : "Error $(result.info)")
    
    if result.M > 0
        println("\nEigenvalues:")
        for i in 1:result.M
            println("  λ[$i] = ", result.lambda[i], "  (residual: ", result.res[i], ")")
        end
    end
    
    return nothing
end

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