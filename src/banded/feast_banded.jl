# Feast banded matrix routines
# Translated from dzfeast_banded.f90


function feast_sbgv!(A::Matrix{T}, B::Matrix{T}, kla::Int, klb::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for banded real symmetric generalized eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are symmetric banded matrices
    # kla, klb are the number of super-diagonals of A and B respectively

    N = size(A, 2)  # For banded storage, second dimension is the matrix size

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    # Validate banded matrix dimensions
    size(A, 1) >= kla + 1 || throw(ArgumentError("A matrix storage insufficient for kla"))
    size(B, 1) >= klb + 1 || throw(ArgumentError("B matrix storage insufficient for klb"))

    # Initialize workspace
    workspace = FeastWorkspaceReal{T}(N, M0)

    # Initialize variables for RCI
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)

    # Banded linear solver workspace
    kl = max(kla, klb)
    ku = kl
    ldab = 2 * kl + ku + 1
    banded_factors = Matrix{Complex{T}}(undef, ldab, N)
    banded_ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
    factorized = false

    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.Aq, workspace.Sq, fpm, epsout, loop,
                    Emin, Emax, M0, workspace.lambda, workspace.q,
                    mode, workspace.res, info)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            factorized = false
            z = Ze[]
            fill_shifted_banded!(banded_factors, A, B, kla, klb, kl, z)
            _, banded_ipiv_tmp, info_lapack = LinearAlgebra.LAPACK.gbtrf!(kl, ku, N, banded_factors)
            if info_lapack != 0
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            banded_ipiv = banded_ipiv_tmp
            factorized = true

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if !factorized
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end

            for col in 1:M0
                symmetric_banded_matvec!(view(workspace.workc, :, col), B, klb, view(workspace.work, :, col))
            end

            info_lapack = LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, banded_factors, banded_ipiv, view(workspace.workc, :, 1:M0))
            if info_lapack != 0
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            M = mode[]
            for col in 1:M
                symmetric_banded_matvec!(view(workspace.work, :, col), A, kla, view(workspace.q, :, col))
            end

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end

    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]

    return FeastResult{T, T}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_sbgvx!(A::Matrix{T}, B::Matrix{T}, kla::Int, klb::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbgv!(A, B, kla, klb, Emin, Emax, M0, fpm)
    end
end

@inline function symmetric_banded_get(A::Matrix{T}, k::Int, i::Int, j::Int) where T
    abs(i - j) > k && return zero(T)
    if i <= j
        row = k + 1 + i - j
        return A[row, j]
    else
        row = k + 1 + j - i
        return A[row, i]
    end
end

function fill_shifted_banded!(dest::Matrix{Complex{T}}, A::Matrix{T}, B::Matrix{T},
                              kla::Int, klb::Int, kl::Int, z::Complex{T}) where T<:Real
    fill!(dest, zero(Complex{T}))
    N = size(dest, 2)
    ku = kl
    offset = ku + 1
    zero_T = zero(T)

    for j in 1:N
        imin = max(1, j - kl)
        imax = min(N, j + ku)
        for i in imin:imax
            a_val = (abs(i - j) <= kla) ? symmetric_banded_get(A, kla, i, j) : zero_T
            b_val = (abs(i - j) <= klb) ? symmetric_banded_get(B, klb, i, j) : zero_T
            dest[offset + i - j, j] = z * b_val - a_val
        end
    end

    return dest
end

function symmetric_banded_matvec!(y::AbstractVector{S}, A::Matrix{T}, k::Int, x::AbstractVector{T}) where {S,T}
    fill!(y, zero(S))
    N = length(x)

    for j in 1:N
        xj = x[j]
        imin = max(1, j - k)
        for i in imin:j
            row = k + 1 + i - j
            val = A[row, j]
            y[i] += convert(S, val * xj)
            if i != j
                y[j] += convert(S, val * x[i])
            end
        end
    end

    return y
end

function feast_hbev!(A::Matrix{Complex{T}}, ka::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for banded complex Hermitian eigenvalue problem
    # Solves: A*q = lambda*q where A is Hermitian banded
    
    N = size(A, 2)
    
    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Validate banded matrix dimensions
    size(A, 1) >= ka + 1 || throw(ArgumentError("A matrix storage insufficient for ka"))
    
    # Initialize workspace
    workspace = FeastWorkspaceComplex{T}(N, M0)
    
    # Initialize variables for RCI
    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    
    # Banded linear solver workspace
    banded_factors = nothing
    
    while true
        # Call Feast RCI kernel
        feast_hrci!(ijob, N, Ze, workspace.work, workspace.workc,
                   workspace.zAq, workspace.zSq, fpm, epsout, loop,
                   Emin, Emax, M0, workspace.lambda, workspace.q, 
                   mode, workspace.res, info)
        
        if ijob[] == Int(Feast_RCI_FACTORIZE)
            # Factorize Ze*I - A for banded Hermitian matrix
            z = Ze[]
            
            # Convert to full matrix (simplified approach)
            A_full = banded_to_full_hermitian(A, ka, N)
            full_matrix = z .* I .- A_full
            
            # LU factorization
            try
                banded_factors = lu(full_matrix)
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_SOLVE)
            # Solve linear systems
            try
                workspace.workc[:, 1:M0] .= banded_factors \ workspace.workc[:, 1:M0]
            catch e
                info[] = Int(Feast_ERROR_LAPACK)
                break
            end
            
        elseif ijob[] == Int(Feast_RCI_MULT_A)
            # Compute A * q for residual calculation
            M = mode[]
            A_full = banded_to_full_hermitian(A, ka, N)
            workspace.work[:, 1:M] .= real.(A_full * workspace.q[:, 1:M])

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            # Unexpected ijob value - error out to prevent infinite loop
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_A($(Int(Feast_RCI_MULT_A))), DONE($(Int(Feast_RCI_DONE)))")
        end
    end
    
    # Extract results
    M = mode[]
    lambda = workspace.lambda[1:M]
    q = workspace.q[:, 1:M]
    res = workspace.res[1:M]
    
    return FeastResult{T, Complex{T}}(lambda, q, M, res, info[], epsout[], loop[])
end

function feast_hbevx!(A::Matrix{Complex{T}}, ka::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hbev!(A, ka, Emin, Emax, M0, fpm)
    end
end

function feast_hbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    N = size(A, 2)
    size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    A_full = banded_to_full_hermitian(A, ka, N)
    B_full = banded_to_full_hermitian(B, kb, N)
    return feast_hegv!(A_full, B_full, Emin, Emax, M0, fpm)
end

function feast_hbgvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hbgv!(A, B, ka, kb, Emin, Emax, M0, fpm)
    end
end

# Helper functions for banded matrix operations

function banded_to_full(A_banded::Matrix{T}, k::Int, N::Int) where T
    # Convert banded matrix to full format
    # A_banded is stored in LAPACK banded format
    
    A_full = zeros(T, N, N)
    
    # Fill the banded matrix
    for j in 1:N
        for i in max(1, j-k):min(N, j+k)
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                A_full[i, j] = A_banded[row_index, j]
            end
        end
    end
    
    return A_full
end

function banded_to_full_hermitian(A_banded::Matrix{Complex{T}}, k::Int, N::Int) where T
    # Convert Hermitian banded matrix to full format
    
    A_full = zeros(Complex{T}, N, N)
    
    # Fill upper triangle from banded storage
    for j in 1:N
        for i in max(1, j-k):j
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                A_full[i, j] = A_banded[row_index, j]
            end
        end
    end
    
    # Fill lower triangle using Hermitian property
    for j in 1:N
        for i in j+1:min(N, j+k)
            A_full[i, j] = conj(A_full[j, i])
        end
    end
    
    return A_full
end

function full_to_banded(A_full::Matrix{T}, k::Int) where T
    # Convert full matrix to banded format
    
    N = size(A_full, 1)
    A_banded = zeros(T, k+1, N)
    
    for j in 1:N
        for i in max(1, j-k):min(N, j+k)
            if i <= j  # Upper triangle for symmetric case
                row_index = k + 1 + i - j
                A_banded[row_index, j] = A_full[i, j]
            end
        end
    end
    
    return A_banded
end

function banded_matvec!(y::Vector{T}, A_banded::Matrix{T}, k::Int, x::Vector{T}) where T
    # Efficient matrix-vector multiplication for banded matrices
    # y = A_banded * x
    
    N = length(x)
    fill!(y, zero(T))
    
    for j in 1:N
        for i in max(1, j-k):min(N, j+k)
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                y[i] += A_banded[row_index, j] * x[j]
            end
        end
    end
    
    return y
end

function banded_hermitian_matvec!(y::Vector{Complex{T}}, A_banded::Matrix{Complex{T}}, 
                                 k::Int, x::Vector{Complex{T}}) where T
    # Efficient matrix-vector multiplication for Hermitian banded matrices
    
    N = length(x)
    fill!(y, zero(Complex{T}))
    
    # Upper triangle contribution
    for j in 1:N
        for i in max(1, j-k):j
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                val = A_banded[row_index, j]
                y[i] += val * x[j]
                if i != j
                    y[j] += conj(val) * x[i]
                end
            end
        end
    end
    
    return y
end

# Banded matrix information
function feast_banded_info(A_banded::Matrix{T}, k::Int, N::Int) where T
    # Print information about banded matrix
    
    total_elements = N * N
    stored_elements = size(A_banded, 1) * size(A_banded, 2)
    bandwidth = 2 * k + 1
    
    println("Banded Matrix Information:")
    println("  Size: $(N) x $(N)")
    println("  Bandwidth: $(bandwidth)")
    println("  Super-diagonals: $(k)")
    println("  Stored elements: $(stored_elements)")
    # Use Printf for formatted percentage
    pct = stored_elements / total_elements * 100
    println("  Storage efficiency: ", Printf.@sprintf("%.1f", pct), "%")

    return (N, bandwidth, stored_elements)
end

# Standard eigenvalue problem variants (B = I)

function feast_sbev!(A::Matrix{T}, ka::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int}) where T<:Real
    # Feast for banded real symmetric standard eigenvalue problem
    # Solves: A*q = lambda*q where A is symmetric banded
    # This is equivalent to feast_sbgv! with B = I

    N = size(A, 2)

    # Create identity matrix in banded format
    # For identity, we only need the diagonal, so k=0
    B = zeros(T, 1, N)
    B[1, :] .= one(T)
    klb = 0

    # Call generalized version with B = I
    return feast_sbgv!(A, B, ka, klb, Emin, Emax, M0, fpm)
end

function feast_sbevx!(A::Matrix{T}, ka::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbev!(A, ka, Emin, Emax, M0, fpm)
    end
end

function feast_gbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    N = size(A, 2)
    size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))
    check_feast_grci_input(N, M0, Emid, r, fpm)
    A_full = banded_to_full(A, ka, N)
    B_full = banded_to_full(B, kb, N)
    return feast_gegv!(A_full, B_full, Emid, r, M0, fpm)
end

function feast_gbgvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gbgv!(A, B, ka, kb, Emid, r, M0, fpm)
    end
end

function feast_gbev!(A::Matrix{Complex{T}}, ka::Int,
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int}) where T<:Real
    N = size(A, 2)
    check_feast_grci_input(N, M0, Emid, r, fpm)
    A_full = banded_to_full(A, ka, N)
    return feast_geev!(A_full, Emid, r, M0, fpm)
end

function feast_gbevx!(A::Matrix{Complex{T}}, ka::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gbev!(A, ka, Emid, r, M0, fpm)
    end
end
