# Feast banded matrix routines
# Translated from dzfeast_banded.f90
#
# Banded arrays use LAPACK-style column storage. Symmetric/Hermitian matrices
# store the diagonal at row k + 1 and upper bands above it; general shifted
# systems are expanded into gbtrf!/gbtrs! storage before factorization.


function feast_sbgv!(A::Matrix{T}, B::Matrix{T}, kla::Int, klb::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for banded real symmetric generalized eigenvalue problem
    # Solves: A*q = lambda*B*q where A and B are symmetric banded matrices
    # kla, klb are the number of super-diagonals of A and B respectively

    N = size(A, 2)  # For banded storage, second dimension is the matrix size

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    # Validate banded matrix dimensions
    size(A, 1) >= kla + 1 || throw(ArgumentError("A matrix storage insufficient for kla"))
    size(B, 1) >= klb + 1 || throw(ArgumentError("B matrix storage insufficient for klb"))

    # Direct solves cache one factorization per contour point. Iterative solves
    # reuse the same shifted matvec closure and update only the current shift.
    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative banded FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

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
    banded_factor_cache = Matrix{Complex{T}}[]
    banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, 0)
    banded_factorized = falses(0)
    if solver_is_direct
        contour = feast_get_custom_contour(T, fpm)
        contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
        banded_factor_cache = [Matrix{Complex{T}}(undef, ldab, N) for _ in eachindex(contour.Zne)]
        banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, length(contour.Zne))
        banded_factorized = falses(length(contour.Zne))
    end
    factorized = false
    current_factor_idx = 0
    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)
    tmpAx = zeros(Complex{T}, N)
    tmpBx = zeros(Complex{T}, N)
    current_shift = Ref(zero(Complex{T}))

    # GMRES sees a dense-looking linear operator, but each multiply touches only
    # the stored band entries of A and B.
    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        symmetric_banded_matvec!(tmpBx, B, klb, x)
        symmetric_banded_matvec!(tmpAx, A, kla, x)
        @. y = current_shift[] * tmpBx - tmpAx
        return y
    end

    # Persistent RCI state (must be reused across calls in the loop)
    srci_state = FeastSRCIState{T}()

    while true
        # Call Feast RCI kernel
        feast_srci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.Aq, workspace.Sq, fpm, epsout, loop,
                    Emin, Emax, M0, workspace.lambda, workspace.q,
                    mode, workspace.res, info; state=srci_state)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            factorized = false
            z = Ze[]
            if solver_is_direct
                factor_idx = fpm[50]
                if !(1 <= factor_idx <= length(banded_factor_cache))
                    info[] = Int(Feast_ERROR_INTERNAL)
                    break
                end
                shifted_factor = banded_factor_cache[factor_idx]
                try
                    if !banded_factorized[factor_idx]
                        fill_shifted_banded!(shifted_factor, A, B, kla, klb, kl, z)
                        # gbtrf! returns (AB, ipiv), not (AB, ipiv, info)
                        _, banded_ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, N,
                                                                     shifted_factor)
                        banded_ipiv_cache[factor_idx] = banded_ipiv
                        banded_factorized[factor_idx] = true
                    end
                    current_factor_idx = factor_idx
                    factorized = true
                catch e
                    if isa(e, LinearAlgebra.SingularException) || isa(e, LinearAlgebra.LAPACKException)
                        info[] = Int(Feast_ERROR_LAPACK)
                        break
                    end
                    rethrow(e)
                end
            else
                current_shift[] = z
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if solver_is_direct && !factorized
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end

            rhs_block = solver_is_direct ? view(workspace.workc, :, 1:M0) : view(rhs_buffer, :, 1:M0)
            for col in 1:M0
                symmetric_banded_matvec!(view(rhs_block, :, col), B, klb,
                                         view(workspace.work, :, col))
            end

            if solver_is_direct
                try
                    # gbtrs! requires m parameter and returns the solution matrix (not info code)
                    LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, N,
                                                banded_factor_cache[current_factor_idx],
                                                banded_ipiv_cache[current_factor_idx],
                                                view(workspace.workc, :, 1:M0))
                catch e
                    if isa(e, LinearAlgebra.SingularException) || isa(e, LinearAlgebra.LAPACKException)
                        info[] = Int(Feast_ERROR_LAPACK)
                        break
                    end
                    rethrow(e)
                end
            else
                success = _solve_banded_shifted!(view(workspace.workc, :, 1:M0), rhs_block,
                                                 shifted_mul!, solver_choice,
                                                 tol_value, solver_maxiter,
                                                 solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
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
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbgv!(A, B, kla, klb, Emin, Emax, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

# Return A[i,j] from upper-stored symmetric banded storage without materializing
# the dense matrix.
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
    # LAPACK's gbtrf! storage reserves the first `kl` rows for fill-in, so the
    # original matrix diagonal lives at row `kl + ku + 1`.
    offset = kl + ku + 1
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

function symmetric_banded_matvec!(y::AbstractVector{S}, A::Matrix{T}, k::Int, x::AbstractVector{X}) where {S,T,X}
    fill!(y, zero(S))
    N = length(x)

    # Walk columns and mirror off-diagonal contributions so symmetric storage is
    # applied as a full matrix-vector product.
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

# General banded storage keeps signed offsets, so lower and upper entries are
# read directly instead of mirrored.
@inline function general_banded_get(A::Matrix{T}, k::Int, i::Int, j::Int) where T
    abs(i - j) > k && return zero(T)
    row = k + 1 + i - j
    if 1 <= row <= size(A, 1)
        return A[row, j]
    else
        return zero(T)
    end
end

function fill_shifted_general_banded!(dest::Matrix{Complex{T}},
                                      A::Matrix{Complex{T}},
                                      B::Union{Matrix{Complex{T}},Nothing},
                                      ka::Int, kb::Int, kl::Int,
                                      z::Complex{T}) where T<:Real
    fill!(dest, zero(Complex{T}))
    N = size(dest, 2)
    ku = kl
    offset = kl + ku + 1

    for j in 1:N
        imin = max(1, j - kl)
        imax = min(N, j + ku)
        for i in imin:imax
            a_val = general_banded_get(A, ka, i, j)
            b_val = B === nothing ?
                    (i == j ? one(Complex{T}) : zero(Complex{T})) :
                    general_banded_get(B, kb, i, j)
            dest[offset + i - j, j] = z * b_val - a_val
        end
    end

    return dest
end

function general_banded_matvec!(y::AbstractVector{S}, A::Matrix{T},
                                k::Int, x::AbstractVector{T}) where {S,T}
    fill!(y, zero(S))
    N = length(x)

    for j in 1:N
        xj = x[j]
        for i in max(1, j - k):min(N, j + k)
            row = k + 1 + i - j
            if 1 <= row <= size(A, 1)
                y[i] += convert(S, A[row, j] * xj)
            end
        end
    end

    return y
end

function _solve_banded_shifted!(dest::AbstractMatrix{Complex{T}},
                                rhs::AbstractMatrix{Complex{T}},
                                apply_shift!::F,
                                solver::Symbol, tol::T,
                                maxiter::Int, restart::Int) where {T<:Real,F}
    # Share dense shifted-solve logic so direct/iterative stopping behavior stays
    # consistent between dense and banded FEAST paths.
    return solve_dense_shifted!(dest, rhs, apply_shift!, solver, tol, maxiter, restart)
end

function feast_hbev!(A::Matrix{Complex{T}}, ka::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    # Feast for banded complex Hermitian eigenvalue problem
    # Solves: A*q = lambda*q where A is Hermitian banded
    
    N = size(A, 2)
    
    # Check inputs
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Validate banded matrix dimensions
    size(A, 1) >= ka + 1 || throw(ArgumentError("A matrix storage insufficient for ka"))
    
    return _feast_banded_complex_hermitian(A, nothing, ka, 0, Emin, Emax, M0, fpm;
                                           solver=solver,
                                           solver_tol=solver_tol,
                                           solver_maxiter=solver_maxiter,
                                           solver_restart=solver_restart)
end

function feast_hbevx!(A::Matrix{Complex{T}}, ka::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hbev!(A, ka, Emin, Emax, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_hbev!(A::Matrix{Complex{T}}, ka::Int,
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_hbev!(A, ka, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_hbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_hbgv!(A, B, ka, kb, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_hbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    N = size(A, 2)
    size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))

    # Apply defaults FIRST before using any fpm values
    feastdefault!(fpm)

    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    return _feast_banded_complex_hermitian(A, B, ka, kb, Emin, Emax, M0, fpm;
                                           solver=solver,
                                           solver_tol=solver_tol,
                                           solver_maxiter=solver_maxiter,
                                           solver_restart=solver_restart)
end

function feast_hbgvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_hbgv!(A, B, ka, kb, Emin, Emax, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

# Helper functions for tests, examples, and compatibility wrappers that need to
# convert between compact FEAST/LAPACK storage and ordinary Julia matrices.

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

function banded_to_full_complex_symmetric(A_banded::Matrix{Complex{T}}, k::Int, N::Int) where T<:Real
    # Convert upper-stored complex-symmetric banded data to full dense form.
    A_full = zeros(Complex{T}, N, N)

    for j in 1:N
        for i in max(1, j - k):j
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                value = A_banded[row_index, j]
                A_full[i, j] = value
                A_full[j, i] = value
            end
        end
    end

    return A_full
end

# Hermitian lookup conjugates the mirrored entry; complex-symmetric lookup below
# mirrors without conjugation. Keeping both helpers explicit prevents accidental
# structure changes in the complex paths.
@inline function hermitian_banded_get(A::Matrix{Complex{T}}, k::Int, i::Int, j::Int) where T<:Real
    abs(i - j) > k && return zero(Complex{T})
    if i <= j
        row = k + 1 + i - j
        return A[row, j]
    else
        row = k + 1 + j - i
        return conj(A[row, i])
    end
end

@inline function complex_symmetric_banded_get(A::Matrix{Complex{T}}, k::Int,
                                              i::Int, j::Int) where T<:Real
    abs(i - j) > k && return zero(Complex{T})
    if i <= j
        row = k + 1 + i - j
        return A[row, j]
    else
        row = k + 1 + j - i
        return A[row, i]
    end
end

function fill_shifted_hermitian_banded!(dest::Matrix{Complex{T}},
                                        A::Matrix{Complex{T}},
                                        B::Union{Matrix{Complex{T}},Nothing},
                                        ka::Int, kb::Int, kl::Int,
                                        z::Complex{T}) where T<:Real
    fill!(dest, zero(Complex{T}))
    N = size(dest, 2)
    ku = kl
    offset = kl + ku + 1

    for j in 1:N
        imin = max(1, j - kl)
        imax = min(N, j + ku)
        for i in imin:imax
            a_val = hermitian_banded_get(A, ka, i, j)
            b_val = B === nothing ?
                    (i == j ? one(Complex{T}) : zero(Complex{T})) :
                    hermitian_banded_get(B, kb, i, j)
            dest[offset + i - j, j] = z * b_val - a_val
        end
    end

    return dest
end

function fill_shifted_complex_symmetric_banded!(dest::Matrix{Complex{T}},
                                                A::Matrix{Complex{T}},
                                                B::Union{Matrix{Complex{T}},Nothing},
                                                ka::Int, kb::Int, kl::Int,
                                                z::Complex{T}) where T<:Real
    fill!(dest, zero(Complex{T}))
    N = size(dest, 2)
    ku = kl
    offset = kl + ku + 1

    for j in 1:N
        imin = max(1, j - kl)
        imax = min(N, j + ku)
        for i in imin:imax
            a_val = complex_symmetric_banded_get(A, ka, i, j)
            b_val = B === nothing ?
                    (i == j ? one(Complex{T}) : zero(Complex{T})) :
                    complex_symmetric_banded_get(B, kb, i, j)
            dest[offset + i - j, j] = z * b_val - a_val
        end
    end

    return dest
end

function _feast_banded_complex_hermitian(A::Matrix{Complex{T}},
                                         B::Union{Matrix{Complex{T}},Nothing},
                                         ka::Int, kb::Int,
                                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                         solver::Symbol = :direct,
                                         solver_tol::Real = 0.0,
                                         solver_maxiter::Int = 500,
                                         solver_restart::Int = 30) where T<:Real
    N = size(A, 2)
    size(A, 1) >= ka + 1 || throw(ArgumentError("A matrix storage insufficient for ka"))
    if B !== nothing
        size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))
        size(B, 1) >= kb + 1 || throw(ArgumentError("B matrix storage insufficient for kb"))
    end

    feastdefault!(fpm)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative banded FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_is_identity = B === nothing
    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    # These workspaces are kept outside the refinement loop to avoid allocating
    # during contour solves, Rayleigh-Ritz projection, and residual checks.
    solutions = similar(Q_basis)
    rhs_buffer = similar(Q_basis)
    zAq = zeros(Complex{T}, M0, M0)
    zSq = zeros(Complex{T}, M0, M0)
    Aq_herm = similar(zAq)
    Sq_herm = similar(zSq)
    lambda_vec = zeros(T, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(solutions)
    res_vec = zeros(T, M0)
    residual_vec = zeros(Complex{T}, N)
    Bq_vec = B_is_identity ? nothing : zeros(Complex{T}, N)

    kl = max(ka, B_is_identity ? 0 : kb)
    ku = kl
    ldab = 2 * kl + ku + 1
    current_shift = Ref(zero(Complex{T}))
    tmpAx = zeros(Complex{T}, N)
    tmpBx = zeros(Complex{T}, N)

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_is_identity
            copyto!(tmpBx, x)
        else
            banded_hermitian_matvec!(tmpBx, B, kb, x)
        end
        banded_hermitian_matvec!(tmpAx, A, ka, x)
        @. y = current_shift[] * tmpBx - tmpAx
        return y
    end

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_contour(Emin, Emax, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    banded_factor_cache = Matrix{Complex{T}}[]
    banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, 0)
    banded_factorized = falses(0)
    if solver_is_direct
        banded_factor_cache = [Matrix{Complex{T}}(undef, ldab, N) for _ in eachindex(Zne)]
        banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, length(Zne))
        banded_factorized = falses(length(Zne))
    end

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    loop_count = 0
    M_found = 0
    active_dim = M0
    Q_proj = zeros(Complex{T}, N, M0)

    @views for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(zAq, zero(Complex{T}))
        fill!(zSq, zero(Complex{T}))
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false
        for (idx, z) in enumerate(Zne)
            weight = 2 * Wne[idx]
            basis_block = view(Q_basis, :, 1:active_dim)
            rhs_block = view(rhs_buffer, :, 1:active_dim)
            solutions_block = view(solutions, :, 1:active_dim)
            qproj_block = view(Q_proj, :, 1:active_dim)

            if B_is_identity
                copyto!(rhs_block, basis_block)
            else
                for col in 1:active_dim
                    banded_hermitian_matvec!(view(rhs_block, :, col), B, kb,
                                             view(basis_block, :, col))
                end
            end

            copyto!(solutions_block, rhs_block)
            if solver_is_direct
                try
                    shifted_factor = banded_factor_cache[idx]
                    if !banded_factorized[idx]
                        fill_shifted_hermitian_banded!(shifted_factor, A, B, ka,
                                                       kb, kl, z)
                        _, banded_ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, N,
                                                                     shifted_factor)
                        banded_ipiv_cache[idx] = banded_ipiv
                        banded_factorized[idx] = true
                    end
                    LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, N, shifted_factor,
                                                banded_ipiv_cache[idx],
                                                solutions_block)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Hermitian banded direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                current_shift[] = z
                success = _solve_banded_shifted!(solutions_block, rhs_block,
                                                 shifted_mul!, solver_choice,
                                                 tol_value, solver_maxiter,
                                                 solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            @. qproj_block += weight * solutions_block
        end

        solve_failed && break

        try
            rank = _feast_qr_compress!(solutions_tmp, Q_proj, active_dim;
                                       rank_tol=sqrt(eps(T)))
            if rank == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            q_rank = view(solutions_tmp, :, 1:rank)
            aq_work = view(rhs_buffer, :, 1:rank)
            bq_work = view(solutions, :, 1:rank)
            zAq_rank = view(zAq, 1:rank, 1:rank)
            zSq_rank = view(zSq, 1:rank, 1:rank)
            Aq_rank = view(Aq_herm, 1:rank, 1:rank)
            Sq_rank = view(Sq_herm, 1:rank, 1:rank)

            for col in 1:rank
                banded_hermitian_matvec!(view(aq_work, :, col), A, ka,
                                         view(q_rank, :, col))
            end
            mul!(zSq_rank, adjoint(q_rank), aq_work)
            _feast_hermitian_part!(Sq_rank, zSq_rank)

            if B_is_identity
                fill!(Aq_rank, zero(Complex{T}))
                for i in 1:rank
                    Aq_rank[i, i] = one(Complex{T})
                end
            else
                for col in 1:rank
                    banded_hermitian_matvec!(view(bq_work, :, col), B, kb,
                                             view(q_rank, :, col))
                end
                mul!(zAq_rank, adjoint(q_rank), bq_work)
                _feast_hermitian_part!(Aq_rank, zAq_rank)
            end

            lambda_red = Vector{T}(undef, 0)
            v_red = Array{Complex{T}}(undef, 0, 0)
            try
                F = eigen(Hermitian(Sq_rank), Hermitian(Aq_rank))
                lambda_red = Vector{T}(F.values)
                v_red = Matrix{Complex{T}}(F.vectors)
            catch e
                if isa(e, PosDefException) || isa(e, LAPACKException)
                    F = eigen(Sq_rank, Aq_rank)
                    lambda_red = Vector{T}(real.(F.values))
                    v_red = Matrix{Complex{T}}(F.vectors)
                else
                    rethrow(e)
                end
            end

            for idx in 1:rank
                mul!(view(solutions, :, idx), q_rank, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda_vec, solutions, perm,
                                             lambda_tmp, solutions_tmp,
                                             Emin, Emax, rank)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for j in 1:M
                vec = view(solutions, :, j)
                nrm = norm(vec)
                nrm > 0 && (vec ./= nrm)
            end

            max_res = zero(T)
            for j in 1:M
                q_col = view(solutions, :, j)
                banded_hermitian_matvec!(residual_vec, A, ka, q_col)
                if B_is_identity
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    banded_hermitian_matvec!(Bq_vec, B, kb, q_col)
                    @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                end
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M
            if epsout_val <= eps_tol
                break
            end

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            active_dim = rank
            Q_basis[:, 1:active_dim] .= solutions[:, 1:active_dim]
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced Hermitian banded eigenproblem failed" exception=err
            break
        end
    end

    lambda = lambda_vec[1:M_found]
    q = solutions[:, 1:M_found]
    res = res_vec[1:M_found]

    return FeastResult{T, Complex{T}}(lambda, q, M_found, res,
                                      info_code, epsout_val, loop_count)
end

"""
    _feast_banded_complex_symmetric(A, B, ka, kb, Emid, r, M0, fpm; solver=:direct)

Complex-symmetric banded FEAST implementation using upper-triangle band
storage. Direct solves keep the shifted matrix in LAPACK general-banded
storage, while the Rayleigh-Ritz projection uses the transpose bilinear form
required for complex-symmetric pencils.
"""
@views function _feast_banded_complex_symmetric(A::Matrix{Complex{T}},
                                                B::Union{Matrix{Complex{T}},Nothing},
                                                ka::Int, kb::Int,
                                                Emid::Complex{T}, r::T,
                                                M0::Int, fpm::Vector{Int};
                                                solver::Symbol = :direct,
                                                solver_tol::Real = 0.0,
                                                solver_maxiter::Int = 500,
                                                solver_restart::Int = 30) where T<:Real
    N = size(A, 2)
    size(A, 1) >= ka + 1 || throw(ArgumentError("A matrix storage insufficient for ka"))
    if B !== nothing
        size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))
        size(B, 1) >= kb + 1 || throw(ArgumentError("B matrix storage insufficient for kb"))
    end

    feastdefault!(fpm)
    check_feast_grci_input(N, M0, Emid, r, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative banded FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_is_identity = B === nothing
    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    shifted_solutions = similar(Q_basis)
    rhs_buffer = similar(Q_basis)
    Q_proj = zeros(Complex{T}, N, M0)
    AQ = Matrix{Complex{T}}(undef, N, M0)
    BQ = Matrix{Complex{T}}(undef, N, M0)
    Ared = Matrix{Complex{T}}(undef, M0, M0)
    Bred = Matrix{Complex{T}}(undef, M0, M0)
    lambda_vec = Vector{Complex{T}}(undef, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    solutions_tmp = similar(shifted_solutions)
    res_vec = zeros(T, M0)
    residual_vec = Vector{Complex{T}}(undef, N)
    Bq_vec = Vector{Complex{T}}(undef, N)

    kl = max(ka, B_is_identity ? 0 : kb)
    ku = kl
    ldab = 2 * kl + ku + 1
    current_shift = Ref(zero(Complex{T}))
    tmpAx = zeros(Complex{T}, N)
    tmpBx = zeros(Complex{T}, N)

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_is_identity
            copyto!(tmpBx, x)
        else
            banded_complex_symmetric_matvec!(tmpBx, B, kb, x)
        end
        banded_complex_symmetric_matvec!(tmpAx, A, ka, x)
        @. y = current_shift[] * tmpBx - tmpAx
        return y
    end

    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_gcontour(Emid, r, fpm))
    Zne = contour.Zne
    Wne = contour.Wne
    banded_factor_cache = Matrix{Complex{T}}[]
    banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, 0)
    banded_factorized = falses(0)
    if solver_is_direct
        banded_factor_cache = [Matrix{Complex{T}}(undef, ldab, N) for _ in eachindex(Zne)]
        banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, length(Zne))
        banded_factorized = falses(length(Zne))
    end

    maxloop = fpm[4]
    eps_tol = feast_tolerance(fpm, T)
    epsout_val = T(Inf)
    loop_count = 0
    info_code = Int(Feast_SUCCESS)
    M_found = 0
    active_dim = M0

    for loop_idx in 0:maxloop
        loop_count = loop_idx
        fill!(Q_proj, zero(Complex{T}))

        solve_failed = false
        for e in eachindex(Zne)
            z = Zne[e]
            weight = Wne[e]
            basis_block = view(Q_basis, :, 1:active_dim)
            rhs_block = view(rhs_buffer, :, 1:active_dim)
            shifted_block = view(shifted_solutions, :, 1:active_dim)
            qproj_block = view(Q_proj, :, 1:active_dim)

            if B_is_identity
                copyto!(rhs_block, basis_block)
            else
                for col in 1:active_dim
                    banded_complex_symmetric_matvec!(view(rhs_block, :, col), B, kb,
                                                     view(basis_block, :, col))
                end
            end

            copyto!(shifted_block, rhs_block)
            if solver_is_direct
                try
                    shifted_factor = banded_factor_cache[e]
                    if !banded_factorized[e]
                        fill_shifted_complex_symmetric_banded!(shifted_factor, A, B,
                                                               ka, kb, kl, z)
                        _, banded_ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, N,
                                                                     shifted_factor)
                        banded_ipiv_cache[e] = banded_ipiv
                        banded_factorized[e] = true
                    end
                    LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, N, shifted_factor,
                                                banded_ipiv_cache[e], shifted_block)
                catch err
                    info_code = Int(Feast_ERROR_LAPACK)
                    @warn "Complex-symmetric banded direct solve failed for shift $z" exception=err
                    solve_failed = true
                    break
                end
            else
                current_shift[] = z
                success = _solve_banded_shifted!(shifted_block, rhs_block,
                                                 shifted_mul!, solver_choice,
                                                 tol_value, solver_maxiter,
                                                 solver_restart)
                if !success
                    info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                    solve_failed = true
                    break
                end
            end

            @. qproj_block += weight * shifted_block
        end
        solve_failed && break

        try
            rank = _feast_qr_compress!(solutions_tmp, Q_proj, active_dim;
                                       rank_tol=sqrt(eps(T)))
            if rank == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            q_rank = view(solutions_tmp, :, 1:rank)
            aq_work = view(AQ, :, 1:rank)
            bq_work = view(BQ, :, 1:rank)
            Ared_rank = view(Ared, 1:rank, 1:rank)
            Bred_rank = view(Bred, 1:rank, 1:rank)

            for col in 1:rank
                banded_complex_symmetric_matvec!(view(aq_work, :, col), A, ka,
                                                 view(q_rank, :, col))
                if B_is_identity
                    copyto!(view(bq_work, :, col), view(q_rank, :, col))
                else
                    banded_complex_symmetric_matvec!(view(bq_work, :, col), B, kb,
                                                     view(q_rank, :, col))
                end
            end
            mul!(Ared_rank, transpose(q_rank), aq_work)
            mul!(Bred_rank, transpose(q_rank), bq_work)

            F = eigen(Ared_rank, Bred_rank)
            lambda_red = F.values
            v_red = F.vectors

            for idx in 1:rank
                mul!(view(shifted_solutions, :, idx), q_rank, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_gcontour!(lambda_vec, shifted_solutions, perm,
                                            lambda_tmp, solutions_tmp,
                                            Emid, r, fpm, rank)
            if M == 0
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            for idx in 1:rank
                vec = view(shifted_solutions, :, idx)
                nrm = norm(vec)
                if nrm > zero(T)
                    vec ./= nrm
                else
                    fill!(vec, zero(Complex{T}))
                    vec[mod1(idx, N)] = one(Complex{T})
                end
            end

            max_res = zero(T)
            for j in 1:M
                q_col = view(shifted_solutions, :, j)
                banded_complex_symmetric_matvec!(residual_vec, A, ka, q_col)
                if B_is_identity
                    @. residual_vec = residual_vec - lambda_vec[j] * q_col
                else
                    banded_complex_symmetric_matvec!(Bq_vec, B, kb, q_col)
                    @. residual_vec = residual_vec - lambda_vec[j] * Bq_vec
                end
                res_val = norm(residual_vec) / max(abs(lambda_vec[j]), one(T))
                res_vec[j] = res_val
                max_res = max(max_res, res_val)
            end

            epsout_val = max_res
            M_found = M
            epsout_val <= eps_tol && break

            if loop_idx == maxloop
                info_code = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            active_dim = rank
            copyto!(view(Q_basis, :, 1:active_dim),
                    view(shifted_solutions, :, 1:active_dim))
        catch err
            info_code = Int(Feast_ERROR_LAPACK)
            @warn "Reduced eigenvalue problem failed during banded complex-symmetric FEAST" exception=err
            break
        end
    end

    if M_found == 0 && info_code == Int(Feast_SUCCESS)
        info_code = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    M_found > 1 && feast_sort_general!(lambda_vec, shifted_solutions, res_vec, M_found)

    lambda = lambda_vec[1:M_found]
    q = shifted_solutions[:, 1:M_found]
    res = res_vec[1:M_found]

    return FeastGeneralResult{T}(lambda, q, M_found, res,
                                 info_code, epsout_val, loop_count)
end

"""
    _feast_banded_general(A, B, ka, kb, Emid, r, M0, fpm; solver=:direct)

General non-Hermitian banded FEAST implementation. Direct solves keep
`zB - A` in LAPACK general-banded storage and apply A/B through banded
matrix-vector products in the RCI loop. Iterative solves continue to delegate
to the dense GMRES path used by the existing dense general solver.
"""
@views function _feast_banded_general(A::Matrix{Complex{T}},
                                      B::Union{Matrix{Complex{T}},Nothing},
                                      ka::Int, kb::Int,
                                      Emid::Complex{T}, r::T,
                                      M0::Int, fpm::Vector{Int};
                                      solver::Symbol = :direct,
                                      solver_tol::Real = 0.0,
                                      solver_maxiter::Int = 500,
                                      solver_restart::Int = 30) where T<:Real
    N = size(A, 2)
    size(A, 1) >= ka + 1 || throw(ArgumentError("A matrix storage insufficient for ka"))
    if B !== nothing
        size(B, 2) == N || throw(ArgumentError("B must have same dimensions as A"))
        size(B, 1) >= kb + 1 || throw(ArgumentError("B matrix storage insufficient for kb"))
    end

    feastdefault!(fpm)
    check_feast_grci_input(N, M0, Emid, r, fpm)

    solver_choice = solver == :iterative ? :gmres : solver
    solver_choice = solver_choice in (:direct, :gmres) ? solver_choice : :invalid
    solver_choice == :invalid &&
        throw(ArgumentError("Unsupported solver '$solver'. Use :direct, :gmres, or :iterative."))
    solver_is_direct = solver_choice == :direct
    solver_is_iterative = !solver_is_direct
    solver_is_iterative && !FEAST_KRYLOV_AVAILABLE[] &&
        throw(ArgumentError("Krylov.jl is required for iterative banded FEAST solves."))
    tol_value = solver_tol == 0.0 ? T(10.0^(-fpm[3])) : T(solver_tol)

    B_is_identity = B === nothing
    workspace = FeastWorkspaceComplex{T}(N, M0)

    ijob = Ref(-1)
    Ze = Ref(zero(Complex{T}))
    epsout = Ref(zero(T))
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    lambda_complex = Vector{Complex{T}}(undef, M0)
    q_complex = Matrix{Complex{T}}(undef, N, M0)
    rhs_buffer = Matrix{Complex{T}}(undef, N, M0)

    kl = max(ka, B_is_identity ? 0 : kb)
    ku = kl
    ldab = 2 * kl + ku + 1
    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && (contour = feast_gcontour(Emid, r, fpm))
    banded_factor_cache = Matrix{Complex{T}}[]
    banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, 0)
    banded_factorized = falses(0)
    if solver_is_direct
        banded_factor_cache = [Matrix{Complex{T}}(undef, ldab, N) for _ in eachindex(contour.Zne)]
        banded_ipiv_cache = Vector{Vector{LinearAlgebra.BlasInt}}(undef, length(contour.Zne))
        banded_factorized = falses(length(contour.Zne))
    end
    factorized = false
    current_factor_idx = 0
    current_shift = Ref(zero(Complex{T}))
    tmpAx = zeros(Complex{T}, N)
    tmpBx = zeros(Complex{T}, N)

    function shifted_mul!(y::Vector{Complex{T}}, x::Vector{Complex{T}})
        if B_is_identity
            copyto!(tmpBx, x)
        else
            general_banded_matvec!(tmpBx, B, kb, x)
        end
        general_banded_matvec!(tmpAx, A, ka, x)
        @. y = current_shift[] * tmpBx - tmpAx
        return y
    end

    grci_state = FeastGRCIState{T}()
    max_rci_iterations = fpm[8] * (fpm[4] + 1) * 10
    rci_iteration_count = 0

    while true
        rci_iteration_count += 1
        if rci_iteration_count > max_rci_iterations
            info[] = Int(Feast_ERROR_NO_CONVERGENCE)
            @warn "General banded FEAST RCI loop exceeded maximum iterations" max_rci_iterations=max_rci_iterations
            break
        end

        feast_grci!(ijob, N, Ze, workspace.work, workspace.workc,
                    workspace.zAq, workspace.zSq, fpm, epsout, loop,
                    Emid, r, M0, lambda_complex, q_complex,
                    mode, workspace.res, info; state=grci_state)

        if ijob[] == Int(Feast_RCI_FACTORIZE)
            factorized = false
            if solver_is_direct
                factor_idx = fpm[50]
                if !(1 <= factor_idx <= length(banded_factor_cache))
                    info[] = Int(Feast_ERROR_INTERNAL)
                    break
                end
                shifted_factor = banded_factor_cache[factor_idx]
                try
                    if !banded_factorized[factor_idx]
                        fill_shifted_general_banded!(shifted_factor, A, B, ka, kb,
                                                     kl, Ze[])
                        _, banded_ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, N,
                                                                     shifted_factor)
                        banded_ipiv_cache[factor_idx] = banded_ipiv
                        banded_factorized[factor_idx] = true
                    end
                    current_factor_idx = factor_idx
                    factorized = true
                catch e
                    if isa(e, LinearAlgebra.SingularException) || isa(e, LinearAlgebra.LAPACKException)
                        info[] = Int(Feast_ERROR_LAPACK)
                        break
                    end
                    rethrow(e)
                end
            else
                current_shift[] = Ze[]
            end

        elseif ijob[] == Int(Feast_RCI_SOLVE)
            if solver_is_direct && !factorized
                info[] = Int(Feast_ERROR_INTERNAL)
                break
            end

            workc_block = view(workspace.workc, :, 1:M0)
            rhs_block = view(rhs_buffer, :, 1:M0)
            if B_is_identity
                copyto!(rhs_block, workc_block)
            else
                for col in 1:M0
                    general_banded_matvec!(view(rhs_buffer, :, col), B, kb,
                                           view(workspace.workc, :, col))
                end
            end

            if solver_is_direct
                try
                    copyto!(workc_block, rhs_block)
                    shifted_factor = banded_factor_cache[current_factor_idx]
                    LinearAlgebra.LAPACK.gbtrs!('N', kl, ku, N, shifted_factor,
                                                banded_ipiv_cache[current_factor_idx],
                                                workc_block)
                catch e
                    if isa(e, LinearAlgebra.SingularException) || isa(e, LinearAlgebra.LAPACKException)
                        info[] = Int(Feast_ERROR_LAPACK)
                        break
                    end
                    rethrow(e)
                end
            else
                success = _solve_banded_shifted!(workc_block, rhs_block,
                                                 shifted_mul!, solver_choice,
                                                 tol_value, solver_maxiter,
                                                 solver_restart)
                if !success
                    info[] = Int(Feast_ERROR_NO_CONVERGENCE)
                    break
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_B)
            M = mode[]
            if B_is_identity
                copyto!(view(workspace.workc, :, 1:M), view(q_complex, :, 1:M))
            else
                for col in 1:M
                    general_banded_matvec!(view(workspace.workc, :, col), B, kb,
                                           view(q_complex, :, col))
                end
            end

        elseif ijob[] == Int(Feast_RCI_MULT_A)
            M = mode[]
            for col in 1:M
                general_banded_matvec!(view(workspace.workc, :, col), A, ka,
                                       view(q_complex, :, col))
            end

        elseif ijob[] == Int(Feast_RCI_DONE)
            break
        else
            error("Unexpected FEAST RCI job code: ijob=$(ijob[]). Expected one of: " *
                  "FACTORIZE($(Int(Feast_RCI_FACTORIZE))), SOLVE($(Int(Feast_RCI_SOLVE))), " *
                  "MULT_B($(Int(Feast_RCI_MULT_B))), MULT_A($(Int(Feast_RCI_MULT_A))), " *
                  "DONE($(Int(Feast_RCI_DONE)))")
        end
    end

    M = mode[]
    lambda = lambda_complex[1:M]
    q = q_complex[:, 1:M]
    res = workspace.res[1:M]

    return FeastGeneralResult{T}(lambda, q, M, res, info[], epsout[], loop[])
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

function full_to_general_banded(A_full::Matrix{T}, k::Int) where T
    # Convert a general banded matrix to the storage used by banded_to_full.
    N = size(A_full, 1)
    size(A_full, 2) == N || throw(ArgumentError("A_full must be square"))
    A_banded = zeros(T, 2 * k + 1, N)

    for j in 1:N
        for i in max(1, j - k):min(N, j + k)
            row_index = k + 1 + i - j
            A_banded[row_index, j] = A_full[i, j]
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

function banded_hermitian_matvec!(y::AbstractVector{Complex{T}}, A_banded::Matrix{Complex{T}},
                                  k::Int, x::AbstractVector{Complex{T}}) where T
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

function banded_complex_symmetric_matvec!(y::AbstractVector{Complex{T}},
                                          A_banded::Matrix{Complex{T}},
                                          k::Int,
                                          x::AbstractVector{Complex{T}}) where T<:Real
    # Apply upper-stored complex-symmetric banded data without conjugating the mirror side.
    N = length(x)
    fill!(y, zero(Complex{T}))

    for j in 1:N
        xj = x[j]
        for i in max(1, j - k):j
            row_index = k + 1 + i - j
            if 1 <= row_index <= size(A_banded, 1)
                val = A_banded[row_index, j]
                y[i] += val * xj
                if i != j
                    y[j] += val * x[i]
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
                     Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
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
    return feast_sbgv!(A, B, ka, klb, Emin, Emax, M0, fpm;
                       solver=solver, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_sbevx!(A::Matrix{T}, ka::Int,
                      Emin::T, Emax::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbev!(A, ka, Emin, Emax, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function difeast_sbgv!(A::Matrix{T}, B::Matrix{T}, kla::Int, klb::Int,
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_sbgv!(A, B, kla, klb, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function difeast_sbev!(A::Matrix{T}, ka::Int,
                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_sbev!(A, ka, Emin, Emax, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_sbgv_complex!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                             ka::Int, kb::Int,
                             Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                             solver::Symbol = :direct,
                             solver_tol::Real = 0.0,
                             solver_maxiter::Int = 500,
                             solver_restart::Int = 30) where T<:Real
    return _feast_banded_complex_symmetric(A, B, ka, kb, Emid, r, M0, fpm;
                                           solver=solver,
                                           solver_tol=solver_tol,
                                           solver_maxiter=solver_maxiter,
                                           solver_restart=solver_restart)
end

function feast_sbgvx_complex!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                              ka::Int, kb::Int,
                              Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                              Zne::AbstractVector{Complex{TZ}},
                              Wne::AbstractVector{Complex{TW}};
                              solver::Symbol = :direct,
                              solver_tol::Real = 0.0,
                              solver_maxiter::Int = 500,
                              solver_restart::Int = 30) where {T<:Real,TZ<:Real,TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbgv_complex!(A, B, ka, kb, Emid, r, M0, fpm;
                            solver=solver, solver_tol=solver_tol,
                            solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function feast_sbev_complex!(A::Matrix{Complex{T}}, ka::Int,
                             Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                             solver::Symbol = :direct,
                             solver_tol::Real = 0.0,
                             solver_maxiter::Int = 500,
                             solver_restart::Int = 30) where T<:Real
    return _feast_banded_complex_symmetric(A, nothing, ka, 0, Emid, r, M0, fpm;
                                           solver=solver,
                                           solver_tol=solver_tol,
                                           solver_maxiter=solver_maxiter,
                                           solver_restart=solver_restart)
end

function feast_sbevx_complex!(A::Matrix{Complex{T}}, ka::Int,
                              Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                              Zne::AbstractVector{Complex{TZ}},
                              Wne::AbstractVector{Complex{TW}};
                              solver::Symbol = :direct,
                              solver_tol::Real = 0.0,
                              solver_maxiter::Int = 500,
                              solver_restart::Int = 30) where {T<:Real,TZ<:Real,TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_sbev_complex!(A, ka, Emid, r, M0, fpm;
                            solver=solver, solver_tol=solver_tol,
                            solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_sbgv_complex!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                               ka::Int, kb::Int,
                               Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                               solver_tol::Real = 0.0,
                               solver_maxiter::Int = 500,
                               solver_restart::Int = 30) where T<:Real
    return feast_sbgv_complex!(A, B, ka, kb, Emid, r, M0, fpm;
                               solver=:gmres, solver_tol=solver_tol,
                               solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_sbev_complex!(A::Matrix{Complex{T}}, ka::Int,
                               Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                               solver_tol::Real = 0.0,
                               solver_maxiter::Int = 500,
                               solver_restart::Int = 30) where T<:Real
    return feast_sbev_complex!(A, ka, Emid, r, M0, fpm;
                               solver=:gmres, solver_tol=solver_tol,
                               solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function feast_gbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    return _feast_banded_general(A, B, ka, kb, Emid, r, M0, fpm;
                                 solver=solver,
                                 solver_tol=solver_tol,
                                 solver_maxiter=solver_maxiter,
                                 solver_restart=solver_restart)
end

function feast_gbgvx!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gbgv!(A, B, ka, kb, Emid, r, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function feast_gbev!(A::Matrix{Complex{T}}, ka::Int,
                     Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                     solver::Symbol = :direct,
                     solver_tol::Real = 0.0,
                     solver_maxiter::Int = 500,
                     solver_restart::Int = 30) where T<:Real
    return _feast_banded_general(A, nothing, ka, 0, Emid, r, M0, fpm;
                                 solver=solver,
                                 solver_tol=solver_tol,
                                 solver_maxiter=solver_maxiter,
                                 solver_restart=solver_restart)
end

function feast_gbevx!(A::Matrix{Complex{T}}, ka::Int,
                      Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int},
                      Zne::AbstractVector{Complex{TZ}},
                      Wne::AbstractVector{Complex{TW}};
                      solver::Symbol = :direct,
                      solver_tol::Real = 0.0,
                      solver_maxiter::Int = 500,
                      solver_restart::Int = 30) where {T<:Real, TZ<:Real, TW<:Real}
    return with_custom_contour(fpm, Zne, Wne) do
        feast_gbev!(A, ka, Emid, r, M0, fpm;
                    solver=solver, solver_tol=solver_tol,
                    solver_maxiter=solver_maxiter, solver_restart=solver_restart)
    end
end

function zifeast_gbgv!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}}, ka::Int, kb::Int,
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_gbgv!(A, B, ka, kb, Emid, r, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end

function zifeast_gbev!(A::Matrix{Complex{T}}, ka::Int,
                       Emid::Complex{T}, r::T, M0::Int, fpm::Vector{Int};
                       solver_tol::Real = 0.0,
                       solver_maxiter::Int = 500,
                       solver_restart::Int = 30) where T<:Real
    return feast_gbev!(A, ka, Emid, r, M0, fpm;
                       solver=:gmres, solver_tol=solver_tol,
                       solver_maxiter=solver_maxiter, solver_restart=solver_restart)
end
