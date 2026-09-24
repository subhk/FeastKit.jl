function _mpi_feast_complex_hermitian!(A::AbstractMatrix{Complex{T}},
                                       B::AbstractMatrix{Complex{T}},
                                       Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                                       comm::MPI.Comm = MPI.COMM_WORLD,
                                       root::Int = 0,
                                       solver::Symbol = :direct,
                                       solver_tol::Real = 0.0,
                                       solver_maxiter::Int = 500,
                                       solver_restart::Int = 30) where T<:Real
    rank = MPI.Comm_rank(comm)
    N = size(A, 1)
    size(A, 2) == N || throw(ArgumentError("Matrix A must be square"))
    size(B) == (N, N) || throw(ArgumentError("Matrix B must match size of A"))
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    feastdefault!(fpm)
    fpm[42] == 1 && throw(ArgumentError("mixed_precision requires the serial dense solver"))
    solver_choice = _mpi_solver_choice(solver)
    tol = _mpi_solver_tolerance(fpm,T,solver_tol)
    krylov_workspace = solver_choice === :gmres ? _feast_krylov_workspace(N, T, solver_restart) : nothing

    contour, _ = _mpi_contour(T,fpm,Emin,Emax,root,comm)
    contour = _feast_complete_hermitian_contour(contour)
    ne = length(contour.Zne)
    Zne_global, Wne_global = contour.Zne, contour.Wne
    keep = Vector{Bool}(undef, M0)

    mpi_state = MPIFeastState{T}(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm),
                                 N, M0, ne, root)
    _mpi_distribute_complex_contour!(mpi_state, Zne_global, Wne_global)

    Q_basis = zeros(Complex{T}, N, M0)
    _feast_seeded_subspace_complex!(Q_basis)
    MPI.Bcast!(Q_basis, root, comm)

    # Direct solves cache local factors only when fpm[10] requests storage.
    # Iterative solves have nothing to cache.
    local_factors = solver_choice == :direct ?
        _mpi_factorize_contour(A,B,mpi_state.local_Zne,comm;store=fpm[10]==1) : nothing
    if solver_choice == :direct && local_factors === nothing
        return FeastResult{T,Complex{T}}(T[],zeros(Complex{T},N,0),0,T[],
            Int(Feast_ERROR_LAPACK),T(Inf),0)
    end
    B_is_identity = (B == I)   # standard problem: skip per-loop identity matmuls
    # Every rank measures on the same broadcast basis; root's value is used.
    res_scale = _mpi_residual_floor(A, B_is_identity ? nothing : B, Q_basis,
                                    _feast_residual_scale(Emin, Emax), root, comm)
    res_scale === nothing && return FeastResult{T,Complex{T}}(T[],zeros(Complex{T},N,0),0,T[],
        solver_choice == :direct ? Int(Feast_ERROR_LAPACK) : Int(Feast_ERROR_NO_CONVERGENCE),
        T(Inf),0)
    BQ_loop = similar(Q_basis)
    Q_proj_local_buf = similar(Q_basis)
    # In-place solve buffer. Sparse factors are UMFPACK and always ComplexF64
    # (it promotes Float32 inputs); dense LU keeps Complex{T}.
    Y_loop = A isa AbstractSparseMatrix ?
        Matrix{Complex{promote_type(T, Float64)}}(undef, N, M0) : similar(Q_basis)

    Aq_herm = Matrix{Complex{T}}(undef, M0, M0)
    Sq_herm = Matrix{Complex{T}}(undef, M0, M0)
    AQc = similar(Q_basis)
    BQc = similar(Q_basis)
    Q_proj = similar(Q_basis)
    solutions = similar(Q_basis)
    solutions_tmp = similar(Q_basis)
    lambda_vec = zeros(T, M0)
    lambda_tmp = similar(lambda_vec)
    perm = Vector{Int}(undef, M0)
    res_vec = zeros(T, M0)
    epsout_val = T(Inf)
    info_code = Int(Feast_SUCCESS)
    M_found = 0
    loop_count = 0

    for loop_idx in 0:fpm[4]
        loop_count = loop_idx
        # Each rank solves its local contour points; only the partial filtered
        # subspace (complex) is needed. The reduced pencil is rebuilt from an
        # ORTHONORMAL basis below — what the old rank-deficient moment path lacked.
        local_success = false
        local_Q_proj = Q_proj_local_buf
        try
            if solver_choice == :direct
                B_is_identity ? copyto!(BQ_loop, Q_basis) : mul!(BQ_loop, B, Q_basis)
                local_success = _mpi_local_projection!(Q_proj_local_buf,local_factors,BQ_loop,
                                                       mpi_state.local_Wne,Y_loop)
            else
                _, _, local_Q_proj, local_success =
                    mpi_compute_complex_hermitian_moments(A, B, Q_basis,
                                                          mpi_state.local_Zne,
                                                          mpi_state.local_Wne, M0,
                                                          solver_choice, _feast_inner_tol(solver_tol == 0.0, tol, epsout_val, loop_idx),
                                                          solver_maxiter, solver_restart,
                                                          comm; workspace=krylov_workspace)
            end
        catch err
            @debug "MPI local Hermitian projection failed" exception=err
            local_success = false
        end
        if _mpi_success_count(local_success, comm) != MPI.Comm_size(comm)
            info_code = solver_choice == :direct ? Int(Feast_ERROR_LAPACK) : Int(Feast_ERROR_NO_CONVERGENCE)
            M_found = 0
            break
        end

        # Direct send/recv Allreduce into the persistent buffer — no fresh
        # receive array per refinement loop.
        MPI.Allreduce!(local_Q_proj, Q_proj, MPI.SUM, comm)

        # Q_proj is the filtered image of the previous loop's Ritz vectors
        # (Q_basis); the root classifies them by filter response and every
        # rank follows. `solutions` still holds those pairs.
        if loop_idx >= _FEAST_SPURIOUS_MIN_LOOPS
            kept = rank == root ?
                _feast_screen_spurious!(keep, Q_proj, Q_basis, res_vec, M_found,
                                        feast_tolerance(fpm, T)) : nothing
            kept = MPI.bcast(kept, root, comm)
            if kept !== nothing
                MPI.Bcast!(keep, root, comm)
                M_found = _feast_compact_pairs!(lambda_vec, solutions, res_vec, keep, M_found)
                epsout_val = M_found > 0 ? maximum(view(res_vec, 1:M_found)) : zero(T)
                info_code = M_found == 0 ? Int(Feast_SUCCESS) :
                            _feast_exit_info(true, M_found, M0, N)
                break
            end
        end

        M = 0
        success = _mpi_collective_try(comm) do
            # Orthonormalize the filtered subspace, then Hermitian Rayleigh-Ritz:
            # Sq = Qᴴ A Q, Aq = Qᴴ B Q. Orthonormality keeps the reduced pencil
            # well-conditioned even when the filtered subspace is rank deficient.
            Qo = Matrix(qr!(copyto!(AQc, Q_proj)).Q)
            mul!(AQc, A, Qo)
            mul!(Sq_herm, adjoint(Qo), AQc)
            if B_is_identity
                # Qo orthonormal ⇒ Qᴴ B Q = I; skip the dense identity matmul.
                fill!(Aq_herm, zero(Complex{T}))
                @inbounds for i in 1:size(Aq_herm, 1)
                    Aq_herm[i, i] = one(Complex{T})
                end
            else
                mul!(BQc, B, Qo)
                mul!(Aq_herm, adjoint(Qo), BQc)
            end
            F = try
                eigen(Hermitian(Sq_herm), Hermitian(Aq_herm))
            catch err
                eigen(Sq_herm, Aq_herm)
            end
            lambda_red = real.(F.values)
            v_red = F.vectors
            for idx in 1:M0
                mul!(view(solutions, :, idx), Qo, view(v_red, :, idx))
                lambda_vec[idx] = lambda_red[idx]
            end

            M = _feast_reorder_by_interval!(lambda_vec, solutions, perm,
                                             lambda_tmp, solutions_tmp,
                                             Emin, Emax, M0)
            if M == 0
                return
            end
            for j in 1:M
                qj = view(solutions, :, j)
                nrm = norm(qj)
                nrm > 0 && (qj ./= nrm)
            end
        end
        if !success
            info_code = Int(Feast_ERROR_LAPACK)
            M_found = 0
            break
        end
        M = MPI.bcast(M,root,comm)
        if M == 0
            # No Ritz value inside the region: it holds no eigenvalues.
            info_code = Int(Feast_SUCCESS)
            epsout_val = zero(T)
            M_found = 0
            break
        end
        MPI.Bcast!(lambda_vec,root,comm)
        MPI.Bcast!(solutions,root,comm)
        if !mpi_compute_complex_residuals!(A,B,lambda_vec,solutions,res_vec,M,comm;
                                           scale=res_scale)
            info_code = Int(Feast_ERROR_LAPACK)
            M_found = 0
            break
        end
        epsout_val = maximum(view(res_vec,1:M))
        M_found = M
        converged,exhausted = MPI.bcast((epsout_val<=feast_tolerance(fpm,T),loop_idx==fpm[4]),root,comm)
        if converged || exhausted
            info_code = _feast_exit_info(converged,M,M0,N)
            break
        end
        copyto!(Q_basis,solutions)
    end

    M_found > 1 && feast_sort!(lambda_vec, solutions, res_vec, M_found)
    return FeastResult{T, Complex{T}}(lambda_vec[1:M_found],
                                      solutions[:, 1:M_found], M_found,
                                      res_vec[1:M_found], info_code,
                                      epsout_val, loop_count)
end

function mpi_feast_hcsrgv!(A::SparseMatrixCSC{Complex{T},Int},
                           B::SparseMatrixCSC{Complex{T},Int},
                           Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    return _mpi_feast_complex_hermitian!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_hcsrev!(A::SparseMatrixCSC{Complex{T},Int},
                           Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                           kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast_hcsrgv!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_hegv!(A::Matrix{Complex{T}},
                         B::Matrix{Complex{T}},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    ishermitian(A) || throw(ArgumentError("Matrix A must be Hermitian"))
    ishermitian(B) || throw(ArgumentError("Matrix B must be Hermitian positive definite"))
    return _mpi_feast_complex_hermitian!(A, B, Emin, Emax, M0, fpm; kwargs...)
end

function mpi_feast_heev!(A::Matrix{Complex{T}},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast_hegv!(A, B, Emin, Emax, M0, fpm; kwargs...)
end
