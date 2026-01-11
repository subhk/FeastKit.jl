# MPI-based parallel FeastKit implementation
# True MPI support for HPC clusters and distributed computing

# Note: MPI should already be loaded when this file is included
using LinearAlgebra
using SparseArrays

# MPI-specific FeastKit state
mutable struct MPIFeastState{T<:Real}
    # MPI communication info
    comm::MPI.Comm
    rank::Int
    size::Int
    root::Int
    
    # Feast parameters
    N::Int
    M0::Int
    ne::Int
    
    # Local contour points assigned to this rank
    local_points::Vector{Int}
    local_Zne::Vector{Complex{T}}
    local_Wne::Vector{Complex{T}}
    
    # Convergence state
    converged::Bool
    loop::Int
    epsout::T
    info::Int
    
    function MPIFeastState{T}(comm::MPI.Comm, N::Int, M0::Int, ne::Int, root::Int=0) where T<:Real
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        
        # Distribute contour points among MPI ranks
        points_per_rank = div(ne, size)
        remainder = ne % size
        
        # Calculate local points for this rank
        start_idx = rank * points_per_rank + min(rank, remainder) + 1
        local_count = points_per_rank + (rank < remainder ? 1 : 0)
        local_points = collect(start_idx:(start_idx + local_count - 1))
        
        new(
            comm, rank, size, root,
            N, M0, ne,
            local_points,
            Vector{Complex{T}}(undef, local_count),
            Vector{Complex{T}}(undef, local_count),
            false, 0, zero(T), 0
        )
    end
end

# Main MPI FeastKit interface
function mpi_feast_sygv!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                         Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                         comm::MPI.Comm = MPI.COMM_WORLD,
                         root::Int = 0) where T<:Real
    # MPI parallel Feast for real symmetric generalized eigenvalue problems
    
    # Initialize MPI if not already done
    if !MPI.Initialized()
        MPI.Init()
    end

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    N = Base.size(A, 1)

    # Input validation (on all ranks)
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Generate integration contour (on root, then broadcast)
    contour = nothing
    if rank == root
        contour = feast_contour(Emin, Emax, fpm)
    end
    
    # Broadcast contour to all ranks
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)
    
    # Create MPI state
    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)
    
    # Distribute contour points
    for (i, global_idx) in enumerate(mpi_state.local_points)
        mpi_state.local_Zne[i] = Zne_global[global_idx]
        mpi_state.local_Wne[i] = Wne_global[global_idx]
    end
    
    # Initialize workspace (all ranks need full workspace)
    workspace = FeastWorkspaceReal{T}(N, M0)
    randn!(workspace.work)  # Random initial guess
    
    # Broadcast initial subspace from root
    MPI.Bcast!(workspace.work, root, comm)
    
    # Initialize Feast parameters
    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Main Feast refinement loop
    for loop in 1:max_loops
        mpi_state.loop = loop

        # Compute local moment contributions (now returns Q_proj too)
        local_Aq, local_Sq, local_Q_proj = mpi_compute_local_moments(A, B, workspace.work,
                                                      mpi_state.local_Zne,
                                                      mpi_state.local_Wne, M0)

        # Reduce (sum) moment matrices AND Q_proj across all ranks
        global_Aq = MPI.Allreduce(local_Aq, MPI.SUM, comm)
        global_Sq = MPI.Allreduce(local_Sq, MPI.SUM, comm)
        global_Q_proj = MPI.Allreduce(local_Q_proj, MPI.SUM, comm)

        # Solve reduced eigenvalue problem (on all ranks for consistency)
        # IMPORTANT: Solve Sq*v = lambda*Aq*v (consistent with FEAST algorithm)
        # Aq = sum(w * Q' * Y), Sq = sum(w * z * Q' * Y)
        try
            # Use Symmetric wrapper - moments should be symmetric for Hermitian problems
            F = try
                eigen(Symmetric(global_Sq), Symmetric(global_Aq))
            catch e
                eigen(global_Sq, global_Aq)
            end
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q
            # This is the core FEAST algorithm - use spectral projector output
            for idx in 1:M0
                workspace.q[:, idx] = global_Q_proj * view(v_red, :, idx)
                workspace.lambda[idx] = lambda_red[idx]
            end

            # Reorder: put eigenvalues inside the interval first
            inside_mask = [feast_inside_contour(workspace.lambda[i], Emin, Emax) for i in 1:M0]
            inside_indices = findall(inside_mask)
            outside_indices = findall(.!inside_mask)
            perm = vcat(inside_indices, outside_indices)

            workspace.lambda[1:M0] = workspace.lambda[perm]
            workspace.q[:, 1:M0] = workspace.q[:, perm]

            M = length(inside_indices)

            if M == 0
                mpi_state.info = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Normalize eigenvectors
            for j in 1:M
                q_norm = norm(view(workspace.q, :, j))
                if q_norm > 0
                    workspace.q[:, j] ./= q_norm
                end
            end

            # Compute residuals only for eigenvalues inside interval
            mpi_compute_residuals!(A, B, workspace.lambda, workspace.q,
                                 workspace.res, M, comm)

            mpi_state.epsout = maximum(workspace.res[1:M])

            # Check convergence
            if mpi_state.epsout <= eps_tolerance
                mpi_state.converged = true
                mpi_state.info = Int(Feast_SUCCESS)
                break
            end

            # Prepare for next iteration - use full M0 subspace
            workspace.work[:, 1:M0] = workspace.q[:, 1:M0]

        catch e
            mpi_state.info = Int(Feast_ERROR_LAPACK)
            break
        end
    end
    
    # Final result processing
    if !mpi_state.converged && mpi_state.info == 0
        mpi_state.info = Int(Feast_ERROR_NO_CONVERGENCE)
    end
    
    # Count final eigenvalues
    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    
    # Sort results
    if M > 0
        feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
    end
    
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M, 
                           workspace.res[1:M], mpi_state.info, 
                           mpi_state.epsout, mpi_state.loop)
end

# Compute local moment contributions on each MPI rank (returns moments and Q_proj)
function mpi_compute_local_moments(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                  work::Matrix{T}, local_Zne::Vector{Complex{T}},
                                  local_Wne::Vector{Complex{T}}, M0::Int) where T<:Real

    local_ne = length(local_Zne)
    N = size(A, 1)
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    Q_proj_local = zeros(T, N, M0)

    # Process each local contour point
    for e in 1:local_ne
        z = local_Zne[e]
        w = local_Wne[e]

        try
            # Form and factorize (z*B - A)
            system_matrix = z * B - A
            F = lu(system_matrix)

            # Right-hand side
            rhs = B * work[:, 1:M0]

            # Solve linear systems
            workc_local = F \ rhs

            # Factor of 2 for half-contour symmetry
            weight = 2 * w

            # Accumulate moment contribution
            for j in 1:M0
                for i in 1:M0
                    inner_product = dot(work[:, i], workc_local[:, j])
                    Aq_local[i, j] += real(weight * inner_product)
                    Sq_local[i, j] += real(weight * z * inner_product)
                end
            end

            # Accumulate filtered subspace contribution
            Q_proj_local .+= real.(weight .* workc_local)

        catch err
            @warn "MPI rank $(MPI.Comm_rank(MPI.COMM_WORLD)): Linear solve failed for contour point $e: $err"
        end
    end

    return Aq_local, Sq_local, Q_proj_local
end

# Distributed residual computation
function mpi_compute_residuals!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                               lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                               M::Int, comm::MPI.Comm) where T<:Real

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Distribute eigenvalue computations among ranks
    eigs_per_rank = div(M, nprocs)
    remainder = M % nprocs

    start_idx = rank * eigs_per_rank + min(rank, remainder) + 1
    local_count = eigs_per_rank + (rank < remainder ? 1 : 0)
    end_idx = start_idx + local_count - 1
    
    # Compute local residuals
    local_res = zeros(T, M)
    for j in start_idx:min(end_idx, M)
        # Residual: ||A*q - lambda*B*q||
        Aq = A * q[:, j]
        Bq = B * q[:, j]
        residual = Aq - lambda[j] * Bq
        local_res[j] = norm(residual)
    end
    
    # Reduce residuals across all ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end

# MPI sparse FeastKit
function mpi_feast_scsrgv!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                          Emin::T, Emax::T, M0::Int, fpm::Vector{Int};
                          comm::MPI.Comm = MPI.COMM_WORLD,
                          root::Int = 0) where T<:Real
    # MPI parallel Feast for sparse matrices
    # Similar structure to dense version but with sparse linear algebra

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    N = Base.size(A, 1)

    # Input validation
    check_feast_srci_input(N, M0, Emin, Emax, fpm)
    
    # Generate and distribute contour
    contour = nothing
    if rank == root
        contour = feast_contour(Emin, Emax, fpm)
    end
    
    ne = MPI.bcast(rank == root ? length(contour.Zne) : 0, root, comm)
    Zne_global = MPI.bcast(rank == root ? contour.Zne : Vector{Complex{T}}(undef, ne), root, comm)
    Wne_global = MPI.bcast(rank == root ? contour.Wne : Vector{Complex{T}}(undef, ne), root, comm)
    
    # Create MPI state and distribute points
    mpi_state = MPIFeastState{T}(comm, N, M0, ne, root)
    for (i, global_idx) in enumerate(mpi_state.local_points)
        mpi_state.local_Zne[i] = Zne_global[global_idx]
        mpi_state.local_Wne[i] = Wne_global[global_idx]
    end
    
    # Initialize workspace
    workspace = FeastWorkspaceReal{T}(N, M0)
    randn!(workspace.work)
    MPI.Bcast!(workspace.work, root, comm)
    
    # Feast parameters
    feastdefault!(fpm)
    eps_tolerance = feast_tolerance(fpm)
    max_loops = fpm[4]
    
    # Main refinement loop
    for loop in 1:max_loops
        mpi_state.loop = loop

        # Compute local sparse moments (now returns Q_proj too)
        local_Aq, local_Sq, local_Q_proj = mpi_compute_sparse_moments(A, B, workspace.work,
                                                       mpi_state.local_Zne,
                                                       mpi_state.local_Wne, M0)

        # Reduce moments AND Q_proj
        global_Aq = MPI.Allreduce(local_Aq, MPI.SUM, comm)
        global_Sq = MPI.Allreduce(local_Sq, MPI.SUM, comm)
        global_Q_proj = MPI.Allreduce(local_Q_proj, MPI.SUM, comm)

        # Solve reduced problem and check convergence (same as dense version)
        # IMPORTANT: Solve Sq*v = lambda*Aq*v (consistent with FEAST algorithm)
        try
            # Use Symmetric wrapper - moments should be symmetric for Hermitian problems
            F = try
                eigen(Symmetric(global_Sq), Symmetric(global_Aq))
            catch e
                eigen(global_Sq, global_Aq)
            end
            lambda_red = real.(F.values)
            v_red = real.(F.vectors)

            # Project ALL eigenvectors using FILTERED subspace (Q_proj), not original Q
            for idx in 1:M0
                workspace.q[:, idx] = global_Q_proj * view(v_red, :, idx)
                workspace.lambda[idx] = lambda_red[idx]
            end

            # Reorder: put eigenvalues inside the interval first
            inside_mask = [feast_inside_contour(workspace.lambda[i], Emin, Emax) for i in 1:M0]
            inside_indices = findall(inside_mask)
            outside_indices = findall(.!inside_mask)
            perm = vcat(inside_indices, outside_indices)

            workspace.lambda[1:M0] = workspace.lambda[perm]
            workspace.q[:, 1:M0] = workspace.q[:, perm]

            M = length(inside_indices)

            if M == 0
                mpi_state.info = Int(Feast_ERROR_NO_CONVERGENCE)
                break
            end

            # Normalize eigenvectors
            for j in 1:M
                q_norm = norm(view(workspace.q, :, j))
                if q_norm > 0
                    workspace.q[:, j] ./= q_norm
                end
            end

            # Sparse residual computation
            mpi_compute_sparse_residuals!(A, B, workspace.lambda, workspace.q,
                                        workspace.res, M, comm)

            mpi_state.epsout = maximum(workspace.res[1:M])

            if mpi_state.epsout <= eps_tolerance
                mpi_state.converged = true
                mpi_state.info = Int(Feast_SUCCESS)
                break
            end

            # Use full M0 subspace for next iteration
            workspace.work[:, 1:M0] = workspace.q[:, 1:M0]

        catch e
            mpi_state.info = Int(Feast_ERROR_LAPACK)
            break
        end
    end

    # Final processing
    if !mpi_state.converged && mpi_state.info == 0
        mpi_state.info = Int(Feast_ERROR_NO_CONVERGENCE)
    end

    M = count(i -> feast_inside_contour(workspace.lambda[i], Emin, Emax), 1:M0)
    
    if M > 0
        feast_sort!(workspace.lambda, workspace.q, workspace.res, M)
    end
    
    return FeastResult{T, T}(workspace.lambda[1:M], workspace.q[:, 1:M], M,
                           workspace.res[1:M], mpi_state.info,
                           mpi_state.epsout, mpi_state.loop)
end

# Sparse moment computation for MPI (returns moments and Q_proj)
function mpi_compute_sparse_moments(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                   work::Matrix{T}, local_Zne::Vector{Complex{T}},
                                   local_Wne::Vector{Complex{T}}, M0::Int) where T<:Real

    local_ne = length(local_Zne)
    N = size(A, 1)
    Aq_local = zeros(T, M0, M0)
    Sq_local = zeros(T, M0, M0)
    Q_proj_local = zeros(T, N, M0)

    for e in 1:local_ne
        z = local_Zne[e]
        w = local_Wne[e]

        try
            # Sparse system formation and solve
            system_matrix = z * B - A
            F = lu(system_matrix)

            rhs = B * work[:, 1:M0]
            workc_local = F \ rhs

            # Factor of 2 for half-contour symmetry
            weight = 2 * w

            # Accumulate moments
            for j in 1:M0
                for i in 1:M0
                    inner_product = dot(work[:, i], workc_local[:, j])
                    Aq_local[i, j] += real(weight * inner_product)
                    Sq_local[i, j] += real(weight * z * inner_product)
                end
            end

            # Accumulate filtered subspace contribution
            Q_proj_local .+= real.(weight .* workc_local)

        catch err
            @warn "MPI rank $(MPI.Comm_rank(MPI.COMM_WORLD)): Sparse solve failed: $err"
        end
    end

    return Aq_local, Sq_local, Q_proj_local
end

# Sparse residual computation for MPI
function mpi_compute_sparse_residuals!(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                                      lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                                      M::Int, comm::MPI.Comm) where T<:Real

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Distribute eigenvalue computations
    eigs_per_rank = div(M, nprocs)
    remainder = M % nprocs

    start_idx = rank * eigs_per_rank + min(rank, remainder) + 1
    local_count = eigs_per_rank + (rank < remainder ? 1 : 0)
    end_idx = start_idx + local_count - 1
    
    # Compute local sparse residuals
    local_res = zeros(T, M)
    for j in start_idx:min(end_idx, M)
        Aq = A * q[:, j]
        Bq = B * q[:, j]
        residual = Aq - lambda[j] * Bq
        local_res[j] = norm(residual)
    end
    
    # Reduce across ranks
    MPI.Allreduce!(local_res, res, MPI.SUM, comm)
end

# High-level MPI FeastKit interface
function mpi_feast(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0) where T<:Real
    # Unified MPI interface that detects matrix type
    
    Emin, Emax = interval
    
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Detect matrix type and call appropriate MPI solver
    if isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return mpi_feast_scsrgv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    else
        return mpi_feast_sygv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    end
end

# Standard eigenvalue problem MPI interface
function mpi_feast(A::AbstractMatrix{T}, interval::Tuple{T,T};
                   M0::Int = 10, fpm::Union{Vector{Int}, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0) where T<:Real
    # MPI interface for standard eigenvalue problems
    
    N = size(A, 1)
    
    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end
    
    # Create identity matrix of appropriate type
    if isa(A, SparseMatrixCSC)
        B = sparse(I, N, N)
    else
        B = Matrix{T}(I, N, N)
    end
    
    return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm, root=root)
end

# MPI performance benchmarking
function mpi_feast_benchmark(A::AbstractMatrix, B::AbstractMatrix, interval::Tuple, M0::Int;
                            comm::MPI.Comm = MPI.COMM_WORLD)
    # Benchmark MPI FeastKit performance

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit Performance Benchmark")
        println("="^40)
        println("Matrix size: $(Base.size(A, 1))")
        println("Search interval: $interval")
        println("MPI processes: $nprocs")
        println("Subspace size: $M0")
    end
    
    # MPI timing
    MPI.Barrier(comm)
    start_time = MPI.Wtime()
    
    result = mpi_feast(A, B, interval, M0=M0, comm=comm)
    
    MPI.Barrier(comm)
    end_time = MPI.Wtime()
    elapsed_time = end_time - start_time
    
    if rank == 0
        println("\nMPI FeastKit Results:")
        println("Time: $(round(elapsed_time, digits=3)) seconds")
        println("Eigenvalues found: $(result.M)")
        println("Convergence loops: $(result.loop)")
        println("Final residual: $(result.epsout)")
        println("Exit status: $(result.info)")
        
        if result.M > 0
            println("\nEigenvalues:")
            for i in 1:min(result.M, 5)  # Show first 5
                println("  λ[$i] = $(result.lambda[i])")
            end
            if result.M > 5
                println("  ... and $(result.M - 5) more")
            end
        end
    end
    
    return result
end

# Utility: Check if MPI is available and initialized
function mpi_feast_available()
    try
        return MPI.Initialized()
    catch
        return false
    end
end

# Utility: Initialize MPI for FeastKit if needed
function mpi_feast_init()
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("MPI FeastKit initialized with $nprocs processes")
    end

    return comm, rank, nprocs
end

# Utility: Clean up MPI
function mpi_feast_finalize()
    if MPI.Initialized()
        MPI.Finalize()
    end
end
