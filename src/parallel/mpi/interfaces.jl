# High-level MPI FeastKit interface
function mpi_feast(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0) where T<:Real
    # Unified MPI interface that detects matrix type

    Emin, Emax = interval

    if fpm === nothing
        fpm = zeros(Int, 64)
        feastinit!(fpm)
    end

    # Detect matrix type and call appropriate MPI solver
    fpm = _ensure_feast_parameters(fpm)
    if isa(A, SparseMatrixCSC) && isa(B, SparseMatrixCSC)
        return mpi_feast_scsrgv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    else
        return mpi_feast_sygv!(A, B, Emin, Emax, M0, fpm, comm=comm, root=root)
    end
end

# Standard eigenvalue problem MPI interface
function mpi_feast(A::AbstractMatrix{T}, interval::Tuple{T,T};
                   M0::Int = 10, fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
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
        B = spdiagm(0 => fill(one(T), N))
    else
        B = Matrix{T}(I, N, N)
    end

    return mpi_feast(A, B, interval, M0=M0, fpm=fpm, comm=comm, root=root)
end

function mpi_feast(A::SparseMatrixCSC{Complex{T},Int},
                   B::SparseMatrixCSC{Complex{T},Int},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   solver::Symbol = :direct,
                   solver_tol::Real = 0.0,
                   solver_maxiter::Int = 500,
                   solver_restart::Int = 30) where T<:Real
    Emin, Emax = interval
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_hcsrgv!(A, B, Emin, Emax, M0, params;
                             comm=comm, root=root, solver=solver,
                             solver_tol=solver_tol,
                             solver_maxiter=solver_maxiter,
                             solver_restart=solver_restart)
end

function mpi_feast(A::SparseMatrixCSC{Complex{T},Int},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast(A, B, interval; M0=M0, fpm=fpm, comm=comm, root=root, kwargs...)
end

function mpi_feast(A::Matrix{Complex{T}},
                   B::Matrix{Complex{T}},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   solver::Symbol = :direct,
                   solver_tol::Real = 0.0,
                   solver_maxiter::Int = 500,
                   solver_restart::Int = 30) where T<:Real
    Emin, Emax = interval
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_hegv!(A, B, Emin, Emax, M0, params;
                           comm=comm, root=root, solver=solver,
                           solver_tol=solver_tol,
                           solver_maxiter=solver_maxiter,
                           solver_restart=solver_restart)
end

function mpi_feast(A::Matrix{Complex{T}},
                   interval::Tuple{T,T}; M0::Int = 10,
                   fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   root::Int = 0,
                   kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast(A, B, interval; M0=M0, fpm=fpm, comm=comm, root=root, kwargs...)
end

function mpi_feast_general(A::SparseMatrixCSC{Complex{T},Int},
                           B::SparseMatrixCSC{Complex{T},Int},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           solver::Symbol = :direct,
                           solver_tol::Real = 0.0,
                           solver_maxiter::Int = 500,
                           solver_restart::Int = 30) where T<:Real
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_gcsrgv!(A, B, center, radius, M0, params;
                             comm=comm, root=root, solver=solver,
                             solver_tol=solver_tol,
                             solver_maxiter=solver_maxiter,
                             solver_restart=solver_restart)
end

function mpi_feast_general(A::SparseMatrixCSC{Complex{T},Int},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           kwargs...) where T<:Real
    B = spdiagm(0 => fill(one(Complex{T}), size(A, 1)))
    return mpi_feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                             comm=comm, root=root, kwargs...)
end

function mpi_feast_general(A::Matrix{Complex{T}},
                           B::Matrix{Complex{T}},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           solver::Symbol = :direct,
                           solver_tol::Real = 0.0,
                           solver_maxiter::Int = 500,
                           solver_restart::Int = 30) where T<:Real
    params = if fpm === nothing
        values = zeros(Int, 64)
        feastinit!(values)
        values
    elseif fpm isa FeastParameters
        fpm.fpm
    else
        fpm
    end

    return mpi_feast_gegv!(A, B, center, radius, M0, params;
                           comm=comm, root=root, solver=solver,
                           solver_tol=solver_tol,
                           solver_maxiter=solver_maxiter,
                           solver_restart=solver_restart)
end

function mpi_feast_general(A::Matrix{Complex{T}},
                           center::Complex{T}, radius::T; M0::Int = 10,
                           fpm::Union{Vector{Int}, FeastParameters, Nothing} = nothing,
                           comm::MPI.Comm = MPI.COMM_WORLD,
                           root::Int = 0,
                           kwargs...) where T<:Real
    B = Matrix{Complex{T}}(I, size(A, 1), size(A, 1))
    return mpi_feast_general(A, B, center, radius; M0=M0, fpm=fpm,
                             comm=comm, root=root, kwargs...)
end
