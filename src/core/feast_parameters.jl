# Feast parameter initialization and management
# Translated from feast_tools.f90

# Sentinel value to indicate uninitialized parameters (matches Fortran FEAST)
const FEAST_UNINITIALIZED = -111

function feastinit!(fpm::Vector{Int})
    length(fpm) >= 64 || throw(ArgumentError("fpm array must have at least 64 elements"))

    # Initialize all parameters to -111 (sentinel value)
    # This matches the Fortran implementation exactly
    # Parameters set to -111 indicate "not set by user" and will be assigned defaults later
    for i in 1:64
        fpm[i] = FEAST_UNINITIALIZED
    end

    return fpm
end

function feastinit()
    fpm = Vector{Int}(undef, 64)
    feastinit!(fpm)
    return FeastParameters(fpm)
end

function feastinit_driver(fpm::Vector{Int}, N::Integer)
    feastinit!(fpm)
    if N > 0
        # Optionally suggest number of contour points based on problem size
        suggested = clamp(Int(ceil(sqrt(Float64(N)))), 8, 64)
        fpm[2] = suggested
    end
    return fpm
end

function feastinit_driver(N::Integer)
    fpm = Vector{Int}(undef, 64)
    return feastinit_driver(fpm, N)
end

function feastdefault!(fpm::Vector{Int})
    # Set default values for Feast parameters
    # Only modifies parameters that are still equal to FEAST_UNINITIALIZED (-111)
    # This allows users to override specific parameters before calling FEAST
    # Matches Fortran implementation in feast_tools.f90::feastdefault

    length(fpm) >= 64 || throw(ArgumentError("fpm array must have at least 64 elements"))

    # fpm[1]: Print level (0=off, 1=on)
    if fpm[1] == FEAST_UNINITIALIZED
        fpm[1] = 0  # Default: comments off
    elseif fpm[1] > 1
        throw(ArgumentError("Invalid fpm[1]=$(fpm[1]): print level must be 0 or 1"))
    end

    # fpm[14]: Must be defined before fpm[2] and fpm[16]
    if fpm[14] == FEAST_UNINITIALIZED
        fpm[14] = 0  # Default: standard iteration
    elseif fpm[14] < 0 || fpm[14] > 2
        throw(ArgumentError("Invalid fpm[14]=$(fpm[14]): must be 0, 1, or 2"))
    end

    # fpm[16]: Integration type (0=Gauss, 1=Trapezoidal, 2=Zolotarev)
    if fpm[16] == FEAST_UNINITIALIZED
        fpm[16] = 0  # Default: Gauss for symmetric/Hermitian
    elseif fpm[16] < 0 || fpm[16] > 2
        throw(ArgumentError("Invalid fpm[16]=$(fpm[16]): must be 0, 1, or 2"))
    end

    # fpm[2]: Number of contour points (half-contour for symmetric/Hermitian)
    if fpm[2] == FEAST_UNINITIALIZED || fpm[2] == 0
        fpm[2] = 8  # Default half-contour
        if fpm[14] == 2
            fpm[2] = 3  # Stochastic estimate
        end
    elseif fpm[2] < 1
        throw(ArgumentError("Invalid fpm[2]=$(fpm[2]): must be positive"))
    elseif (fpm[16] == 0 || fpm[16] == 2) && fpm[2] > 20
        # Gauss or Zolotarev restrictions
        throw(ArgumentError("Invalid fpm[2]=$(fpm[2]): max 20 for Gauss/Zolotarev integration"))
    end

    # fpm[3]: Convergence tolerance (stopping criteria: 10^(-fpm[3]))
    if fpm[3] == FEAST_UNINITIALIZED || fpm[3] == 0
        fpm[3] = 12  # Default: 10^-12
    elseif fpm[3] < 0 || fpm[3] > 16
        throw(ArgumentError("Invalid fpm[3]=$(fpm[3]): must be between 0 and 16"))
    end

    # fpm[4]: Maximum number of refinement loops
    if fpm[4] == FEAST_UNINITIALIZED || fpm[4] == 0
        fpm[4] = 20  # Default
    elseif fpm[4] < 1
        throw(ArgumentError("Invalid fpm[4]=$(fpm[4]): must be positive"))
    end

    # fpm[5]: Initial subspace (0=random, 1=user-provided)
    if fpm[5] == FEAST_UNINITIALIZED
        fpm[5] = 0  # Default: random initial guess
    elseif fpm[5] < 0 || fpm[5] > 1
        throw(ArgumentError("Invalid fpm[5]=$(fpm[5]): must be 0 or 1"))
    end

    # fpm[6]: Convergence criteria (0=relative, 1=absolute)
    if fpm[6] == FEAST_UNINITIALIZED
        fpm[6] = 0  # Default: relative convergence
    elseif fpm[6] < 0 || fpm[6] > 1
        throw(ArgumentError("Invalid fpm[6]=$(fpm[6]): must be 0 or 1"))
    end

    # fpm[7]: Error trace estimate
    if fpm[7] == FEAST_UNINITIALIZED
        fpm[7] = 5  # Default
    end

    # fpm[8]: Check input matrices
    if fpm[8] == FEAST_UNINITIALIZED
        fpm[8] = 1  # Default: check
    end

    # fpm[9]: MPI communicator (for PFEAST)
    if fpm[9] == FEAST_UNINITIALIZED
        fpm[9] = 0
    end

    # fpm[10]-[13]: Internal use
    for i in 10:13
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[15]: Custom contour (0=default, 1=custom)
    if fpm[15] == FEAST_UNINITIALIZED
        fpm[15] = 0  # Default: use standard contour
    end

    # fpm[17]: Orthogonalization
    if fpm[17] == FEAST_UNINITIALIZED
        fpm[17] = 1
    end

    # fpm[18]: Ellipse aspect ratio (percentage * 100, e.g., 100 = circle)
    if fpm[18] == FEAST_UNINITIALIZED
        fpm[18] = 100  # Default: circle
    elseif fpm[18] < 0
        throw(ArgumentError("Invalid fpm[18]=$(fpm[18]): aspect ratio must be positive"))
    end

    # fpm[19]-[20]: Internal use
    for i in 19:20
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[21]-[64]: Internal use and output parameters
    for i in 21:64
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    return fpm
end

# Get tolerance for convergence
function feast_tolerance(fpm::Vector{Int})
    if fpm[3] < 1 || fpm[3] > 15
        return 1e-12  # Default tolerance
    end
    return 10.0^(-fpm[3])
end

# Get machine epsilon
function feast_epsilon(::Type{Float64})
    return eps(Float64)
end

function feast_epsilon(::Type{Float32})
    return eps(Float32)
end

# Check if we should use custom contour
function feast_use_custom_contour(fpm::Vector{Int})
    return fpm[15] != 0
end

# Get number of integration points
function feast_integration_points(fpm::Vector{Int})
    return fpm[2]
end
