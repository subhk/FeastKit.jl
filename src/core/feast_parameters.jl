# Feast parameter initialization and management
# Translated from feast_tools.f90

function feastinit!(fpm::Vector{Int})
    length(fpm) >= 64 || throw(ArgumentError("fpm array must have at least 64 elements"))
    
    # Initialize all parameters to zero
    fill!(fpm, 0)
    
    # Set default values
    fpm[1] = 1    # Print level
    fpm[2] = 8    # Number of contour points
    fpm[3] = 12   # Stopping criteria
    fpm[4] = 20   # Maximum number of refinement loops
    fpm[5] = 0    # Initial guess
    fpm[6] = 0    # Convergence criteria
    fpm[7] = 5    # Error trace
    fpm[8] = 1    # Check input matrices
    fpm[9] = 0    # Single/double precision
    fpm[10] = 0   # Matrix type
    fpm[11] = 1   # Solver type
    fpm[12] = 0   # Direct solver
    fpm[13] = 0   # Matrix storage
    fpm[14] = 1   # Sub-space iteration
    fpm[15] = 0   # Custom contour
    fpm[16] = 0   # Integration type
    fpm[17] = 1   # Orthogonalization
    fpm[18] = 1   # Eigen-decomposition
    fpm[19] = 1   # Return mode
    fpm[20] = 1   # Use residual
    
    # Initialize remaining parameters to default values
    for i in 21:64
        fpm[i] = 0
    end
    
    return fpm
end

function feastinit()
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    return FeastParameters(fpm)
end

function feastdefault!(fpm::Vector{Int})
    # Validate and set default values for Feast parameters
    length(fpm) >= 64 || throw(ArgumentError("fpm array must have at least 64 elements"))
    
    # Validate fpm[1] - print level
    if fpm[1] < 0 || fpm[1] > 1
        @warn "Invalid print level fpm[1]=$(fpm[1]), setting to default (1)"
        fpm[1] = 1
    end
    
    # Validate fpm[2] - number of contour points
    if fpm[2] < 3
        @warn "Invalid number of contour points fpm[2]=$(fpm[2]), setting to default (8)"
        fpm[2] = 8
    end
    
    # Validate fpm[3] - stopping criteria
    if fpm[3] < 1 || fpm[3] > 15
        @warn "Invalid stopping criteria fpm[3]=$(fpm[3]), setting to default (12)"
        fpm[3] = 12
    end
    
    # Validate fpm[4] - maximum number of refinement loops
    if fpm[4] < 1
        @warn "Invalid max refinement loops fpm[4]=$(fpm[4]), setting to default (20)"
        fpm[4] = 20
    end
    
    # Set other parameters if they are zero
    if fpm[7] == 0
        fpm[7] = 5
    end
    
    if fpm[8] == 0
        fpm[8] = 1
    end
    
    if fpm[11] == 0
        fpm[11] = 1
    end
    
    if fpm[14] == 0
        fpm[14] = 1
    end
    
    if fpm[17] == 0
        fpm[17] = 1
    end
    
    if fpm[18] == 0
        fpm[18] = 1
    end
    
    if fpm[19] == 0
        fpm[19] = 1
    end
    
    if fpm[20] == 0
        fpm[20] = 1
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