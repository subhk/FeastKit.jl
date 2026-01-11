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

    # Extract routine code digits for conditional defaults (like Fortran)
    # fpm[30] format: d1 d2 d3 d4 d5 d6
    # d1: 1=FEAST, 2=PFEAST; d2: precision; d3: 1=FEAST,2=IFEAST;
    # d4: 1=S,2=H,3=G; d5: interface; d6: variant
    dig = zeros(Int, 6)
    if fpm[30] != FEAST_UNINITIALIZED && fpm[30] > 0
        rem = fpm[30]
        for i in 1:6
            dig[7-i] = rem % 10
            rem = rem ÷ 10
        end
    end

    # fpm[1]: Print level (0=off, 1=on, <0=write to file)
    if fpm[1] == FEAST_UNINITIALIZED
        fpm[1] = 0  # Default: comments off
    elseif fpm[1] > 1
        throw(ArgumentError("Invalid fpm[1]=$(fpm[1]): print level must be 0, 1, or negative for file"))
    end

    # fpm[14]: Must be defined before fpm[2], fpm[8], and fpm[16]
    # FEAST execution options: 0=normal, 1=return subspace only, 2=stochastic estimate
    if fpm[14] == FEAST_UNINITIALIZED
        fpm[14] = 0  # Default: normal FEAST execution
    elseif fpm[14] < 0 || fpm[14] > 2
        throw(ArgumentError("Invalid fpm[14]=$(fpm[14]): must be 0, 1, or 2"))
    end

    # fpm[16]: Integration type (0=Gauss, 1=Trapezoidal, 2=Zolotarev)
    # Must be defined before fpm[2] and fpm[8]
    if fpm[16] == FEAST_UNINITIALIZED
        fpm[16] = 0  # Default: Gauss for FEAST symmetric/Hermitian
        if dig[3] == 2
            fpm[16] = 1  # Trapezoid default for IFEAST
        end
        # Trapezoid for non-Hermitian problems
        if dig[4] == 3
            fpm[16] = 1  # Trapezoid for non-symmetric eigenvalue
        end
        if dig[4] == 1 && dig[2] == 4
            fpm[16] = 1  # Trapezoid for complex symmetric eigenvalue
        end
    elseif fpm[16] < 0 || fpm[16] > 2
        throw(ArgumentError("Invalid fpm[16]=$(fpm[16]): must be 0, 1, or 2"))
    end
    # No Zolotarev for non-Hermitian problems
    if fpm[16] == 2
        if dig[4] == 3 || (dig[4] == 1 && dig[2] == 4)
            throw(ArgumentError("Invalid fpm[16]=2: Zolotarev not allowed for non-Hermitian problems"))
        end
    end

    # fpm[2]: Number of contour points (half-contour for symmetric/Hermitian)
    # Treat 0 as uninitialized (allows zero-initialized arrays)
    if fpm[2] == FEAST_UNINITIALIZED || fpm[2] <= 0
        fpm[2] = 8  # Default half-contour for symmetric/Hermitian
        if dig[3] == 2
            fpm[2] = 4  # IFEAST uses fewer nodes
        end
        if fpm[14] == 2
            fpm[2] = 3  # Stochastic estimate
        end
    elseif (fpm[16] == 0 || fpm[16] == 2) && fpm[2] > 20
        # Gauss or Zolotarev have restrictions (allow specific larger values)
        allowed_large = [24, 32, 40, 48, 56]
        if !(fpm[2] in allowed_large)
            throw(ArgumentError("Invalid fpm[2]=$(fpm[2]): max 20 for Gauss/Zolotarev, or use $allowed_large"))
        end
    end

    # fpm[3]: Convergence tolerance (stopping criteria: 10^(-fpm[3]))
    # Treat 0 as uninitialized (allows zero-initialized arrays)
    if fpm[3] == FEAST_UNINITIALIZED || fpm[3] == 0
        fpm[3] = 12  # Default: 10^-12
    elseif fpm[3] < 0 || fpm[3] > 16
        throw(ArgumentError("Invalid fpm[3]=$(fpm[3]): must be between 0 and 16"))
    end

    # fpm[4]: Maximum number of refinement loops
    # Treat 0 as uninitialized (allows zero-initialized arrays)
    if fpm[4] == FEAST_UNINITIALIZED || fpm[4] <= 0
        fpm[4] = 20  # Default
        if dig[3] == 2
            fpm[4] = 50  # IFEAST needs more iterations
        end
    elseif fpm[4] < 0
        throw(ArgumentError("Invalid fpm[4]=$(fpm[4]): must be non-negative"))
    end

    # fpm[5]: Initial subspace (0=random, 1=user-provided)
    if fpm[5] == FEAST_UNINITIALIZED
        fpm[5] = 0  # Default: random initial guess
    elseif fpm[5] != 0 && fpm[5] != 1
        throw(ArgumentError("Invalid fpm[5]=$(fpm[5]): must be 0 or 1"))
    end

    # fpm[6]: Convergence criteria (0=trace, 1=relative residual)
    # Fortran default is 1 (residual-based convergence)
    if fpm[6] == FEAST_UNINITIALIZED
        fpm[6] = 1  # Default: convergence on eigenvector relative residual
    elseif fpm[6] != 0 && fpm[6] != 1
        throw(ArgumentError("Invalid fpm[6]=$(fpm[6]): must be 0 or 1"))
    end

    # fpm[7]: Single precision tolerance (deprecated in FEAST v4.0)
    if fpm[7] == FEAST_UNINITIALIZED
        fpm[7] = 5  # Default
    elseif fpm[7] < 0 || fpm[7] > 7
        throw(ArgumentError("Invalid fpm[7]=$(fpm[7]): must be between 0 and 7"))
    end

    # fpm[8]: Number of contour points (full-contour for non-Hermitian)
    # Treat 0 as uninitialized (allows zero-initialized arrays)
    if fpm[8] == FEAST_UNINITIALIZED || fpm[8] <= 0
        fpm[8] = 16  # Default full-contour for non-Hermitian
        if dig[3] == 2
            fpm[8] = 8  # IFEAST uses fewer nodes
        end
        if fpm[14] == 2
            fpm[8] = 6  # Stochastic estimate
        end
    elseif fpm[8] < 2
        throw(ArgumentError("Invalid fpm[8]=$(fpm[8]): must be at least 2"))
    elseif fpm[16] == 0 && fpm[8] > 40
        # Gauss for non-Hermitian: full contour = 2 * half-Gauss contour
        allowed_large = [48, 64, 80, 96, 112]  # 2 * [24,32,40,48,56]
        if !(fpm[8] in allowed_large)
            throw(ArgumentError("Invalid fpm[8]=$(fpm[8]): max 40 for Gauss, or use $allowed_large"))
        end
    end

    # fpm[9]: MPI L2 communicator (for PFEAST)
    # Default: MPI_COMM_WORLD if MPI, 0 otherwise
    if fpm[9] == FEAST_UNINITIALIZED
        fpm[9] = 0
    end

    # fpm[10]: Store factorizations (0=no, 1=yes)
    if fpm[10] == FEAST_UNINITIALIZED
        fpm[10] = 1  # Default: store factorizations
        if dig[5] == 1
            fpm[10] = 0  # Direct call to RCI routines
        end
    elseif fpm[10] != 0 && fpm[10] != 1
        throw(ArgumentError("Invalid fpm[10]=$(fpm[10]): must be 0 or 1"))
    end

    # fpm[11]-[12]: Reserved for internal use
    if fpm[11] == FEAST_UNINITIALIZED
        fpm[11] = 0
    end
    if fpm[12] == FEAST_UNINITIALIZED
        fpm[12] = 0
    end

    # fpm[13]: Customize RCI interface (expert users)
    # 0=default, 1=reduced eigenvalue, 2=inner product, 3=both
    if fpm[13] == FEAST_UNINITIALIZED
        fpm[13] = 0
    elseif fpm[13] < 0 || fpm[13] > 3
        throw(ArgumentError("Invalid fpm[13]=$(fpm[13]): must be 0, 1, 2, or 3"))
    end

    # fpm[15]: Contour schemes for non-Hermitian FEAST
    # 0=two-sided contour (default for complex/real non-symmetric)
    # 1=one-sided standard
    # 2=one-sided complex symmetric (default for complex symmetric)
    if fpm[15] == FEAST_UNINITIALIZED
        fpm[15] = 0  # Default: two-sided contour for non-symmetric
        if dig[4] == 1  # Symmetric eigenvalue problem
            fpm[15] = 2  # One contour with left=right*
        end
    elseif fpm[15] < 0 || fpm[15] > 2
        throw(ArgumentError("Invalid fpm[15]=$(fpm[15]): must be 0, 1, or 2"))
    end
    if fpm[14] == 2
        fpm[15] = 1  # Compute only right contour for stochastic estimate
    end

    # fpm[17]: Deprecated in v4.0 (was integration type for non-symmetric)
    if fpm[17] == FEAST_UNINITIALIZED
        fpm[17] = 0
    end

    # fpm[18]: Ellipsoid contour ratio a/b * 100 (b is [Emin-Emax])
    # For symmetric/Hermitian: default 30 (narrow ellipse), 100=circle
    if fpm[18] == FEAST_UNINITIALIZED
        fpm[18] = 100  # Default: circle
        # For direct FEAST (dig[3]=1) and linear eigenvalue (dig[6]<=5)
        if dig[3] == 1 && dig[6] <= 5
            if dig[4] == 2  # Hermitian
                fpm[18] = 30
            end
            if dig[4] == 1 && dig[2] != 3 && dig[2] != 4  # Real symmetric
                fpm[18] = 30
            end
        end
    elseif fpm[18] < 0
        throw(ArgumentError("Invalid fpm[18]=$(fpm[18]): aspect ratio must be non-negative"))
    end

    # fpm[19]: Rotation angle in degrees for ellipsoid contour [-180,180]
    if fpm[19] == FEAST_UNINITIALIZED
        fpm[19] = 0  # Default: no rotation
    elseif fpm[19] < -180 || fpm[19] > 180
        throw(ArgumentError("Invalid fpm[19]=$(fpm[19]): must be between -180 and 180"))
    end

    # fpm[20]-[28]: Internal use (RCI state, linear system info, etc.)
    for i in 20:28
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[29]: Custom contour type flag
    # 0=user-provided custom contour, 1=generated using default contour
    if fpm[29] == FEAST_UNINITIALIZED
        fpm[29] = 0
    end

    # fpm[30]: Routine name code (set by caller)
    # fpm[31]: FEAST version * 10 (internal)
    fpm[31] = 40  # FEAST v4.0

    # fpm[32]: Stochastic estimate - number of steps/trials
    if fpm[32] == FEAST_UNINITIALIZED
        fpm[32] = 10
    end

    # fpm[33]-[35]: Reserved for factorization ID and parallel info
    for i in 33:35
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[36]: Bi-orthogonalization flag for non-symmetric FEAST (0=no, 1=yes)
    if fpm[36] == FEAST_UNINITIALIZED
        fpm[36] = 1  # Default: perform bi-orthogonalization
    end

    # fpm[37]: Internal use for bi-orthogonality test
    if fpm[37] == FEAST_UNINITIALIZED
        fpm[37] = 0
    end

    # fpm[38]: Spurious detection for non-symmetric FEAST (0=no, 1=yes)
    if fpm[38] == FEAST_UNINITIALIZED
        fpm[38] = 1  # Default: detect spurious eigenvalues
    end

    # fpm[39]: Solve standard vs generalized reduced system (0=generalized, 1=standard)
    if fpm[39] == FEAST_UNINITIALIZED
        fpm[39] = 0
    end

    # fpm[40]: Search interval option (0=user-defined, 1=largest, -1=smallest)
    if fpm[40] == FEAST_UNINITIALIZED
        fpm[40] = 0
    end

    # fpm[41]: Matrix scaling (0=no, 1=yes) for sparse drivers
    if fpm[41] == FEAST_UNINITIALIZED
        fpm[41] = 1  # Default: scale matrix
    end

    # fpm[42]: Mixed precision (0=double, 1=single solver)
    if fpm[42] == FEAST_UNINITIALIZED
        fpm[42] = 1  # Default: mixed precision
    end

    # fpm[43]: Switch FEAST to IFEAST interfaces (0=FEAST, 1=IFEAST)
    if fpm[43] == FEAST_UNINITIALIZED
        fpm[43] = 0
    end

    # fpm[44]: Iterative solver type for IFEAST (0=BiCGstab)
    if fpm[44] == FEAST_UNINITIALIZED
        fpm[44] = 0
    end

    # fpm[45]: Iterative solver accuracy 10^(-fpm[45])
    if fpm[45] == FEAST_UNINITIALIZED
        fpm[45] = 1
    end

    # fpm[46]: Maximum iterations for inner solver
    if fpm[46] == FEAST_UNINITIALIZED
        fpm[46] = 40
    end

    # fpm[47]: Enforced load balancing (0=off, 1=on)
    if fpm[47] == FEAST_UNINITIALIZED
        fpm[47] = 0
    end

    # fpm[48]: Vector convergence criteria
    if fpm[48] == FEAST_UNINITIALIZED
        fpm[48] = 0
    end

    # fpm[49]: L3 communicator (for PFEAST row distribution)
    if fpm[49] == FEAST_UNINITIALIZED
        fpm[49] = 0
    end

    # fpm[50]-[58]: Internal use for state, counters, etc.
    for i in 50:58
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[59]: Global L1 communicator (NEW_COMM_WORLD from pfeastinit)
    if fpm[59] == FEAST_UNINITIALIZED
        fpm[59] = 0
    end

    # fpm[60]: Output - counts BiCGstab iterations
    if fpm[60] == FEAST_UNINITIALIZED
        fpm[60] = 0
    end

    # fpm[61]-[63]: Reserved
    for i in 61:63
        if fpm[i] == FEAST_UNINITIALIZED
            fpm[i] = 0
        end
    end

    # fpm[64]: Additional feast parameters flag (0=64 params, 1=extended)
    if fpm[64] == FEAST_UNINITIALIZED
        fpm[64] = 0
    end

    return fpm
end

# Get tolerance for convergence
# fpm[3] is the tolerance exponent: tolerance = 10^(-fpm[3])
# Valid range is [0, 16] matching feastdefault! validation
function feast_tolerance(fpm::Vector{Int})
    if fpm[3] < 0 || fpm[3] > 16
        return 1e-12  # Default tolerance for out-of-range values
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

# Check if a custom contour mode is active
# Note: This is a simple check based on fpm[29] which indicates custom contour type:
#   fpm[29] = 0: default contour (ellipsoid based on Emin, Emax, fpm[18], etc.)
#   fpm[29] = 1: custom contour generated using default parameters
# For a complete check whether a custom contour object exists, use:
#   feast_get_custom_contour(fpm) !== nothing  (from feast_aux.jl)
# Note: fpm[15] is for contour schemes (two-sided vs one-sided), not custom contours
function feast_use_custom_contour(fpm::Vector{Int})
    return fpm[29] != 0
end

# Get number of integration points
function feast_integration_points(fpm::Vector{Int})
    return fpm[2]
end
