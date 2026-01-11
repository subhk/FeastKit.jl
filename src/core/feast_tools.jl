# Feast utility and contour generation tools
# Translated from feast_tools.f90

using Random

# Helper functions for integration nodes
function gauss_legendre_point(n::Int, k::Int)
    # Use FastGaussQuadrature directly for robust Gauss–Legendre nodes/weights
    x, w = FastGaussQuadrature.gausslegendre(n)
    return x[k], w[k]
end

@inline function _feast_seeded_subspace!(work::AbstractMatrix{T}) where T<:Real
    N, M0 = size(work)
    seed = hash((N, M0))
    rng = MersenneTwister(seed)
    for j in 1:M0
        randn!(rng, view(work, :, j))
        col_norm = norm(view(work, :, j))
        if col_norm == 0
            work[1, j] = one(T)
            col_norm = one(T)
        end
        work[:, j] ./= col_norm
    end
    return work
end

@inline function _feast_seeded_subspace_complex!(work::AbstractMatrix{Complex{T}}) where T<:Real
    # Use REAL random values for initialization (zero imaginary parts)
    # This is correct for Hermitian eigenvalue problems because eigenvectors are real
    # (or can be chosen to be real for degenerate eigenvalues)
    # Using complex random values causes poor initial overlap with real eigenvectors
    N, M0 = size(work)
    M0 == 0 && return work
    seed = hash((N, M0, :complex))
    rng = MersenneTwister(seed)
    for j in 1:M0
        for i in 1:N
            work[i, j] = Complex{T}(randn(rng, T), zero(T))  # Real values only
        end
        col_norm = norm(view(work, :, j))
        if col_norm == 0
            work[1, j] = Complex{T}(one(T), zero(T))
            col_norm = one(T)
        end
        work[:, j] ./= col_norm
    end
    return work
end

# Zolotarev quadrature tables - precomputed optimal rational approximation points
# Translated from FEAST libnum.f90 (Stefan Guettel, Eric Polizzi 2013-2015)
# Format: ZOLOTAREV_TABLES[n] = (we0, [(xe1, we1), (xe2, we2), ...])
# where we0 is the weight for k=0 (initialization), and (xek, wek) are nodes/weights for k=1..n

const ZOLOTAREV_TABLES = Dict{Int, Tuple{ComplexF64, Vector{Tuple{ComplexF64, ComplexF64}}}}(
    # n=1: rate = 9.92e-1
    1 => (complex(-0.49800399400799011, 0.0),
          [(complex(0.0, 1.0), complex(0.0, 0.99800399400799011))]),

    # n=2: rate = 7.18e-1
    2 => (complex(0.41805096443248230, 0.0),
          [(complex(-0.99900149850137365, 0.044676682867128663), complex(-0.040933604666346268, 0.0018306055366585177)),
           (complex(0.99900149850137365, 0.044676682867128663), complex(0.040933604666346268, 0.0018306055366585177))]),

    # n=3: rate = 3.58e-1
    3 => (complex(-0.26356075833756432, 0.0),
          [(complex(-0.99992162986865651, 0.012519349855705481), complex(-0.0094403480868925395, 0.00011819628351771121)),
           (complex(0.0, 1.0), complex(0.0, 0.74467858236516826)),
           (complex(0.99992162986865651, 0.012519349855705481), complex(0.0094403480868925395, 0.00011819628351771121))]),

    # n=4: rate = 1.71e-1
    4 => (complex(0.14575533545956809, 0.0),
          [(complex(-0.99997862836826079, 0.0065377983092055223), complex(-0.0040449611415998383, 2.6445705300228226e-5)),
           (complex(-0.95625772508631735, 0.29252549156054936), complex(-0.16550649051159086, 0.050629517778791600)),
           (complex(0.95625772508631735, 0.29252549156054936), complex(0.16550649051159086, 0.050629517778791600)),
           (complex(0.99997862836826079, 0.0065377983092055223), complex(0.0040449611415998383, 2.6445705300228226e-5))]),

    # n=5: rate = 8.39e-2
    5 => (complex(-0.077374834133583259, 0.0),
          [(complex(-0.99999051974059949, 0.0043543574641737573), complex(-0.0023046399755639315, 1.0035321417280235e-5)),
           (complex(-0.99543837740544383, 0.095406691528514928), complex(-0.045688379331906018, 0.0043789522408392290)),
           (complex(0.0, 1.0), complex(0.0, 0.48097001541667800)),
           (complex(0.99543837740544383, 0.095406691528514928), complex(0.045688379331906018, 0.0043789522408392290)),
           (complex(0.99999051974059949, 0.0043543574641737573), complex(0.0023046399755639315, 1.0035321417280235e-5))]),

    # n=6: rate = 4.23e-2
    6 => (complex(0.040601929796073799, 0.0),
          [(complex(-0.99999466072397813, 0.0032678010245045879), complex(-0.0015423558061466516, 5.0401387941688375e-6)),
           (complex(-0.99900149850137365, 0.044676682867128663), complex(-0.017985701250813335, 0.00080434461022422783)),
           (complex(-0.85293349191434553, 0.52201959577280344), complex(-0.17924652624997453, 0.10970398051397839)),
           (complex(0.85293349191434553, 0.52201959577280344), complex(0.17924652624997453, 0.10970398051397839)),
           (complex(0.99900149850137365, 0.044676682867128663), complex(0.017985701250813335, 0.00080434461022422783)),
           (complex(0.99999466072397813, 0.0032678010245045879), complex(0.0015423558061466516, 5.0401387941688375e-6))]),

    # n=7: rate = 2.17e-2
    7 => (complex(-0.021237706090261987, 0.0),
          [(complex(-0.99999655648000429, 0.0026243147931738105), complex(-0.0011399193084610077, 2.9915174055686861e-6)),
           (complex(-0.99966343892137466, 0.025942414765997054), complex(-0.0089861107855352777, 0.00023319989914114602)),
           (complex(-0.97434899659958563, 0.22504229119296795), complex(-0.075755925208015523, 0.017497105287481524)),
           (complex(0.0, 1.0), complex(0.0, 0.34547899051275072)),
           (complex(0.97434899659958563, 0.22504229119296795), complex(0.075755925208015523, 0.017497105287481524)),
           (complex(0.99966343892137466, 0.025942414765997054), complex(0.0089861107855352777, 0.00023319989914114602)),
           (complex(0.99999655648000429, 0.0026243147931738105), complex(0.0011399193084610077, 2.9915174055686861e-6))]),

    # n=8: rate = 1.12e-2 (DEFAULT for symmetric/Hermitian problems)
    8 => (complex(0.011099137041258145, 0.0),
          [(complex(-0.99999758153396057, 0.0021993013049440135), complex(-0.00089892014626439772, 1.9770010320296091e-6)),
           (complex(-0.99985147448075562, 0.017234528675274002), complex(-0.0052457912271928649, 9.0422169329207065e-5)),
           (complex(-0.99333587640998278, 0.11525552757595411), complex(-0.034625385252140740, 0.0040175404307143140)),
           (complex(-0.73983485714849262, 0.67278851368618764), complex(-0.15051737271560608, 0.13687697801045523)),
           (complex(0.73983485714849262, 0.67278851368618764), complex(0.15051737271560608, 0.13687697801045523)),
           (complex(0.99333587640998278, 0.11525552757595411), complex(0.034625385252140740, 0.0040175404307143140)),
           (complex(0.99985147448075562, 0.017234528675274002), complex(0.0052457912271928649, 9.0422169329207065e-5)),
           (complex(0.99999758153396057, 0.0021993013049440135), complex(0.00089892014626439772, 1.9770010320296091e-6))]),

    # n=10: rate = 3.04e-3
    10 => (complex(0.0030298206493477586, 0.0),
           [(complex(-0.99999860413950714, 0.0016708438099384962), complex(-0.00063052218358594105, 1.0535055580202612e-6)),
            (complex(-0.99995317824297625, 0.0096768446184941678), complex(-0.0023906025222613010, 2.3134572353828454e-5)),
            (complex(-0.99900149850137365, 0.044676682867128663), complex(-0.010809177648659182, 0.00048340087836508665)),
            (complex(-0.97930459779150147, 0.20239195820097600), complex(-0.047956018043409793, 0.0099110250490149158)),
            (complex(-0.64113075594177837, 0.76743166066140622), complex(-0.11904157484648165, 0.14249242081357993)),
            (complex(0.64113075594177837, 0.76743166066140622), complex(0.11904157484648165, 0.14249242081357993)),
            (complex(0.97930459779150147, 0.20239195820097600), complex(0.047956018043409793, 0.0099110250490149158)),
            (complex(0.99900149850137365, 0.044676682867128663), complex(0.010809177648659182, 0.00048340087836508665)),
            (complex(0.99995317824297625, 0.0096768446184941678), complex(0.0023906025222613010, 2.3134572353828454e-5)),
            (complex(0.99999860413950714, 0.0016708438099384962), complex(0.00063052218358594105, 1.0535055580202612e-6))]),

    # n=12: rate = 8.28e-4
    12 => (complex(0.00082698721091084559, 0.0),
           [(complex(-0.99999908436863361, 0.0013532412550538205), complex(-0.00048687164580549528, 6.5885540028861806e-7)),
            (complex(-0.99997862836826079, 0.0065377983092055223), complex(-0.0013784196581295379, 9.0120223119170229e-6)),
            (complex(-0.99971931159972960, 0.023691728821737014), complex(-0.0047923676064111223, 0.00011357135190625232)),
            (complex(-0.99645774817202859, 0.084094923199501848), complex(-0.016899924452230135, 0.0014262499855059792)),
            (complex(-0.95625772508631735, 0.29252549156054936), complex(-0.056400393497717298, 0.017253249201870668)),
            (complex(-0.56039535450830469, 0.82822523907781964), complex(-0.093578819652718012, 0.13830296710345763)),
            (complex(0.56039535450830469, 0.82822523907781964), complex(0.093578819652718012, 0.13830296710345763)),
            (complex(0.95625772508631735, 0.29252549156054936), complex(0.056400393497717298, 0.017253249201870668)),
            (complex(0.99645774817202859, 0.084094923199501848), complex(0.016899924452230135, 0.0014262499855059792)),
            (complex(0.99971931159972960, 0.023691728821737014), complex(0.0047923676064111223, 0.00011357135190625232)),
            (complex(0.99997862836826079, 0.0065377983092055223), complex(0.0013784196581295379, 9.0120223119170229e-6)),
            (complex(0.99999908436863361, 0.0013532412550538205), complex(0.00048687164580549528, 6.5885540028861806e-7))]),

    # n=16: rate = 6.16e-5
    16 => (complex(6.1610602189565711e-5, 0.0),
           [(complex(-0.99999951365073736, 0.00098625467737379399), complex(-0.00033720271377405092, 3.3256791542695477e-7)),
            (complex(-0.99999232430038310, 0.0039180786512652823), complex(-0.00066519220239713778, 2.6062953723407513e-6)),
            (complex(-0.99994317254293952, 0.010660754418012596), complex(-0.0016400960309060023, 1.7485654672736178e-5)),
            (complex(-0.99961437632196248, 0.027768663101670359), complex(-0.0042082918803256564, 0.00011690372030085671)),
            (complex(-0.99741833534204793, 0.071809918002307196), complex(-0.010834902847825161, 0.00078006735739269233)),
            (complex(-0.98285560240079251, 0.18437696393360870), complex(-0.027404178905123672, 0.0051408358392403394)),
            (complex(-0.89067323792710495, 0.45464401815095595), complex(-0.061233481833219840, 0.031256621441574164)),
            (complex(-0.44227617016573273, 0.89687891563105204), complex(-0.059982325962616087, 0.12163640524928188)),
            (complex(0.44227617016573273, 0.89687891563105204), complex(0.059982325962616087, 0.12163640524928188)),
            (complex(0.89067323792710495, 0.45464401815095595), complex(0.061233481833219840, 0.031256621441574164)),
            (complex(0.98285560240079251, 0.18437696393360870), complex(0.027404178905123672, 0.0051408358392403394)),
            (complex(0.99741833534204793, 0.071809918002307196), complex(0.010834902847825161, 0.00078006735739269233)),
            (complex(0.99961437632196248, 0.027768663101670359), complex(0.0042082918803256564, 0.00011690372030085671)),
            (complex(0.99994317254293952, 0.010660754418012596), complex(0.0016400960309060023, 1.7485654672736178e-5)),
            (complex(0.99999232430038310, 0.0039180786512652823), complex(0.00066519220239713778, 2.6062953723407513e-6)),
            (complex(0.99999951365073736, 0.00098625467737379399), complex(0.00033720271377405092, 3.3256791542695477e-7))]),

    # n=20: rate = 5.32e-6
    20 => (complex(5.3206481313953855e-6, 0.0),
           [(complex(-0.99999969803306787, 0.00077719987397691893), complex(-0.00026587073269648168, 2.0663574649551199e-7)),
            (complex(-0.99999527008414571, 0.0030749612451655949), complex(-0.00046027251440315831, 1.4155766905424915e-6)),
            (complex(-0.99996337012912316, 0.0085586003234654379), complex(-0.00097668527058618232, 8.3610108892660893e-6)),
            (complex(-0.99971931159972960, 0.023691728821737014), complex(-0.0021356851685330131, 5.0603291695653665e-5)),
            (complex(-0.99857010050426131, 0.053469972795268089), complex(-0.0046698618773687553, 0.00026491556046687499)),
            (complex(-0.99371422051491652, 0.11192883656509048), complex(-0.010038792633579628, 0.0011191135813893063)),
            (complex(-0.96909936851788621, 0.21556858668095908), complex(-0.020726655887702879, 0.0042084686679282303)),
            (complex(-0.89067323792710495, 0.45464401815095595), complex(-0.039414579440067499, 0.013614046698279809)),
            (complex(-0.64113075594177837, 0.76743166066140622), complex(-0.060380633098199696, 0.054553700551279195)),
            (complex(-0.19656148888024621, 0.98049167291399236), complex(-0.031556785908093972, 0.12036696715174568)),
            (complex(0.19656148888024621, 0.98049167291399236), complex(0.031556785908093972, 0.12036696715174568)),
            (complex(0.64113075594177837, 0.76743166066140622), complex(0.060380633098199696, 0.054553700551279195)),
            (complex(0.89067323792710495, 0.45464401815095595), complex(0.039414579440067499, 0.013614046698279809)),
            (complex(0.96909936851788621, 0.21556858668095908), complex(0.020726655887702879, 0.0042084686679282303)),
            (complex(0.99371422051491652, 0.11192883656509048), complex(0.010038792633579628, 0.0011191135813893063)),
            (complex(0.99857010050426131, 0.053469972795268089), complex(0.0046698618773687553, 0.00026491556046687499)),
            (complex(0.99971931159972960, 0.023691728821737014), complex(0.0021356851685330131, 5.0603291695653665e-5)),
            (complex(0.99996337012912316, 0.0085586003234654379), complex(0.00097668527058618232, 8.3610108892660893e-6)),
            (complex(0.99999527008414571, 0.0030749612451655949), complex(0.00046027251440315831, 1.4155766905424915e-6)),
            (complex(0.99999969803306787, 0.00077719987397691893), complex(0.00026587073269648168, 2.0663574649551199e-7))]),
)

function zolotarev_point(n::Int, k::Int)
    # Zolotarev rational approximation points and weights
    # Uses precomputed optimal tables from FEAST libnum.f90

    if haskey(ZOLOTAREV_TABLES, n)
        we0, nodes = ZOLOTAREV_TABLES[n]
        if k == 0  # Special case for initialization
            return complex(0.0, 0.0), we0
        elseif 1 <= k <= length(nodes)
            xe, we = nodes[k]
            return xe, we
        end
    end

    # Fallback for unsupported n values - use trapezoidal-like approximation
    # with warning (this should rarely happen in practice)
    @warn "Zolotarev quadrature not available for n=$n, using approximation" maxlog=1

    if k == 0
        return complex(0.0, 0.0), complex(1.0, 0.0)
    end

    # Approximate using points on unit circle (trapezoidal-like)
    theta = π * (2*k - 1) / (2*n)
    xe = complex(cos(theta), sin(theta))
    we = complex(0.0, π/n)

    return xe, we
end

function feast_contour(Emin::T, Emax::T, fpm::Vector{Int}) where T<:Real
    # Ensure fpm parameters are initialized
    # If fpm[2] is still -111 (uninitialized), apply defaults first
    # NOTE: This should only happen if feastdefault! wasn't called by the caller
    if fpm[2] == FEAST_UNINITIALIZED || fpm[2] <= 0
        @debug "feast_contour calling feastdefault! because fpm[2]=$(fpm[2])"
        feastdefault!(fpm)
    end

    ne = fpm[2]   # Number of integration points (half-contour)
    fpm16 = fpm[16]  # Integration type: 0=Gauss, 1=Trapezoidal, 2=Zolotarev
    fpm18 = fpm[18]  # Ellipse ratio a/b * 100 (default: 100 = circle)

    # Parameters from Fortran implementation (matches zfeast_contour)
    r = (Emax - Emin) / T(2)  # Semi-major axis (horizontal)
    Emid = Emin + r           # Center point
    aspect_ratio = fpm18 * T(0.01)  # a/b ratio (vertical/horizontal)

    # Integration limits for parametrization (matches Fortran ba, ab)
    ba = -T(π) / 2
    ab = T(π) / 2

    # Generate half-contour (symmetric about real axis)
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)

    # Precompute Gauss–Legendre nodes/weights if needed
    x_gl = nothing
    w_gl = nothing
    if fpm16 == 0
        x_gl, w_gl = FastGaussQuadrature.gausslegendre(ne)
    end

    for e in 1:ne
        if fpm16 == 0  # Gauss-Legendre integration (matches Fortran exactly)
            xe = x_gl[e]
            we = w_gl[e]

            # Map x ∈ [-1, 1] to θ: theta = ba*xe + ab = -π/2*xe + π/2
            # This gives θ ∈ [π, 0] as xe goes from -1 to 1 (Fortran convention)
            theta = ba * xe + ab

            # Elliptical contour point: z(θ) = Emid + r*cos(θ) + i*r*aspect*sin(θ)
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta)

            # Jacobian (Fortran formula): jac = r*i*sin(θ) + r*aspect*cos(θ)
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)

            # Weight: Wne = (1/4) * we * jac (matches Fortran exactly)
            Wne[e] = T(0.25) * we * jac

        elseif fpm16 == 2  # Zolotarev integration (optimal for ellipses)
            zxe, zwe = zolotarev_point(ne, e)
            Zne[e] = zxe * r + Emid
            Wne[e] = zwe * r

        else  # Trapezoidal integration (fpm16 == 1)
            # Fortran: theta = π - (π/ne)/2 - (π/ne)*(e-1)
            theta = T(π) - (T(π)/ne)/2 - (T(π)/ne) * (e-1)

            # Elliptical contour point
            Zne[e] = Emid + r * cos(theta) + im * r * aspect_ratio * sin(theta)

            # Jacobian (Fortran formula)
            jac = r * im * sin(theta) + r * aspect_ratio * cos(theta)

            # Weight: Wne = (1/(2*ne)) * jac (matches Fortran)
            Wne[e] = (one(T) / (2 * ne)) * jac
        end
    end

    return FeastContour{T}(Zne, Wne)
end

function feast_gcontour(Emid::Complex{T}, r::T, fpm::Vector{Int}) where T<:Real
    # Ensure fpm parameters are initialized
    if fpm[8] == FEAST_UNINITIALIZED || fpm[8] <= 0
        feastdefault!(fpm)
    end

    # For non-Hermitian problems, use fpm[8] for full-contour point count
    ne = fpm[8]      # Number of integration points (full contour)
    fpm16 = fpm[16]  # Integration type: 0=Gauss, 1=Trapezoidal
    fpm18 = fpm[18]  # Ellipse ratio a/b * 100 (100 = circle)
    fpm19 = fpm[19]  # Rotation angle in degrees [-180, 180]

    aspect_ratio = fpm18 * T(0.01)  # a/b ratio

    # Ellipse axis rotation (convert degrees to radians)
    rotation_theta = (fpm19 / T(180)) * T(π)
    nr = r * (cos(rotation_theta) + im * sin(rotation_theta))

    # Integration limits (matches Fortran ba, ab for half-contour)
    ba = -T(π) / 2
    ab = T(π) / 2

    # Generate full contour in complex plane (matches zfeast_gcontour)
    Zne = Vector{Complex{T}}(undef, ne)
    Wne = Vector{Complex{T}}(undef, ne)

    if fpm16 == 0  # Gauss-Legendre integration
        # Fortran: uses two half-contours (upper and lower)
        x_gl_upper, w_gl_upper = FastGaussQuadrature.gausslegendre(ne ÷ 2)
        x_gl_lower, w_gl_lower = FastGaussQuadrature.gausslegendre(ne - ne ÷ 2)

        # Upper half (e = 1 to ne/2)
        for e in 1:(ne ÷ 2)
            xe = x_gl_upper[e]
            we = w_gl_upper[e]

            # theta = ba*xe + ab = -π/2*xe + π/2 (upper half: θ ∈ [π, 0])
            theta = ba * xe + ab

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/4) * we * jac
            Wne[e] = T(0.25) * we * jac
        end

        # Lower half (e = ne/2+1 to ne)
        for e in (ne ÷ 2 + 1):ne
            idx = e - ne ÷ 2
            xe = x_gl_lower[idx]
            we = w_gl_lower[idx]

            # theta = -ba*xe - ab = π/2*xe - π/2 (lower half: θ ∈ [0, -π] -> [-π, 0])
            theta = -ba * xe - ab

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian (same formula)
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/4) * we * jac
            Wne[e] = T(0.25) * we * jac
        end

    else  # Trapezoidal integration (fpm16 == 1)
        # Fortran: theta = π - (2π/ne)/2 - (2π/ne)*(e-1)
        for e in 1:ne
            theta = T(π) - (2 * T(π) / ne) / 2 - (2 * T(π) / ne) * (e - 1)

            # Elliptical contour with rotation
            Zne[e] = Emid + nr * cos(theta) + nr * im * aspect_ratio * sin(theta)

            # Jacobian
            jac = nr * im * sin(theta) + nr * aspect_ratio * cos(theta)

            # Weight: (1/ne) * jac (full contour trapezoidal)
            Wne[e] = (one(T) / ne) * jac
        end
    end

    return FeastContour{T}(Zne, Wne)
end

# Overload for real Emid (convenience function)
function feast_gcontour(Emid::T, r::T, fpm::Vector{Int}) where T<:Real
    return feast_gcontour(Complex{T}(Emid, zero(T)), r, fpm)
end

function feast_customcontour(Zne::Vector{Complex{T}},
                           fpm::Vector{Int}) where T<:Real
    ne = length(Zne)
    fpm[2] = ne  # Set number of integration points
    # Note: fpm[29] = 0 means user-provided custom contour (not generated by default)
    # fpm[15] is for contour schemes (two-sided vs one-sided), not custom contour flag
    # We leave fpm[29] = 0 since user is providing custom nodes

    # Compute weights using trapezoidal rule
    Wne = Vector{Complex{T}}(undef, ne)

    for i in 1:ne
        i_prev = i == 1 ? ne : i - 1
        i_next = i == ne ? 1 : i + 1

        # Trapezoidal rule weight
        Wne[i] = (Zne[i_next] - Zne[i_prev]) / (2 * ne)
    end

    return FeastContour{T}(Zne, Wne)
end

# Advanced contour generation functions following Fortran interface

"""
    feast_contour_expert(Emin, Emax, ne, integration_type, ellipse_ratio)

Generate Feast integration contour with expert-level control matching original Fortran implementation.

# Arguments
- `Emin, Emax`: Search interval bounds
- `ne`: Number of integration points (half-contour)
- `integration_type`: 0=Gauss-Legendre, 1=Trapezoidal, 2=Zolotarev  
- `ellipse_ratio`: Aspect ratio a/b * 100 (100 = circle)

# Returns
- `FeastContour` with integration nodes and weights
"""
function feast_contour_expert(Emin::T, Emax::T, ne::Int, 
                            integration_type::Int=0, 
                            ellipse_ratio::Int=100) where T<:Real
    fpm = zeros(Int, 64)
    fpm[2] = ne
    fpm[16] = integration_type  
    fpm[18] = ellipse_ratio
    
    return feast_contour(Emin, Emax, fpm)
end

"""
    feast_contour_custom_weights!(Zne, Wne)

Custom contour integration with user-provided nodes and weights.
Follows the Fortran expert interface pattern.

# Arguments  
- `Zne`: Complex integration nodes
- `Wne`: Complex integration weights (modified in-place)

# Returns
- `FeastContour` using provided nodes and computed/modified weights
"""  
function feast_contour_custom_weights!(Zne::Vector{Complex{T}}, 
                                     Wne::Vector{Complex{T}}) where T<:Real
    ne = length(Zne)
    
    # Validate input
    if length(Wne) != ne
        throw(ArgumentError("Zne and Wne must have same length"))
    end
    
    # User is responsible for providing correct weights
    # This interface allows maximum flexibility like the Fortran version
    
    return FeastContour{T}(copy(Zne), copy(Wne))
end

"""
    feast_rationalx(Zne, Wne, lambda)

Compute rational function values using custom integration nodes and weights.
Direct translation of dfeast_rationalx from Fortran.

For eigenvalues inside the contour, the rational function returns ≈1.
For eigenvalues outside, it returns ≈0.

# Arguments
- `Zne`: Integration nodes (half-contour)
- `Wne`: Integration weights (half-contour)
- `lambda`: Real eigenvalues to evaluate

# Returns
- Vector of rational function values
"""
function feast_rationalx(Zne::Vector{Complex{T}},
                         Wne::Vector{Complex{T}},
                         lambda::Vector{T}) where T<:Real
    ne = length(Zne)
    M = length(lambda)
    f = zeros(T, M)

    # Compute rational function (matches dfeast_rationalx exactly)
    # f(λ) = 2 * Re(Σ Wne[e] / (Zne[e] - λ))
    # Factor of 2 accounts for symmetric half-contour
    for j in 1:M
        for e in 1:ne
            f[j] += 2 * real(Wne[e] / (Zne[e] - lambda[j]))
        end
    end

    return f
end

"""
    feast_rational(Emin, Emax, fpm, lambda)

Compute rational function values for real eigenvalues using default ellipsoid contour.
Direct translation of dfeast_rational from Fortran.

# Arguments
- `Emin, Emax`: Search interval bounds
- `fpm`: FEAST parameters (fpm[2]=nodes, fpm[16]=integration type, fpm[18]=aspect ratio)
- `lambda`: Real eigenvalues to evaluate

# Returns
- Vector of rational function values
"""
function feast_rational(lambda::Vector{T}, Emin::T, Emax::T,
                        fpm::Vector{Int}) where T<:Real
    # Generate contour (matches dfeast_rational calling zfeast_contour)
    contour = feast_contour(Emin, Emax, fpm)

    # Compute rational using feast_rationalx
    f = feast_rationalx(contour.Zne, contour.Wne, lambda)

    # Add Zolotarev initialization if needed (matches Fortran)
    if fpm[16] == 2  # Zolotarev
        _, zwe = zolotarev_point(fpm[2], 0)
        f .+= real(zwe)
    end

    return f
end

"""
    feast_grationalx(Zne, Wne, lambda)

Compute rational function values for complex eigenvalues using custom contour.
Direct translation of zfeast_grationalx from Fortran.

# Arguments
- `Zne`: Integration nodes (full contour)
- `Wne`: Integration weights (full contour)
- `lambda`: Complex eigenvalues to evaluate

# Returns
- Vector of complex rational function values
"""
function feast_grationalx(Zne::Vector{Complex{T}},
                          Wne::Vector{Complex{T}},
                          lambda::Vector{Complex{T}}) where T<:Real
    ne = length(Zne)
    M = length(lambda)
    f = Vector{Complex{T}}(undef, M)

    # Compute rational function (matches zfeast_grationalx exactly)
    # f(λ) = Σ Wne[e] / (Zne[e] - λ)  (full contour, no factor of 2)
    for j in 1:M
        f[j] = zero(Complex{T})
        for e in 1:ne
            f[j] += Wne[e] / (Zne[e] - lambda[j])
        end
    end

    return f
end

"""
    feast_grational(Emid, r, fpm, lambda)

Compute rational function values for complex eigenvalues using default ellipsoid contour.
Direct translation of zfeast_grational from Fortran.

# Arguments
- `Emid`: Center of search region (complex)
- `r`: Radius of search region
- `fpm`: FEAST parameters (fpm[8]=nodes, fpm[16]=integration type, etc.)
- `lambda`: Complex eigenvalues to evaluate

# Returns
- Vector of complex rational function values
"""
function feast_grational(lambda::Vector{Complex{T}}, Emid::Complex{T},
                         r::T, fpm::Vector{Int}) where T<:Real
    # Generate contour (matches zfeast_grational calling zfeast_gcontour)
    contour = feast_gcontour(Emid, r, fpm)

    # Compute rational using feast_grationalx
    return feast_grationalx(contour.Zne, contour.Wne, lambda)
end

# Convenience overloads for type flexibility
function feast_rationalx(lambda::Vector{T},
                         Zne::AbstractVector{Complex{TZ}},
                         Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    ne = length(Zne)
    ne == length(Wne) || throw(ArgumentError("Zne and Wne must have the same length"))
    base = promote_type(T, TZ, TW)
    return feast_rationalx(
        Vector{Complex{base}}(Zne),
        Vector{Complex{base}}(Wne),
        Vector{base}(lambda))
end

function feast_grationalx(lambda::Vector{Complex{T}},
                          Zne::AbstractVector{Complex{TZ}},
                          Wne::AbstractVector{Complex{TW}}) where {T<:Real, TZ<:Real, TW<:Real}
    ne = length(Zne)
    ne == length(Wne) || throw(ArgumentError("Zne and Wne must have the same length"))
    base = promote_type(T, TZ, TW)
    return feast_grationalx(
        Vector{Complex{base}}(Zne),
        Vector{Complex{base}}(Wne),
        Vector{Complex{base}}(lambda))
end

# Legacy alias for backward compatibility
const feast_rational_expert = feast_rationalx

# Check if eigenvalue is inside contour
function feast_inside_contour(lambda::T, Emin::T, Emax::T) where T<:Real
    return Emin <= lambda <= Emax
end

function feast_inside_gcontour(lambda::Complex{T}, Emid::Complex{T}, r::T) where T<:Real
    return abs(lambda - Emid) <= r
end

# Sort eigenvalues and eigenvectors
function feast_sort!(lambda::Vector{T}, q::Matrix{VT}, 
                    res::Vector{T}, M::Int) where {T<:Real, VT}
    # Sort by eigenvalue magnitude
    perm = sortperm(lambda[1:M])
    lambda[1:M] = lambda[perm]
    q[:, 1:M] = q[:, perm]
    res[1:M] = res[perm]
    return nothing
end

# Sort function for complex eigenvalues (general case)
function feast_sort_general!(lambda::Vector{Complex{T}}, q::Matrix{Complex{T}}, 
                           res::Vector{T}, M::Int) where T<:Real
    # Sort by eigenvalue magnitude |lambda|
    perm = sortperm(abs.(lambda[1:M]))
    lambda[1:M] = lambda[perm]
    q[:, 1:M] = q[:, perm]
    res[1:M] = res[perm]
    return nothing
end

# Compute residual norms
function feast_residual!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                        lambda::Vector{T}, q::Matrix{T}, res::Vector{T}, 
                        M::Int) where T
    N = size(A, 1)
    
    for j in 1:M
        # Compute residual: r = A*q - lambda*B*q
        r = A * q[:, j] - lambda[j] * (B * q[:, j])
        res[j] = norm(r)
    end
    
    return nothing
end

# Extract name from Feast code (similar to feast_name subroutine)
function feast_name(code::Int)
    digits = Vector{Int}(undef, 6)
    rem = code
    
    for i in 1:6
        digits[6-i+1] = rem % 10
        rem = rem ÷ 10
    end
    
    name = ""
    
    # Parallel/serial
    if digits[1] == 2
        name *= "p"
    end
    
    # Precision
    if digits[2] == 1
        name *= "s"
    elseif digits[2] == 2
        name *= "d"
    elseif digits[2] == 3
        name *= "c"
    elseif digits[2] == 4
        name *= "z"
    end
    
    # Direct/iterative
    if digits[3] == 2
        name *= "i"
    end
    
    name *= "feast_"
    
    # Matrix type
    if digits[4] == 1
        name *= "s"
    elseif digits[4] == 2
        name *= "h"
    elseif digits[4] == 3
        name *= "g"
    end
    
    # Interface type
    if digits[5] == 1
        name *= "rci"
    elseif digits[5] == 2
        name *= "y"
    elseif digits[5] == 3
        name *= "b"
    elseif digits[5] == 4
        name *= "csr"
    elseif digits[5] == 5
        name *= "e"
    end
    
    # Variant
    if digits[6] == 1
        name *= "x"
    elseif digits[6] == 2
        name *= "ev"
    elseif digits[6] == 3
        name *= "evx"
    elseif digits[6] == 4
        name *= "gv"
    elseif digits[6] == 5
        name *= "gvx"
    elseif digits[6] == 6
        name *= "pev"   # Polynomial eigenvalue (matches Fortran output)
    elseif digits[6] == 7
        name *= "pevx"  # Polynomial eigenvalue with custom contour
    end
    
    return name
end
