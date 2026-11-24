# Feast utility and contour generation tools
# Translated from feast_tools.f90

using Random

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

    throw(ArgumentError("Zolotarev quadrature is tabulated only for " *
                        "$(_FEAST_ZOLOTAREV_POINTS) contour points; got n=$n"))
end

# Integer or mixed bounds are natural to type -- feast_contour_expert(-2, 2, 16)
# used to die with InexactError deep inside, on `T(0.01)`. Promote first: the
# contour is complex-valued regardless, so there is nothing to gain from an
# integer element type.
function feast_contour(Emin::Real, Emax::Real, fpm::Vector{Int})
    T = float(promote_type(typeof(Emin), typeof(Emax)))
    return feast_contour(T(Emin), T(Emax), fpm)
end

function feast_contour(Emin::T, Emax::T, fpm::Vector{Int}) where T<:AbstractFloat
    # Resolve defaults if ANY parameter this builder reads is still the
    # -111 sentinel, not just the point count. Guarding on fpm[2] alone meant
    # `feastinit!(fpm); fpm[2] = 12` left fpm[18] (ellipse aspect ratio, in
    # percent) at -111, which mirrors the contour, reverses its orientation and
    # flips the rational filter to -1 inside.
    # A zero aspect ratio (what a bare `zeros(Int, 64)` leaves) collapses the
    # contour onto the real axis, and an unknown fpm[16] used to fall through
    # to the trapezoidal rule; feastdefault! repairs the first and rejects the
    # second.
    if fpm[2] == FEAST_UNINITIALIZED || fpm[2] <= 0 ||
       fpm[16] == FEAST_UNINITIALIZED || !(fpm[16] in (0, 1, 2)) ||
       fpm[18] == FEAST_UNINITIALIZED || fpm[18] <= 0
        @debug "feast_contour calling feastdefault!" fpm2=fpm[2] fpm16=fpm[16] fpm18=fpm[18]
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

    # Precompute Gauss–Legendre nodes/weights if needed. Typed empty fallback
    # keeps x_gl/w_gl concrete (Vector{Float64}) instead of Union{Nothing, ...}.
    x_gl, w_gl = fpm16 == 0 ? FastGaussQuadrature.gausslegendre(ne) :
                              (Float64[], Float64[])

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

# True when the nodes and weights already form a conjugate-closed (full) path:
# every node off the real axis pairs with a DISTINCT node carrying the
# conjugate weight, and there is at least one such pair. A node on the real
# axis is its own conjugate and needs a real weight (a closed path with an odd
# number of nodes crosses the axis at one). A Hermitian half contour lies on
# one side of the real axis, so it has no pairs and reads as "not yet
# completed", which is also the safe direction: the documented public
# convention is a half contour.
function _feast_hermitian_contour_is_closed(contour::FeastContour{T}) where T
    Zne, Wne = contour.Zne, contour.Wne
    n = length(Zne)
    n > 1 || return false
    all(isfinite, Zne) && all(isfinite, Wne) || return false
    # Tolerances are relative to the contour's own extent and to its largest
    # weight, so tiny contours and tiny weights still need distinct conjugate
    # partners and a large real translation cannot mask a mismatch. They must
    # absorb the rounding of sin/cos in generated nodes (a dozen ulps), which
    # is far below the separation between distinct nodes.
    center = sum(Zne) / n
    ztol = sqrt(eps(T)) * maximum(z -> abs(z - center), Zne)
    wtol = sqrt(eps(T)) * maximum(abs, Wne)
    matched = falses(n)
    npairs = 0
    for i in 1:n
        matched[i] && continue
        if abs(imag(Zne[i])) <= ztol
            abs(imag(Wne[i])) <= wtol || return false
            matched[i] = true
            continue
        end
        zc = conj(Zne[i])
        wc = conj(Wne[i])
        partner = 0
        for j in 1:n
            (j == i || matched[j]) && continue
            signbit(imag(Zne[i])) == signbit(imag(Zne[j])) && continue
            if abs(real(Zne[j]) - real(zc)) <= ztol &&
               abs(imag(Zne[j]) - imag(zc)) <= ztol &&
               abs(Wne[j] - wc) <= wtol
                partner = j
                break
            end
        end
        partner == 0 && return false
        matched[i] = true
        matched[partner] = true
        npairs += 1
    end
    return npairs > 0
end

# Public Hermitian contours contain one half of the conjugate-symmetric path.
# For a complex trial block, the lower-half solve is NOT the conjugate of the
# upper-half solution: both must be applied to the same B*Q. Expand the shifts
# explicitly so direct, iterative, banded and RCI drivers share that contract.
# Already-closed paths are returned untouched: `feast_hrcix!` and
# `feast_set_custom_contour!` take arbitrary user paths, and completing one of
# those twice would double every weight in the projector.
function _feast_complete_hermitian_contour(contour::FeastContour{T}) where T
    _feast_hermitian_contour_is_closed(contour) && return contour
    return FeastContour{T}(vcat(contour.Zne, conj.(contour.Zne)),
                           vcat(contour.Wne, conj.(contour.Wne)))
end

function feast_gcontour(Emid::Complex{T}, r::T, fpm::Vector{Int}) where T<:AbstractFloat
    # Same sentinel resolution as feast_contour: fpm[8] alone is not enough,
    # this builder also reads fpm[16], fpm[18] and the rotation fpm[19].
    if fpm[8] == FEAST_UNINITIALIZED || fpm[8] <= 0 ||
       fpm[16] == FEAST_UNINITIALIZED || !(fpm[16] in (0, 1, 2)) ||
       fpm[18] == FEAST_UNINITIALIZED || fpm[18] <= 0 ||
       fpm[19] == FEAST_UNINITIALIZED
        feastdefault!(fpm)
    end
    # Zolotarev rules approximate the step function of a real interval; there
    # is no full-contour version, and fpm[16] = 2 used to become trapezoidal.
    fpm[16] == 2 && throw(ArgumentError(
        "fpm[16]=2 (Zolotarev) applies to symmetric/Hermitian interval contours; " *
        "use 0 (Gauss) or 1 (trapezoidal) for a full contour"))

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
            # Continue from the upper half's right endpoint around the lower
            # half to the left. Quadrature is unchanged, but the node sequence
            # must trace a closed polygon when reused as a custom contour.
            idx = ne - e + 1
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

# Constant term of the filter a built-in interval contour stands for. Zolotarev
# rules approximate the step function by w0 + Σ w_e/(z_e - λ) (+ conjugates), and
# `feast_rational` has always included w0; the solvers add w0 X to the filtered
# block, which halves the filter's worst response outside the interval
# relative to its smallest response inside (0.41 -> 0.17 for 4 nodes, 0.023 ->
# 0.011 for 8). Gauss, trapezoidal and custom contours have no constant term.
function _feast_filter_offset(::Type{T}, fpm::Vector{Int}, custom::Bool) where T<:Real
    (custom || fpm[16] != 2) && return zero(T)
    return T(real(zolotarev_point(fpm[2], 0)[2]))
end

# Convenience overloads: a real center, and integer or mixed-precision inputs.
function feast_gcontour(Emid::Real, r::Real, fpm::Vector{Int})
    T = float(promote_type(typeof(Emid), typeof(r)))
    return feast_gcontour(Complex{T}(Emid, zero(T)), T(r), fpm)
end

function feast_gcontour(Emid::Complex, r::Real, fpm::Vector{Int})
    T = float(promote_type(real(typeof(Emid)), typeof(r)))
    return feast_gcontour(Complex{T}(Emid), T(r), fpm)
end

function feast_customcontour(Zne::Vector{Complex{T}},
                           fpm::Vector{Int}) where T<:Real
    ne = length(Zne)
    ne >= 3 || throw(ArgumentError("A closed custom contour needs at least 3 nodes, got $ne"))
    # Record the count both for interval (fpm[2]) and full-contour (fpm[8]) use.
    # fpm[29] (the custom contour registry ID) is left alone: this function
    # returns a contour rather than registering one.
    fpm[2] = ne
    fpm[8] = ne

    # Trapezoidal weights for the Cauchy integral (1/2πi) ∮ f(z) dz, taking the
    # nodes as samples of a closed curve z(t) at equally spaced parameter values
    # t_j = 2πj/ne, in counterclockwise order: w_j = z'(t_j) (2π/ne) / (2πi). The
    # derivative is that of the trigonometric interpolant through the nodes:
    # exact for circles and ellipses, whose weights then match feast_circle and
    # feast_ellipse, and spectrally accurate for other smooth curves. Central
    # differences of neighbouring nodes, used before,
    # are only second order: on a circle they scale the filter by
    # sin(2π/ne)/(2π/ne), 0.90 at 8 nodes, and two nodes gave zero weights.
    # A constant translation has zero derivative. Remove it before the DFT
    # so rounding in the nonzero-frequency sums is set by the contour's extent.
    origin = first(Zne)
    coeffs = Vector{Complex{T}}(undef, ne)
    for k in 1:(ne - 1)
        acc = zero(Complex{T})
        for j in 0:(ne - 1)
            acc += (Zne[j + 1] - origin) * cis(-T(2π) * k * j / ne)
        end
        coeffs[k + 1] = acc / ne
    end
    # Wavenumber of DFT index k; the Nyquist mode of an even count has no
    # well-defined derivative and is dropped.
    wavenumber(k) = 2k < ne ? k : (2k == ne ? 0 : k - ne)
    Wne = Vector{Complex{T}}(undef, ne)
    for j in 0:(ne - 1)
        deriv = zero(Complex{T})
        for k in 1:(ne - 1)
            κ = wavenumber(k)
            κ == 0 && continue
            deriv += (im * κ) * coeffs[k + 1] * cis(T(2π) * κ * j / ne)
        end
        Wne[j + 1] = deriv / (im * ne)
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
- `ellipse_ratio`: Vertical/horizontal semiaxis ratio * 100
  (100 = circle, 50 = half as tall as it is wide)

# Returns
- `FeastContour` with integration nodes and weights
"""
function feast_contour_expert(Emin::Real, Emax::Real, ne::Int,
                            integration_type::Int=0,
                            ellipse_ratio::Int=100)
    T = float(promote_type(typeof(Emin), typeof(Emax)))
    return _feast_contour_expert(T(Emin), T(Emax), ne, integration_type, ellipse_ratio)
end

function _feast_contour_expert(Emin::T, Emax::T, ne::Int,
                               integration_type::Int,
                               ellipse_ratio::Int) where T<:AbstractFloat
    if ne <= 0
        throw(ArgumentError("Number of integration points ne must be positive, got $ne"))
    end
    if integration_type < 0 || integration_type > 2
        throw(ArgumentError("integration_type must be 0 (Gauss), 1 (Trapezoidal), or 2 (Zolotarev), got $integration_type"))
    end
    if ellipse_ratio <= 0
        throw(ArgumentError("ellipse_ratio must be positive, got $ellipse_ratio"))
    end

    fpm = zeros(Int, 64)
    feastinit!(fpm)
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
function feast_rationalx(Zne::AbstractVector{Complex{T}},
                         Wne::AbstractVector{Complex{T}},
                         lambda::AbstractVector{<:Real}) where T<:Real
    ne = length(Zne)
    ne == length(Wne) ||
        throw(ArgumentError("Zne and Wne must have the same length; got $(ne) and $(length(Wne)). " *
                            "The argument order is (Zne, Wne, lambda)."))
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
function feast_rational(lambda::AbstractVector{T}, Emin::T, Emax::T,
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

`lambda` keeps the same element type as the nodes on purpose: there is a second
method taking `(lambda, Zne, Wne)` in the opposite order, and when every
argument is a complex vector the two are told apart only by how tightly this
one binds its type parameters.

# Arguments
- `Zne`: Integration nodes (full contour)
- `Wne`: Integration weights (full contour)
- `lambda`: Complex eigenvalues to evaluate

# Returns
- Vector of complex rational function values
"""
function feast_grationalx(Zne::AbstractVector{Complex{T}},
                          Wne::AbstractVector{Complex{T}},
                          lambda::AbstractVector{Complex{T}}) where T<:Real
    ne = length(Zne)
    ne == length(Wne) ||
        throw(ArgumentError("Zne and Wne must have the same length; got $(ne) and $(length(Wne)). " *
                            "The argument order is (Zne, Wne, lambda)."))
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
function feast_grational(lambda::AbstractVector{Complex{T}}, Emid::Complex{T},
                         r::T, fpm::Vector{Int}) where T<:Real
    # Generate contour (matches zfeast_grational calling zfeast_gcontour)
    contour = feast_gcontour(Emid, r, fpm)

    # Compute rational using feast_grationalx
    return feast_grationalx(contour.Zne, contour.Wne, lambda)
end

# Convenience overloads for type flexibility
function feast_rationalx(lambda::AbstractVector{T},
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

function feast_grationalx(lambda::AbstractVector{Complex{T}},
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

function feast_inside_gcontour(lambda::Complex{T}, Emid::Complex{T}, r::T;
                               fpm::Union{Vector{Int},Nothing} = nothing) where T<:Real
    w = lambda - Emid

    # Extract ellipse parameters from fpm if available
    aspect_ratio = one(T)
    rotation_theta = zero(T)
    if fpm !== nothing && length(fpm) >= 19
        fpm18 = fpm[18]
        fpm19 = fpm[19]
        if fpm18 > 0
            aspect_ratio = fpm18 * T(0.01)
        end
        if fpm19 != 0
            rotation_theta = (fpm19 / T(180)) * T(π)
        end
    end

    # Un-rotate the displacement to align with ellipse axes
    if rotation_theta != zero(T)
        w *= exp(-im * rotation_theta)
    end

    # Check (Re(w)/r)^2 + (Im(w)/(r*aspect))^2 <= 1
    x = real(w) / r
    y = imag(w) / (r * aspect_ratio)
    return x * x + y * y <= one(T)
end

function feast_inside_gcontour(lambda::T, Emid::Complex{T}, r::T;
                               fpm::Union{Vector{Int},Nothing} = nothing) where T<:Real
    return feast_inside_gcontour(Complex{T}(lambda), Emid, r; fpm=fpm)
end

# Sort eigenvalues and eigenvectors
function feast_sort!(lambda::Vector{T}, q::Matrix{VT}, 
                    res::Vector{T}, M::Int) where {T<:Real, VT}
    # Sort by eigenvalue without allocating permutation or column-copy buffers.
    @inbounds for i in 2:M
        λ = lambda[i]
        r = res[i]
        insert_at = i
        while insert_at > 1 && isless(λ, lambda[insert_at - 1])
            insert_at -= 1
        end
        insert_at == i && continue

        for j in (i - 1):-1:insert_at
            lambda[j + 1] = lambda[j]
            res[j + 1] = res[j]
        end
        lambda[insert_at] = λ
        res[insert_at] = r

        for row in axes(q, 1)
            q_value = q[row, i]
            for j in (i - 1):-1:insert_at
                q[row, j + 1] = q[row, j]
            end
            q[row, insert_at] = q_value
        end
    end
    return nothing
end

# Sort function for complex eigenvalues (general case)
function feast_sort_general!(lambda::Vector{Complex{T}}, q::Matrix{Complex{T}}, 
                           res::Vector{T}, M::Int) where T<:Real
    # Sort by eigenvalue magnitude without allocating abs/permutation buffers.
    @inbounds for i in 2:M
        λ = lambda[i]
        r = res[i]
        λ_abs2 = abs2(λ)
        insert_at = i
        while insert_at > 1 && isless(λ_abs2, abs2(lambda[insert_at - 1]))
            insert_at -= 1
        end
        insert_at == i && continue

        for j in (i - 1):-1:insert_at
            lambda[j + 1] = lambda[j]
            res[j + 1] = res[j]
        end
        lambda[insert_at] = λ
        res[insert_at] = r

        for row in axes(q, 1)
            q_value = q[row, i]
            for j in (i - 1):-1:insert_at
                q[row, j + 1] = q[row, j]
            end
            q[row, insert_at] = q_value
        end
    end
    return nothing
end

# Residuals are measured as ||A x - λ B x|| / (||B x|| max(|λ|, σ)), where σ is
# the spectral scale of the pencil. An eigenpair with |λ| >= σ is judged
# relative to its eigenvalue; one with |λ| < σ by backward error, which is what
# floating point -- and an inexact inner solver -- can actually deliver. Because
# σ scales with the pencil, rescaling A (a change of units) leaves every
# residual, and so every convergence decision, unchanged. The previous floor
# was a fixed 1, which made the test absolute for small-magnitude problems:
# any vector passed at scale 1e-8, and near-zero eigenvalues of large matrices
# could never converge.

# σ estimated from a probe block R as ||A R||_F / ||B R||_F: the root-mean-square
# eigenvalue magnitude for a normal pencil. Returns zero when the probes give
# no usable scale (for example A = 0), so the caller can fall back.
function _feast_spectral_scale(AR::AbstractMatrix, BR::AbstractMatrix)
    na = norm(AR)
    nb = norm(BR)
    σ = nb > zero(nb) ? na / nb : zero(na)
    return isfinite(σ) ? σ : zero(σ)
end

# Driver-side estimate for solver loops that hold A and B themselves.
_feast_spectral_scale(A, B, R::AbstractMatrix) =
    _feast_spectral_scale(A * R, B === nothing ? R : B * R)

# The same estimate for storage that is only reachable through mat-vecs, such
# as LAPACK band storage: `applyA!(y, x)` and `applyB!(y, x)` write A*x and B*x.
function _feast_banded_spectral_scale(R::AbstractMatrix, applyA!, applyB!)
    AR = similar(R)
    BR = similar(R)
    for j in axes(R, 2)
        applyA!(view(AR, :, j), view(R, :, j))
        applyB!(view(BR, :, j), view(R, :, j))
    end
    return _feast_spectral_scale(AR, BR)
end

"""
    _feast_polynomial_balance(norms) -> (γ, δ)

Balance a polynomial eigenproblem `Σ λ^i A_i` before linearizing it, from the
coefficient sizes `norms[i+1] = ‖A_i‖`: solve for `μ = λ/γ` with coefficients
`δ γ^i A_i`, where `γ = (‖A_0‖/‖A_d‖)^(1/d)` balances the extreme coefficients and
`δ` makes the largest scaled coefficient 1 (the parameter scaling of Fan, Lin
and Van Dooren, generalized by Betcke). Rescaling every coefficient, or
measuring `λ` in other units, yields the same balanced problem, so the
companion solve and its convergence test do not depend on either.
"""
function _feast_polynomial_balance(norms::AbstractVector{T}) where T<:Real
    d = length(norms) - 1
    γ = d >= 1 && norms[1] > zero(T) && norms[end] > zero(T) ?
        (norms[1] / norms[end])^(one(T) / d) : one(T)
    (isfinite(γ) && γ > zero(T)) || (γ = one(T))
    largest = zero(T)
    p = one(T)
    for c in norms
        largest = max(largest, p * c)
        p *= γ
    end
    δ = largest > zero(T) && isfinite(largest) ? inv(largest) : one(T)
    return γ, δ
end

# Run `f(fpm_inner)` on a copy of `fpm` whose registered custom contour, if any,
# is rescaled for the balanced variable μ = λ/γ; `f` sees the scaled region.
# The caller's registration is left untouched.
function _feast_with_scaled_contour(f, fpm::Vector{Int}, ::Type{T}, γ::T) where T<:Real
    inner = copy(fpm)
    contour = feast_get_custom_contour(T, fpm)
    contour === nothing && return f(inner)
    inner[29] = 0   # a fresh registration must not replace the caller's
    scaled = FeastContour{T}(contour.Zne ./ γ, contour.Wne ./ γ,
                             contour.vertices === nothing ? nothing : contour.vertices ./ γ)
    feast_set_custom_contour!(inner, scaled)
    try
        return f(inner)
    finally
        feast_clear_custom_contour!(inner)
    end
end

# Magnitude of the search region: the fallback floor when no spectral scale is
# available (FEAST's own convention, α = max(|Emin|, |Emax|)). A closed contour
# uses its farthest node, which also covers user-registered contours that
# (Emid, r) do not describe.
_feast_residual_scale(Emin::Real, Emax::Real) = max(abs(Emin), abs(Emax))
_feast_residual_scale(Zne::AbstractVector{<:Complex}) =
    isempty(Zne) ? zero(real(eltype(Zne))) : maximum(abs, Zne)

# The residual floor: the measured spectral scale, or the region magnitude.
_feast_residual_floor(spectral::Real, region::Real) =
    spectral > zero(spectral) ? oftype(float(region), spectral) : float(region)

# Normalize by B*q so a common rescaling of A and B cannot change convergence,
# and by max(|λ|, scale) so that rescaling A alone cannot either (see above).
# The floor also handles zero eigenvalues without dividing by norm(A*q). A
# common null vector of A and B cannot certify a finite eigenvalue, so a zero
# B*q is rejected.
@inline function _feast_scaled_residual(residual, Bq, lambda, scale)
    bnorm = norm(Bq)
    denom = max(abs(lambda), abs(scale))
    return bnorm > zero(bnorm) && denom > zero(denom) ?
        (norm(residual) / bnorm) / denom : oftype(bnorm, Inf)
end

# A contour filter maps an eigenvector with eigenvalue λ to ρ(λ) times itself:
# |ρ(λ)| is roughly 1 inside the region (about 1/2 on the boundary) for the
# built-in quadratures, and small outside. A Ritz vector whose filtered image
# has shrunk below this fraction of what the filter does at its own Ritz value
# is dominated by out-of-region eigenvectors, even though its Rayleigh quotient
# happens to land inside the region.
const _FEAST_SPURIOUS_RESPONSE = 0.25

"""
    _feast_filter_response(lambda, Zne, Wne, half)

Magnitude `|ρ(λ)|` of the rational filter `ρ(λ) = Σ Wne[e] / (Zne[e] - λ)` that a
contour sweep applies to an eigenvector with eigenvalue `lambda`. `half=true`
is for kernels that integrate over the upper half contour only and keep the
real part of `2 Σ Wne[e] (Zne[e] B - A)⁻¹ B`. `offset` is the filter's constant
term ([`_feast_filter_offset`](@ref)), which the Zolotarev filters have.

The spurious-pair screen measures responses against this value rather than
against 1: a flat ellipse with few nodes, or user weights that are rescaled,
give a filter whose in-region value is far from 1 without making FEAST any
less valid. A value that cannot be evaluated (a Ritz value on a node) falls
back to 1.
"""
function _feast_filter_response(lambda::Number, Zne::AbstractVector,
                                Wne::AbstractVector, half::Bool; offset::Real = 0)
    acc = zero(promote_type(eltype(Wne), eltype(Zne), typeof(lambda)))
    @inbounds for e in eachindex(Zne, Wne)
        acc += Wne[e] / (Zne[e] - lambda)
    end
    ρ = half ? abs(2 * real(acc) + offset) : abs(acc + offset)
    return isfinite(ρ) ? ρ : one(ρ)
end

# Refinement loops that must have completed before a Ritz pair may be declared
# spurious. By then every in-region eigendirection has been amplified relative
# to the out-of-region ones at least twice, so a genuine pair that is still
# converging has a filter response near |ρ(λ)| >= 1/2.
const _FEAST_SPURIOUS_MIN_LOOPS = 2

"""
    _feast_filter_rank(Y, X; scale=1)

Number of eigendirections in `span(X)` that the contour filter keeps: the
eigenvalues `μ` of the filter restricted to that subspace, `G = X⁺ Y` with
`Y = ρ(B⁻¹A) X`, that reach `_FEAST_SPURIOUS_RESPONSE * scale` in magnitude,
where `scale` is the filter's in-region response. Once the
subspace holds the region's eigenvectors, `G` approximates the restriction of
the spectral projector, whose eigenvalues are 1 on in-region directions and 0
elsewhere — independently of how the Rayleigh-Ritz basis mixes them.

`G` comes from the normal equations `(XᴴX) G = XᴴY`, which touch `X` and `Y`
only through two small products. When `XᴴX` is too ill-conditioned for that,
a pivoted QR of `X` drops its dependent columns instead.
"""
function _feast_filter_rank(Y::AbstractMatrix, X::AbstractMatrix; scale::Real = 1)
    m = size(X, 2)
    m == 0 && return 0
    RT = real(eltype(X))
    G = nothing
    gram = X' * X
    chol = cholesky!(Hermitian(gram); check=false)
    if issuccess(chol)
        dU = abs.(diag(chol.U))
        # The Gram matrix squares the conditioning of X; accept it only while
        # X itself stays well inside sqrt(eps)-conditioning.
        if minimum(dU) > eps(RT)^(1 // 4) * maximum(dU)
            G = chol \ (X' * Y)
        end
    end
    if G === nothing
        F = qr(X, ColumnNorm())
        d = abs.(diag(F.R))
        (isempty(d) || !(d[1] > zero(RT))) && return 0
        r = count(>(sqrt(eps(RT)) * d[1]), d)
        p = F.p[1:r]
        # X[:, p] = Q R, so X_r⁺ = R_r⁻¹ Q_rᴴ for the leading r pivoted columns.
        QtY = (F.Q' * Y[:, p])[1:r, :]
        G = UpperTriangular(F.R[1:r, 1:r]) \ QtY
    end
    μ = try
        eigvals(G)
    catch err
        @debug "Restricted filter eigenvalues failed; skipping the rank check" exception=err
        return m
    end
    all(isfinite, μ) || return m
    return count(v -> abs(v) >= _FEAST_SPURIOUS_RESPONSE * scale, μ)
end

# Keep the residual checks local when a large translation makes the usual
# near-convergence window cover the whole contour. Both limits remain in
# spectral-relative units, with an allowance for cancellation in A*x - λ*B*x.
function _feast_contour_residual_limits(tol::Real, residual_scale, Zne::AbstractVector)
    near_tol = sqrt(tol)
    residual_scale === nothing && return tol, near_tol
    center, radius = _feast_contour_extent(Zne)
    if zero(radius) < radius < near_tol * abs(center)
        scale = max(residual_scale, abs(center) + radius)
        local_limit = max(near_tol * (radius / scale), 8eps(typeof(radius)))
        return min(tol, local_limit), min(near_tol, local_limit)
    end
    return tol, near_tol
end

"""
    _feast_screen_spurious!(keep, Y, X, res, M, tol, lambda, Zne, Wne;
                            half=false, filter_rank=false)

Classify the previous loop's in-region Ritz pairs using the contour sweep that
has just filtered their vectors: column `j` of `Y` is `ρ(B⁻¹A) * X[:, j]`, and
`lambda[j]` is the pair's Ritz value. `Zne`, `Wne` are the nodes and weights of
that sweep, and `half` says whether it covered only the upper half contour
(see [`_feast_filter_response`](@ref)).

Converged pairs are kept. For a translated narrow contour, the convergence
and near-convergence limits also account for the extent of the region.
An unconverged pair whose filter
response `‖Y[:, j]‖ / ‖X[:, j]‖` is below `_FEAST_SPURIOUS_RESPONSE` times the
filter's own value `|ρ(lambda[j])|` is spurious: its vector is made of
out-of-region eigenvectors mixed so that its Rayleigh quotient falls inside the
region. Measuring against `|ρ(lambda[j])|` keeps the test independent of how
the filter is normalized. Nothing further filtering can do separates
such a mixture when the out-of-region eigenvalues are damped equally, so left
alone it blocks convergence until the loop budget runs out, and is reported as
an eigenvalue.

For a non-normal pencil the one-sided (oblique) Rayleigh-Ritz can also build a
spurious vector that contains an in-region eigenvector already represented by a
converged pair, so its own response stays high. `filter_rank=true` then counts
the in-region directions of the whole subspace instead
([`_feast_filter_rank`](@ref), over the columns of `X` and `Y`): when there are
no more of them than converged pairs, every unconverged pair is spurious.

Returns the number of genuine pairs when every unconverged pair is spurious,
so that the caller can finish with the converged ones (`keep` marks them), and
`nothing` while some genuine pair is still converging. Kernels that integrate
over half the contour must pass the real part of their accumulator, which is
the filtered vector.
"""
function _feast_screen_spurious!(keep::AbstractVector{Bool}, Y::AbstractMatrix,
                                 X::AbstractMatrix, res::AbstractVector, M::Int,
                                 tol::Real, lambda::AbstractVector,
                                 Zne::AbstractVector, Wne::AbstractVector;
                                 half::Bool = false, filter_rank::Bool = false,
                                 residual_scale = nothing, boundary = nothing,
                                 offset::Real = 0)
    M > 0 || return nothing
    nconverged = 0
    all_weak = true
    # Smallest filter value among the converged pairs, which are genuine
    # in-region eigenvalues: the scale against which the subspace's filter
    # eigenvalues are counted below.
    ρmin = oftype(float(real(one(eltype(Zne)))), Inf)
    # Near a node of a full contour the filter's value at a Ritz value can be
    # many times its value in the rest of the region, and changes faster than
    # the Ritz value of a pair that is still converging is known: such a pair
    # responds like the region, not like the node. So the response a pair is
    # held to is at most the region's floor. (On the half contour the Ritz
    # values are real, where the filter has no such peaks.)
    ρfloor = half ? oftype(ρmin, Inf) :
        boundary === nothing ?
            oftype(ρmin, _FEAST_INREGION_RESPONSE *
                         _feast_filter_response(first(_feast_contour_extent(Zne)), Zne, Wne,
                                                false; offset=offset)) :
            oftype(ρmin, _feast_filter_floor(boundary, Zne, Wne, false; offset=offset))
    tol, near_tol = _feast_contour_residual_limits(tol, residual_scale, Zne)
    edge = boundary === nothing ? Zne : boundary
    @inbounds for j in 1:M
        ρ = _feast_filter_response(lambda[j], Zne, Wne, half; offset=offset)
        if res[j] <= tol
            keep[j] = true
            nconverged += 1
            ρmin = min(ρmin, ρ)
            continue
        end
        keep[j] = false
        # An approximate eigenpair can have a weak response in a nonnormal
        # pencil. Protect pairs with small residuals on the contour's scale,
        # allowing for roundoff on the scale of the eigenproblem.
        res[j] <= near_tol && return nothing
        # A spurious pair mixes eigenvectors from outside the region, so its
        # residual ‖A x - λ B x‖ / ‖B x‖ is at least the distance from λ to the
        # nearest of them, hence to the region's boundary. A smaller residual
        # means an eigenvalue nearby inside the region (Bauer-Fike for a
        # normal pencil); for a non-normal one such a pair can sit on an
        # in-region invariant subspace that has more eigenvalues than subspace
        # columns, and its response says nothing. `residual_scale` is the
        # σ that `res` was divided by (with max(|λ|, σ)); `boundary` samples
        # the region's boundary (see `_feast_region_boundary`).
        if residual_scale !== nothing
            absres = res[j] * max(abs(lambda[j]), residual_scale)
            absres < _feast_polygon_distance(lambda[j], edge) && return nothing
        end
        xnorm = norm(view(X, :, j))
        ynorm = norm(view(Y, :, j))
        (xnorm > zero(xnorm) && ynorm < _FEAST_SPURIOUS_RESPONSE * min(ρ, ρfloor) * xnorm) ||
            (all_weak = false)
    end
    nconverged == M && return nothing
    all_weak && return nconverged
    # A pair with a strong response is either still converging or, for a
    # non-normal pencil, an oblique mixture that reuses a converged direction.
    # With nothing converged there is nothing to reuse, and a subspace with any
    # in-region direction could not pass the count, so skip it.
    nconverged > 0 && filter_rank &&
        _feast_filter_rank(Y, X; scale=ρmin) <= nconverged && return nconverged
    return nothing
end

# A Ritz value whose filter value is below this fraction of the filter's value
# at the region's center lies where the filter has already removed everything:
# no in-region eigenvector could have been outweighed by eigenvectors there.
const _FEAST_EMPTY_FAR_RESPONSE = 1e-3
# The smallest filter value inside the region, relative to its value at the
# centre: 1/2, reached on the boundary, for every circle and ellipse rule
# (Gauss and trapezoidal, 4 to 32 nodes). A genuine in-region eigenpair
# responds at least this strongly wherever it sits. Polygon corners go lower
# (about 1/4 at a right angle); see `_feast_filter_floor`, which measures it.
const _FEAST_INREGION_RESPONSE = 0.5

"""
    _feast_region_looks_empty(ritz, center, radius, Zne, Wne, half, prev_dist;
                              real_spectrum=true, nearest_residual=nothing, tol=0)

Decide what a Rayleigh-Ritz step that put no Ritz value inside the region
means. `ritz` holds all of its Ritz values, `center` and `radius` describe the
region, and `prev_dist` is the distance returned by the previous call if the
loop before this one also found nothing (negative otherwise).

An empty Ritz set does not by itself show that the region is empty: when
eigenvalues crowd the boundary from outside and the subspace is small, the
first few subspaces are dominated by them and an enclosed eigenvalue appears
only after further filtering. Two things do settle it:

- the filter is negligible (`_FEAST_EMPTY_FAR_RESPONSE`) everywhere at least as
  far from the center as the nearest Ritz value, so nothing near the boundary
  is competing. On the real axis the bound is sampled
  ([`_feast_filter_tail_bound`](@ref)), not read off at the Ritz value itself:
  Zolotarev and custom filters ripple outside the region and can vanish at an
  isolated point. For a complex spectrum the Ritz values themselves must all
  be in the negligible zone, and the nearest pair must also be a converged
  eigenpair; or
- the nearest Ritz value has stopped moving between two consecutive loops, so
  the subspace has converged onto out-of-region eigenvectors. For a
  non-Hermitian pencil (`real_spectrum=false`) that also needs the nearest pair
  to be a converged eigenpair (`nearest_residual(i)` for its index `i` in `ritz`
  at most `tol`, or the eps/ρ accuracy of a subspace the filter damped by ρ):
  with more eigenvalues inside than subspace columns the filter is
  close to the identity on the in-region eigenspace, the subspace can stall on
  a non-invariant part of it, and oblique Ritz values of such a subspace may
  lie outside the region and stay put.

Returns `(empty, dist)`; when `empty` is false the caller should filter again
and pass `dist` back in.
"""
function _feast_region_looks_empty(ritz::AbstractVector, center::Number, radius::Real,
                                   Zne::AbstractVector, Wne::AbstractVector,
                                   half::Bool, prev_dist::Real;
                                   real_spectrum::Bool = true,
                                   nearest_residual = nothing, tol::Real = 0,
                                   filtered_ratio = nothing, offset::Real = 0,
                                   boundary = nothing)
    RT = typeof(float(radius))
    isempty(ritz) && return true, -one(RT)
    dist = RT(Inf)
    inear = 0
    @inbounds for (i, v) in enumerate(ritz)
        d = RT(abs(v - center))
        d < dist && ((dist, inear) = (d, i))
    end
    isfinite(dist) || return true, -one(RT)
    ρcenter = _feast_filter_response(center, Zne, Wne, half; offset=offset)
    # `filtered_ratio` is ‖ρ(B⁻¹A) X‖ / ‖X‖ for the sweep's input X, passed once
    # X has itself been filtered: any in-region content then dominates X and
    # keeps the ratio near the filter's in-region value, so a negligible ratio
    # means the region holds nothing, wherever the Ritz values are.
    filtered_ratio !== nothing && filtered_ratio < _FEAST_EMPTY_FAR_RESPONSE * ρcenter &&
        return true, dist
    # How strongly the filter keeps what the subspace holds. On the real axis
    # the sampled tail bound covers everything beyond the nearest Ritz value;
    # a complex spectrum can sit in any direction, and sampling whole circles
    # there is too pessimistic (Gauss nodes leave the filter largest near the
    # axis), so the Ritz values themselves are used: all of them must be in
    # the negligible zone.
    reach = real_spectrum ?
        _feast_filter_tail_bound(center, dist, Zne, Wne, half; real_spectrum=true, offset=offset) :
        maximum(v -> isfinite(v) ? RT(_feast_filter_response(v, Zne, Wne, half; offset=offset)) : zero(RT), ritz)
    # Pair i as an eigenpair: a subspace the filter has damped by ρ is only
    # known to about eps/ρ, since the contour sum that produced it cancelled
    # terms that much larger. (`nearest_residual` gives any pair's residual.)
    # Its variables are declared local: a name it shares with this function
    # would otherwise be the same variable, overwritten by every call.
    function converged(i)
        nearest_residual === nothing && return true
        local ρi = RT(_feast_filter_response(ritz[i], Zne, Wne, half; offset=offset))
        local accuracy = ρi > zero(RT) ? 10 * eps(RT) * ρcenter / ρi : RT(Inf)
        return nearest_residual(i) <= max(tol, accuracy)
    end
    nearest_converged() = converged(inear)
    # For a non-Hermitian pencil the Ritz values alone prove nothing: oblique
    # Ritz values of an in-region subspace can sit far outside the region.
    # A filter with a constant term (Zolotarev) keeps an offset-weighted copy
    # of the trial block; when the spectrum spans many orders of magnitude,
    # the first Ritz values are dragged towards its far end however small that
    # weight, so they are read only once a second sweep has squared it.
    first_look = iszero(offset) || prev_dist >= zero(prev_dist)
    reach < _FEAST_EMPTY_FAR_RESPONSE * ρcenter && first_look &&
        (real_spectrum || nearest_converged()) && return true, dist
    settled = prev_dist >= zero(prev_dist) && abs(dist - prev_dist) <= sqrt(eps(RT)) * radius
    # A settled subspace shows the region empty only if an in-region
    # eigenvector would have displaced one of its pairs: some converged pair
    # must be amplified less than the filter's floor in the region (`boundary`,
    # see `_feast_filter_floor`; without it, a quarter of the central value).
    # Then an in-region component would have grown against that pair at every
    # sweep, and could not have fallen to what a converged residual allows. An
    # unconverged pair proves nothing: its Ritz value is wherever the oblique
    # projection put it this sweep. On the real axis the filter is below the
    # floor everywhere outside, but off it an eigenvalue just outside, near a
    # node, can be amplified far more than the region itself; a subspace with
    # too few columns then settles on such pairs whatever the region holds.
    # The same holds for the subspace as a whole: if the last sweep kept less
    # of it than the floor (`filtered_ratio`), some direction of it is weaker
    # than any in-region eigenvector, which would have taken its place.
    if settled && !real_spectrum
        ρfloor = boundary === nothing ? RT(_FEAST_INREGION_RESPONSE / 2) * ρcenter :
                 RT(_feast_filter_floor(boundary, Zne, Wne, half; offset=offset))
        displaced = filtered_ratio !== nothing && filtered_ratio < ρfloor
        for i in eachindex(ritz)
            displaced && break
            isfinite(ritz[i]) || continue
            RT(_feast_filter_response(ritz[i], Zne, Wne, half; offset=offset)) < ρfloor || continue
            converged(i) && (displaced = true; break)
        end
        displaced || return false, dist
    end
    return settled && nearest_converged(), dist
end

"""
    _feast_filter_tail_bound(center, dist, Zne, Wne, half; real_spectrum=true)

Largest filter magnitude `|ρ(λ)|` found by sampling points at distance `dist`
to `10⁴ dist` from `center`: on the real axis on both sides for a real
spectrum, on circles otherwise. The built-in Gauss and trapezoidal filters
decay monotonically away from the region, but Zolotarev and user contours
ripple, so the value at one point says little about the values beyond it.
"""
function _feast_filter_tail_bound(center::Number, dist::Real, Zne::AbstractVector,
                                  Wne::AbstractVector, half::Bool;
                                  real_spectrum::Bool = true, offset::Real = 0)
    RT = typeof(float(dist))
    dist > zero(RT) || return _feast_filter_response(center, Zne, Wne, half; offset=offset)
    bound = zero(RT)
    nradii = 32
    nangles = real_spectrum ? 2 : 16
    for k in 0:(nradii - 1)
        d = dist * exp(RT(log(1e4)) * k / (nradii - 1))
        for m in 0:(nangles - 1)
            direction = real_spectrum ? (m == 0 ? one(RT) : -one(RT)) : cis(RT(2π) * m / nangles)
            bound = max(bound, RT(_feast_filter_response(center + d * direction, Zne, Wne, half;
                                                         offset=offset)))
        end
    end
    return bound
end

# Scaled residual of one Ritz pair (λ, x), for the empty-region test of the
# non-Hermitian drivers; `apply_A!(y, x)` and `apply_B!(y, x)` are their matvecs.
function _feast_pair_residual(apply_A!, apply_B!, x::AbstractVector, λ, scale)
    Ax = similar(x)
    Bx = similar(x)
    apply_A!(Ax, x)
    apply_B!(Bx, x)
    return _feast_scaled_residual(Ax .- λ .* Bx, Bx, λ, scale)
end

# A lower bound for the filter's modulus inside the region, which every
# in-region eigenvector is amplified by at least. The filter is analytic inside
# the contour and has no zeros there, so its smallest modulus is on the
# boundary: sampled on `boundary` (see `_feast_region_boundary`) and along each
# of its edges, with a 10% margin for what falls between samples. About half
# the central value for the built-in circles and ellipses, lower at the
# corners of a polygon.
function _feast_filter_floor(boundary::AbstractVector, Zne::AbstractVector,
                             Wne::AbstractVector, half::Bool; offset::Real = 0)
    RT = float(real(eltype(Zne)))
    floor = RT(Inf)
    n = length(boundary)
    @inbounds for i in 1:n
        a = boundary[i]
        b = boundary[i == n ? 1 : i + 1]
        for t in (0, 1, 2, 3)
            z = a + (t / 4) * (b - a)
            floor = min(floor, RT(_feast_filter_response(z, Zne, Wne, half; offset=offset)))
        end
    end
    return RT(0.9) * floor
end

# Center and radius of a closed contour, for kernels whose region is described
# by its nodes rather than by an interval.
function _feast_contour_extent(Zne::AbstractVector{Complex{T}}) where T<:Real
    isempty(Zne) && return zero(Complex{T}), zero(T)
    center = sum(Zne) / length(Zne)
    return center, maximum(z -> abs(z - center), Zne)
end

# Retain the latest Ritz pairs, including unresolved candidates. If the check
# lost directions that already converged, return the saved pairs and report
# incomplete verification.
function _feast_seed_check_result!(lambda::AbstractVector, q::AbstractMatrix,
                                   res::AbstractVector, M::Int, converged::Bool, state)
    M >= state.seed_M && return M, converged
    return _feast_restore_seed_pairs!(lambda, q, res, state), false
end

function _feast_restore_seed_pairs!(lambda::AbstractVector, q::AbstractMatrix,
                                    res::AbstractVector, state)
    Ms = state.seed_M
    copyto!(view(lambda, 1:Ms), state.seed_lambda)
    copyto!(view(q, :, 1:Ms), state.seed_q)
    copyto!(view(res, 1:Ms), state.seed_res)
    return Ms
end

# Keep the converged pairs a seed check is about to verify (see above).
function _feast_save_seed_pairs!(state, lambda::AbstractVector, q::AbstractMatrix,
                                 res::AbstractVector, M::Int)
    state.seed_M = M
    state.seed_lambda = lambda[1:M]
    state.seed_q = q[:, 1:M]
    state.seed_res = res[1:M]
    return state
end

# Points along the boundary of a full-contour search region: a custom contour's
# vertices (or nodes), else 256 points of the built-in ellipse. The polygon
# through the few quadrature nodes is a poor stand-in: inscribed in the
# ellipse, it cuts off up to 1 - cos(π/ne) of the radius.
function _feast_region_boundary(::Type{T}, center::Number, radius::Real,
                                fpm::Vector{Int}) where T<:Real
    contour = feast_get_custom_contour(T, fpm)
    contour !== nothing && return contour.vertices === nothing ? contour.Zne : contour.vertices
    aspect = (fpm[18] > 0 ? fpm[18] : 100) * T(0.01)
    rot = cis(T(fpm[19] == FEAST_UNINITIALIZED ? 0 : fpm[19]) * T(π) / 180)
    m = 256
    return [Complex{T}(center) + T(radius) * rot * complex(cos(θ), aspect * sin(θ))
            for θ in (T(2π) .* (0:(m - 1)) ./ m)]
end

# Distance from z to the closed polygon through `nodes`.
function _feast_polygon_distance(z::Number, nodes::AbstractVector)
    n = length(nodes)
    n == 0 && return oftype(abs(float(z)), Inf)
    n == 1 && return abs(z - nodes[1])
    best = oftype(abs(float(z)), Inf)
    @inbounds for i in 1:n
        a = nodes[i]
        b = nodes[i == n ? 1 : i + 1]
        edge = b - a
        len2 = abs2(edge)
        t = len2 > 0 ? clamp(real(conj(edge) * (z - a)) / len2, 0, 1) : zero(real(edge))
        best = min(best, abs(z - (a + t * edge)))
    end
    return best
end

# Move the pairs marked in `keep[1:M]` to the front of `lambda`, the columns of
# `vectors`, and `res`, preserving their order. Returns how many were kept.
function _feast_compact_pairs!(lambda::AbstractVector, vectors::AbstractMatrix,
                               res::AbstractVector, keep::AbstractVector{Bool}, M::Int)
    kept = 0
    @inbounds for j in 1:M
        keep[j] || continue
        kept += 1
        if kept != j
            lambda[kept] = lambda[j]
            res[kept] = res[j]
            copyto!(view(vectors, :, kept), view(vectors, :, j))
        end
    end
    return kept
end

# Defective eigenvalues. The in-region Ritz vectors of an eigenvalue with a
# Jordan block of size k converge to eigenvectors of the rounding-perturbed
# pencil, which lie within about eps^(1/k) of each other, so the block becomes
# numerically dependent although the invariant subspace it spans is well
# conditioned. Swept as they are, the Ritz vectors are compressed to fewer
# columns at the next rank compression (`rank_tol = sqrt(eps)`): the general
# solvers then reported one eigenvalue where the region held k, and the
# complex-symmetric ones -- whose one-dimensional Rayleigh quotient xᵀAx / xᵀx
# is 0/0 at such an eigenvector -- stopped converging. When the in-region Ritz
# vectors are this close to dependent, the next sweep filters an orthonormal
# basis Z of their span instead; the Ritz vectors are kept, with X = Z * R, for
# the tests that read them.
_feast_ritz_dependence_tol(::Type{T}) where T<:Real = eps(T)^(one(T) / 4)

# Whether vectors of unit norm with Gram matrix G (M×M, overwritten) are within
# `_feast_ritz_dependence_tol` of dependent. The Cholesky factor's diagonal is
# the |R_jj| of their QR factorization; through the Gram matrix it is accurate
# to about sqrt(eps), well below the threshold. Allocation-free, since it runs
# every refinement loop.
function _feast_gram_dependent!(G::AbstractMatrix{CT}) where CT<:Complex
    M = size(G, 1)
    _, info = LinearAlgebra.LAPACK.potrf!('U', G)
    info > 0 && return true
    tol = _feast_ritz_dependence_tol(real(CT))
    @inbounds for j in 1:M
        real(G[j, j]) < tol && return true
    end
    return false
end

# For the solvers that run their own refinement loop: if the leading `M`
# columns of `Q_basis` -- the in-region Ritz vectors, of unit norm -- are nearly
# dependent, copy the `active` Ritz vectors to `Xr`, the triangular factor to
# `Rin`, and replace the block by an orthonormal basis of its span. Returns the
# size of the block replaced, or 0 if `Q_basis` was left as it was.
function _feast_orthonormal_ritz_block!(Q_basis::AbstractMatrix{CT},
                                        Xr::AbstractMatrix{CT},
                                        Rin::AbstractMatrix{CT},
                                        M::Int, active::Int) where CT<:Complex
    M >= 2 || return 0
    X = view(Q_basis, :, 1:M)
    G = view(Rin, 1:M, 1:M)
    mul!(G, adjoint(X), X)
    _feast_gram_dependent!(G) || return 0
    F = qr(X)
    copyto!(view(Xr, :, 1:active), view(Q_basis, :, 1:active))
    R = view(Rin, 1:M, 1:M)
    fill!(R, zero(CT))
    @inbounds for j in 1:M, i in 1:j
        R[i, j] = F.R[i, j]
    end
    Z = view(Q_basis, :, 1:M)
    fill!(Z, zero(CT))
    @inbounds for i in 1:M
        Z[i, i] = one(CT)
    end
    lmul!(F.Q, Z)
    return M
end

# After a sweep: the filtered images of the previous loop's Ritz vectors and the
# vectors themselves. `Q_proj` holds the filtered image of `Q_basis`; with a
# block of `k` orthonormalized columns (see `_feast_orthonormal_ritz_block!`),
# the images of the Ritz vectors are Q_proj * R on that block, written to `Ybuf`.
function _feast_ritz_images(Q_proj::AbstractMatrix, Q_basis::AbstractMatrix,
                            Xr::AbstractMatrix, Rin::AbstractMatrix, k::Int,
                            active::Int, Ybuf::AbstractMatrix)
    k == 0 && return view(Q_proj, :, 1:active), view(Q_basis, :, 1:active)
    Yr = view(Ybuf, :, 1:active)
    copyto!(view(Yr, :, (k + 1):active), view(Q_proj, :, (k + 1):active))
    mul!(view(Yr, :, 1:k), view(Q_proj, :, 1:k), view(Rin, 1:k, 1:k))
    return Yr, view(Xr, :, 1:active)
end

# Without a search region to measure against, fall back to the largest Ritz
# value so that the residual stays unit-free.
function _feast_default_residual_scale(lambda::AbstractVector, M::Int)
    T = real(eltype(lambda))
    M > 0 || return one(T)
    scale = maximum(abs, view(lambda, 1:M))
    return scale > zero(scale) ? T(scale) : one(T)
end

# Compute residual norms. `scale` is the residual floor σ (see
# `_feast_scaled_residual`); solvers pass their spectral scale, and without it
# the largest |λ| among the pairs is used so the result stays unit-free.
function feast_residual!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                        lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                        M::Int; scale = nothing) where T<:Real
    N = size(A, 1)
    Aq = Vector{T}(undef, N)
    Bq = Vector{T}(undef, N)
    residual = Vector{T}(undef, N)
    return feast_residual!(A, B, lambda, q, res, M, Aq, Bq, residual; scale=scale)
end

function feast_residual!(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                        lambda::Vector{T}, q::Matrix{T}, res::Vector{T},
                        M::Int, Aq::AbstractVector{T}, Bq::AbstractVector{T},
                        residual::AbstractVector{T}; scale = nothing) where T<:Real
    N = size(A, 1)
    @boundscheck begin
        size(A, 2) == N || throw(DimensionMismatch("A must be square"))
        size(B) == size(A) || throw(DimensionMismatch("A and B must have the same size"))
        size(q, 1) == N || throw(DimensionMismatch("q row count must match A"))
        length(lambda) >= M || throw(BoundsError(lambda, M))
        length(res) >= M || throw(BoundsError(res, M))
        length(Aq) >= N || throw(BoundsError(Aq, N))
        length(Bq) >= N || throw(BoundsError(Bq, N))
        length(residual) >= N || throw(BoundsError(residual, N))
    end

    α = scale === nothing ? _feast_default_residual_scale(lambda, M) : scale
    for j in 1:M
        # Relative residual, independent of the scaling of the pencil and q.
        qj = view(q, :, j)
        mul!(Aq, A, qj)
        mul!(Bq, B, qj)
        λ = lambda[j]
        @inbounds @simd for i in 1:N
            residual[i] = Aq[i] - λ * Bq[i]
        end
        res[j] = _feast_scaled_residual(residual, Bq, λ, α)
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
