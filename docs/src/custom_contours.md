# Custom Contour Integration

Define a circle, ellipse, or rectangle around the eigenvalues you want, then
pass that contour to `feast`. Start with the complete examples below; custom
weights, half-contours, and integration theory follow later in this guide.

```@contents
Pages = ["custom_contours.md"]
Depth = 2
```

---

## Built-in Circle, Ellipse, and Box

Use these exported constructors; they generate the nodes and normalized
weights for you. A rectangle is also called a box in this guide.

| Shape | Geometry arguments | Resolution |
|:--|:--|:--|
| `feast_circle(center, radius)` | Center `x + y*im`; positive radius | `n=64` total nodes by default |
| `feast_ellipse(center, a, b)` | Positive **semiaxes**: horizontal `a`, vertical `b` before rotation | `n=64` total nodes by default |
| `feast_rectangle(xmin, xmax, ymin, ymax)` | Real-axis bounds first, then imaginary-axis bounds | `points_per_edge=32` by default: **128 total nodes** |

For example, `center=2+3im` places a circle or ellipse at real coordinate 2 and
imaginary coordinate 3. A rectangle with arguments `(1, 3, 2, 4)` spans real
coordinates 1 to 3 and imaginary coordinates 2 to 4. Ellipse `rotation` is
counterclockwise in **radians**; `rotation=π/6` means 30 degrees. Geometry must
be finite, rectangle bounds must increase, and `n` must be at least 3.

This complete example defines all three shapes and solves the same problem
with each. Copy the constructor and solve line for the shape you need:

```@example standard_shapes
using FeastKit, LinearAlgebra

A = Matrix(Diagonal(ComplexF64[-0.3+0.2im, 0.4-0.1im, 2.5]))

circle = feast_circle(0, 1; n=64)
ellipse = feast_ellipse(0, 2, 1; n=64, rotation=π/6)
rectangle = feast_rectangle(-1, 1, -1, 1; points_per_edge=32)

circle_result = feast(A, circle; subspace_size=3)
ellipse_result = feast(A, ellipse; subspace_size=3)
rectangle_result = feast(A, rectangle; subspace_size=3)

results = (circle_result, ellipse_result, rectangle_result)
for result in results
    @assert result.converged && result.M == 2
    @assert sort(result.values; by=real) ≈ [-0.3+0.2im, 0.4-0.1im]
end
[result.values for result in results]
```

The contour controls integration and eigenvalue selection: `2.5` is excluded
here. Registration and cleanup are automatic, including after an exception.
For a generalized problem, use `feast(A, B, contour; ...)`.
All three constructors produce full closed contours. These overloads use the
general eigensolver and return complex eigenvalues in `FeastGeneralResult`,
including when `A` is real. Access eigenvalues through `result.values` and
eigenvectors through `result.vectors`.
See [Matrix types: detected or declared?](@ref matrix-properties) for how
real, complex, symmetric, and nonsymmetric inputs select solver paths.

Use `subspace_size` larger than the expected enclosed count, or equal to the
matrix dimension for a small full-space example like this one. The node count
controls contour integration accuracy; it is separate from `subspace_size`.
Keep eigenvalues away from contour boundaries and increase the node count when
needed, especially for boxes. Set the count in the contour constructor;
`quadrature_points`, if also supplied, must match the contour's node count.

For a real symmetric or complex Hermitian problem on a real interval, the
simpler call is `feast(A, (Emin, Emax))`; it creates its contour internally.
The [half-contour workflow](#Advanced-Half-Contours-for-Interval-Solves) is for
customizing that interval solve.

---

## Contour Integration Theory

### Mathematical Foundation

The FEAST algorithm uses contour integration to compute spectral projectors. For a matrix **A** with eigenvalues λᵢ and eigenvectors xᵢ, the spectral projector is:

$$P = \frac{1}{2\pi i} \oint_\Gamma (z\mathbf{I} - \mathbf{A})^{-1} dz$$

Where Γ is a contour enclosing the eigenvalues of interest.

### Moment-Based Approach

FeastKit computes moments of the spectral projector:

$$\mathbf{S}_k = \frac{1}{2\pi i} \oint_\Gamma z^k (z\mathbf{B} - \mathbf{A})^{-1} \mathbf{B}\mathbf{Q} dz$$

The numerical integration becomes:

$$\mathbf{S}_k \approx \sum_{e=1}^{n_e} w_e z_e^k (z_e\mathbf{B} - \mathbf{A})^{-1} \mathbf{B}\mathbf{Q}$$

!!! note "Key Insight"
    The quality of eigenvalue extraction depends critically on:
    - **Contour placement**: Must enclose target eigenvalues
    - **Integration accuracy**: Affects moment computation quality  
    - **Node distribution**: Should resolve eigenvalue clustering

---

## Built-in Integration Methods

### Gauss-Legendre Integration

**Best for**: High accuracy with minimal points, smooth integrands

```@example quadrature_gauss
using FeastKit
using FeastKit

# High-accuracy Gauss-Legendre with 16 points
contour = feast_contour_expert(-1.0, 1.0, 16, 0, 100)

println("Gauss-Legendre contour:")
println("Nodes: ", contour.Zne[1:5])  # Show first 5 nodes
println("Weights: ", contour.Wne[1:5]) # Show first 5 weights
@assert all(isfinite, contour.Zne) && all(isfinite, contour.Wne)
```

**Advantages**:
- Useful for smooth contour integrands
- Accuracy depends on the integrand and node count
- Well-established theory

**Disadvantages**:  
- Points not uniformly distributed
- May miss isolated eigenvalues

### Trapezoidal Rule Integration  

**Best for**: Robust integration, debugging, educational purposes

```@example quadrature_trapezoid
using FeastKit
# Trapezoidal rule with 12 points
contour = feast_contour_expert(-2.0, 2.0, 12, 1, 100)

# Visualize uniform node distribution
θ = [angle(z) for z in contour.Zne]
println("Angles: ", sort(θ))  # Should be uniformly spaced
@assert all(isfinite, contour.Zne) && all(isfinite, contour.Wne)
```

**Advantages**:
- Uniformly distributed points
- Simple and robust
- Good for debugging

**Disadvantages**:
- Accuracy depends on contour shape and distance to the spectrum
- Near-boundary eigenvalues may need more points

### Zolotarev Integration

**Use for**: Symmetric interval filtering; compare against Gauss or trapezoidal integration for your problem

```@example quadrature_zolotarev
using FeastKit
# Zolotarev integration for a symmetric interval
contour = feast_contour_expert(0.0, 4.0, 12, 2, 100)

println("Zolotarev contour characteristics:")
println("Nodes: ", length(contour.Zne))
@assert all(isfinite, contour.Zne) && all(isfinite, contour.Wne)
```

**Advantages**:
- Rational-filter construction for symmetric interval problems
- Precomputed nodes for supported counts
- May help distinguish eigenvalues near interval endpoints

**Disadvantages**:
- More complex implementation
- Problem-specific optimization

---

## Custom Contour Design

### Advanced Half-Contours for Interval Solves

Half-contours from `feast_contour_expert` are intended for the symmetric interval
solver, rather than the full-contour overload. Register one for the duration of the call
using `FeastKit.with_custom_contour`, passing the **same parameter vector** to
both the helper and the solver:

```@example custom_symmetric
using FeastKit, LinearAlgebra

A = Matrix(Diagonal([0.5, 1.0, 1.5, 3.0]))
interval = (0.0, 2.0)
contour = feast_contour_expert(interval..., 16, 0, 100)
fpm = feastinit().fpm  # Raw vector required by with_custom_contour

result = FeastKit.with_custom_contour(fpm, contour) do
    feast(A, interval; M0=4, fpm=fpm, backend=:serial)
end

@assert result.info == 0
@assert result.lambda ≈ [0.5, 1.0, 1.5]
result.lambda
```

For a generalized problem, use `feast(A, B, interval; ...)` inside the same
block. The helper restores the previous contour settings when the block exits,
including on exceptions. It is module-qualified because it is not exported.
Use a separate `fpm` vector for each concurrent solve.

!!! important "Half-contour versus full contour"
    `feast(A, interval)` solves real symmetric or complex Hermitian problems on a real
    interval. Its kernels account for conjugate symmetry using a half-contour,
    as returned by `feast_contour_expert`. Use `feast_general` with a full
    closed contour for complex search regions, or pass it to `feast(A, contour)`. Complex
    Hermitian solvers internally add the conjugate nodes and weights and solve
    both halves against the same trial block. Continue supplying only the
    half-contour; do not double its weights or append its conjugate yourself.
    Threaded/distributed real interval backends and MPI drivers also consume
    registered contours within their supported problem types.

Choose `M0` larger than the expected number of enclosed eigenvalues, or use
`M0 == size(A, 1)` for a small full-space solve. A saturated smaller subspace
returns `Feast_ERROR_M0` because completeness cannot be certified.

### Full-Contour Weight Convention

For a counterclockwise full contour, supply quadrature weights for
`dz / (2π * im)`, **not** for `dz` alone. For a parametrization `z(t)`, this is
`z′(t) * Δt / (2π * im)`. `feast_contour_custom_weights!` copies the nodes and
weights into a contour; it does not normalize the weights or register the contour.
Do not divide the weights returned by FeastKit's built-in contour generators again.

### Designing Your Own Contour

```@example custom_rectangle
using FeastKit, LinearAlgebra

function create_rectangular_contour(xmin, xmax, ymin, ymax, nx, ny)
    """
    Create rectangular contour for eigenvalues in complex rectangle.
    
    Parameters:
    - xmin, xmax: Real axis bounds
    - ymin, ymax: Imaginary axis bounds  
    - nx, ny: Number of points on horizontal/vertical segments
    """
    
    # Midpoint quadrature on each edge, traversed counterclockwise.
    # Each weight includes the normalization dz / (2π * im).
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    bottom_x = [xmin + (j - 0.5) * dx for j in 1:nx]
    bottom_nodes = [x + im*ymin for x in bottom_x]
    bottom_weights = fill(dx / (2π * im), nx)
    
    # Right edge: ymin to ymax  
    right_y = [ymin + (j - 0.5) * dy for j in 1:ny]
    right_nodes = [xmax + im*y for y in right_y]
    right_weights = fill(im * dy / (2π * im), ny)
    
    # Top edge: xmax to xmin (reverse direction)
    top_x = [xmax - (j - 0.5) * dx for j in 1:nx]
    top_nodes = [x + im*ymax for x in top_x] 
    top_weights = fill(-dx / (2π * im), nx)
    
    # Left edge: ymax to ymin (reverse direction)
    left_y = [ymax - (j - 0.5) * dy for j in 1:ny]
    left_nodes = [xmin + im*y for y in left_y]
    left_weights = fill(-im * dy / (2π * im), ny)
    
    # Combine all segments
    all_nodes = vcat(bottom_nodes, right_nodes, top_nodes, left_nodes)
    all_weights = vcat(bottom_weights, right_weights, top_weights, left_weights)
    
    return all_nodes, all_weights
end

# Example: Rectangle around complex eigenvalues
nodes, weights = create_rectangular_contour(-1, 3, -2, 2, 20, 16)
println("Created rectangular contour with $(length(nodes)) points")

# Use with FeastKit
corners = ComplexF64[-1-2im, 3-2im, 3+2im, -1+2im]
contour = FeastKit.FeastContour{Float64}(nodes, weights, corners)
A = Matrix(Diagonal(ComplexF64[0.5+0.1im, 2+im, 5]))
result = feast(A, contour; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 2+im]; atol=1e-9)
result.values
```

Without explicit vertices, general membership uses the polygon through the
quadrature nodes. For rectangles, prefer `feast_rectangle`, which retains the
actual corners separately.

### Circular Contour for Complex Eigenvalues

```@example custom_general
using FeastKit, LinearAlgebra

function create_circular_contour(center, radius, n_points)
    """
    Create circular contour for general eigenvalue problems.
    """
    θ = range(0, 2π, length=n_points+1)[1:end-1]  # Exclude 2π (same as 0)
    
    nodes = [center + radius * exp(im * θᵢ) for θᵢ in θ]
    # dz / (2π * im) = radius * exp(im * θ) / n_points
    weights = [radius * exp(im * θᵢ) / n_points for θᵢ in θ]
    
    return nodes, weights
end

# Example usage
center = 1.0 + 0.5im  
radius = 2.0
nodes, weights = create_circular_contour(center, radius, 64)

contour = feast_contour_custom_weights!(nodes, weights)

A = Matrix(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 + 0.5im, 4.0 + 0.5im]))
fpm = feastinit().fpm
fpm[16] = 1  # Trapezoidal nodes, rather than Gauss node-count validation

result = FeastKit.with_custom_contour(fpm, contour) do
    feast_general(A, center, radius; M0=3, fpm=fpm, backend=:serial)
end

@assert result.info == 0
@assert result.M == 2
@assert sort(result.lambda; by=real) ≈ [0.5 + 0.1im, 1.0 + 0.5im]
result.lambda
```

For a generalized problem, call `feast_general(A, B, center, radius; ...)`.
The `center` and `radius` arguments are still required by the interface; for this
circular example they match the custom contour. The registered contour supplies
the integration nodes and weights.

### Adaptive Contour Generation

This illustrative heuristic chooses settings from supplied eigenvalue estimates;
it does not discover the spectrum or guarantee convergence. Custom node-count
bounds must be valid for the selected integration rule.

```@example custom_adaptive
using FeastKit
function adaptive_elliptical_contour(Emin, Emax, eigenvalue_estimates; 
                                    min_points=8, max_points=20)
    """
    Create elliptical contour adapted to eigenvalue distribution.
    """
    
    isempty(eigenvalue_estimates) && throw(ArgumentError("Supply eigenvalue estimates"))
    3 <= min_points <= max_points || throw(ArgumentError("Invalid node-count bounds"))
    # Analyze eigenvalue clustering
    λ_center = (Emax + Emin) / 2
    λ_spread = (Emax - Emin) / 2
    
    # Estimate clustering near center
    center_density = sum(abs.(eigenvalue_estimates .- λ_center) .< λ_spread/4)
    total_estimates = length(eigenvalue_estimates)
    
    if center_density / total_estimates > 0.7
        # High clustering near center - use more points, Zolotarev method
        n_points = max_points
        method = 2  # Zolotarev
        aspect_ratio = 50  # Flatter ellipse
        println("Detected clustering: using $n_points Zolotarev points")
        
    elseif total_estimates < 5  
        # Few eigenvalues - use fewer points
        n_points = min_points
        method = 0  # Gauss-Legendre 
        aspect_ratio = 100  # Circle
        println("Few eigenvalues: using $n_points Gauss-Legendre points")
        
    else
        # Moderate distribution - standard approach
        n_points = clamp(16, min_points, max_points)
        method = 0  # Gauss-Legendre
        aspect_ratio = 100
        println("Standard distribution: using $n_points Gauss-Legendre points")
    end
    
    return feast_contour_expert(Emin, Emax, n_points, method, aspect_ratio)
end

# Example with eigenvalue estimates
λ_estimates = [0.5, 0.52, 0.54, 1.8, 1.82, 1.84]  # Two clusters
contour = adaptive_elliptical_contour(0.0, 2.5, λ_estimates)
@assert length(contour.Zne) == 16
@assert length(adaptive_elliptical_contour(0.0, 2.5, [1.2, 1.25, 1.3]).Zne) == 20
@assert length(adaptive_elliptical_contour(0.0, 2.5, [0.2, 2.2]).Zne) == 8
```

---

## Advanced Applications

### Multi-Level Contour Strategy

This helper is for real `Float64` symmetric matrices and disjoint intervals.
It concatenates the results, so overlapping intervals would duplicate eigenpairs:

```@example custom_multiregion
using FeastKit, LinearAlgebra
function multi_level_feast(A::AbstractMatrix{Float64}, eigenvalue_regions; M0_per_region=10)
    """
    Apply FeastKit to multiple regions with customized contours.
    """
    
    all_eigenvalues = Float64[]
    all_eigenvectors = Matrix{Float64}(undef, size(A, 1), 0)
    
    for (i, (region_min, region_max, description)) in enumerate(eigenvalue_regions)
        println("Processing region $i: $description")
        println("Interval: [$region_min, $region_max]")
        
        # Customize contour for this region
        width = region_max - region_min
        
        if width < 0.01  # Very narrow region
            n_points = 20
            method = 2  # Zolotarev for high precision
        elseif width > 10  # Very wide region  
            n_points = 12
            method = 1  # Trapezoidal for robustness
        else
            n_points = 16
            method = 0  # Gauss-Legendre standard
        end
        
        contour = feast_contour_expert(region_min, region_max, 
                                     n_points, method, 100)
        
        # Apply FeastKit to this region
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[2] = n_points
        fpm[16] = method
        
        result = FeastKit.with_custom_contour(fpm, contour) do
            feast(A, (region_min, region_max); M0=M0_per_region, fpm=fpm,
                  backend=:serial)
        end
        
        result.converged || error(result.message)
        println("Found $(result.M) eigenvalues in region $i")
        
        if result.M > 0
            append!(all_eigenvalues, result.lambda[1:result.M])
            all_eigenvectors = hcat(all_eigenvectors, result.q[:, 1:result.M])
        end
    end
    
    # Sort combined results
    perm = sortperm(all_eigenvalues)
    all_eigenvalues = all_eigenvalues[perm]
    all_eigenvectors = all_eigenvectors[:, perm]
    
    println("\nTotal eigenvalues found: $(length(all_eigenvalues))")
    return all_eigenvalues, all_eigenvectors
end

# Example usage
regions = [
    (0.01, 0.1, "Low frequency modes"),
    (0.8, 1.2, "Mid-range cluster"), 
    (4.5, 5.5, "High frequency modes")
]

A = Matrix(Diagonal([0.05, 0.85, 1.0, 1.1, 5.0, 8.0]))
eigenvalues, eigenvectors = multi_level_feast(A, regions)
@assert isapprox(eigenvalues, [0.05, 0.85, 1.0, 1.1, 5.0]; atol=1e-9)
@assert norm(A*eigenvectors - eigenvectors*Diagonal(eigenvalues)) < 1e-8
```

### Rational Filters and Eigenvalue Counts

`feast_rational` and `feast_rational_expert` evaluate a contour's rational
filter at the supplied points. They do not inspect `A`. Counting grid points
where the filter is near one counts sample points, not eigenvalues.

For a matrix-dependent estimate, use `feast_estimate_count`:

```@example contour_count_estimate
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
interval = (0.5, 2.5)
estimate = feast_estimate_count(A, interval; nprobe=16)
# Leave headroom and verify the result; the estimate is stochastic.
result = feast(A, interval; subspace_size=3)
@assert result.converged && result.M == 2
(estimate=estimate, found=result.M)
```

---

## Advanced Contour Shapes

### Star-Shaped Contours

For eigenvalues with radial distribution:

```@example custom_star
using FeastKit, LinearAlgebra
function create_star_contour(center, radius, n_spikes, n_points_per_spike)
    """
    Create star-shaped contour for eigenvalues with radial symmetry.
    """
    nodes = ComplexF64[]
    weights = ComplexF64[]
    
    n = n_spikes * n_points_per_spike
    Δθ = 2π / n
    for j in 0:n-1
        # A smooth closed curve, not disconnected radial spokes.
        θ = (j + 0.5) * Δθ
        r = radius * (1 + 0.3 * cos(n_spikes * θ))
        dr = -0.3 * radius * n_spikes * sin(n_spikes * θ)
        push!(nodes, center + r * exp(im * θ))
        dz_dθ = (dr + im * r) * exp(im * θ)
        push!(weights, dz_dθ * Δθ / (2π * im))
    end
    
    return nodes, weights
end

nodes, weights = create_star_contour(0, 2, 5, 32)
contour = FeastKit.FeastContour{Float64}(nodes, weights)
A = Matrix(Diagonal(ComplexF64[-0.3+0.2im, 0.4-0.1im, 4]))
result = feast(A, contour; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [-0.3+0.2im, 0.4-0.1im]; atol=1e-9)
@assert abs(sum(weights)) < 1e-12
result.values
```

### Lens-Shaped Contours

For eigenvalues in bimodal distributions:

```@example custom_lens
using FeastKit, LinearAlgebra
function create_lens_contour(focus1, focus2, width, n_points)
    """
    Create lens-shaped (elliptical) contour between two focal points.
    """
    # Ellipse parameters
    center = (focus1 + focus2) / 2
    focus_distance = abs(focus2 - focus1)
    width > 0 || throw(ArgumentError("width must be positive"))
    minor_axis = width
    major_axis = sqrt(focus_distance^2 + width^2)
    rotation = cis(angle(complex(focus2 - focus1)))
    
    θ = range(0, 2π, length=n_points+1)[1:end-1]
    
    nodes = ComplexF64[]
    weights = ComplexF64[]
    
    for θᵢ in θ
        # Parametric ellipse
        x = (major_axis/2) * cos(θᵢ)
        y = (minor_axis/2) * sin(θᵢ)
        
        z = center + rotation * (x + im * y)
        push!(nodes, z)
        
        # Derivative for weight calculation
        dx_dθ = -(major_axis/2) * sin(θᵢ)  
        dy_dθ = (minor_axis/2) * cos(θᵢ)
        dz_dθ = rotation * (dx_dθ + im * dy_dθ)
        
        weight = dz_dθ * (2π / n_points) / (2π * im)
        push!(weights, weight)
    end
    
    return nodes, weights
end

nodes, weights = create_lens_contour(-1, 1, 2, 64)
contour = FeastKit.FeastContour{Float64}(nodes, weights)
A = Matrix(Diagonal(ComplexF64[-0.3+0.2im, 0.4-0.1im, 4]))
result = feast(A, contour; subspace_size=3)
@assert result.converged && result.M == 2
@assert isapprox(sort(result.values; by=real), [-0.3+0.2im, 0.4-0.1im]; atol=1e-9)
@assert abs(sum(weights)) < 1e-12
result.values
```

---

## Troubleshooting Contour Issues

### Diagnostic Tools

The closure check below is for a **full** contour, not a symmetric
half-contour. Singular values make this a small dense-problem diagnostic.

```@example custom_diagnostics
using FeastKit, LinearAlgebra


function diagnose_contour_quality(contour, A)
    """
    Analyze contour quality for eigenvalue computation.
    """
    
    println("Contour Quality Diagnostics")
    println("="^40)
    
    Zne, Wne = contour.Zne, contour.Wne
    n_points = length(Zne)
    
    # 1. Check contour closure
    contour_sum = sum(Wne)
    closure_error = abs(contour_sum)
    println("Contour closure error: $closure_error")
    
    if closure_error > 1e-12
        @warn "Contour may not be properly closed"
    end
    
    # 2. Check node distribution
    min_spacing = minimum([abs(Zne[i] - Zne[j]) 
                          for i in 1:n_points for j in i+1:n_points])
    avg_spacing = sum(abs(Zne[i+1] - Zne[i]) for i in 1:n_points-1) / (n_points-1)
    
    println("Minimum node spacing: $min_spacing")
    println("Average node spacing: $avg_spacing")
    
    if min_spacing < avg_spacing * 0.1
        @warn "Nodes may be too close together"
    end
    
    # 3. Test integration accuracy with known function
    # Integrate f(z) = 1 (should give 0 for closed contour)
    integral_one = sum(Wne)
    println("∮ 1 dz = $integral_one (should be ≈ 0)")
    
    # 4. Estimate condition number at integration points
    condition_numbers = Float64[]
    for z in Zne
        try
            # Approximate condition number of (zI - A)
            shift_matrix = z * I - A
            σ = svdvals(Matrix(shift_matrix))
            cond_approx = maximum(σ) / minimum(σ)
            push!(condition_numbers, cond_approx)
        catch
            push!(condition_numbers, Inf)
        end
    end
    
    max_cond = maximum(condition_numbers)
    avg_cond = sum(condition_numbers) / length(condition_numbers)
    
    println("Max condition number: $max_cond")
    println("Average condition number: $avg_cond") 
    
    if max_cond > 1e12
        @warn "Some integration points may be too close to eigenvalues"
    end
    
    return (closure_error, min_spacing, avg_spacing, max_cond, avg_cond)
end

# Example usage
A = Matrix(Diagonal([-1.0, 0.5, 3.0]))
contour = feast_circle(0, 2; n=32)

diagnostics = diagnose_contour_quality(contour, A)
@assert diagnostics[1] < 1e-12 && all(isfinite, diagnostics)
```

### Common Issues and Solutions

!!! warning "Issue: No eigenvalues found"
    **Causes**: Contour doesn't enclose eigenvalues
    
    **Solutions**:
    ```@example custom_empty
    using FeastKit, LinearAlgebra
    A = Matrix(Diagonal(collect(1.0:40.0)))
    interval = (0.5, 3.5)
    # 1. Check eigenvalue bounds
    bounds = feast_validate_interval(A, interval)
    println("Estimated bounds: $bounds")
    
    # 2. Use wider interval
    wider_interval = (bounds[1] - 0.1, bounds[2] + 0.1)
    result = feast(A, wider_interval; subspace_size=size(A, 1))
    @assert result.converged && result.M == 40
    contour = feast_contour_expert(interval..., 16)
    
    # 3. Visualize rational function
    test_points = range(interval[1], interval[2], length=100)
    rational_vals = feast_rational_expert(contour.Zne, contour.Wne, test_points)
    # Plot rational_vals vs test_points to inspect the filter, not the spectrum
    ```

!!! warning "Issue: Integration not converging"
    **Causes**: Too few integration points, poor contour shape
    
    **Solutions**:
    ```@example custom_refine
    using FeastKit, LinearAlgebra
    A = Matrix(Diagonal(collect(1.0:40.0)))
    interval = (0.5, 3.5)
    # 1. Increase integration points
    contour = feast_contour_expert(interval[1], interval[2], 32, 0, 100)
    
    # 2. Use Zolotarev integration for difficult problems
    contour = feast_contour_expert(interval[1], interval[2], 20, 2, 100)
    
    # 3. Adjust ellipse aspect ratio for eigenvalue distribution
    contour = feast_contour_expert(interval[1], interval[2], 16, 0, 50)  # Flatter
    @assert length(contour.Zne) == 16
    ```

!!! warning "Issue: Spurious eigenvalues"
    **Causes**: Numerical errors, ill-conditioned linear systems
    
    **Solutions**:
    ```@example custom_spurious
    using FeastKit, LinearAlgebra
    A = Matrix(Diagonal(collect(1.0:40.0)))
    interval = (0.5, 3.5)
    # 1. Increase precision
    fpm = feastinit().fpm
    fpm[3] = 14  # Higher tolerance (10^-14)
    result = feast(A, interval; fpm=fpm)
    
    # 2. Check residuals
    for i in 1:result.M
        residual = norm(A * result.q[:, i] - result.lambda[i] * result.q[:, i])
        println("λ[$(i)]: residual = $residual")
    end
    
    # 3. Use iterative refinement
    fpm[4] = 50  # More refinement iterations
    @assert result.converged && result.M == 3
    ```

---

## Expert reverse-communication state

The expert RCI wrappers `feast_srcix!`, `feast_hrcix!`, and `feast_grcix!`
accept a `state` keyword just like their ordinary-contour counterparts.
Construct a `FeastSRCIState{T}`, `FeastHRCIState{T}`, or `FeastGRCIState{T}`
once and pass the same object with `state=state` on every call in the RCI loop.
The contour arguments alone do not preserve the trial subspace or iteration
phase between calls.

---

**Master advanced contour integration techniques with FeastKit.jl**

← [Performance](performance.md) | [API Reference](api_reference.md) →
