# Custom Contour Integration {#custom_contours}

```@id custom_contours
```


Advanced guide to customizing FeastKit's contour integration for optimal performance and accuracy.

## Table of Contents

1. [Contour Integration Theory](#contour-integration-theory)
2. [Built-in Integration Methods](#built-in-integration-methods)  
3. [Custom Contour Design](#custom-contour-design)
4. [Advanced Applications](#advanced-applications)
5. [Troubleshooting Contour Issues](#troubleshooting-contour-issues)

---

## Contour Integration Theory

### Mathematical Foundation

The FEAST algorithm uses contour integration to compute spectral projectors. For a matrix **A** with eigenvalues λᵢ and eigenvectors xᵢ, the spectral projector is:

$$P = \frac{1}{2\pi i} \oint_\Gamma (z\mathbf{I} - \mathbf{A})^{-1} dz$$

Where Γ is a contour enclosing the eigenvalues of interest.

### Moment-Based Approach

FeastKit computes moments of the spectral projector:

$$\mathbf{S}_k = \frac{1}{2\pi i} \oint_\Gamma z^k (z\mathbf{B} - \mathbf{A})^{-1} \mathbf{Q} dz$$

The numerical integration becomes:

$$\mathbf{S}_k \approx \sum_{e=1}^{n_e} w_e z_e^k (z_e\mathbf{B} - \mathbf{A})^{-1} \mathbf{Q}$$

!!! note "Key Insight"
    The quality of eigenvalue extraction depends critically on:
    - **Contour placement**: Must enclose target eigenvalues
    - **Integration accuracy**: Affects moment computation quality  
    - **Node distribution**: Should resolve eigenvalue clustering

---

## Built-in Integration Methods

### Gauss-Legendre Integration

**Best for**: High accuracy with minimal points, smooth integrands

```julia
using FeastKit

# High-accuracy Gauss-Legendre with 16 points
contour = feast_contour_expert(-1.0, 1.0, 16, 0, 100)

println("Gauss-Legendre contour:")
println("Nodes: ", contour.Zne[1:5])  # Show first 5 nodes
println("Weights: ", contour.Wne[1:5]) # Show first 5 weights
```

**Advantages**:
- Highest accuracy per integration point
- Optimal for smooth problems
- Well-established theory

**Disadvantages**:  
- Points not uniformly distributed
- May miss isolated eigenvalues

### Trapezoidal Rule Integration  

**Best for**: Robust integration, debugging, educational purposes

```julia
# Trapezoidal rule with 12 points
contour = feast_contour_expert(-2.0, 2.0, 12, 1, 100)

# Visualize uniform node distribution
θ = [angle(z) for z in contour.Zne]
println("Angles: ", sort(θ))  # Should be uniformly spaced
```

**Advantages**:
- Uniformly distributed points
- Simple and robust
- Good for debugging

**Disadvantages**:
- Lower accuracy per point
- Requires more points for precision

### Zolotarev Integration

**Best for**: Elliptical domains, clustered eigenvalues

```julia  
# Zolotarev integration (optimal for ellipses)
contour = feast_contour_expert(0.0, 4.0, 12, 2, 100)

println("Zolotarev contour characteristics:")
println("Optimized for elliptical regions")
println("Excellent for clustered eigenvalues")
```

**Advantages**:
- Theoretically optimal for ellipses  
- Excellent for clustered eigenvalues
- Adaptive node placement

**Disadvantages**:
- More complex implementation
- Problem-specific optimization

---

## Custom Contour Design

### Designing Your Own Contour

```julia
using FeastKit, LinearAlgebra

function create_rectangular_contour(xmin, xmax, ymin, ymax, nx, ny)
    """
    Create rectangular contour for eigenvalues in complex rectangle.
    
    Parameters:
    - xmin, xmax: Real axis bounds
    - ymin, ymax: Imaginary axis bounds  
    - nx, ny: Number of points on horizontal/vertical segments
    """
    
    # Bottom edge: xmin to xmax
    bottom_x = range(xmin, xmax, length=nx)
    bottom_nodes = [x + im*ymin for x in bottom_x]
    bottom_weights = fill((xmax - xmin) / nx, nx)
    
    # Right edge: ymin to ymax  
    right_y = range(ymin, ymax, length=ny)[2:end] # Skip corner
    right_nodes = [xmax + im*y for y in right_y]
    right_weights = fill(im * (ymax - ymin) / ny, ny-1)
    
    # Top edge: xmax to xmin (reverse direction)
    top_x = range(xmax, xmin, length=nx)[2:end] # Skip corner
    top_nodes = [x + im*ymax for x in top_x] 
    top_weights = fill(-(xmax - xmin) / nx, nx-1)
    
    # Left edge: ymax to ymin (reverse direction)
    left_y = range(ymax, ymin, length=ny)[2:end-1] # Skip corners
    left_nodes = [xmin + im*y for y in left_y]
    left_weights = fill(-im * (ymax - ymin) / ny, ny-2)
    
    # Combine all segments
    all_nodes = vcat(bottom_nodes, right_nodes, top_nodes, left_nodes)
    all_weights = vcat(bottom_weights, right_weights, top_weights, left_weights)
    
    return all_nodes, all_weights
end

# Example: Rectangle around complex eigenvalues
nodes, weights = create_rectangular_contour(-1, 3, -2, 2, 20, 16)
println("Created rectangular contour with $(length(nodes)) points")

# Use with FeastKit
contour = feast_contour_custom_weights!(nodes, weights)
```

### Circular Contour for Complex Eigenvalues

```julia
function create_circular_contour(center, radius, n_points)
    """
    Create circular contour for general eigenvalue problems.
    """
    θ = range(0, 2π, length=n_points+1)[1:end-1]  # Exclude 2π (same as 0)
    
    nodes = [center + radius * exp(im * θᵢ) for θᵢ in θ]
    weights = [im * radius * exp(im * θᵢ) * (2π / n_points) for θᵢ in θ]
    
    return nodes, weights
end

# Example usage
center = 1.0 + 0.5im  
radius = 2.0
nodes, weights = create_circular_contour(center, radius, 16)

contour = feast_contour_custom_weights!(nodes, weights)
```

### Adaptive Contour Generation

```julia
function adaptive_elliptical_contour(Emin, Emax, eigenvalue_estimates; 
                                    min_points=8, max_points=32)
    """
    Create elliptical contour adapted to eigenvalue distribution.
    """
    
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
        n_points = 16
        method = 0  # Gauss-Legendre
        aspect_ratio = 100
        println("Standard distribution: using $n_points Gauss-Legendre points")
    end
    
    return feast_contour_expert(Emin, Emax, n_points, method, aspect_ratio)
end

# Example with eigenvalue estimates
λ_estimates = [0.5, 0.52, 0.54, 1.8, 1.82, 1.84]  # Two clusters
contour = adaptive_elliptical_contour(0.0, 2.5, λ_estimates)
```

---

## Advanced Applications

### Multi-Level Contour Strategy

For problems with eigenvalues at different scales:

```julia
function multi_level_feast(A, eigenvalue_regions; M0_per_region=10)
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
            n_points = 32
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
        
        result = feast(A, (region_min, region_max), M0=M0_per_region, fpm=fpm)
        
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
    
    println("\\nTotal eigenvalues found: $(length(all_eigenvalues))")
    return all_eigenvalues, all_eigenvectors
end

# Example usage
regions = [
    (0.01, 0.1, "Low frequency modes"),
    (0.8, 1.2, "Mid-range cluster"), 
    (4.5, 5.5, "High frequency modes")
]

A = create_test_matrix(1000)
eigenvalues, eigenvectors = multi_level_feast(A, regions)
```

### Contour Optimization via Rational Function

```julia
function optimize_contour_placement(A, initial_interval, target_count; 
                                   max_iterations=5)
    """
    Optimize contour placement using rational function evaluation.
    """
    
    Emin, Emax = initial_interval
    
    for iter in 1:max_iterations
        println("Iteration $iter: interval [$Emin, $Emax]")
        
        # Generate test points in current interval
        test_points = range(Emin, Emax, length=50)
        
        # Evaluate rational function to estimate eigenvalue count
        contour = feast_contour_expert(Emin, Emax, 16, 0, 100)
        rational_values = feast_rational_expert(contour.Zne, contour.Wne, test_points)
        
        # Count estimated eigenvalues (rational function ≈ 1 near eigenvalues)
        estimated_count = sum(rational_values .> 0.5)
        
        println("Estimated eigenvalues in interval: $estimated_count")
        println("Target count: $target_count")
        
        if abs(estimated_count - target_count) <= 1
            println("Converged to optimal interval!")
            break
        end
        
        # Adjust interval based on estimate
        if estimated_count > target_count
            # Too many eigenvalues - shrink interval
            width = Emax - Emin
            center = (Emin + Emax) / 2
            new_width = width * 0.8
            Emin = center - new_width/2
            Emax = center + new_width/2
            
        else  # estimated_count < target_count
            # Too few eigenvalues - expand interval
            width = Emax - Emin  
            expansion = 1.2
            Emin -= width * (expansion - 1) / 2
            Emax += width * (expansion - 1) / 2
        end
    end
    
    return (Emin, Emax)
end

# Example usage
A = SymTridiagonal(2.0 * ones(500), -1.0 * ones(499))
initial_interval = (0.5, 1.5)
optimized_interval = optimize_contour_placement(A, initial_interval, 10)

println("Optimized interval: $optimized_interval")
```

---

## Advanced Contour Shapes

### Star-Shaped Contours

For eigenvalues with radial distribution:

```julia
function create_star_contour(center, radius, n_spikes, n_points_per_spike)
    """
    Create star-shaped contour for eigenvalues with radial symmetry.
    """
    nodes = ComplexF64[]
    weights = ComplexF64[]
    
    for spike in 1:n_spikes
        # Base angle for this spike
        θ_base = 2π * (spike - 1) / n_spikes
        
        # Create points along this spike
        for i in 1:n_points_per_spike
            # Vary radius from center to maximum
            r = radius * i / n_points_per_spike
            θ = θ_base + 0.1 * sin(4π * i / n_points_per_spike)  # Add slight perturbation
            
            z = center + r * exp(im * θ)
            push!(nodes, z)
            
            # Approximate weight (tangent direction)
            dz_dt = (radius / n_points_per_spike) * exp(im * θ) * 
                   (1 + im * 0.4 * π * cos(4π * i / n_points_per_spike) / n_points_per_spike)
            push!(weights, dz_dt)
        end
    end
    
    return nodes, weights
end
```

### Lens-Shaped Contours

For eigenvalues in bimodal distributions:

```julia
function create_lens_contour(focus1, focus2, width, n_points)
    """
    Create lens-shaped (elliptical) contour between two focal points.
    """
    # Ellipse parameters
    center = (focus1 + focus2) / 2
    focus_distance = abs(focus2 - focus1)
    major_axis = focus_distance + width
    minor_axis = width
    
    θ = range(0, 2π, length=n_points+1)[1:end-1]
    
    nodes = ComplexF64[]
    weights = ComplexF64[]
    
    for θᵢ in θ
        # Parametric ellipse
        x = (major_axis/2) * cos(θᵢ)
        y = (minor_axis/2) * sin(θᵢ)
        
        z = center + x + im * y
        push!(nodes, z)
        
        # Derivative for weight calculation
        dx_dθ = -(major_axis/2) * sin(θᵢ)  
        dy_dθ = (minor_axis/2) * cos(θᵢ)
        dz_dθ = dx_dθ + im * dy_dθ
        
        weight = dz_dθ * (2π / n_points)
        push!(weights, weight)
    end
    
    return nodes, weights
end
```

---

## Troubleshooting Contour Issues

### Diagnostic Tools

```julia
function diagnose_contour_quality(contour, A, interval)
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
            σ_min = minimum(svdvals(shift_matrix))  # Smallest singular value
            cond_approx = 1.0 / σ_min
            push!(condition_numbers, cond_approx)
        catch
            push!(condition_numbers, Inf)
        end
    end
    
    max_cond = maximum(condition_numbers)
    avg_cond = mean(condition_numbers)
    
    println("Max condition number: $max_cond")
    println("Average condition number: $avg_cond") 
    
    if max_cond > 1e12
        @warn "Some integration points may be too close to eigenvalues"
    end
    
    return (closure_error, min_spacing, avg_spacing, max_cond, avg_cond)
end

# Example usage
A = randn(100, 100); A = A + A'  # Symmetric test matrix
interval = (-2, 2)
contour = feast_contour_expert(interval[1], interval[2], 16, 0, 100)

diagnostics = diagnose_contour_quality(contour, A, interval)
```

### Common Issues and Solutions

!!! warning "Issue: No eigenvalues found"
    **Causes**: Contour doesn't enclose eigenvalues
    
    **Solutions**:
    ```julia
    # 1. Check eigenvalue bounds
    bounds = feast_validate_interval(A, interval)
    println("Estimated bounds: $bounds")
    
    # 2. Use wider interval
    wider_interval = (bounds[1] - 0.1, bounds[2] + 0.1)
    result = feast(A, wider_interval)
    
    # 3. Visualize rational function
    test_points = range(interval[1], interval[2], length=100)
    rational_vals = feast_rational_expert(contour.Zne, contour.Wne, test_points)
    # Plot rational_vals vs test_points (peaks indicate eigenvalues)
    ```

!!! warning "Issue: Integration not converging"
    **Causes**: Too few integration points, poor contour shape
    
    **Solutions**:
    ```julia
    # 1. Increase integration points
    contour = feast_contour_expert(interval[1], interval[2], 32, 0, 100)
    
    # 2. Use Zolotarev integration for difficult problems
    contour = feast_contour_expert(interval[1], interval[2], 24, 2, 100)
    
    # 3. Adjust ellipse aspect ratio for eigenvalue distribution
    contour = feast_contour_expert(interval[1], interval[2], 16, 0, 50)  # Flatter
    ```

!!! warning "Issue: Spurious eigenvalues"
    **Causes**: Numerical errors, ill-conditioned linear systems
    
    **Solutions**:
    ```julia
    # 1. Increase precision
    fpm = zeros(Int, 64)
    fpm[3] = 14  # Higher tolerance (10^-14)
    result = feast(A, interval, fmp=fpm)
    
    # 2. Check residuals
    for i in 1:result.M
        residual = norm(A * result.q[:, i] - result.lambda[i] * result.q[:, i])
        println("λ[$(i)]: residual = $residual")
    end
    
    # 3. Use iterative refinement
    fpm[4] = 50  # More refinement iterations
    ```

---

<div align="center">
  <p><strong>Master advanced contour integration techniques with FeastKit.jl</strong></p>
  ← [Performance](@ref "performance_guide") | [API Reference](@ref "api_reference") →
</div>
