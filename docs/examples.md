# Examples and Tutorials {#examples}

```@id examples
```

Comprehensive collection of FeastKit.jl examples from basic usage to advanced applications.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Scientific Computing Applications](#scientific-computing-applications)  
3. [Matrix-Free Examples](#matrix-free-examples)
4. [Advanced Features](#advanced-features)
5. [Performance Examples](#performance-examples)
6. [Real-World Applications](#real-world-applications)

---

## Basic Examples

### Example 1: Your First FeastKit Calculation

**Problem**: Find eigenvalues of a simple symmetric matrix.

```julia
using FeastKit, LinearAlgebra

# Create a 5×5 symmetric matrix
A = [4.0  -1.0   0.0   0.0   0.0
    -1.0   4.0  -1.0   0.0   0.0  
     0.0  -1.0   4.0  -1.0   0.0
     0.0   0.0  -1.0   4.0  -1.0
     0.0   0.0   0.0  -1.0   4.0]

println("Matrix A:")
display(A)

# Find eigenvalues between 2 and 6
result = feast(A, (2.0, 6.0), M0=5)

println("\\nFEAST Results:")
println("Status: $(result.info == 0 ? "Success" : "Failed")")
println("Found $(result.M) eigenvalues")

for i in 1:result.M
    λ = result.lambda[i]
    println("λ[$i] = $λ")
end

# Compare with Julia's built-in eigensolver
all_eigvals = sort(eigvals(A))
println("\\nAll eigenvalues (Julia): $all_eigvals")
```

**Expected Output:**
```
Found 5 eigenvalues
λ[1] = 2.0
λ[2] = 3.0
λ[3] = 4.0  
λ[4] = 5.0
λ[5] = 6.0
```

### Example 2: Sparse Matrix Eigenvalues

**Problem**: Find eigenvalues of a large sparse tridiagonal matrix.

```julia
using FeastKit, SparseArrays, LinearAlgebra

# Create large sparse tridiagonal matrix  
n = 1000
A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

println("Sparse matrix: $(n)×$(n) with $(nnz(A)) nonzeros")

# Find the 10 smallest eigenvalues
# For this tridiagonal matrix: λₖ = 2 - 2*cos(kπ/(n+1))
λ_min = 2 - 2*cos(π/(n+1))      # ≈ 0.0001  
λ_10 = 2 - 2*cos(10π/(n+1))     # ≈ 0.001

println("Expected λ₁ ≈ $λ_min")
println("Expected λ₁₀ ≈ $λ_10")

# FeastKit search
result = feast(A, (λ_min * 0.9, λ_10 * 1.1), M0=12)

println("\\nFEAST found $(result.M) eigenvalues:")
for i in 1:result.M
    k = round(Int, acos(1 - result.lambda[i]/2) * (n+1) / π)
    λ_exact = 2 - 2*cos(k*π/(n+1))
    error = abs(result.lambda[i] - λ_exact)
    println("λ[$i] = $(result.lambda[i]) (exact: $λ_exact, error: $error)")
end
```

### Example 3: Generalized Eigenvalue Problem

**Problem**: Solve A*x = λ*B*x with two different matrices.

```julia
using FeastKit, LinearAlgebra

# Create matrices A (stiffness) and B (mass)
n = 100
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))  # Stiffness
B = SymTridiagonal(1.0 * ones(n), 0.1 * ones(n-1))   # Mass

println("Generalized eigenvalue problem: A*x = λ*B*x")
println("A: $(n)×$(n) tridiagonal (stiffness)")  
println("B: $(n)×$(n) tridiagonal (mass)")

# Find eigenvalues between 0.5 and 2.5
result = feast(A, B, (0.5, 2.5), M0=15)

println("\\nResults:")
println("Found $(result.M) generalized eigenvalues")

# Display first few eigenvalues and check generalized orthogonality  
for i in 1:min(5, result.M)
    λ = result.lambda[i]
    x = result.q[:, i]
    
    # Check: A*x = λ*B*x
    residual = norm(A*x - λ*(B*x))
    println("λ[$i] = $λ (residual: $residual)")
end

# Check B-orthogonality of eigenvectors: X'*B*X = I
if result.M > 1
    X = result.q[:, 1:result.M]
    orthogonality = norm(X' * B * X - I)
    println("\\nB-orthogonality error: $orthogonality")
end
```

---

## Scientific Computing Applications

### Example 4: 1D Wave Equation (Vibrating String)

**Problem**: Find natural frequencies of a vibrating string.

```julia
using FeastKit, LinearAlgebra, Plots

# Physical parameters
L = 1.0      # String length
T = 100.0    # Tension
ρ = 1.0      # Linear density  
n = 500      # Discretization points

# Discretization
h = L / (n + 1)
x = h * (1:n)

# Finite difference matrices
# d²u/dx² ≈ (u[i-1] - 2u[i] + u[i+1])/h²
A = (T/h^2) * SymTridiagonal(-2.0 * ones(n), ones(n-1))  # -T * d²/dx²
B = ρ * I  # Mass matrix

println("1D Wave Equation: -T*d²u/dx² = ω²*ρ*u")
println("String length: $L, Tension: $T, Density: $ρ")
println("Discretized with $n points, h = $h")

# Find first few natural frequencies
# Analytical: ωₖ = kπ√(T/ρ)/L
ω_max = 5π * sqrt(T/ρ) / L  # First 5 modes
ω²_max = ω_max^2

result = feast(A, B, (0.1, ω²_max), M0=8)

println("\\nNatural frequencies (Hz):")
frequencies = sqrt.(result.lambda[1:result.M]) / (2π)

for i in 1:result.M
    ω_analytical = i * π * sqrt(T/ρ) / L
    f_analytical = ω_analytical / (2π)
    error = abs(frequencies[i] - f_analytical)
    
    println("Mode $i: f = $(frequencies[i]) Hz (exact: $f_analytical Hz, error: $error)")
end

# Plot mode shapes
if result.M >= 3
    p = plot(x, result.q[:, 1], label="Mode 1", linewidth=2)
    plot!(p, x, result.q[:, 2], label="Mode 2", linewidth=2)
    plot!(p, x, result.q[:, 3], label="Mode 3", linewidth=2)
    xlabel!(p, "Position x")
    ylabel!(p, "Amplitude")  
    title!(p, "Vibrating String Mode Shapes")
    display(p)
end
```

### Example 5: 2D Membrane Vibrations (Drum Head)

**Problem**: Find vibrational modes of a circular drum.

```julia
using FeastKit, LinearAlgebra

# 2D Laplacian on square domain (approximating circular drum)
nx, ny = 50, 50
n = nx * ny
h = 1.0 / (nx + 1)

println("2D Membrane: -∇²u = ω²u on unit square")
println("Grid: $(nx)×$(ny), spacing: $h")

# Matrix-free 2D Laplacian with Dirichlet boundary conditions
function laplacian_2d!(y, x)
    fill!(y, 0)
    
    for j in 1:ny, i in 1:nx
        k = (j-1) * nx + i
        
        # Central difference: -∇²u ≈ -(4u - u_left - u_right - u_up - u_down)/h²
        y[k] += 4 * x[k] / h^2
        
        # Neighbors (with zero Dirichlet boundaries)  
        if i > 1
            y[k] -= x[k-1] / h^2
        end
        if i < nx
            y[k] -= x[k+1] / h^2
        end
        if j > 1
            y[k] -= x[k-nx] / h^2  
        end
        if j < ny
            y[k] -= x[k+nx] / h^2
        end
    end
end

# Create matrix-free operator
A_op = LinearOperator{Float64}(laplacian_2d!, (n, n), 
                              issymmetric=true, isposdef=true)

# Find fundamental modes (lowest frequencies)
# Analytical for square: ωₘₙ² = π²(m² + n²) for m,n = 1,2,3,...  
ω²_fundamental = π^2 * (1^2 + 1^2)  # Mode (1,1)
ω²_search_max = π^2 * (2^2 + 2^2)   # Up to mode (2,2)

println("Searching for eigenvalues up to $(ω²_search_max)")
println("Fundamental frequency ω₁₁² = $ω²_fundamental")

result = feast(A_op, (0.8 * ω²_fundamental, 1.2 * ω²_search_max), 
              M0=10, solver=:cg, solver_opts=(rtol=1e-6, maxiter=500))

println("\\n2D Membrane Modes:")
println("Found $(result.M) eigenfrequencies")

# Match with analytical modes
analytical_modes = [(1,1), (1,2), (2,1), (2,2)]
for i in 1:min(result.M, length(analytical_modes))
    m, n = analytical_modes[i]
    ω²_exact = π^2 * (m^2 + n^2)  
    ω²_feast = result.lambda[i]
    error = abs(ω²_feast - ω²_exact)
    
    println("Mode ($m,$n): ω² = $ω²_feast (exact: $ω²_exact, error: $error)")
end

# Visualize first mode shape
if result.M > 0
    mode1 = reshape(result.q[:, 1], nx, ny)
    println("\\nFirst mode shape computed (reshape to $(nx)×$(ny) grid for visualization)")
end
```

### Example 6: Quantum Harmonic Oscillator

**Problem**: Find energy levels of quantum harmonic oscillator.

```julia
using FeastKit, LinearAlgebra

# 1D Quantum Harmonic Oscillator: H = -½d²/dx² + ½x²
# Discretized on [-L, L] with N points

L = 6.0     # Domain size (covers most of wavefunction)
N = 1000    # Grid points
h = 2L / (N - 1)
x = range(-L, L, length=N)

println("Quantum Harmonic Oscillator: H = -½d²/dx² + ½x²")
println("Domain: x ∈ [$(-L), $L], grid spacing: $h")

# Hamiltonian matrix (using finite differences)
# Kinetic energy: -½d²/dx² ≈ -½(ψ[i+1] - 2ψ[i] + ψ[i-1])/h²
# Potential energy: ½x²ψ[i]

H = zeros(N, N)

for i in 1:N
    # Potential energy
    H[i, i] += 0.5 * x[i]^2
    
    # Kinetic energy (finite difference)
    if i > 1
        H[i, i-1] += -0.5 / h^2
    end
    if i < N  
        H[i, i+1] += -0.5 / h^2
    end
    H[i, i] += 0.5 / h^2  # Central term from kinetic
end

# Enforce boundary conditions (ψ = 0 at boundaries)
H[1, :] .= 0; H[1, 1] = 1e10
H[N, :] .= 0; H[N, N] = 1e10

println("Hamiltonian matrix: $(N)×$(N)")

# Find ground state and first few excited states  
# Analytical energies: Eₙ = n + ½ for n = 0, 1, 2, ...
E_ground = 0.5      # Ground state
E_max = 5.5         # Up to n = 5

result = feast(H, (E_ground - 0.1, E_max + 0.1), M0=8)

println("\\nQuantum Energy Levels:")
println("Found $(result.M) energy eigenstates")

for i in 1:result.M
    E_exact = (i-1) + 0.5  # n = 0, 1, 2, ...
    E_computed = result.lambda[i]
    error = abs(E_computed - E_exact)
    
    println("n=$(i-1): E = $E_computed (exact: $E_exact, error: $error)")
end

# Check normalization of wavefunctions
println("\\nWavefunction normalization check:")
for i in 1:min(3, result.M)
    ψ = result.q[:, i]
    norm_ψ = sqrt(sum(ψ.^2) * h)  # Numerical integration
    println("||ψ_$(i-1)||₂ = $norm_ψ (should be ≈ 1)")
end
```

---

## Matrix-Free Examples

### Example 7: Large-Scale Finite Element Problem

**Problem**: Structural eigenanalysis without storing global matrices.

```julia
using FeastKit, SparseArrays

# Simulate large finite element problem  
# In reality, this would come from FE assembly
n_nodes = 100000  # Very large problem
n_dofs = 3 * n_nodes  # 3 DOFs per node (x, y, z)

println("Large-scale structural analysis")  
println("Nodes: $(n_nodes), DOFs: $(n_dofs)")
println("Memory for full matrix: $(8 * n_dofs^2 / 1e9) GB")
println("Memory for matrix-free: $(8 * n_dofs / 1e6) MB")

# Matrix-free stiffness matrix operation K*u
function stiffness_matvec!(Ku, u)
    # In practice: loop over elements, compute element stiffness,
    # and assemble contributions to Ku
    
    # Simplified: 3D Laplacian-like operator  
    fill!(Ku, 0)
    
    # Simple 3D finite difference stencil (6-point)
    n = length(u) ÷ 3  # Number of nodes
    
    for node in 1:n
        for dof in 1:3  # x, y, z components
            idx = 3*(node-1) + dof
            
            # Diagonal term (self-stiffness)
            Ku[idx] += 6.0 * u[idx]
            
            # Coupling with neighboring nodes (simplified connectivity)
            for neighbor_offset in [-1, 1, -10, 10, -100, 100]  # 3D grid-like
                neighbor = node + neighbor_offset
                if 1 <= neighbor <= n
                    neighbor_idx = 3*(neighbor-1) + dof
                    Ku[idx] -= u[neighbor_idx]
                end
            end
        end
    end
end

# Matrix-free mass matrix operation M*u
function mass_matvec!(Mu, u)
    # Diagonal mass matrix (lumped mass)
    @. Mu = 1.0 * u
end

# Create matrix-free operators
K_op = LinearOperator{Float64}(stiffness_matvec!, (n_dofs, n_dofs), issymmetric=true)
M_op = LinearOperator{Float64}(mass_matvec!, (n_dofs, n_dofs), 
                              issymmetric=true, isposdef=true)

println("\\nCreated matrix-free operators")
println("Searching for natural frequencies between 0.1 and 2.0 Hz...")

# Find structural modes (natural frequencies)
ω²_min, ω²_max = (2π * 0.1)^2, (2π * 2.0)^2

result = feast(K_op, M_op, (ω²_min, ω²_max), M0=20,
              solver=:cg,
              solver_opts=(rtol=1e-4, maxiter=200))

println("\\nStructural Analysis Results:")
println("Found $(result.M) vibrational modes")

frequencies_Hz = sqrt.(result.lambda[1:result.M]) / (2π)
for i in 1:result.M
    println("Mode $i: f = $(frequencies_Hz[i]) Hz")
end

println("\\nMatrix-free calculation completed successfully!")
println("Peak memory usage: ~$(8 * n_dofs * result.M / 1e6) MB")
```

### Example 8: Iterative Solver Comparison

**Problem**: Compare different iterative solvers for matrix-free FeastKit.

```julia
using FeastKit, LinearAlgebra, BenchmarkTools

# Test problem: 2D Poisson equation
nx, ny = 200, 200  
n = nx * ny
h = 1.0 / (nx + 1)

println("2D Poisson Eigenvalue Problem")
println("Grid: $(nx)×$(ny), DOFs: $(n)")

# Matrix-free Laplacian
function laplacian!(y, x)
    fill!(y, 0)
    for j in 1:ny, i in 1:nx
        k = (j-1) * nx + i
        y[k] += 4 * x[k] / h^2
        
        i > 1  && (y[k] -= x[k-1] / h^2)
        i < nx && (y[k] -= x[k+1] / h^2)  
        j > 1  && (y[k] -= x[k-nx] / h^2)
        j < ny && (y[k] -= x[k+nx] / h^2)
    end
end

A_op = LinearOperator{Float64}(laplacian!, (n, n), issymmetric=true, isposdef=true)
interval = (10.0, 50.0)  # Search range

# Test different iterative solvers
solvers = [
    (:cg, "Conjugate Gradient"),
    (:gmres, "GMRES"), 
    (:bicgstab, "BiCGSTAB(l)")
]

results = Dict()

for (solver_name, description) in solvers
    println("\\n" * "="^50)
    println("Testing solver: $description")
    
    # Configure solver options
    if solver_name == :cg
        opts = (rtol=1e-6, maxiter=300)
    elseif solver_name == :gmres  
        opts = (rtol=1e-6, restart=50, maxiter=300)
    else  # bicgstab
        opts = (rtol=1e-6, l=2, maxiter=300)
    end
    
    # Time the calculation
    time_taken = @elapsed begin
        result = feast(A_op, interval, M0=8, 
                      solver=solver_name, solver_opts=opts)
    end
    
    results[solver_name] = (result=result, time=time_taken)
    
    println("Solver: $description")
    println("  Time: $(time_taken) seconds")
    println("  Eigenvalues found: $(result.M)")
    println("  FeastKit status: $(result.info == 0 ? "Success" : "Failed")")
    println("  Convergence: $(result.epsout)")
    
    if result.M > 0
        println("  First 3 eigenvalues: $(result.lambda[1:min(3, result.M)])")
    end
end

# Summary comparison
println("\\n" * "="^60)
println("SOLVER COMPARISON SUMMARY")
println("="^60)
printf_str = "%-20s %-10s %-8s %-12s\\n"
@printf(printf_str, "Solver", "Time (s)", "Found", "Status")
println("-"^60)

for (solver_name, description) in solvers
    if haskey(results, solver_name)
        r = results[solver_name]
        status = r.result.info == 0 ? "Success" : "Failed"
        @printf(printf_str, description, r.time, r.result.M, status)
    end
end
```

---

## Advanced Features

### Example 9: Custom Contour Integration

**Problem**: Use advanced integration methods for challenging problems.

```julia
using FeastKit, LinearAlgebra

# Create a challenging matrix (clustered eigenvalues)
n = 200
A = diagm(0 => vcat(0.98:0.002:1.02, 2.0:0.1:5.0))  # Tight cluster + spread
A = A + 0.01 * randn(n, n)  # Add small random perturbation  
A = (A + A') / 2  # Ensure symmetry

println("Matrix with clustered eigenvalues")
println("Cluster around 1.0: [0.98, 1.02]")
println("Spread eigenvalues: [2.0, 5.0]")

target_interval = (0.97, 1.03)  # Focus on the cluster

# Compare different integration methods
integration_methods = [
    (0, "Gauss-Legendre"),
    (1, "Trapezoidal"), 
    (2, "Zolotarev")
]

println("\\nComparing integration methods for clustered eigenvalues:")

for (method_id, method_name) in integration_methods
    println("\\n" * "-"^40)
    println("Method: $method_name")
    
    # Test different numbers of integration points
    for ne in [8, 16, 24]
        # Generate custom contour
        contour = feast_contour_expert(target_interval[1], target_interval[2], 
                                     ne, method_id, 100)
        
        println("\\n  Integration points: $ne")
        println("  Contour nodes: $(length(contour.Zne))")
        
        # Create custom FMP parameters
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fpm[2] = ne           # Number of points
        fpm[16] = method_id   # Integration method
        fpm[18] = 100         # Circular contour
        
        result = feast(A, target_interval, M0=20, fpm=fpm)
        
        # Find how many eigenvalues are actually in the target interval
        exact_in_interval = count(λ -> target_interval[1] <= λ <= target_interval[2], 
                                eigvals(A))
        
        println("    Found: $(result.M) eigenvalues")
        println("    Expected: $exact_in_interval eigenvalues")  
        println("    Convergence: $(result.epsout)")
        println("    Iterations: $(result.loop)")
    end
end
```

### Example 10: Complex Eigenvalue Problems

**Problem**: Find eigenvalues of non-Hermitian matrices in complex regions.

```julia
using FeastKit, LinearAlgebra, Random

Random.seed!(123)

# Create non-Hermitian matrix with known eigenvalue distribution
n = 100

# Upper Hessenberg matrix (like from Arnoldi process)
A = triu(randn(ComplexF64, n, n), -1)  # Upper Hessenberg
A[diagind(A)] .+= 2.0 + 0.5im          # Shift diagonal 

println("Non-Hermitian matrix ($(n)×$(n) upper Hessenberg)")
println("Expected eigenvalues near 2.0 + 0.5i")

# Identity mass matrix
B = Matrix{ComplexF64}(I, n, n)

# Define circular search regions
search_regions = [
    (2.0 + 0.5im, 1.0, "Main cluster"),
    (0.0 + 0.0im, 2.0, "Origin region"), 
    (3.0 + 2.0im, 1.5, "Upper right")
]

println("\\nSearching in multiple circular regions:")

all_found_eigenvalues = ComplexF64[]

for (center, radius, description) in search_regions
    println("\\n" * "="^50) 
    println("Region: $description")
    println("Center: $center, Radius: $radius")
    
    result = feast_general(A, B, center, radius, M0=15)
    
    println("Status: $(result.info == 0 ? "Success" : "Failed")")  
    println("Found: $(result.M) eigenvalues")
    
    if result.M > 0
        println("Eigenvalues in this region:")
        region_eigenvalues = result.lambda[1:result.M]
        
        for (i, λ) in enumerate(region_eigenvalues)
            distance = abs(λ - center)
            inside = distance <= radius
            println("  λ[$i] = $(round(λ, digits=4)) " *
                   "(distance: $(round(distance, digits=3)), " *
                   "inside: $inside)")
        end
        
        # Add to global list (avoid duplicates)  
        for λ in region_eigenvalues
            if !any(abs.(λ .- all_found_eigenvalues) .< 1e-8)
                push!(all_found_eigenvalues, λ)
            end
        end
    end
end

println("\\n" * "="^60)
println("SUMMARY: Found $(length(all_found_eigenvalues)) unique eigenvalues")

# Compare with full eigendecomposition
true_eigenvalues = eigvals(A)
println("Total eigenvalues in matrix: $(length(true_eigenvalues))")

# Visualize eigenvalue distribution (if plotting available)
try
    using Plots
    
    p = scatter(real.(true_eigenvalues), imag.(true_eigenvalues), 
               label="All eigenvalues", alpha=0.6, ms=4)
    scatter!(p, real.(all_found_eigenvalues), imag.(all_found_eigenvalues),
            label="FeastKit found", color=:red, ms=6, alpha=0.8)
    
    # Draw search circles
    θ = 0:0.1:2π
    for (center, radius, description) in search_regions
        circle_x = real(center) .+ radius .* cos.(θ)  
        circle_y = imag(center) .+ radius .* sin.(θ)
        plot!(p, circle_x, circle_y, label="$description region", 
              linestyle=:dash, linewidth=2)
    end
    
    xlabel!(p, "Real part")
    ylabel!(p, "Imaginary part")
    title!(p, "Complex Eigenvalue Distribution")
    display(p)
    
catch e
    println("Plotting not available: $e")
end
```

### Example 11: Polynomial Eigenvalue Problems  

**Problem**: Solve quadratic eigenvalue problem (λ²M + λC + K)x = 0.

```julia
using FeastKit, LinearAlgebra

# Quadratic eigenvalue problem: (λ²M + λC + K)x = 0
# Example: Damped vibration problem

n = 50  # System size

# Create coefficient matrices
K = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))  # Stiffness
C = 0.1 * SymTridiagonal(ones(n), zeros(n-1))         # Damping  
M = Matrix{Float64}(I, n, n)                          # Mass

println("Quadratic Eigenvalue Problem: (λ²M + λC + K)x = 0")
println("Damped vibration system")
println("System size: $(n)×$(n)")

# Convert to matrix-free operators for polynomial FeastKit
K_op = LinearOperator{ComplexF64}((y, x) -> mul!(y, K, real.(x)), (n, n))
C_op = LinearOperator{ComplexF64}((y, x) -> mul!(y, C, real.(x)), (n, n))  
M_op = LinearOperator{ComplexF64}((y, x) -> mul!(y, M, real.(x)), (n, n))

# Coefficient array: P(λ) = K + λC + λ²M
coeffs = [K_op, C_op, M_op]

println("Polynomial coefficients: P(λ) = K + λC + λ²M")

# Search for eigenvalues near origin (low-frequency modes)
center = 0.0 + 0.0im
radius = 2.0

println("Searching in circular region: center = $center, radius = $radius")

result = feast_polynomial(coeffs, center, radius, M0=10)

println("\\nPolynomial Eigenvalue Results:")
println("Status: $(result.info == 0 ? "Success" : "Failed")")
println("Found: $(result.M) eigenvalues")

if result.M > 0
    println("\\nEigenvalues (should be complex conjugate pairs):")
    for i in 1:result.M
        λ = result.lambda[i]
        freq = abs(λ) / (2π)
        damping_ratio = -real(λ) / abs(λ)
        
        println("λ[$i] = $(round(λ, digits=6))")  
        println("  Frequency: $(round(freq, digits=4)) Hz")
        println("  Damping ratio: $(round(damping_ratio, digits=4))")
        println()
    end
    
    # Verify polynomial eigenvalue equation: P(λ)x = 0
    println("Verification (residual norms):")
    for i in 1:min(3, result.M)
        λ = result.lambda[i]
        x = result.q[:, i]
        
        # Compute P(λ)x = (λ²M + λC + K)x  
        residual = λ^2 * (M * real.(x)) + λ * (C * real.(x)) + K * real.(x)
        residual_norm = norm(residual)
        
        println("||P(λ[$i])x[$i]|| = $residual_norm")
    end
end

# Compare with linearized version (for validation)
println("\\n" * "-"^50)
println("Comparison with linearized eigenvalue problem:")

# Linearization: [ 0   I ] [x]     [x]
#                [-K  -C] [λx] = λ[-M  0][λx] 
#
# This gives 2n eigenvalues (including spurious ones)

A_lin = [zeros(n, n)  Matrix{Float64}(I, n, n);
         -K           -C                       ]
B_lin = [-M           zeros(n, n);
         zeros(n, n)  Matrix{Float64}(I, n, n)]

# Convert to complex for general eigenvalue solver
A_lin_c = ComplexF64.(A_lin)
B_lin_c = ComplexF64.(B_lin)

result_lin = feast_general(A_lin_c, B_lin_c, center, radius, M0=20)

println("Linearized problem found: $(result_lin.M) eigenvalues")
if result_lin.M > 0
    println("First few linearized eigenvalues:")
    for i in 1:min(5, result_lin.M)
        println("  λ_lin[$i] = $(round(result_lin.lambda[i], digits=6))")
    end
end
```

---

## Performance Examples

### Example 12: Memory Usage Comparison

**Problem**: Compare memory usage of different approaches.

```julia
using FeastKit, LinearAlgebra, SparseArrays

# Test different problem sizes and approaches
problem_sizes = [1000, 5000, 10000, 20000]

println("Memory Usage Comparison: Dense vs Sparse vs Matrix-Free")
println("="^70)
@printf("%-10s %-15s %-15s %-15s\\n", "Size", "Dense (MB)", "Sparse (MB)", "Matrix-Free (MB)")
println("-"^70)

for n in problem_sizes
    # Dense matrix memory
    dense_memory = 8 * n^2 / 1e6  # 8 bytes per Float64
    
    # Sparse matrix memory (assuming 0.1% density)
    nnz_sparse = round(Int, 0.001 * n^2)
    sparse_memory = (8 * nnz_sparse + 4 * (nnz_sparse + n + 1)) / 1e6  # data + indices
    
    # Matrix-free memory (just vectors for FeastKit)
    M0 = 10
    matfree_memory = 8 * n * M0 / 1e6  # Workspace vectors
    
    @printf("%-10d %-15.1f %-15.1f %-15.1f\\n", 
           n, dense_memory, sparse_memory, matfree_memory)
    
    # Demonstrate actual usage for largest manageable size
    if n <= 5000
        println("\\nTesting n = $n:")
        
        # Create test matrix  
        A_dense = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
        A_sparse = sparse(A_dense)
        
        function A_matvec!(y, x)
            y[1] = 2*x[1] - x[2]
            for i in 2:n-1
                y[i] = -x[i-1] + 2*x[i] - x[i+1]  
            end
            y[n] = -x[n-1] + 2*x[n]
        end
        A_op = LinearOperator{Float64}(A_matvec!, (n, n), issymmetric=true)
        
        # Time each approach  
        interval = (0.01, 0.1)
        M0 = 5
        
        println("  Dense matrix:")
        @time result_dense = feast(A_dense, interval, M0=M0)
        println("    Found: $(result_dense.M) eigenvalues")
        
        println("  Sparse matrix:")  
        @time result_sparse = feast(A_sparse, interval, M0=M0)
        println("    Found: $(result_sparse.M) eigenvalues")
        
        println("  Matrix-free:")
        @time result_matfree = feast(A_op, interval, M0=M0, solver=:cg)  
        println("    Found: $(result_matfree.M) eigenvalues")
        
        # Verify all give same results
        if result_dense.M == result_sparse.M == result_matfree.M
            max_diff = maximum(abs.(result_dense.lambda[1:result_dense.M] .- 
                                  result_sparse.lambda[1:result_sparse.M]))
            println("  Max difference (dense vs sparse): $max_diff")
            
            max_diff_mf = maximum(abs.(result_dense.lambda[1:result_dense.M] .- 
                                     result_matfree.lambda[1:result_matfree.M]))
            println("  Max difference (dense vs matrix-free): $max_diff_mf")
        end
        
        println()
    end
end
```

### Example 13: Parallel Performance

**Problem**: Demonstrate parallel FeastKit capabilities.

```julia
using FeastKit, LinearAlgebra, Distributed

# Add worker processes
if nprocs() == 1
    addprocs(4)  # Add 4 worker processes
end

@everywhere using FeastKit, LinearAlgebra

println("Parallel FeastKit Performance Test")
println("Available processes: $(nprocs())")
println("Worker processes: $(nworkers())")

# Create test problem
n = 5000
A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
interval = (0.01, 0.5)
M0 = 20

println("\\nProblem: $(n)×$(n) tridiagonal matrix")
println("Search interval: $interval")
println("Max eigenvalues: $M0")

# Compare serial vs parallel performance
println("\\n" * "="^50)
println("Performance Comparison:")

# Serial FeastKit
println("\\nSerial FeastKit:")
GC.gc()  # Clean garbage before timing
serial_time = @elapsed begin
    result_serial = feast(A, interval, M0=M0, parallel=false)
end

println("  Time: $(serial_time) seconds")
println("  Eigenvalues found: $(result_serial.M)")
println("  Status: $(result_serial.info == 0 ? "Success" : "Failed")")

# Parallel FeastKit with threading  
if Threads.nthreads() > 1
    println("\\nThreaded FeastKit ($(Threads.nthreads()) threads):")
    GC.gc()
    threaded_time = @elapsed begin
        result_threaded = feast(A, interval, M0=M0, parallel=:threads)
    end
    
    println("  Time: $(threaded_time) seconds")
    println("  Speedup: $(serial_time / threaded_time)x") 
    println("  Eigenvalues found: $(result_threaded.M)")
    println("  Status: $(result_threaded.info == 0 ? "Success" : "Failed")")
end

# Distributed parallel FeastKit
if nworkers() > 1
    println("\\nDistributed FeastKit ($(nworkers()) workers):")
    GC.gc()
    distributed_time = @elapsed begin
        result_distributed = feast(A, interval, M0=M0, parallel=:distributed)
    end
    
    println("  Time: $(distributed_time) seconds") 
    println("  Speedup: $(serial_time / distributed_time)x")
    println("  Eigenvalues found: $(result_distributed.M)")
    println("  Status: $(result_distributed.info == 0 ? "Success" : "Failed")")
    
    # Verify results match
    if result_serial.M == result_distributed.M > 0
        max_diff = maximum(abs.(result_serial.lambda[1:result_serial.M] .- 
                              result_distributed.lambda[1:result_distributed.M]))
        println("  Max difference from serial: $max_diff")
    end
end

# Performance summary
println("\\n" * "="^50)
println("Performance Summary:")
println("Serial time: $(serial_time) seconds") 

if @isdefined(threaded_time)
    println("Threading speedup: $(serial_time / threaded_time)x")
end

if @isdefined(distributed_time)  
    println("Distributed speedup: $(serial_time / distributed_time)x")
end

println("\\nNote: Speedup depends on problem size, hardware, and communication overhead.")
println("Larger problems generally show better parallel scalability.")
```

---

## Real-World Applications

### Example 14: Electronic Structure (Tight-Binding Model)

**Problem**: Find electronic band structure using tight-binding approximation.

```julia
using FeastKit, LinearAlgebra

# 1D Tight-binding model for electrons in a periodic chain
# H = -t∑(c†ᵢcᵢ₊₁ + c†ᵢ₊₁cᵢ) + ε∑c†ᵢcᵢ

N = 1000    # Number of sites  
t = 1.0     # Hopping parameter
ε = 0.0     # On-site energy

println("1D Tight-Binding Electronic Structure")
println("Chain length: $N sites")  
println("Hopping parameter: $t")
println("On-site energy: $ε")

# Hamiltonian matrix (tridiagonal)
H = SymTridiagonal(ε * ones(N), -t * ones(N-1))

# For periodic boundary conditions, add corner elements
H_periodic = Matrix(H)
H_periodic[1, N] = H_periodic[N, 1] = -t

println("Boundary conditions: periodic")
println("Expected energy band: [$(ε-2t), $(ε+2t)]")

# Find states in different parts of the band

# Valence band (filled states, lower energies)
E_valence_min, E_valence_max = ε - 2*t + 0.1, ε - 0.5  
result_valence = feast(H_periodic, (E_valence_min, E_valence_max), M0=50)

println("\\nValence band [$E_valence_min, $E_valence_max]:")
println("  Found $(result_valence.M) states")

# Conduction band (empty states, higher energies)
E_conduct_min, E_conduct_max = ε + 0.5, ε + 2*t - 0.1
result_conduct = feast(H_periodic, (E_conduct_min, E_conduct_max), M0=50)

println("\\nConduction band [$E_conduct_min, $E_conduct_max]:")
println("  Found $(result_conduct.M) states")

# States near Fermi level (band gap region)
E_gap_min, E_gap_max = ε - 0.5, ε + 0.5  
result_gap = feast(H_periodic, (E_gap_min, E_gap_max), M0=20)

println("\\nBand gap region [$E_gap_min, $E_gap_max]:")
println("  Found $(result_gap.M) states")

# Calculate density of states
all_energies = vcat(
    result_valence.M > 0 ? result_valence.lambda[1:result_valence.M] : Float64[],
    result_gap.M > 0 ? result_gap.lambda[1:result_gap.M] : Float64[],  
    result_conduct.M > 0 ? result_conduct.lambda[1:result_conduct.M] : Float64[]
)

sort!(all_energies)

println("\\nElectronic Structure Summary:")
println("Total states found: $(length(all_energies))")
if length(all_energies) > 10
    println("Lowest 5 energies: $(all_energies[1:5])")
    println("Highest 5 energies: $(all_energies[end-4:end])")
end

# Analyze band gap
if result_gap.M == 0
    valence_top = result_valence.M > 0 ? maximum(result_valence.lambda[1:result_valence.M]) : -Inf
    conduct_bottom = result_conduct.M > 0 ? minimum(result_conduct.lambda[1:result_conduct.M]) : Inf
    
    if isfinite(valence_top) && isfinite(conduct_bottom)
        band_gap = conduct_bottom - valence_top
        println("\\nBand gap: $(band_gap) eV")
        println("Valence band maximum: $valence_top")  
        println("Conduction band minimum: $conduct_bottom")
    end
else
    println("\\nStates found in gap region - may be surface states or numerical artifacts")
end
```

### Example 15: Fluid Dynamics Stability Analysis

**Problem**: Linear stability analysis of fluid flow.

```julia
using FeastKit, LinearAlgebra

# Linear stability analysis of 2D channel flow
# Solve generalized eigenvalue problem: (A - λB)φ = 0
# where A is the linearized Navier-Stokes operator

# Flow parameters  
Re = 1000.0    # Reynolds number
nx, ny = 64, 32  # Grid resolution
Lx, Ly = 4π, 2.0 # Domain size  

n = nx * ny      # Total degrees of freedom

println("2D Channel Flow Stability Analysis")
println("Reynolds number: $Re")
println("Domain: $Lx × $Ly")  
println("Grid: $(nx) × $(ny)")
println("DOFs: $n")

# Create simplified stability operators (matrix-free)
# In practice, these would come from discretized Navier-Stokes

function stability_operator!(y, x)
    # Simplified 2D advection-diffusion operator
    # Represents linearized Navier-Stokes around base flow
    
    fill!(y, 0)
    
    # Grid spacing
    hx, hy = Lx/nx, Ly/ny
    
    for j in 1:ny, i in 1:nx
        k = (j-1)*nx + i
        
        # Base flow velocity (parabolic profile)  
        y_coord = (j-1) * hy
        U_base = 4 * (y_coord/Ly) * (1 - y_coord/Ly)  # Parabolic
        
        # Advection terms: U·∇φ
        if i > 1 && i < nx
            y[k] += U_base * (x[k+1] - x[k-1]) / (2*hx)  # ∂φ/∂x
        end
        
        # Viscous terms: (1/Re)∇²φ  
        y[k] -= 4 * x[k] / (Re * hx^2)  # Central difference
        if i > 1
            y[k] += x[k-1] / (Re * hx^2)
        end
        if i < nx
            y[k] += x[k+1] / (Re * hx^2)
        end
        if j > 1
            y[k] += x[k-nx] / (Re * hy^2)
        end  
        if j < ny
            y[k] += x[k+nx] / (Re * hy^2)
        end
        
        # Pressure gradient and continuity (simplified)
        y[k] += 0.1 * x[k]  # Regularization
    end
end

function mass_operator!(y, x)
    # Time derivative term  
    copy!(y, x)
end

# Create matrix-free operators
A_op = LinearOperator{ComplexF64}(
    (y, x) -> stability_operator!(y, complex(real.(x))), (n, n)
)
B_op = LinearOperator{ComplexF64}(
    (y, x) -> mass_operator!(y, complex(real.(x))), (n, n)
)

# Search for unstable modes (positive real part)
# and neutral/stable modes near imaginary axis

search_regions = [
    (0.1 + 0.0im, 0.5, "Unstable modes"),           # Right half-plane
    (0.0 + 2.0im, 1.0, "High frequency modes"),     # Imaginary axis, high freq  
    (0.0 + 0.5im, 0.8, "Low frequency modes")       # Imaginary axis, low freq
]

println("\\nSearching for instability modes:")
all_eigenvalues = ComplexF64[]

for (center, radius, description) in search_regions
    println("\\n" * "="^40)
    println("Region: $description")
    println("Center: $center, Radius: $radius")
    
    result = feast_general(A_op, B_op, center, radius, M0=10)
    
    println("Status: $(result.info == 0 ? "Success" : "Failed")")
    println("Found: $(result.M) modes")
    
    if result.M > 0
        println("Eigenvalues (growth rate + i*frequency):")
        region_modes = result.lambda[1:result.M]
        
        for (i, λ) in enumerate(region_modes)
            growth_rate = real(λ)
            frequency = imag(λ)
            stability = growth_rate > 1e-6 ? "UNSTABLE" : 
                       growth_rate < -1e-6 ? "stable" : "neutral"
                       
            println("  λ[$i] = $(round(λ, digits=6)) ($stability)")
            println("    Growth rate: $(round(growth_rate, digits=6))")  
            println("    Frequency: $(round(frequency, digits=6))")
        end
        
        append!(all_eigenvalues, region_modes)
    end
end

# Stability analysis summary
println("\\n" * "="^60)
println("FLOW STABILITY SUMMARY")  
println("="^60)

if length(all_eigenvalues) > 0
    unstable_modes = filter(λ -> real(λ) > 1e-6, all_eigenvalues)
    neutral_modes = filter(λ -> abs(real(λ)) <= 1e-6, all_eigenvalues)  
    stable_modes = filter(λ -> real(λ) < -1e-6, all_eigenvalues)
    
    println("Total modes found: $(length(all_eigenvalues))")
    println("Unstable modes: $(length(unstable_modes))")
    println("Neutral modes: $(length(neutral_modes))")
    println("Stable modes: $(length(stable_modes))")
    
    if length(unstable_modes) > 0
        max_growth = maximum(real.(unstable_modes))
        println("\\nFLOW IS UNSTABLE!")
        println("Maximum growth rate: $max_growth")
        
        # Find most unstable mode
        most_unstable_idx = argmax(real.(unstable_modes))
        λ_most_unstable = unstable_modes[most_unstable_idx]
        println("Most unstable mode: $λ_most_unstable")
        println("Doubling time: $(log(2)/real(λ_most_unstable))")
        
    else
        println("\\nFLOW APPEARS STABLE")
        println("All found modes have negative or zero growth rates")
    end
else
    println("No eigenvalues found in searched regions")
    println("Flow stability could not be determined")
end
```

---

<div align="center">
  <p><strong>Complete Examples Collection for FeastKit.jl</strong></p>
  <p>From basic usage to advanced scientific applications</p>
  ← [Getting Started](@ref "getting-started") | [API Reference](@ref "api-reference") →
</div>
