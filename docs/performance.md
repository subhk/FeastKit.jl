# Performance Guide

```@id performance-guide
```

Optimize FeastKit.jl for maximum performance in your eigenvalue calculations.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Memory Optimization](#memory-optimization)
3. [Computational Efficiency](#computational-efficiency)
4. [Parallel Computing](#parallel-computing)
5. [Problem-Specific Optimizations](#problem-specific-optimizations)
6. [Benchmarking and Profiling](#benchmarking-and-profiling)

---

## Performance Overview

### FEAST Algorithm Complexity

The FEAST algorithm has the following computational complexity:

| Operation | Complexity | Description |
|-----------|------------|-------------|
| **Contour Integration** | O(ne × M₀ × T_solve) | ne integration points, M₀ subspace size |
| **Reduced Eigenvalue Problem** | O(M₀³) | Dense eigenvalue problem |
| **Refinement Iterations** | O(k × above) | k ≈ 1-5 typically |

Where T_solve is the cost of solving (zB - A)X = Y linear systems.

!!! performance "Key Performance Factors"
    - **Linear solver efficiency**: Dominates total cost (80-95%)
    - **Number of integration points**: Linear scaling with ne
    - **Subspace size M₀**: Memory O(N×M₀), computation O(M₀³)
    - **Matrix structure**: Dense vs sparse vs matrix-free

---

## Memory Optimization

### Memory Usage Patterns

```julia
using FeastKit, LinearAlgebra

# Problem sizes and memory requirements
function memory_analysis(N, M0, ne=8)
    println("Memory Analysis for N=$N, M₀=$M0")
    println("="^50)
    
    # Workspace memory (dominant term)
    workspace_mb = 8 * N * M0 / 1e6  # Float64 vectors
    
    # Integration weights/nodes  
    integration_mb = 16 * ne / 1e6  # Complex{Float64}
    
    # Reduced matrices
    reduced_mb = 8 * M0^2 / 1e6
    
    println("Workspace vectors: $(workspace_mb) MB")
    println("Integration data: $(integration_mb) MB") 
    println("Reduced matrices: $(reduced_mb) MB")
    println("Total FeastKit memory: $(workspace_mb + integration_mb + reduced_mb) MB")
    
    # Matrix storage (if not matrix-free)
    matrix_dense_gb = 8 * N^2 / 1e9
    matrix_sparse_mb = 8 * 0.001 * N^2 / 1e6  # Assuming 0.1% density
    
    println("\nMatrix storage (not matrix-free):")
    println("Dense matrix: $(matrix_dense_gb) GB")
    println("Sparse matrix (~0.1%): $(matrix_sparse_mb) MB")
    
    return (workspace_mb, matrix_dense_gb)
end

# Analyze different problem sizes
for N in [1000, 10000, 100000, 1000000]
    memory_analysis(N, 20)
    println()
end
```

### Memory Optimization Strategies

!!! tip "Memory Optimization Tips"
    1. **Use matrix-free methods** for large problems
    2. **Choose M₀ carefully**: Balance accuracy vs memory  
    3. **Reduce integration points** if convergence allows
    4. **Use iterative refinement** instead of high precision
    5. **Consider domain decomposition** for very large problems

#### Strategy 1: Matrix-Free Implementation

```julia
# Instead of storing the full matrix
A_matrix = create_large_matrix(n)  # Requires O(n²) memory
result = feast(A_matrix, interval)

# Use matrix-free operations  
function A_mult!(y, x)
    # Your matrix-vector product (O(n) memory)
    apply_operator!(y, x, parameters)
end

A_op = LinearOperator{Float64}(A_mult!, (n, n), issymmetric=true)
result = feast(A_op, interval, solver=:cg)
```

#### Strategy 2: Adaptive Subspace Sizing

```julia
function adaptive_feast(A, interval; M0_initial=10, max_M0=50)
    M0 = M0_initial
    
    while M0 <= max_M0
        result = feast(A, interval, M0=M0)
        
        if result.info == 0
            # Check if we found all eigenvalues in interval
            if result.M < M0 * 0.8  # Found significantly fewer than M0
                println("Found $(result.M) eigenvalues with M0=$M0")
                return result
            end
        end
        
        M0 = min(M0 + 10, max_M0)
        println("Increasing M0 to $M0")
    end
    
    @warn "May not have found all eigenvalues"
    return result
end
```

---

## Computational Efficiency

### Integration Method Selection

Different integration methods have varying computational costs:

```julia
using FeastKit, BenchmarkTools

function benchmark_integration_methods(A, interval)
    methods = [
        (0, "Gauss-Legendre"),
        (1, "Trapezoidal"), 
        (2, "Zolotarev")
    ]
    
    println("Integration Method Comparison")
    println("="^40)
    
    results = []
    
    for (method_id, method_name) in methods
        println("\n$method_name:")
        
        # Test different numbers of integration points
        for ne in [8, 12, 16, 24]
            contour = feast_contour_expert(interval[1], interval[2], ne, method_id, 100)
            
            fpm = zeros(Int, 64)
            feastinit!(fmp)
            fpm[2] = ne
            fpm[16] = method_id
            
            time = @elapsed begin
                result = feast(A, interval, M0=10, fpm=fpm)
            end
            
            push!(results, (method_name, ne, time, result.M, result.info))
            println("  ne=$ne: $(time) s, found $(result.M), status $(result.info)")
        end
    end
    
    return results
end

# Run benchmark
A = SymTridiagonal(2.0 * ones(1000), -1.0 * ones(999))
results = benchmark_integration_methods(A, (0.1, 1.0))
```

!!! performance "Integration Method Guidelines"
    - **Gauss-Legendre**: Highest accuracy per point, best for smooth problems
    - **Zolotarev**: Optimal for elliptical contours, excellent for clustered eigenvalues
    - **Trapezoidal**: Simplest, good for initial experiments

### Linear Solver Optimization

The linear solver dominates computational cost. Choose wisely:

```julia
function benchmark_solvers(A_op, B_op, interval)
    solvers = [
        (:cg, "Conjugate Gradient", (rtol=1e-8, maxiter=1000)),
        (:gmres, "GMRES", (rtol=1e-8, restart=30, maxiter=1000)),
        (:bicgstab, "BiCGSTAB", (rtol=1e-8, l=2, maxiter=1000))
    ]
    
    println("Linear Solver Comparison")
    println("="^30)
    
    for (solver, name, opts) in solvers
        println("\n$name:")
        
        try
            time = @elapsed begin
                result = feast(A_op, B_op, interval, M0=10, 
                              solver=solver, solver_opts=opts)
            end
            
            println("  Time: $(time) s")
            println("  Found: $(result.M) eigenvalues")  
            println("  Status: $(result.info == 0 ? "Success" : "Failed")")
            println("  Final residual: $(result.epsout)")
            
        catch e
            println("  Failed: $e")
        end
    end
end
```

!!! tip "Solver Selection Guidelines"
    - **CG**: Best for symmetric positive definite (SPD) systems
    - **GMRES**: General purpose, works for any nonsingular system
    - **BiCGSTAB**: Good for systems with complex eigenvalues near shifts

---

## Parallel Computing

### Shared Memory Parallelization

```julia
using FeastKit, LinearAlgebra
BLAS.set_num_threads(8)  # Use 8 threads for BLAS operations

# Enable threading in FeastKit
result = feast(A, interval, M0=20, parallel=:threads)
```

### Distributed Memory Parallelization

```julia
using Distributed, FeastKit

# Add worker processes
addprocs(4)

@everywhere using FeastKit

# Distributed FeastKit
result = feast(A, interval, M0=20, parallel=:mpi)
```

### Hybrid Parallelization

```julia
# Use both MPI and threading
BLAS.set_num_threads(4)  # 4 threads per MPI process

result = feast(A, interval, M0=20, parallel=:hybrid)
```

### Performance Scaling Analysis

```julia
function scaling_study(A, interval)
    thread_counts = [1, 2, 4, 8, 16]
    results = []
    
    println("Parallel Scaling Study")
    println("="^30)
    
    # Serial baseline
    BLAS.set_num_threads(1)
    serial_time = @elapsed begin
        result_serial = feast(A, interval, M0=20, parallel=false)
    end
    
    println("Serial: $(serial_time) s (baseline)")
    
    # Test different thread counts
    for nthreads in thread_counts[2:end]
        BLAS.set_num_threads(nthreads)
        
        parallel_time = @elapsed begin
            result_parallel = feast(A, interval, M0=20, parallel=:threads)
        end
        
        speedup = serial_time / parallel_time
        efficiency = speedup / nthreads
        
        println("$nthreads threads: $(parallel_time) s, speedup $(speedup), efficiency $(efficiency)")
        push!(results, (nthreads, parallel_time, speedup, efficiency))
    end
    
    return results
end
```

---

## Problem-Specific Optimizations

### Sparse Matrix Optimizations

```julia
using SparseArrays, FeastKit

function optimize_sparse_feast(A_sparse, interval)
    println("Sparse Matrix Optimization")
    println("Matrix: $(size(A_sparse, 1))×$(size(A_sparse, 2)), nnz: $(nnz(A_sparse))")
    
    # Strategy 1: Direct sparse solver (for small-medium problems)
    if size(A_sparse, 1) < 50000
        println("\nUsing direct sparse solver:")
        time_direct = @elapsed begin
            result_direct = feast(A_sparse, interval, M0=20)
        end
        println("Time: $(time_direct) s, Found: $(result_direct.M)")
    end
    
    # Strategy 2: Matrix-free with iterative solver
    println("\nUsing matrix-free iterative solver:")
    A_mult!(y, x) = mul!(y, A_sparse, x)
    A_op = LinearOperator{Float64}(A_mult!, size(A_sparse), issymmetric=issymmetric(A_sparse))
    
    time_iterative = @elapsed begin
        result_iterative = feast(A_op, interval, M0=20, 
                               solver=:gmres, 
                               solver_opts=(rtol=1e-6, restart=50))
    end
    println("Time: $(time_iterative) s, Found: $(result_iterative.M)")
    
    # Strategy 3: Preconditioned iterative solver
    println("\nUsing preconditioned iterative solver:")
    P = create_preconditioner(A_sparse)  # Your preconditioner
    
    time_precond = @elapsed begin
        result_precond = feast(A_op, interval, M0=20,
                              solver=:gmres,
                              solver_opts=(rtol=1e-6, Pl=P, restart=30))
    end
    println("Time: $(time_precond) s, Found: $(result_precond.M)")
end

function create_preconditioner(A)
    # Example: Incomplete LU preconditioner
    return ilu(A, τ=0.01)  # Requires Preconditioners.jl
end
```

### Eigenvalue Distribution Optimization

```julia
function optimize_for_distribution(A, interval, eigenvalue_density="uniform")
    if eigenvalue_density == "clustered"
        # Use more integration points and Zolotarev method
        contour = feast_contour_expert(interval[1], interval[2], 24, 2, 100)
        
        fpm = zeros(Int, 64)
        feastinit!(fpm)
        fmp[2] = 24      # More integration points
        fpm[16] = 2      # Zolotarev integration
        fpm[3] = 14      # Higher precision
        
        result = feast(A, interval, M0=30, fpm=fpm)
        
    elseif eigenvalue_density == "sparse"
        # Use fewer integration points, lower precision
        result = feast(A, interval, M0=10, 
                      integration_points=8, 
                      tolerance=1e-8)
        
    else  # uniform
        # Default settings work well
        result = feast(A, interval, M0=20)
    end
    
    return result
end
```

---

## Benchmarking and Profiling

### Comprehensive Benchmarking Suite

```julia
using FeastKit, BenchmarkTools, Profile

function feast_benchmark_suite()
    println("FeastKit.jl Comprehensive Benchmark Suite")
    println("="^50)
    
    # Test problems of increasing size
    problems = [
        (1000, "Small"),
        (10000, "Medium"), 
        (50000, "Large"),
        (100000, "Very Large")
    ]
    
    results = Dict()
    
    for (n, size_name) in problems
        println("\n$size_name Problem (N=$n)")
        println("-"^30)
        
        # Create test matrix
        A = SymTridiagonal(2.0 * ones(n), -1.0 * ones(n-1))
        interval = (0.01, 0.1)
        
        # Benchmark different approaches
        approaches = [
            ("Dense", () -> feast(Matrix(A), interval, M0=10)),
            ("Sparse", () -> feast(sparse(A), interval, M0=10)),
            ("Matrix-Free", () -> begin
                A_mult!(y, x) = mul!(y, A, x)
                A_op = LinearOperator{Float64}(A_mult!, (n, n), issymmetric=true)
                feast(A_op, interval, M0=10, solver=:cg)
            end)
        ]
        
        results[size_name] = []
        
        for (approach_name, approach_func) in approaches
            try
                # Warm up
                if n <= 10000  # Skip warmup for very large problems
                    approach_func()
                end
                
                # Benchmark
                benchmark = @benchmark $approach_func() seconds=30 samples=5
                median_time = median(benchmark).time / 1e9  # Convert to seconds
                
                println("$approach_name: $(median_time) s")
                push!(results[size_name], (approach_name, median_time))
                
            catch e
                println("$approach_name: Failed ($e)")
                push!(results[size_name], (approach_name, Inf))
            end
        end
    end
    
    # Summary table
    println("\n" * "="^60)
    println("BENCHMARK SUMMARY")  
    println("="^60)
    @printf("%-12s %-12s %-12s %-12s\\n", "Size", "Dense (s)", "Sparse (s)", "Matrix-Free (s)")
    println("-"^60)
    
    for (size_name, size_results) in results
        times = Dict(name => time for (name, time) in size_results)
        @printf("%-12s %-12.3f %-12.3f %-12.3f\\n", 
                size_name, 
                get(times, "Dense", Inf),
                get(times, "Sparse", Inf), 
                get(times, "Matrix-Free", Inf))
    end
    
    return results
end

# Run comprehensive benchmark
results = feast_benchmark_suite()
```

### Profiling FeastKit Performance

```julia
using Profile, ProfileView

function profile_feast(A, interval)
    println("Profiling FeastKit Performance")
    
    # Profile a typical FeastKit run
    @profile result = feast(A, interval, M0=20)
    
    # Show profile results
    Profile.print()
    
    # Interactive profile view (if ProfileView.jl is available)
    try
        ProfileView.view()
    catch
        println("Install ProfileView.jl for interactive profiling")
    end
    
    return result
end

# Example usage
A = SymTridiagonal(2.0 * ones(5000), -1.0 * ones(4999))
profile_feast(A, (0.1, 1.0))
```

### Memory Profiling

```julia
function memory_profile_feast(A, interval)
    println("Memory Profiling FeastKit")
    
    # Track memory allocation
    GC.gc()  # Clean up before profiling
    initial_memory = Base.gc_bytes()
    
    result = feast(A, interval, M0=20)
    
    GC.gc()
    final_memory = Base.gc_bytes()
    
    allocated_mb = (final_memory - initial_memory) / 1e6
    println("Total memory allocated: $(allocated_mb) MB")
    
    # Detailed allocation tracking
    @time result = feast(A, interval, M0=20)
    
    return result
end
```

---

## Performance Checklist

!!! success "Pre-Optimization Checklist"
    - [ ] **Problem size**: Use matrix-free for N > 50,000
    - [ ] **Matrix type**: Sparse for <1% density, matrix-free for larger
    - [ ] **Eigenvalue distribution**: Clustered → more integration points
    - [ ] **Search interval**: Tight intervals → faster convergence
    - [ ] **Subspace size M₀**: 1.5-2× expected number of eigenvalues

!!! tip "Solver Optimization Checklist"
    - [ ] **CG for SPD**: Symmetric positive definite problems
    - [ ] **GMRES for general**: Non-symmetric or indefinite problems
    - [ ] **Preconditioner**: For poorly conditioned systems
    - [ ] **Tolerance**: Balance accuracy vs speed (1e-6 to 1e-12)
    - [ ] **Max iterations**: Increase for difficult problems

!!! performance "Parallelization Checklist" 
    - [ ] **BLAS threads**: Set to number of physical cores
    - [ ] **FeastKit parallel**: Use :threads for shared memory
    - [ ] **MPI**: For distributed memory systems  
    - [ ] **Load balancing**: Ensure even work distribution
    - [ ] **Communication**: Minimize for distributed systems

---

<div align="center">
  <p><strong>Optimize your FeastKit.jl calculations for maximum performance</strong></p>
  ← [Examples](@ref "examples") | [API Reference](@ref "api-reference") →
</div>
