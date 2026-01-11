# Testing Guide

This guide covers how to run and write tests for FeastKit.jl.

## Table of Contents

- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Writing Tests](#writing-tests)
- [Test Categories](#test-categories)
- [Debugging Tests](#debugging-tests)
- [Continuous Integration](#continuous-integration)

---

## Running Tests

### Full Test Suite

```bash
# Standard test run
julia --project -e 'using Pkg; Pkg.test()'

# With multiple threads
julia --project --threads=auto -e 'using Pkg; Pkg.test()'

# With verbose output
julia --project -e 'using Pkg; Pkg.test(coverage=true)'
```

### Specific Tests

```julia
# Interactive testing
julia --project

julia> using Test
julia> include("test/runtests.jl")

# Run specific test file
julia> include("test/test_matrix_free.jl")
```

### Test with Specific Configuration

```bash
# Keep artifacts local
JULIA_DEPOT_PATH=$PWD/.julia julia --project -e 'using Pkg; Pkg.test()'

# Test with MPI (if available)
FEASTKIT_ENABLE_MPI=true mpiexec -n 4 julia --project test/runtests.jl
```

---

## Test Structure

### Directory Layout

```
test/
├── runtests.jl           # Main test file
├── test_matrix_free.jl   # Matrix-free interface tests
└── (future test files)
```

### Main Test File

`test/runtests.jl` contains the primary test suite organized by functionality:

```julia
using Test
using FeastKit
using LinearAlgebra
using SparseArrays

@testset "FeastKit.jl" begin
    @testset "Core Types" begin
        # Type tests
    end

    @testset "Dense Solvers" begin
        # Dense matrix tests
    end

    @testset "Sparse Solvers" begin
        # Sparse matrix tests
    end

    # ... more testsets
end
```

---

## Writing Tests

### Basic Test Patterns

```julia
@testset "Feature Name" begin
    # Setup
    n = 100
    A = create_test_matrix(n)

    # Test basic functionality
    @test result.info == 0
    @test result.M > 0

    # Test numerical accuracy
    @test isapprox(computed, expected, rtol=1e-10)

    # Test types
    @test result isa FeastResult

    # Test exceptions
    @test_throws ArgumentError bad_function_call()
end
```

### Testing Eigenvalue Accuracy

```julia
@testset "Eigenvalue accuracy" begin
    n = 50
    A = SymTridiagonal(2*ones(n), -ones(n-1))
    B = Matrix(1.0I, n, n)

    # Compute reference eigenvalues
    λ_ref = eigvals(A)

    # FEAST result
    result = feast(A, (0.0, 1.0), M0=20)

    # Check eigenvalues match
    for i in 1:result.M
        λ = result.lambda[i]
        # Find closest reference eigenvalue
        idx = argmin(abs.(λ_ref .- λ))
        @test isapprox(λ, λ_ref[idx], rtol=1e-10)
    end
end
```

### Testing Eigenvector Residuals

```julia
@testset "Eigenvector residuals" begin
    n = 100
    A = Symmetric(randn(n, n))
    B = Matrix(1.0I, n, n)

    result = feast(A, (-1.0, 1.0), M0=20)

    for i in 1:result.M
        λ = result.lambda[i]
        x = result.q[:, i]

        # Residual: ||Ax - λBx|| / ||Ax||
        Ax = A * x
        residual = norm(Ax - λ * B * x) / norm(Ax)
        @test residual < 1e-10
    end
end
```

### Testing Orthogonality

```julia
@testset "Eigenvector orthogonality" begin
    result = feast(A, B, interval, M0=20)

    Q = result.q[:, 1:result.M]

    # For standard problem (B=I): Q'Q ≈ I
    @test isapprox(Q' * Q, I, atol=1e-10)

    # For generalized problem: Q'BQ ≈ I
    @test isapprox(Q' * B * Q, I, atol=1e-10)
end
```

---

## Test Categories

### Core Type Tests

```julia
@testset "FeastResult" begin
    # Test construction
    result = FeastResult{Float64, Float64}(
        [1.0, 2.0, 3.0],        # lambda
        randn(10, 3),           # q
        3,                       # M
        [1e-12, 1e-12, 1e-12],  # res
        0,                       # info
        1e-12,                   # epsout
        5                        # loop
    )

    @test length(result.lambda) == 3
    @test result.M == 3
    @test result.info == 0
end
```

### Dense Solver Tests

```julia
@testset "Dense Solvers" begin
    @testset "feast_sygv! - Real symmetric generalized" begin
        n = 50
        A = Symmetric(randn(n, n))
        B = Symmetric(randn(n, n) + 5I)
        fpm = zeros(Int, 64)
        feastinit!(fpm)

        result = feast_sygv!(Matrix(A), Matrix(B), -1.0, 1.0, 15, fpm)

        @test result.info == 0
        @test 0 <= result.M <= 15
    end

    @testset "feast_heev! - Complex Hermitian" begin
        n = 50
        A = randn(ComplexF64, n, n)
        A = (A + A') / 2  # Hermitian
        fpm = zeros(Int, 64)
        feastinit!(fpm)

        result = feast_heev!(copy(A), -1.0, 1.0, 15, fpm)

        @test result.info == 0
        @test all(isreal.(result.lambda[1:result.M]))
    end
end
```

### Sparse Solver Tests

```julia
@testset "Sparse Solvers" begin
    @testset "feast_scsrgv! - Sparse symmetric generalized" begin
        n = 500
        A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
        B = sparse(1.0I, n, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)

        result = feast_scsrgv!(A, B, 0.0, 1.0, 20, fpm)

        @test result.info == 0
        @test result.M > 0
    end
end
```

### Parallel Tests

```julia
@testset "Parallel Computing" begin
    @testset "Threading" begin
        if Threads.nthreads() > 1
            n = 500
            A = Symmetric(randn(n, n))
            B = Matrix(1.0I, n, n)

            result = feast_parallel(A, B, (0.0, 1.0), M0=20, use_threads=true)

            @test result.info == 0
            @test result.M > 0
        else
            @info "Skipping threading tests (single thread)"
        end
    end
end
```

### Matrix-Free Tests

```julia
@testset "Matrix-Free Interface" begin
    n = 200

    # Define operators
    function A_mul!(y, x)
        y .= 2 .* x
        y[1:end-1] .-= x[2:end]
        y[2:end] .-= x[1:end-1]
    end

    A_op = LinearOperator{Float64}(A_mul!, (n, n), issymmetric=true)

    @testset "LinearOperator creation" begin
        @test size(A_op) == (n, n)
        @test A_op.issymmetric == true
    end

    @testset "Matrix-free solve" begin
        result = feast(A_op, (0.0, 1.0), M0=10, solver=:cg)
        @test result.info == 0 || result.M > 0
    end
end
```

### Error Handling Tests

```julia
@testset "Error Handling" begin
    @testset "Invalid interval" begin
        A = randn(10, 10)
        A = A + A'
        # Emin >= Emax should fail
        @test_throws ArgumentError feast(A, (1.0, 0.0), M0=5)
    end

    @testset "Invalid M0" begin
        A = randn(10, 10)
        A = A + A'
        @test_throws ArgumentError feast(A, (0.0, 1.0), M0=0)
        @test_throws ArgumentError feast(A, (0.0, 1.0), M0=-1)
    end
end
```

---

## Debugging Tests

### Verbose Testing

```julia
@testset verbose=true "Detailed Tests" begin
    @testset "Subtest 1" begin
        @test true
    end
    @testset "Subtest 2" begin
        @test true
    end
end
```

### Debugging Failing Tests

```julia
# Add debugging output
@testset "Debug example" begin
    result = feast(A, (0.0, 1.0), M0=10)

    @show result.info
    @show result.M
    @show result.epsout
    @show result.loop

    if result.M > 0
        @show result.lambda[1:result.M]
    end

    @test result.info == 0
end
```

### Isolating Failures

```julia
# Run single test interactively
julia --project
```

```julia
using FeastKit, LinearAlgebra, Test

# Set up test case
n = 100
A = SymTridiagonal(2*ones(n), -ones(n-1))

# Run the failing test
result = feast(A, (0.5, 1.5), M0=10)

# Inspect results
result.info
result.M
result.lambda
```

---

## Continuous Integration

### GitHub Actions Configuration

Tests run automatically on pull requests via `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: ['1.10', '1.11']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

### Test Coverage

Coverage is collected during CI runs:

```julia
# Run tests with coverage
Pkg.test(coverage=true)
```

### CI Environment Detection

```julia
@testset "CI-specific tests" begin
    if get(ENV, "CI", "false") == "true"
        @info "Running on CI"
        # CI-specific tests
    else
        @info "Running locally"
        # Local-only tests (e.g., longer benchmarks)
    end
end
```

---

## Best Practices

### Test Design

1. **Keep tests fast**: Prefer small matrices when possible
2. **Be deterministic**: Use `Random.seed!()` for reproducibility
3. **Test edge cases**: Empty intervals, single eigenvalue, etc.
4. **Clean up**: Don't leave temporary files

### Reproducibility

```julia
using Random

@testset "Reproducible tests" begin
    Random.seed!(42)
    A = Symmetric(randn(50, 50))
    result = feast(A, (-1.0, 1.0), M0=10)
    @test result.M == expected_M  # Should always match
end
```

### Numerical Tolerance

```julia
# Use appropriate tolerances
@test isapprox(computed, expected, rtol=1e-10)  # Relative
@test isapprox(computed, expected, atol=1e-12)  # Absolute
@test abs(computed - expected) < 1e-10          # Manual

# For eigenvalues with different scales
for (λ_computed, λ_expected) in zip(computed, expected)
    if abs(λ_expected) > 1
        @test isapprox(λ_computed, λ_expected, rtol=1e-10)
    else
        @test isapprox(λ_computed, λ_expected, atol=1e-12)
    end
end
```

---

<div align="center">
  <p><strong>Ensuring FeastKit.jl quality through comprehensive testing</strong></p>
  <a href="contributing.md">Contributing Guide</a> · <a href="developer_guide.md">Developer Guide</a>
</div>
