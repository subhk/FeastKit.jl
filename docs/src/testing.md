# Testing Guide

This guide covers how to run and write tests for FeastKit.jl.

```@contents
Pages = ["testing.md"]
Depth = 2
```

---

## Running Tests

### Full Test Suite

```bash
# Standard test run
julia --project -e 'using Pkg; Pkg.test()'

# With multiple threads
julia --project --threads=auto -e 'using Pkg; Pkg.test()'

# Collect local coverage data
julia --project -e 'using Pkg; Pkg.test(coverage=true)'
```

### Specific Tests

`Pkg.test()` prepares the test extras automatically. To include an individual
file manually, first create a temporary environment containing this checkout
and its optional solver packages (run from the repository root):

```julia
using Pkg
checkout = pwd()
Pkg.activate(temp=true)
Pkg.develop(path=checkout)
Pkg.add(["Krylov", "MPI"])
include(joinpath(checkout, "test", "matrixfree", "operators_and_solvers.jl"))
```

### Test with Specific Configuration

```bash
# Keep artifacts local
JULIA_DEPOT_PATH=$PWD/.julia julia --project -e 'using Pkg; Pkg.test()'

# Test distributed workers
FEASTKIT_TEST_DISTRIBUTED=true julia --project -e 'using Distributed; addprocs(2; exeflags=`--project=$(Base.active_project())`); @everywhere using FeastKit, LinearAlgebra, SparseArrays; include("test/backends/execution.jl")'

# MPI tests need an environment with both MPI and Krylov.
# See the mpi-backend job in .github/workflows/ci.yml for the complete launcher command.
```

---

## Test Structure

### Directory Layout

```
test/
├── runtests.jl           # Includes suites grouped by feature
├── support/             # Common imports and deterministic backend fixtures
├── api/                 # Public dispatch, options, compatibility, result wrappers
├── core/                # Parameters, utilities, allocation and CI coverage checks
├── contours/            # Contour generation, shapes, and membership
├── storage/             # Dense, sparse, and banded drivers
├── rci/                 # RCI protocol tests
├── matrixfree/          # Operator and shifted-solver tests
├── numerics/            # Eigenpair accuracy, scaling, projection, solver failures
├── backends/            # Threaded, worker, MPI, and hybrid execution
└── docs/                # Published snippets and standalone example programs
```

### Main Test File

`test/runtests.jl` is an include manifest. Add new tests to the relevant feature
directory and include them in that feature's testset. Existing numerical
regressions remain part of the ordinary suite.

`test/support/setup.jl` provides imports for the extracted storage and RCI suites.
`test/support/fixtures.jl` provides the shared small tridiagonal problem used by
API and worker/MPI tests. Test files with their own imports can also run directly
in a prepared test environment.

The full CI job enables `FEAST_RUN_LONG_TESTS`, `FEAST_RUN_PARALLEL_TESTS`, and
`FEASTKIT_TEST_PARALLEL`. Worker and MPI execution are enabled in dedicated jobs
using `FEASTKIT_TEST_DISTRIBUTED` and `FEASTKIT_TEST_MPI`. MPI fault-injection
scripts (`backends/mpi_faults.jl` and `backends/mpi_rhs_faults.jl`) run in separate
processes with timeouts because they deliberately replace numerical methods.
They must not be included in the ordinary test runner.

`test/docs/example_scripts.jl` is an additional standalone example-program check;
`docs/check_parallel_examples.jl` executes the parallel scripts printed in the
manual. The strict documentation build executes the `@example` blocks.

---

## Writing Tests

### Basic Test Patterns

```@example test_basics
using Test, FeastKit, LinearAlgebra
@testset "Small standard problem" begin
    A = Matrix(Diagonal([1.0, 2.0, 4.0]))
    result = feast(A, (0.5, 2.5); subspace_size=3)
    @test result.converged
    @test result.M == 2
    @test result.values ≈ [1.0, 2.0] atol=1e-9
    @test result isa FeastResult
    @test_throws ArgumentError feast(A, (2.5, 0.5))
end
```

### Testing Eigenvalue Accuracy

Place interval endpoints in gaps between eigenvalues. This matrix has an
eigenvalue exactly at `1.0`, so use `1.05` as the upper endpoint to avoid
roundoff-dependent inclusion of that eigenvalue.

```@example test_eigenvalues
using Test, FeastKit, LinearAlgebra
@testset "Eigenvalue accuracy" begin
    n = 50
    A = SymTridiagonal(2*ones(n), -ones(n-1))
    interval = (0.0, 1.05)

    # Compute reference eigenvalues
    λ_ref = eigvals(A)

    # FEAST result
    result = feast(A, interval, M0=20)

    expected = filter(λ -> interval[1] < λ < interval[2], λ_ref)
    @test length(expected) == 17
    @test result.converged
    @test result.M == length(expected)
    @test result.values ≈ expected atol=1e-9

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

```@example test_residuals
using Test, FeastKit, LinearAlgebra
@testset "Eigenvector residuals" begin
    n = 100
    A = Matrix(SymTridiagonal(2.0*ones(n), -ones(n-1)))
    B = Matrix(1.0I, n, n)

    result = feast(A, (0.0, 0.05), M0=20)

    @test result.converged && result.M == 7
    for i in 1:result.M
        λ = result.lambda[i]
        x = result.q[:, i]

        # Scaled residual: ||Ax - λBx|| / (||Bx|| * max(|λ|, 1))
        Ax = A * x
        residual = norm(Ax - λ * B * x) / norm(B * x) / max(abs(λ), 1)
        @test residual < 1e-10
    end
end
```

### Testing Orthogonality

```@example test_orthogonality
using Test, FeastKit, LinearAlgebra
A = Matrix(Diagonal([2.0, 6.0, 16.0]))
B = Matrix(Diagonal([2.0, 3.0, 4.0]))
interval = (0.5, 2.5)
@testset "Eigenvector orthogonality" begin
    result = feast(A, B, interval, M0=20)

    Q = result.q[:, 1:result.M]

    @test result.converged
    # Returned vectors have unit Euclidean norm. For a symmetric/Hermitian
    # definite pencil with distinct eigenvalues, the B-Gram matrix is diagonal.
    @test all(j -> isapprox(norm(Q[:, j]), 1; atol=1e-10), axes(Q, 2))
    gram = Q' * B * Q
    @test isapprox(gram, Diagonal(diag(gram)); atol=1e-10)
    # Normalize explicitly if downstream work requires Q'BQ = I.
    Q_B = Q * Diagonal(inv.(sqrt.(real.(diag(gram)))))
    @test isapprox(Q_B' * B * Q_B, I; atol=1e-10)
end
```

---

## Test Categories

### Core Type Tests

```@example test_result_type
using Test, FeastKit
@testset "FeastResult" begin
    # Test construction
    result = FeastResult{Float64, Float64}(
        [1.0, 2.0, 3.0],        # lambda
        zeros(10, 3),           # q
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

```@example test_dense_drivers
using Test, FeastKit, LinearAlgebra
@testset "Dense drivers" begin
    B = Matrix(Diagonal([2.0, 3.0, 4.0]))
    A = B * Diagonal([1.0, 2.0, 4.0])
    result = feast_sygv!(A, B, 0.5, 2.5, 3, feastinit().fpm)
    @test result.converged && result.M == 2
    @test result.values ≈ [1.0, 2.0] atol=1e-9

    H = ComplexF64[2 im; -im 2]
    result = feast_heev!(H, 0.5, 3.5, 2, feastinit().fpm)
    @test result.converged && result.M == 2
    @test result.values ≈ [1.0, 3.0] atol=1e-9
end
```

### Sparse Solver Tests

```@example test_sparse_drivers
using Test, FeastKit, LinearAlgebra, SparseArrays
@testset "Sparse Solvers" begin
    @testset "feast_scsrgv! - Sparse symmetric generalized" begin
        n = 500
        A = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
        B = sparse(1.0I, n, n)
        fpm = zeros(Int, 64)
        feastinit!(fpm)

        result = feast_scsrgv!(A, B, 0.0, 1.05*(2-2cos(10π/(n+1))), 20, fpm)

        @test result.info == 0
        @test result.M == 10
    end
end
```

### Parallel Tests

```@example test_threaded_drivers
using Test, FeastKit, LinearAlgebra
@testset "Parallel Computing" begin
    @testset "Threading" begin
        if Threads.nthreads() > 1
            n = 20
            A = Matrix(Diagonal(collect(1.0:n)))
            B = Matrix(1.0I, n, n)

            result = feast_parallel(A, B, (0.5, 2.5), M0=4, use_threads=true)

            @test result.info == 0
            @test result.M == 2
        else
            @info "Skipping threading tests (single thread)"
        end
    end
end
```

### Matrix-Free Tests

```@example test_matfree
using Test, FeastKit, LinearAlgebra
using FeastKit, Krylov, Test
@testset "Matrix-Free Interface" begin
    n = 12

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
        result = feast(A_op, (0.1, 1.0); subspace_size=5, tol=1e-9, solver=:gmres,
                       solver_opts=(rtol=1e-12, maxiter=200, restart=16))
        @test result.converged
        @test result.M == 3
    end
end
```

### Error Handling Tests

```@example test_errors
using Test, FeastKit, LinearAlgebra
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

```@example test_verbose
using Test, FeastKit, LinearAlgebra
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

```@example test_debugging
using Test, FeastKit, LinearAlgebra
A = Matrix(Diagonal([0.1, 0.4, 0.9, 2.0]))
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
    @test result.M == 3
end
```

### Isolating Failures

```bash
julia --project
```

```@example test_isolated
using FeastKit, LinearAlgebra, Test

# Set up test case
n = 100
A = Matrix(SymTridiagonal(2*ones(n), -ones(n-1)))

# Reproduce the failing call. Count what the interval holds first: (0.5, 1.5)
# contains 19 eigenvalues here, so M0 = 10 cannot establish completeness (status 2 or 5).
exact(k) = 2 - 2cos(k * π / (n + 1))
expected = count(k -> 0.5 <= exact(k) <= 1.5, 1:n)
result = feast(A, (0.5, 1.5), M0 = 2 * expected)

# Inspect results
@test result.converged
@test result.M == expected
@test result.values ≈ filter(λ -> 0.5 < λ < 1.5, exact.(1:n)) atol=1e-9
result.values
```

---

## Continuous Integration

### GitHub Actions Configuration

The actual workflow is `.github/workflows/ci.yml`. It runs Julia 1.10 and 1.11
on Linux, macOS, and Windows with two threads and enables the optional long and
parallel test groups. Separate jobs exercise Julia workers and two MPI ranks,
including collective fault tests. `.github/workflows/pages.yml` builds the web
docs against the checkout and fails on broken examples or references.

To enable the same optional test groups locally:

```sh
FEAST_RUN_LONG_TESTS=true FEAST_RUN_PARALLEL_TESTS=true FEASTKIT_TEST_PARALLEL=true julia --project=. --threads=2 -e 'using Pkg; Pkg.test()'
```

### Test Coverage

Request coverage explicitly for a local run (the main CI job does not currently enable it):

```julia
using Pkg
# Run tests with coverage
Pkg.test(coverage=true)
```

### CI Environment Detection

```@example test_ci_environment
using Test, FeastKit, LinearAlgebra
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

```@example test_reproducible
using Test, FeastKit, LinearAlgebra
using Random

@testset "Reproducible tests" begin
    Random.seed!(42)
    A = Symmetric(randn(50, 50))
    # Take the expected count from a dense reference rather than a magic number,
    # so the test states what it means and survives a reseed.
    expected_M = count(λ -> -1.0 <= λ <= 1.0, eigvals(A))
    result = feast(A, (-1.0, 1.0), M0=2 * expected_M)
    @test result.info == 0
    @test result.M == expected_M
end
```

### Numerical Tolerance

```@example test_tolerances
using FeastKit, LinearAlgebra, Test

A = Matrix(Diagonal([1.0e-6, 2.0e-6, 1.0, 2.0, 100.0]))
expected = filter(λ -> 0.0 < λ < 3.0, sort(eigvals(A)))
result = feast(A, (0.0, 3.0), M0 = 8)
computed = sort(result.lambda[1:result.M])
@test result.converged && result.M == length(expected)

# Use appropriate tolerances
@test isapprox(computed, expected, rtol=1e-10)  # Relative
@test isapprox(computed, expected, atol=1e-12)  # Absolute
@test all(abs.(computed .- expected) .< 1e-10)  # Manual

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

**Ensuring FeastKit.jl quality through comprehensive testing**

[Contributing Guide](contributing.md) · [Developer Guide](developer_guide.md)
