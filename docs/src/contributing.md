# Contributing to FeastKit.jl

We welcome contributions to FeastKit.jl! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## Getting Started

### Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with a minimal reproducible example
- **Feature Requests**: Have an idea? Open an issue to discuss
- **Bug Fixes**: Submit a PR with the fix and a test case
- **New Features**: Discuss in an issue first, then implement
- **Documentation**: Improve docs, examples, or docstrings
- **Performance**: Optimize code or add benchmarks

### Before You Start

1. Check existing [issues](https://github.com/subhk/FeastKit.jl/issues) to avoid duplicates
2. For significant changes, open an issue first to discuss
3. Read the [Developer Guide](developer_guide.md) to understand the codebase

---

## Development Setup

### Clone and Install

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/FeastKit.jl.git
cd FeastKit.jl

# Add upstream remote
git remote add upstream https://github.com/subhk/FeastKit.jl.git
```

### Set Up Development Environment

```julia
# Start Julia in project mode
julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()
Pkg.develop(path=".")

# Verify setup
using FeastKit
```

### Run Tests

```bash
# Full test suite
julia --project -e 'using Pkg; Pkg.test()'

# With threading
julia --project --threads=auto -e 'using Pkg; Pkg.test()'

# Specific tests (interactive)
julia --project
```

```julia
using Test
include("test/runtests.jl")
```

---

## Making Changes

### Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### Write Code

1. **Follow existing patterns**: Look at similar code in the codebase
2. **Keep changes focused**: One feature or fix per PR
3. **Add tests**: Every new feature needs tests
4. **Update docs**: Document new public functions

### Test Your Changes

```julia
# Run full test suite
using Pkg
Pkg.test()

# Run specific tests interactively
include("test/runtests.jl")
```

### Commit Changes

```bash
# Stage changes
git add src/your_file.jl test/test_your_file.jl

# Commit with descriptive message
git commit -m "Add feature X for solving Y problems

- Implement new algorithm in src/kernel/...
- Add tests for edge cases
- Update documentation"
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New code has docstrings
- [ ] Tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

### Submit PR

1. Push your branch: `git push origin feature/your-feature-name`
2. Open a Pull Request on GitHub
3. Fill out the PR template with:
   - Description of changes
   - Related issue(s)
   - Test coverage
   - Documentation updates

### PR Review

- Maintainers will review your PR
- Address feedback with additional commits
- Once approved, your PR will be merged

### After Merge

```bash
# Clean up local branch
git checkout main
git pull upstream main
git branch -d feature/your-feature-name
```

---

## Code Style

### General Guidelines

- **Formatting**: 4-space indentation, no trailing whitespace
- **Line length**: Keep lines under 92 characters when possible
- **Naming**:
  - Functions: `snake_case` (`feast_sygv!`, `create_solver`)
  - Types: `PascalCase` (`FeastResult`, `LinearOperator`)
  - Constants: `SCREAMING_SNAKE_CASE` (`FEAST_KRYLOV_AVAILABLE`)
  - Mutating functions: end with `!`

### Julia-Specific

```julia
# Good: Type-stable, explicit
function solve_problem(A::Matrix{T}, B::Matrix{T}) where T<:Real
    result = zeros(T, size(A, 1))
    # ...
    return result
end

# Avoid: Type-unstable
function solve_problem(A, B)
    result = similar(A[:, 1])  # Type inferred at runtime
    # ...
end
```

### Docstrings

```julia
"""
    my_function(A, B; tol=1e-12)

Brief description of what the function does.

# Arguments
- `A::Matrix{T}`: Description of A
- `B::Matrix{T}`: Description of B

# Keyword Arguments
- `tol::Float64=1e-12`: Convergence tolerance

# Returns
- `FeastResult`: Description of return value

# Example
```julia
result = my_function(A, B, tol=1e-10)
```

See also [`related_function`](@ref).
"""
function my_function(A::Matrix{T}, B::Matrix{T}; tol=1e-12) where T
    # Implementation
end
```

---

## Testing Guidelines

### Test Structure

Tests are in `test/runtests.jl` and `test/test_*.jl`:

```julia
@testset "Feature Name" begin
    @testset "Subfeature" begin
        @test some_condition
        @test another_condition
    end
end
```

### What to Test

1. **Basic functionality**: Normal use cases
2. **Edge cases**: Empty inputs, single elements, boundary values
3. **Error handling**: Invalid inputs should throw appropriate errors
4. **Numerical accuracy**: Compare with known solutions
5. **Type stability**: Test with different numeric types

### Example Test

```julia
@testset "feast_sygv!" begin
    # Setup
    n = 100
    A = SymTridiagonal(2*ones(n), -ones(n-1))
    B = Matrix(1.0I, n, n)
    fpm = zeros(Int, 64)
    feastinit!(fpm)

    @testset "Basic functionality" begin
        result = feast_sygv!(A, B, 0.0, 1.0, 10, fpm)
        @test result.info == 0
        @test result.M > 0
        @test all(0.0 .<= result.lambda[1:result.M] .<= 1.0)
    end

    @testset "Residual accuracy" begin
        result = feast_sygv!(A, B, 0.0, 1.0, 10, fpm)
        for i in 1:result.M
            λ, x = result.lambda[i], result.q[:, i]
            residual = norm(A*x - λ*B*x)
            @test residual < 1e-10
        end
    end

    @testset "Error handling" begin
        @test_throws ArgumentError feast_sygv!(A, B, 1.0, 0.0, 10, fpm)  # Invalid interval
    end
end
```

---

## Documentation

### Building Docs Locally

```bash
cd docs

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Build
julia --project=. make.jl

# Serve locally
julia --project=. -e 'using LiveServer; serve(dir="build")'
```

### Documentation Guidelines

1. **Every public function needs a docstring**
2. **Include examples** in docstrings when helpful
3. **Link related functions** using `@ref`
4. **Update API reference** when adding new exports
5. **Add to relevant guide** (e.g., parallel_computing.md for parallel features)

### Adding a New Documentation Page

1. Create `docs/src/your_page.md`
2. Add to `docs/make.jl` in the `pages` array
3. Link from related pages

---

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs for similar discussions
- See the [Developer Guide](developer_guide.md) for architecture details

---

<div align="center">
  <p><strong>Thank you for contributing to FeastKit.jl!</strong></p>
  <a href="developer_guide.md">Developer Guide</a> · <a href="testing.md">Testing Guide</a>
</div>
