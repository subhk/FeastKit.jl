# FEAST.jl

[![Build Status](https://github.com/subhk/FEAST.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/FEAST.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia implementation of the FEAST eigenvalue solver for finding eigenvalues and eigenvectors of large-scale eigenvalue problems within a specified region.

## Overview

FEAST.jl is a pure Julia translation of the original FEAST (Fast Eigenvalue Algorithm using Spectral Transformations) library. FEAST is a numerical algorithm for solving both standard and generalized eigenvalue problems by computing eigenvalues located inside a given region in the complex plane.

### Key Features

- **Interval-based eigenvalue computation**: Find eigenvalues in specified intervals or regions
- **Multiple matrix formats**: Support for dense, sparse, and banded matrices  
- **Generalized eigenvalue problems**: Solve both Ax = λx and Ax = λBx problems
- **Complex arithmetic**: Handle both real symmetric/Hermitian and general complex matrices
- **Reverse Communication Interface (RCI)**: Advanced interface for custom linear solvers
- **Polynomial eigenvalue problems**: Support for polynomial eigenvalue problems

## Installation

```julia
using Pkg
Pkg.add("FEAST")
```

## Quick Start

### Basic Usage

```julia
using FEAST
using LinearAlgebra

# Create a test matrix
n = 100
A = diagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))

# Find eigenvalues in the interval [0.5, 1.5]
result = feast(A, (0.5, 1.5), M0=10)

println("Found $(result.M) eigenvalues")
println("Eigenvalues: ", result.lambda)
```

### Generalized Eigenvalue Problems

```julia
# For generalized problem Ax = λBx
B = diagm(0 => ones(n))
result = feast(A, B, (0.5, 1.5), M0=10)
```

### Sparse Matrices

```julia
using SparseArrays

# Create sparse matrix
A_sparse = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
result = feast(A_sparse, (0.5, 1.5), M0=10)
```

### Complex Eigenvalue Problems

```julia
# For general complex matrices, use circular search region
A_complex = randn(ComplexF64, 50, 50)
B_complex = Matrix{ComplexF64}(I, 50, 50)

# Search in circle centered at origin with radius 2
center = 0.0 + 0.0im
radius = 2.0
result = feast_general(A_complex, B_complex, center, radius, M0=15)
```

## Advanced Usage

### Custom FEAST Parameters

```julia
# Initialize FEAST parameters
fpm = feastinit()

# Customize parameters
feast_set_defaults!(fpm.fpm, 
                   print_level=1,
                   integration_points=16,  # More integration points
                   tolerance_exp=14,       # Higher precision
                   max_refinement=30)      # More refinement loops

# Use custom parameters
result = feast(A, (0.5, 1.5), M0=10, fpm=fpm.fpm)
```

### Banded Matrices

```julia
# For banded matrices stored in LAPACK format
n = 100
k = 2  # Number of super-diagonals
A_banded = zeros(k+1, n)
# ... fill A_banded with appropriate values ...

result = feast_banded(A_banded, k, (0.5, 1.5), M0=10)
```

### Matrix-Free Operations

```julia
# Define matrix-vector operations
function A_mul!(y, x)
    # Implement y = A*x
    y .= A * x
end

function B_mul!(y, x)
    # Implement y = B*x  
    y .= B * x
end

# Use matrix-free interface
result = feast_matvec(A_mul!, B_mul!, n, (0.5, 1.5), M0=10)
```

## Algorithm Overview

FEAST uses contour integration in the complex plane to compute eigenvalues. The key steps are:

1. **Contour Definition**: Define an integration contour enclosing the desired eigenvalues
2. **Moment Computation**: Compute spectral projector moments using numerical integration  
3. **Subspace Extraction**: Extract eigenspace using computed moments
4. **Refinement**: Iteratively refine the solution until convergence

The algorithm is particularly effective for:
- Large sparse matrices
- Finding eigenvalues in specific intervals
- Parallel computation (via the original PFEAST routines)

## Result Structure

FEAST returns a `FeastResult` object containing:

```julia
struct FeastResult{T<:Real, VT}
    lambda::Vector{T}      # Computed eigenvalues
    q::Matrix{VT}          # Computed eigenvectors  
    M::Int                 # Number of eigenvalues found
    res::Vector{T}         # Residual norms
    info::Int              # Exit status (0 = success)
    epsout::T              # Final residual
    loop::Int              # Number of refinement loops
end
```

## Error Codes

- `info = 0`: Successful convergence
- `info = 1`: Invalid matrix size
- `info = 2`: Invalid search subspace size  
- `info = 3`: Invalid search interval
- `info = 5`: No convergence achieved
- `info = 6`: Memory allocation error
- `info = 8`: LAPACK error

## Performance Tips

1. **Choose appropriate M0**: Set M0 slightly larger than expected number of eigenvalues
2. **Integration points**: More points (fpm[2]) improve accuracy but increase cost
3. **Sparse matrices**: Use sparse format for large problems with few non-zeros
4. **Initial guess**: Provide good initial guess when available (fpm[5] = 1)

## Limitations

This is a Julia translation of the original FEAST library. Some features from the original may not be fully implemented:

- Custom contour integration is partially implemented
- Some advanced PFEAST (parallel) features are not included
- Matrix-free interface is simplified compared to the original

## References

1. E. Polizzi, "Density-matrix-based algorithm for solving eigenvalue problems", Physical Review B 79, 115112 (2009)
2. FEAST official website: http://www.feast-solver.org

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the same terms as the original FEAST library.
