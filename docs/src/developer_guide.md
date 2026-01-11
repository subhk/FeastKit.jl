# Developer Guide

This guide provides an overview of the FeastKit.jl project layout, codebase architecture, and development workflow.

## Table of Contents

- [Project Layout](#project-layout)
- [Codebase Architecture](#codebase-architecture)
- [Module Structure](#module-structure)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)

---

## Project Layout

```
FeastKit.jl/
├── Project.toml          # Package manifest and dependencies
├── Manifest.toml         # Dependency lock file
├── README.md             # Project overview
├── LICENSE               # License file
├── src/                  # Main source code
├── test/                 # Test suite
├── docs/                 # Documentation (Documenter.jl)
├── examples/             # Usage examples
└── .github/              # CI/CD workflows
```

---

## Codebase Architecture

The source code is organized into modular directories by functionality:

```
src/
├── FeastKit.jl                    # Main module (exports, initialization)
├── deprecations.jl                # Deprecated function handling
│
├── core/                          # Core types and utilities
│   ├── feast_types.jl             # Type definitions (FeastResult, FeastContour, etc.)
│   ├── feast_parameters.jl        # Parameter handling and initialization
│   ├── feast_tools.jl             # Core FEAST algorithm tools
│   ├── feast_aux.jl               # Auxiliary/helper functions
│   └── feast_backend_utils.jl     # Parallel backend utilities
│
├── kernel/                        # Core algorithm implementation
│   └── feast_kernel.jl            # FEAST kernel (contour integration, RCI)
│
├── dense/                         # Dense matrix solvers
│   └── feast_dense.jl             # feast_sygv!, feast_heev!, feast_gegv!, etc.
│
├── sparse/                        # Sparse matrix solvers
│   └── feast_sparse.jl            # feast_scsrgv!, feast_hcsrgv!, etc.
│
├── banded/                        # Banded matrix solvers
│   └── feast_banded.jl            # feast_sbgv!, feast_hbgv!, feast_gbgv!, etc.
│
├── interfaces/                    # User-facing interfaces
│   ├── feast_interfaces.jl        # High-level feast(), feast_general()
│   └── feast_matfree.jl           # Matrix-free interface (LinearOperator)
│
└── parallel/                      # Parallel computation backends
    ├── feast_parallel.jl          # Threaded/distributed parallel FEAST
    ├── feast_parallel_rci.jl      # Parallel RCI (Reverse Communication Interface)
    ├── feast_mpi.jl               # MPI-based parallel FEAST
    └── feast_mpi_interface.jl     # MPI high-level interface
```

---

## Module Structure

### Core Types (`src/core/feast_types.jl`)

Defines fundamental data structures:

| Type | Description |
|------|-------------|
| `FeastResult{T,VT}` | Result for Hermitian/symmetric problems |
| `FeastGeneralResult{T}` | Result for general non-Hermitian problems |
| `FeastContour{T}` | Integration contour (nodes and weights) |
| `FeastParameters` | Parameter array wrapper |
| `FeastWorkspaceReal{T}` | Workspace for real symmetric problems |
| `FeastWorkspaceComplex{T}` | Workspace for complex Hermitian problems |
| `FeastRCIJob` | RCI job identifiers (enum) |
| `FeastError` | Error codes (enum) |

### Parameters (`src/core/feast_parameters.jl`)

Parameter initialization and management:

- `feastinit!(fpm)` - Initialize parameter array
- `feastdefault!(fpm)` - Reset to defaults
- `feast_set_defaults!(fpm; ...)` - Set parameters by name

### Kernel (`src/kernel/feast_kernel.jl`)

Core FEAST algorithm implementation:

- Contour integration routines
- RCI (Reverse Communication Interface) state machine
- Spectral projection computation
- Eigenvalue extraction and refinement

### Dense Solvers (`src/dense/feast_dense.jl`)

Direct solvers for dense matrices:

| Function | Problem Type |
|----------|--------------|
| `feast_sygv!`, `feast_syev!` | Real symmetric |
| `feast_heev!`, `feast_hegv!` | Complex Hermitian |
| `feast_geev!`, `feast_gegv!` | General non-Hermitian |
| `feast_sypev!`, `feast_hepev!` | Polynomial eigenvalue |

### Sparse Solvers (`src/sparse/feast_sparse.jl`)

Solvers for sparse matrices (CSR format):

| Function | Problem Type |
|----------|--------------|
| `feast_scsrgv!`, `feast_scsrev!` | Real symmetric sparse |
| `feast_hcsrgv!`, `feast_hcsrev!` | Complex Hermitian sparse |
| `feast_gcsrgv!`, `feast_gcsrev!` | General sparse |
| `feast_scsrpev!`, `feast_hcsrpev!` | Sparse polynomial |

### Banded Solvers (`src/banded/feast_banded.jl`)

Solvers for banded matrices:

| Function | Problem Type |
|----------|--------------|
| `feast_sbgv!`, `feast_sbev!` | Real symmetric banded |
| `feast_hbgv!`, `feast_hbev!` | Complex Hermitian banded |
| `feast_gbgv!`, `feast_gbev!` | General banded |

### High-Level Interfaces (`src/interfaces/feast_interfaces.jl`)

User-friendly wrappers:

- `feast(A, interval; ...)` - Standard eigenvalue problem
- `feast(A, B, interval; ...)` - Generalized eigenvalue problem
- `feast_general(A, B, center, radius; ...)` - Non-Hermitian problems
- `feast_banded(A, kl, interval; ...)` - Banded matrices
- `eigvals_feast(...)`, `eigen_feast(...)` - LinearAlgebra-style interfaces

### Matrix-Free Interface (`src/interfaces/feast_matfree.jl`)

For large-scale problems without explicit matrices:

| Type/Function | Description |
|---------------|-------------|
| `MatrixFreeOperator{T}` | Abstract operator type |
| `LinearOperator{T}` | Concrete matrix-free operator |
| `MatrixVecFunction{T}` | Operator with data storage |
| `feast_matfree_srci!` | Matrix-free RCI for symmetric |
| `feast_matfree_grci!` | Matrix-free RCI for general |
| `create_iterative_solver` | Create Krylov solver |

### Parallel Computing (`src/parallel/`)

Multiple parallel backends:

| File | Description |
|------|-------------|
| `feast_parallel.jl` | Threaded/distributed FEAST (`feast_parallel()`) |
| `feast_parallel_rci.jl` | Parallel RCI state machine (`pfeast_srci!`) |
| `feast_mpi.jl` | MPI FEAST (`mpi_feast()`) |
| `feast_mpi_interface.jl` | MPI high-level interface (`feast_hybrid()`) |

---

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/subhk/FeastKit.jl.git
cd FeastKit.jl

# Start Julia with project environment
julia --project=.
```

```julia
# Install dependencies
using Pkg
Pkg.instantiate()

# Load for development
using FeastKit
```

### Running Tests

```bash
# Full test suite
julia --project -e 'using Pkg; Pkg.test()'

# Specific test file
julia --project test/runtests.jl

# With threading
julia --project --threads=auto test/runtests.jl
```

### Building Documentation

```bash
cd docs

# Install doc dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Build documentation locally
julia --project=. make.jl

# Serve locally (requires LiveServer.jl)
julia --project=. -e 'using LiveServer; serve(dir="build")'
```

Documentation is automatically deployed to GitHub Pages on push to `main`.

### Running Examples

```julia
# From project root
include("examples/feast/run_feast_examples.jl")
include("examples/matrix_free_examples.jl")
include("examples/custom_contour_integration.jl")
```

---

## Code Style Guidelines

### Naming Conventions

- **Functions**: `snake_case` (e.g., `feast_sygv!`, `create_iterative_solver`)
- **Types**: `PascalCase` (e.g., `FeastResult`, `LinearOperator`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `FEAST_KRYLOV_AVAILABLE`)
- **Mutating functions**: End with `!` (e.g., `feastinit!`, `feast_srci!`)

### Function Signatures

Follow FEAST convention for solver functions:
```julia
# Standard: feast_<type><format>[x]!
# Examples:
feast_sygv!(A, B, ...)   # Symmetric, generalized, dense
feast_scsrgv!(A, B, ...) # Symmetric, CSR sparse, generalized
feast_hbev!(A, ...)      # Hermitian, banded, standard
```

Suffix meanings:
- No suffix: Use default parameters
- `x` suffix: Expert interface with custom parameters

### Error Handling

Use `FeastError` enum codes:
```julia
if result.info != Int(Feast_SUCCESS)
    # Handle error based on code
end
```

### Type Stability

Ensure type-stable code paths:
```julia
# Good: explicit type parameters
function solve{T<:Real}(A::Matrix{T}, ...)
    workspace = zeros(T, n, m)
    ...
end

# Avoid: type instability
function solve(A, ...)
    workspace = similar(A)  # Type inferred at runtime
    ...
end
```

### Documentation

Use docstrings for public functions:
```julia
"""
    feast_sygv!(A, B, Emin, Emax, M0, fpm)

Solve the generalized symmetric eigenvalue problem A*x = λ*B*x.

# Arguments
- `A::Matrix{T}`: Symmetric matrix
- `B::Matrix{T}`: Symmetric positive definite matrix
- `Emin::T`: Lower bound of search interval
- `Emax::T`: Upper bound of search interval
- `M0::Int`: Maximum number of eigenvalues to find
- `fpm::Vector{Int}`: FEAST parameter array

# Returns
- `FeastResult{T,T}`: Result containing eigenvalues and eigenvectors

# Example
```julia
result = feast_sygv!(A, B, 0.0, 1.0, 10, fpm)
```
"""
function feast_sygv!(A, B, Emin, Emax, M0, fpm)
    ...
end
```

---

## Dependencies

### Required
- `LinearAlgebra` - Standard linear algebra operations
- `SparseArrays` - Sparse matrix support
- `Distributed` - Distributed computing
- `FastGaussQuadrature` - Gauss quadrature for contour integration

### Optional
- `Krylov.jl` - Iterative solvers for matrix-free interface
- `MPI.jl` - MPI parallel computing (requires `FEASTKIT_ENABLE_MPI=true`)

### Development
- `Documenter.jl` - Documentation generation
- `Test` - Testing framework

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     User-Facing Interfaces                        │
│  feast() · feast_general() · feast_banded() · feast_matvec()     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Dense Solvers│   │ Sparse Solvers│   │ Banded Solvers│
│  feast_sygv!  │   │ feast_scsrgv! │   │  feast_sbgv!  │
│  feast_heev!  │   │ feast_hcsrgv! │   │  feast_hbev!  │
│  feast_gegv!  │   │ feast_gcsrgv! │   │  feast_gbgv!  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FEAST Kernel                                 │
│  Contour Integration · RCI State Machine · Spectral Projection   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Threading   │   │  Distributed  │   │      MPI      │
│  @threads     │   │  @spawnat     │   │  MPI.Reduce   │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## File Statistics

| Directory | Files | Approximate Lines |
|-----------|-------|-------------------|
| `src/core/` | 5 | ~2,100 |
| `src/kernel/` | 1 | ~1,200 |
| `src/dense/` | 1 | ~1,100 |
| `src/sparse/` | 1 | ~1,500 |
| `src/banded/` | 1 | ~650 |
| `src/interfaces/` | 2 | ~1,300 |
| `src/parallel/` | 4 | ~2,300 |
| **Total** | **16** | **~9,200** |

---

<div align="center">
  <p><strong>Ready to contribute?</strong> See <a href="contributing.md">Contributing Guidelines</a></p>
  <a href="testing.md">Testing Guide</a> · <a href="api_reference.md">API Reference</a>
</div>
