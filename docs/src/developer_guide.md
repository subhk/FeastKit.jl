# Developer Guide

This guide provides an overview of the FeastKit.jl project layout, codebase architecture, and development workflow.

```@contents
Pages = ["developer_guide.md"]
Depth = 2
```

---

## Project Layout

```
FeastKit.jl/
├── Project.toml          # Package metadata and dependencies
├── Manifest.toml         # Generated local resolution (gitignored)
├── README.md             # Project overview
├── LICENSE               # License file
├── src/                  # Main source code
├── ext/                  # Optional Krylov/MPI package extensions
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
├── FeastKit.jl                    # Public exports and dependency-ordered includes
├── deprecations.jl                # Compatibility aliases
├── core/
│   ├── feast_types.jl             # Results, workspaces, and RCI state
│   ├── feast_parameters.jl        # Legacy FEAST controls
│   ├── feast_tools.jl             # Numerical helpers
│   ├── feast_aux.jl               # Contour and auxiliary routines
│   ├── feast_contour_shapes.jl    # Circle, ellipse, rectangle
│   ├── feast_rci_drivers.jl       # Shared serial real/Hermitian drivers
│   ├── feast_initial_subspace.jl  # Private seed copying and normalization
│   ├── feast_mixed_precision.jl   # Residual inverse correction and LU fallback
│   ├── feast_backend_policy.jl   # Availability, compatibility, fallback policy
│   └── feast_backend_utils.jl    # Serial storage dispatch and capability reports
├── kernel/                       # RCI state machines
├── dense/                        # Dense storage drivers
├── sparse/                       # Sparse storage drivers
├── banded/                       # Banded storage drivers
├── interfaces/
│   ├── feast_options.jl          # Named controls and solver option validation
│   ├── feast_preparation.jl      # Shared setup, materialization, scalar promotion
│   ├── feast_auto_subspace.jl    # Count estimate and bounded retries
│   ├── feast_validation.jl       # Interval validation and spectral bounds
│   ├── feast_interfaces.jl       # Assembled feast() and feast_general()
│   ├── feast_banded_interface.jl # Banded convenience calls
│   ├── feast_polynomial_interface.jl # Assembled polynomial convenience calls
│   ├── feast_matfree.jl          # Matrix-free public entry points
│   ├── feast_contour_interface.jl # Contour convenience calls
│   ├── feast_results.jl          # Display, summaries, and checked wrappers
│   └── feast_precision_aliases.jl # FEAST-compatible aliases
├── matrixfree/
│   ├── operators.jl             # Operator types and multiplication methods
│   ├── workspace.jl             # Workspace allocation
│   ├── solvers.jl               # Shifted iterative solver factories
│   ├── rci_drivers.jl           # Matrix-free RCI execution
│   └── polynomial.jl            # Companion operators and polynomial solves
└── parallel/
    ├── feast_backend_execution.jl # Backend execution and runtime fallback
    ├── feast_parallel.jl         # Threaded/distributed include manifest
    ├── shared.jl                 # Moment storage and contour distribution
    ├── dense.jl                  # Dense factorization and driver
    ├── sparse.jl                 # Sparse/worker factorization and driver
    ├── dense_moments.jl          # Dense contour-point calculations
    ├── sparse_moments.jl         # Sparse contour-point calculations
    ├── diagnostics.jl            # Distribution reporting and benchmarks
    ├── feast_parallel_rci.jl     # Parallel RCI state machine
    ├── feast_parallel_comparison.jl # Backend comparison utilities
    ├── feast_mpi_stubs.jl        # Optional MPI entry-point declarations
    ├── feast_mpi.jl              # MPI include manifest, loaded by extension
    ├── feast_mpi_interface.jl    # Hybrid execution wrapper
    └── mpi/                      # MPI implementation, inside FeastKitMPIExt
        ├── shared.jl             # Collective guards and contour setup
        ├── real_dense.jl         # Dense real driver, moments, residuals
        ├── real_sparse.jl        # Sparse real driver, moments, residuals
        ├── complex_projection.jl # Complex projections and residuals
        ├── hermitian.jl          # Complex Hermitian driver
        ├── general.jl            # General complex driver
        ├── interfaces.jl         # Public MPI wrappers
        └── runtime.jl            # MPI lifecycle and benchmarks
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
- `feastdefault!(fpm)` - Validate settings and fill unset defaults
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

Solvers for Julia `SparseMatrixCSC` matrices (the FEAST-compatible names retain `csr`):

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

### High-Level Interfaces (`src/interfaces/`)

Public wrappers call shared preparation in `feast_preparation.jl`. Named controls
and inner-solver options are validated once, while each entry point retains its
problem-specific symmetry and shape checks. Standard problems keep their
specialized storage drivers.

User-friendly wrappers:

- `feast(A, interval; ...)` - Standard eigenvalue problem
- `feast(A, B, interval; ...)` - Generalized eigenvalue problem
- `feast_general(A, B, center, radius; ...)` - Non-Hermitian problems
- `feast_banded(A, kl, interval; ...)` - Banded matrices
- `eigvals_feast(...)`, `eigen_feast(...)` - LinearAlgebra-style interfaces

### Matrix-Free Interface (`src/matrixfree/`)

Operator definitions, workspaces, shifted solvers, RCI drivers, and polynomial
linearization have separate files. Public overloads live in
`src/interfaces/feast_matfree.jl` and use the same named-option preparation as
assembled and banded overloads.

For large-scale problems without explicit matrices:

| Type/Function | Description |
|---------------|-------------|
| `MatrixFreeOperator{T}` | Abstract operator type |
| `LinearOperator{T}` | Concrete matrix-free operator |
| `MatrixVecFunction{T}` | Callback wrapper; capture payload data in its callable |
| `feast_matfree_srci!` | Matrix-free RCI for symmetric |
| `feast_matfree_grci!` | Matrix-free RCI for general |
| `create_iterative_solver` | Create Krylov solver |

### Parallel Computing (`src/parallel/`)

Multiple parallel backends:

| File | Description |
|------|-------------|
| `feast_parallel.jl` | Loads shared, dense, sparse, moment, and diagnostic implementations |
| `feast_parallel_rci.jl` | Parallel RCI state machine (`pfeast_srci!`) |
| `feast_mpi.jl` | Loads the `mpi/` implementation inside `FeastKitMPIExt` |
| `feast_mpi_stubs.jl` | MPI declarations; implementations load through `FeastKitMPIExt` |

Backend rules live in `core/feast_backend_policy.jl`: request normalization,
resource availability, supported problem/storage/solver combinations, and strict
versus automatic fallback. `parallel/feast_backend_execution.jl` invokes the
selected driver and handles runtime failures. Add compatibility rules to the
policy file and exercise them through public API tests.

All ordinary implementation files share the `FeastKit` module; splitting files
does not introduce new public namespaces. MPI implementation files share the
extension module and load only when the optional MPI dependency is loaded.
Numerical iteration and collective synchronization remain in the driver files.


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

# With threading (Pkg.test also installs test-only dependencies)
julia --project --threads=auto -e 'using Pkg; Pkg.test()'
```

### Building Documentation

```bash
# From the repository root:
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs --threads=2 docs/make.jl
```

Open `docs/build/index.html` locally. The documentation workflow publishes
main-branch and tagged builds; local builds do not publish.

### Running Examples

Run these files as programs. `include` loads their definitions but does not
invoke the script entry points. The docs environment supplies Krylov for the
matrix-free examples.

```sh
# From the repository root, after preparing the docs environment above:
julia --project=docs --threads=2 examples/feast/run_feast_examples.jl
julia --project=docs --threads=2 examples/matrix_free_examples.jl
julia --project=docs --threads=2 examples/custom_contour_integration.jl
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
feast_sygv!(A, B, Emin, Emax, M0, fpm)   # Symmetric, generalized, dense
feast_scsrgv!(A, B, Emin, Emax, M0, fpm) # Symmetric, CSC storage, generalized
feast_hbev!(AB, k, Emin, Emax, M0, fpm) # Hermitian, banded, standard
```

Suffix meanings:
- Ordinary drivers accept the FEAST parameter vector `fpm`.
- `x` variants accept explicit contour nodes and weights as additional arguments.

### Error Handling

Use `FeastError` enum codes:
```julia
if result.info != Int(Feast_SUCCESS)
    # Handle error based on code
end
```

### Type Stability

Ensure type-stable code paths:
```@example developer_type_stability
using LinearAlgebra
# Explicit type parameters and a workspace with a concrete element type
function solve(A::Matrix{T}, b::Vector{T}) where {T<:AbstractFloat}
    workspace = zeros(T, length(b))
    ldiv!(workspace, lu(A), b)
    return workspace
end
@assert solve([2.0 0.0; 0.0 4.0], [2.0, 8.0]) == [1.0, 2.0]
```

### Documentation

This is a docstring/implementation template, not another solver definition to
add to a user script. Replace the body and example for the function you write.

Use docstrings for public functions:
````julia
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
    # Implementation
end
````

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

## Source Inventory

The include order in `src/FeastKit.jl` is the authoritative module inventory.
Optional package integrations live in `ext/`. Use `rg --files src ext` to list
current files; static file and line counts become stale as solvers evolve.

---

**Ready to contribute?** See [Contributing Guidelines](contributing.md)

[Testing Guide](testing.md) · [API Reference](api_reference.md)
