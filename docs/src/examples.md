# Examples and Tutorials

Each example below runs during the documentation build and checks its numerical
result. The models are small so the examples can be reproduced locally. For
larger problems, choose storage, subspace size, and shifted solvers using
[Problem Setup](problem_setup.md) and [Performance](performance.md).

## Basic Examples

### Example 1: Your First FeastKit Calculation

A symmetric tridiagonal matrix has eigenvalues `4 - 2cos(kπ/6)`:

```@example ex1
using FeastKit, LinearAlgebra
A = Matrix(SymTridiagonal(fill(4.0, 5), fill(-1.0, 4)))
result = feast(A, (2.0, 6.0); subspace_size=5)
@assert result.converged result.message
@assert result.values ≈ [4 - 2cos(k * π / 6) for k in 1:5]
result
```

Here the subspace spans the full five-dimensional problem. A smaller saturated
subspace would return a nonzero status even if its computed residuals were small.

### Example 2: Sparse Matrix Eigenvalues

Find the ten smallest eigenvalues without converting the matrix to dense form:

```@example ex2
using FeastKit, SparseArrays, LinearAlgebra
n = 1000
A = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
expected = [2 - 2cos(k * π / (n+1)) for k in 1:10]
result = feast(A, (0.9expected[1], 1.1expected[end]); subspace_size=12, tol=1e-10)
@assert result.converged result.message
@assert result.M == 10
@assert isapprox(result.values, expected; atol=1e-10)
result.values
```

### Example 3: Generalized Eigenvalue Problem

For a symmetric definite pencil, eigenvectors have unit Euclidean norm. Their
mass-weighted Gram matrix is diagonal for distinct eigenvalues, but need not be
identity. Normalize explicitly when downstream work needs `Q'BQ = I`.

```@example ex3
using FeastKit, LinearAlgebra
n = 40
A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
B = Matrix(Diagonal(collect(range(1.0, 2.0; length=n))))
interval = (0.0, 0.15)
result = feast(A, B, interval; subspace_size=10, tol=1e-10)
expected = filter(x -> interval[1] <= x <= interval[2], eigvals(Symmetric(A), Symmetric(B)))
@assert result.converged result.message
@assert isapprox(result.values, expected; atol=1e-9)
Q = result.vectors
G = Q' * B * Q
@assert norm(G - Diagonal(diag(G))) < 1e-8
Q_B = Q * Diagonal(inv.(sqrt.(diag(G))))
@assert isapprox(Q_B' * B * Q_B, I; atol=1e-8)
result.values
```

## Scientific Computing Applications

### Example 4: 1D Wave Equation (Vibrating String)

For fixed endpoints, `-T*u'' = ω²*ρ*u`. The stiffness stencil is positive on
the diagonal and negative off the diagonal. `B` must be a matrix, rather than
an unsized `UniformScaling` object such as `ρ*I`.

```@example string_modes
using FeastKit, LinearAlgebra, SparseArrays
L, tension, density = 1.0, 100.0, 2.0
n = 80
h = L / (n+1)
K = (tension / h^2) * sparse(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
M = spdiagm(0 => fill(density, n))
upper = 1.01 * (5π * sqrt(tension/density) / L)^2
result = feast(K, M, (0.1, upper); subspace_size=8, tol=1e-9)
expected = [(tension/density) * (2 - 2cos(k*π/(n+1))) / h^2 for k in 1:5]
@assert result.converged result.message
@assert isapprox(result.values, expected; rtol=1e-8)
frequencies_hz = sqrt.(result.values) / (2π)
frequencies_hz
```

The continuum frequencies are `k*sqrt(tension/density)/(2L)`. The finite-difference
values approach them as the grid is refined; they are not identical on a finite grid.

### Example 5: Square Membrane Vibrations

This models a **square** membrane with zero Dirichlet boundaries. Its discrete
spectrum contains repeated modes; subspace capacity must include multiplicities.

```@example membrane_modes
using FeastKit, LinearAlgebra, SparseArrays
nside = 8
h = 1 / (nside+1)
D = sparse(SymTridiagonal(fill(2.0, nside), fill(-1.0, nside-1)))
Id = spdiagm(0 => ones(nside))
K = (kron(Id, D) + kron(D, Id)) / h^2
interval = (10.0, 85.0)
result = feast(K, interval; subspace_size=6, tol=1e-9)
all_values = sort([(4/h^2) * (sin(p*π/(2(nside+1)))^2 + sin(q*π/(2(nside+1)))^2)
                   for p in 1:nside for q in 1:nside])
expected = filter(x -> interval[1] <= x <= interval[2], all_values)
@assert result.converged result.message
@assert isapprox(result.values, expected; atol=1e-8)
result.values
```

### Example 6: Quantum Harmonic Oscillator

Discretize `H = -½*d²/dx² + ½*x²` at interior points, eliminating the fixed
boundary values. This preserves symmetry. The kinetic diagonal is `1/h²`.

```@example harmonic_modes
using FeastKit, LinearAlgebra, SparseArrays
L, n = 6.0, 160
h = 2L / (n+1)
x = [-L + i*h for i in 1:n]
H = sparse(SymTridiagonal(1/h^2 .+ 0.5 .* x.^2, fill(-0.5/h^2, n-1)))
result = feast(H, (0.4, 3.6); subspace_size=8, tol=1e-10)
@assert result.converged result.message
@assert isapprox(result.values, [0.5, 1.5, 2.5, 3.5]; atol=0.02)
# FEAST returns Euclidean-normalized vectors. Rescale for grid integration.
psi = result.vectors[:, 1] / sqrt(h)
@assert sum(abs2, psi) * h ≈ 1
result.values
```

## Matrix-Free Examples

### Example 7: A Structured Shifted Solver

The callback below applies diagonal operators without constructing matrices and
solves the supplied complex shifted systems exactly.

```@example structured_matfree
using FeastKit, LinearAlgebra
entries = collect(1.0:12.0)
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x),
                              (12, 12); issymmetric=true)
shifted_solve!(Y, z, X) = (Y .= X ./ (z .- entries))
result = feast(A_op, (0.5, 2.5); subspace_size=4, solver=shifted_solve!)
@assert result.converged result.message
@assert result.values ≈ [1.0, 2.0]
result.values
```

See [Matrix-Free Interface](matrix_free_interface.md) for generalized callbacks,
payload storage, preconditioning, and workspace reuse.

### Example 8: Iterative Solver Comparison

Load Krylov to enable the built-in solvers. Compare both convergence and
residuals when benchmarking; a faster unsuccessful solve is not a speedup.

```@example iterative_comparison
using FeastKit, Krylov, LinearAlgebra
entries = collect(1.0:8.0)
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x),
                              (8, 8); issymmetric=true)
results = map((:gmres, :bicgstab)) do solver
    result = feast(A_op, (0.5, 2.5); subspace_size=4, tol=1e-8, solver=solver,
                   solver_opts=(rtol=1e-12, maxiter=500, restart=16))
    @assert result.converged result.message
    @assert isapprox(result.values, [1.0, 2.0]; atol=1e-8)
    (solver=solver, residual=result.epsout)
end
results
```

## Advanced Features

### Example 9: Contour Integration Rules

Here all 21 eigenvalues in the tight cluster must fit in the trial subspace.
The contour node count is independent of that eigenvalue count.

```@example integration_rules
using FeastKit, LinearAlgebra
entries = vcat(collect(0.98:0.002:1.02), collect(2.0:0.1:5.0))
A = Matrix(Diagonal(entries))
results = map((0, 1, 2)) do rule
    fpm = feastinit().fpm
    fpm[16] = rule
    result = feast(A, (0.97, 1.03); subspace_size=26, quadrature_points=16,
                   tol=1e-10, fpm=fpm)
    @assert result.converged result.message
    @assert isapprox(result.values, entries[1:21]; atol=1e-9)
    (rule=rule, count=result.M, residual=result.epsout)
end
results
```

### Example 10: Complex Eigenvalues in a Rectangle

A non-Hermitian matrix can be searched directly with a full contour:

```@example rectangle_eigenvalues
using FeastKit, LinearAlgebra
A = Matrix(Diagonal(ComplexF64[0.5+0.1im, 1.0+0.2im, 3.0, -2.0]))
A[1, 2] = 0.3
A[2, 3] = 0.2im
contour = feast_rectangle(0.0, 1.5, -0.5, 0.5)
result = feast(A, contour; subspace_size=3, tol=1e-10)
@assert result.converged result.message
@assert isapprox(sort(result.values; by=real), [0.5+0.1im, 1.0+0.2im]; atol=1e-9)
result.values
```

### Example 11: Polynomial Eigenvalues

For `P(λ) = λ²I - diag(1,4,9)`, select the positive roots 1 and 2. Validate
against the original polynomial as well as checking the solver status.

```@example polynomial_roots
using FeastKit, LinearAlgebra
K = -Matrix(Diagonal(ComplexF64[1, 4, 9]))
C = zeros(ComplexF64, 3, 3)
M = Matrix{ComplexF64}(I, 3, 3)
fpm = feastinit().fpm
fpm[8] = 32
fpm[16] = 1
result = feast_polynomial([K, C, M], 1.5+0im, 0.75; M0=3, fpm=fpm)
@assert result.converged result.message
@assert isapprox(sort(real.(result.values)), [1.0, 2.0]; atol=1e-9)
@assert all(j -> norm((K + result.values[j]^2 * M) * result.vectors[:, j]) < 1e-8,
            eachindex(result.values))
result.values
```

## Performance Examples

### Example 12: Backend Selection

`backend=:auto` can use the available supported backend or fall back to serial.
Use an explicit backend when execution on it is a requirement.

```@example backend_selection
using FeastKit, LinearAlgebra, SparseArrays
A = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
result = feast(A, (0.5, 2.5); subspace_size=3, backend=:auto)
@assert result.converged result.message
@assert result.values ≈ [1.0, 2.0]
result.values
```

Threaded and distributed backends support real symmetric problems. MPI also
supports complex Hermitian and general problems. See
[Parallel Computing](parallel_computing.md) for worker setup and complete MPI
scripts, and [Performance](performance.md) for benchmarking and storage costs.

## Checked Convenience Wrappers

When you only need an eigenvalue vector or `LinearAlgebra.Eigen`, enable the
optional status check. The default remains `check=false` for compatibility.

```@example checked_convenience
using FeastKit, LinearAlgebra
A = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0]))
values = eigvals_feast(A, (0.5, 2.5); subspace_size=3, check=true)
decomposition = eigen_feast(A, (0.5, 2.5); subspace_size=3, check=true)
@assert values ≈ [1.0, 2.0]
@assert decomposition.values ≈ values
values
```

`check=true` rejects every nonzero status, including subspace saturation. Use
`feast` directly to inspect partial results and `result.message` on failure.
