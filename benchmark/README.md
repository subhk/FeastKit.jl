# Solver performance checks

- `alloc_check.jl`: quick allocation checks for serial and threaded solvers.
- `solver_performance.jl`: accuracy-checked timing and allocation comparisons.

Run from the repository root after instantiating the docs environment:

```sh
julia --project=docs --threads=2 benchmark/solver_performance.jl
```

The script fixes BLAS to one thread, warms up each case, and reports the median
of three solves with a GC between samples. It checks convergence, the known
complete eigenvalue count, eigenvalues, and explicitly recomputed residuals.
Allocated bytes are cumulative allocations, not peak resident memory.

The first five cases compare the default iterative paths and a direct-solver
control. GMRES uses `restart=N, maxiter=2N` so both versions can converge at the
same requested accuracy. These small structured fixtures are reproducible
regression benchmarks, not predictions for large application matrices.

The remaining cases compare dense direct and mixed precision, a cold and
seeded related solve, and automatic sizing including its estimator cost.
The cost of the previous solve that supplied the seed is excluded from the
warm row. A warm start can require an additional completeness sweep.

For comparison against a checkout without the new public controls, set
`ENV["FEAST_BENCH_CONTROLS"]="false"` before including the script. Use the same
Julia version, dependencies, hardware, and thread settings for both runs.

## Local results (2026-09-22)

macOS ARM64, Julia 1.13.0, Krylov 0.10.10, two Julia threads and one BLAS
thread. Baseline is a source snapshot immediately before this performance
change, including the preceding structural refactor. Both runs used the same
installed dependencies and ran without concurrent regression jobs.

| Case | Before (s) | After (s) | Before allocations (MB) | After allocations (MB) | Allocation reduction |
|:--|--:|--:|--:|--:|--:|
| dense_symmetric_gmres | 3.5778 | 3.6064 | 27.15 | 1.32 | 95.1% |
| sparse_symmetric_gmres | 5.7394 | 5.4935 | 309.02 | 6.57 | 97.9% |
| dense_hermitian_gmres | 2.2306 | 2.0999 | 64.77 | 1.85 | 97.1% |
| sparse_general_gmres | 1.0506 | 0.8277 | 22.47 | 2.16 | 90.4% |
| dense_symmetric_direct | 0.0213 | 0.0214 | 12.53 | 12.53 | 0.0% |

The general sparse GMRES case was about 21% faster; the symmetric/Hermitian
runtime changes were small compared with the allocation reductions. The
explicit residuals were below `3e-11` in all four GMRES cases, with identical
eigenvalue counts. The general case used four refinements instead of three:
looser early inner solves can trade more outer work for less inner work.
The direct control had unchanged allocations and essentially unchanged time.

The following cases use an `N=600` tridiagonal dense matrix with 22 enclosed
eigenvalues, `tol=1e-10`, and 30 columns for fixed-width solves:

| Control | Time (s) | Allocations (MB) | Refinements | Explicit residual |
|:--|--:|--:|--:|--:|
| controls_dense_direct | 0.1481 | 48.87 | 2 | 4.460e-11 |
| controls_dense_mixed | 0.2044 | 76.01 | 2 | 4.456e-11 |
| controls_related_cold | 0.1459 | 48.87 | 2 | 4.508e-11 |
| controls_related_warm | 0.4310 | 53.45 | 19 | 1.797e-15 |
| controls_auto | 0.4080 | 98.56 | 9 | 2.177e-15 |

Mixed precision was slower here: correction checks and additional buffers
outweighed cheaper Float32 LU. Allocation totals also include the temporary
Float64 shifted matrices; lower retained LU storage is not the same as lower
cumulative allocation or lower peak memory.

The warm solve included completeness probing and needed more refinements on
this fixture. Automatic sizing included a count-estimation sweep and chose a
wider subspace. Neither is guaranteed to reduce time relative to a well-chosen
fixed width. Keep these controls opt-in and compare on the application's
actual sequence of pencils. These measurements do not establish scaling or
speedups for larger problems, different spectra, or other hardware.
