# Changelog

## Unreleased

### Correctness

- Residuals are now measured against the spectral scale of the pencil,
  `‖Ax - λBx‖ / (‖Bx‖ max(|λ|, σ))`, instead of a fixed floor of 1. The old
  floor made the convergence test absolute for small-magnitude problems, so a
  change of units could turn a converged solve into one that accepted
  inaccurate or spurious eigenpairs with `info = 0`. `σ` is measured on random
  probes before the first contour sweep, which adds one `MULT_A`/`MULT_B`
  request at the start of every RCI loop; callers that answer those requests
  generically need no change.
- Spurious Ritz pairs — mixtures of out-of-region eigenvectors whose Rayleigh
  quotient falls inside the region — are recognized by their filter response
  and no longer counted. They used to block convergence for the whole loop
  budget (`info = 5`) or be returned as eigenvalues, including by
  `eigvals_feast` and `eigen_feast`.
- A search region with no eigenvalues now returns `info = 0` with `M = 0`
  instead of `info = 5`. A caller-supplied starting subspace is first checked
  with independent probes.
- `eigvals_feast` and `eigen_feast` with `check=false` log a warning when the
  solve did not converge.
- The generalized interval check bounds the spectrum of the pencil `(A, B)`,
  not of `A`, so it no longer warns about intervals that contain eigenvalues.
- `feast_matvec` adapts its inner GMRES tolerance to the outer residual, so it
  can reach the default tolerance; its GMRES options are now keywords.
- The sparse complex-symmetric solver keeps a caller's `initial_subspace`
  instead of overwriting it.

### Other fixes

- MPI is detected automatically again when `FEASTKIT_ENABLE_MPI=true`, MPI is
  loaded and initialized; the check used to run before the extension loaded,
  so it never succeeded. Calling an MPI entry point without `using MPI` now
  says what to load.
- Added `FeastLinearOperator`, an unambiguous alias for `LinearOperator` when
  LinearOperators.jl is also loaded.
- `ParallelFeastState` accepts the unset contour count of `feastinit().fpm`.
- Fixed the README examples, which are now executed by the test suite together
  with the example scripts; documented the differences from Fortran FEAST.
- Removed the unused SharedArrays dependency.

### Earlier unreleased changes

- Reuse GMRES workspaces across contour nodes and sweeps; reuse matrix-free
  BiCGSTAB workspaces across right-hand sides and shifts.
- Adapt default inner tolerances in general, complex-symmetric, banded,
  matrix-free, and complex MPI solves; explicit tolerances remain fixed.
- Add serial/matrix-free `initial_subspace` warm starts, opt-in
  `subspace_size=:auto` with `max_subspace_size`, and full-contour count estimation.
- Add opt-in dense serial mixed precision with Float64 residual checks and
  fallback. `fpm[42]` now defaults to 0; unsupported drivers reject 1.

- Added named high-level solver controls, direct full-contour overloads, and
  readable result aliases and status messages.
- Added optional `check=true` to `eigvals_feast` and `eigen_feast`.
- Corrected threaded projection, generalized residual scaling, rectangle
  membership near corners, and worker-local distributed factorizations.
- Updated the web documentation and executable examples to match these APIs.

## Release history

See [GitHub releases](https://github.com/subhk/FeastKit.jl/releases) and
[version tags](https://github.com/subhk/FeastKit.jl/tags) for published versions.
The package version is defined in `Project.toml`; this section does not imply
that the unreleased changes above are already in a registry release.
