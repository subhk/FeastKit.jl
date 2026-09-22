# Changelog

## Unreleased

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
