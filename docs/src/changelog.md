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
- For non-Hermitian and complex-symmetric problems, a spurious pair built by
  the oblique Rayleigh-Ritz from an in-region eigenvector plus an unresolved
  conjugate pair is also recognized: the filter restricted to the subspace
  shows no more in-region directions than converged pairs.
- Polynomial eigenproblems no longer depend on units. The companion-based
  solvers (`feast_polynomial`, `feast_gepev!` and relatives, the sparse
  `*csrpev!` family) balance the coefficients before linearizing, and the
  moment-method kernel behind `feast_srcipev!`/`feast_grcipev!` measures
  residuals as backward errors `‖P(λ)x‖ / (Σ|λ|ᵏ‖Aₖ‖ ‖x‖)`. That kernel measures
  the coefficient sizes with extra `MULT_A` requests before its first
  factorization. `validate_companion_matrices` judges residuals relative to
  the size of their terms.
- A search region with no eigenvalues now returns `info = 0` with `M = 0`
  instead of `info = 5`. A caller-supplied starting subspace is first checked
  with independent probes.
- `eigvals_feast` and `eigen_feast` with `check=false` log a warning when the
  solve did not converge.
- The generalized interval check bounds the spectrum of the pencil `(A, B)`,
  not of `A`, so it no longer warns about intervals that contain eigenvalues.
- `feast_matvec` reaches the default tolerance (its shifted solves are
  warm-started, see below); its GMRES options are now keywords.
- The sparse complex-symmetric solver keeps a caller's `initial_subspace`
  instead of overwriting it.
- An empty Ritz set is taken as an empty search region only once the subspace
  has converged outside it: the nearest Ritz pair has converged and stopped
  moving, the filter is negligible wherever the Ritz values are, or the
  filtered block itself is negligible. For a non-Hermitian pencil the settled
  subspace must also hold a converged pair that the filter amplifies less than
  anything inside the region, or have been kept less than that as a whole,
  because an eigenvalue just outside the contour, near a node, can be amplified
  more than the region itself. This applies to every driver. It fixes
  `info = 0, M = 0` for intervals beside more eigenvalues than `M0` and for
  strongly non-normal pencils with `M0` below the eigenvalue count (none in
  1,200 such fuzz cases, against about 90 before).
- Defective eigenvalues keep their algebraic multiplicity in the general and
  complex-symmetric solvers. The Ritz vectors of a Jordan block of size k
  converge to within about eps^(1/k) of each other, and the rank compression of
  the next sweep kept only one of them: the general solvers reported fewer
  eigenvalues than the region holds as a success (every block of size 3 or 4
  tested), and the complex-symmetric ones, whose Rayleigh quotient
  `xᵀAx / xᵀx` is 0/0 at such an eigenvector, stopped converging. A nearly
  dependent block of in-region Ritz vectors is now swept through an orthonormal
  basis of its span. As with LAPACK, the k copies agree only to about
  eps^(1/k) and their eigenvectors are nearly parallel.
- The spurious-pair screen compares a pair's filter response with the filter's
  value at its own Ritz value, so rescaled weights, flat ellipses and few-node
  contours keep their eigenpairs. Near a node of a full contour, where that
  value peaks, a pair is held to the filter's floor inside the region instead.
  Pairs within `sqrt(tol)` of converging, and pairs whose residual bound places
  an eigenvalue inside the region, are never screened.
- Zolotarev quadrature (`fpm[16] = 2`) applies its constant term in every
  Hermitian driver, so the filter is the tabulated rational function. Without
  it the filter's worst response outside the interval was about twice as large
  relative to its response inside.
- The polynomial moment kernel (`feast_srcipev!`, `feast_grcipev!` and their
  `x` and iterative twins) now refines: each loop corrects every Ritz pair
  through its residual instead of reproducing the first sweep's quadrature
  error. An empty contour is recognized from the moments of every order up to
  `d - 1`; when `S0` alone cancels (roots inside together with their
  negatives), the kernel reports `info = 5` after the first sweep.
- Iterative shifted solves (`difeast_*`, `solver=:gmres`, and matrix-free
  solvers made by `create_iterative_solver`) are warm-started from the trial
  vectors with a fixed default inner tolerance of `1e-3`. They solve for a
  correction that shrinks with the eigen-residual, so they no longer stall
  short of the default outer tolerance. The matrix-free drivers take a
  `warm_start` keyword.
- The threaded and real MPI drivers run as many refinement loops as the serial
  drivers; they ran one fewer.
- Exhausting the loop budget before verifying a user seed reports `info = 5`.
  The result includes unresolved Ritz candidates and recovers saved converged
  pairs if the check found fewer directions.
- `feast_circle` and `feast_ellipse` select by the shape they describe.
  `FeastParameters()` means all defaults; it used to set the tolerance to 1.
  Threaded backends honour a registered custom contour. Closed contours are no
  longer doubled by Hermitian completion.
- `feast_customcontour` uses spectrally accurate trapezoidal weights. The
  central differences used before scaled the filter by 0.90 on an 8-node
  circle, and two nodes gave zero weights. Translating the nodes leaves the
  weights unchanged to working precision. It needs at least three nodes and
  records the count in both `fpm[2]` and `fpm[8]`. Registered contours use
  their supplied node counts.
- `feast_hbev!` returns `info = 2` when `M0` is too small, like the other
  drivers. An explicit Float32 tolerance such as `fpm[3] = 6` is honoured.
  `feastinit_driver(N)` always produces a node count that
  `feastdefault!` accepts, and `feast_scsrgv_iterative!` resolves defaults
  before reading the tolerance.

### Inputs

- Sparse matrices with `Int32` indices, mixed dense/sparse pencils such as
  `feast(sparse(A), Diagonal(b), interval)`, `Rational` and `Complex{Int}`
  matrices, a real centre in `feast_general`, integer storage in
  `feast_banded`, contour objects in `eigvals_feast` and `eigen_feast`, and
  custom contours with any node count in the `x` drivers now work.
  `feast_polynomial` with its default `M0` works on problems smaller than 10.
- `SymTridiagonal`, `Tridiagonal`, `Bidiagonal`, `Diagonal`, and `Symmetric`,
  `Hermitian`, adjoint and transpose wrappers of sparse matrices keep sparse
  storage instead of being densified. The interval check of a wrapped sparse
  matrix costs O(nnz) instead of O(N²).
- Unsupported element types such as `BigFloat` raise an `ArgumentError` that
  says what to convert to; they used to end in `info = 8`. NaN or infinite
  intervals are argument errors.
- Interval solvers accept one or two half-contour points.

### Parallel execution

- `backend=:distributed`, and `:auto` once workers exist, runs dense real
  symmetric problems on the workers instead of serially. `feast_parallel_comparison`
  skips backends that do not support the problem instead of throwing. The
  distributed RCI path keeps each worker's factorizations for the whole solve.
- Overlapping threaded solves no longer leave BLAS at one thread.
  `feast_with_backend(...; strict_backend=true)` throws when MPI is unavailable.

### Performance

- `create_iterative_solver` no longer makes a dynamic call per right-hand-side
  element, which halves the allocations of a small matrix-free GMRES solve. The
  matrix-free drivers form `B*Q` once per contour sweep instead of at every
  contour point.
- `feast_parallel` starts from the default subspace like the serial solvers.
  It used to treat that subspace as one supplied by the caller and check it
  with independent probes before accepting the answer, which took one or two
  extra refinement loops per solve: twice the serial loop count on easy
  problems. A subspace a caller passes to `pfeast_srci!` in `work` is still
  checked; set `state.user_subspace = false` to start from the default one.
- The threaded, distributed and MPI real symmetric solvers (`backend=:threads`,
  `:distributed` and `:mpi`, `pfeast_sygv!`, `pfeast_scsrgv!`, `mpi_feast` and
  `feast_hybrid`) keep the trial subspace and the reduced eigenproblem real, as
  the serial solvers do; only the shifted solves are complex. A product of a
  real dense matrix with a complex block does not use BLAS: at N = 2000 the
  three such products of a refinement loop took longer than all of its contour
  solves. The MPI solvers reduce half as much data per sweep, and the threaded
  and distributed solvers start from the serial solvers' subspace.

### Behaviour changes

- General banded drivers (`feast_gbev!` and relatives) need `2ka+1` rows and
  reject upper-only storage.
- Zolotarev quadrature is restricted to its tabulated counts (1-8, 10, 12,
  16); other counts throw instead of using a broken fallback.
- An empty region beside more eigenvalues than `M0` columns can end as
  `info = 2` or `5` with `M = 0` instead of `info = 0`. This happens hard
  against a cluster on the real axis, or beside an eigenvalue just outside a
  complex contour near a node. The solver cannot tell such a region from one
  whose eigenvalues it has not reached; increase `M0`.
- `feast(A, sparse(B), interval; backend=:threads)` solves instead of throwing.
- `mixed_precision=true` densifies banded structured matrices and still
  rejects sparse storage.
- A failing matrix-free solver callback logs a warning instead of silently
  returning `info = 8`.
- Custom callbacks passed to the matrix-free drivers still receive `B*X`.
  Solvers made by `create_iterative_solver` receive warm-started right-hand
  sides; `warm_start=false` restores `B*X`.

### Other fixes

- MPI is detected automatically again when `FEASTKIT_ENABLE_MPI=true`, MPI is
  loaded and initialized; the check used to run before the extension loaded,
  so it never succeeded. Calling an MPI entry point without `using MPI` now
  says what to load.
- Added `FeastLinearOperator`, an unambiguous alias for `LinearOperator` when
  LinearOperators.jl is also loaded.
- `ParallelFeastState` accepts the unset contour count of `feastinit().fpm`.
- `pfeast_rci_benchmark` reports the number of integration points its runs
  use; it printed 0.
- Fixed the README examples, which are now executed by the test suite together
  with the example scripts; documented the differences from Fortran FEAST.
- Removed the unused SharedArrays dependency.
- `feast_memory_estimate` no longer overflows for very large problems.
- `feast_contour` and `feast_gcontour` validate `fpm[16]` and `fpm[18]`;
  `feast_gcontour` rejects Zolotarev, which is defined for intervals only.
- `feast_inside_contourx` tests the polygon through the nodes; it accepted
  points in the notches of a concave contour (anywhere in the nodes' convex
  hull). It rejects a contour with no nodes.

### Earlier unreleased changes

- Reuse GMRES workspaces across contour nodes and sweeps; reuse matrix-free
  BiCGSTAB workspaces across right-hand sides and shifts.
- Default inner tolerances in general, complex-symmetric, banded,
  matrix-free, and complex MPI solves (now a fixed `1e-3` on warm-started
  solves, see above); explicit tolerances remain fixed.
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
