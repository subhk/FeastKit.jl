# Hermitian and API Review Fixes

**Goal:** Fix the three reproduced review findings without changing public solver signatures.

**Architecture:** Expand Hermitian half contours into explicit conjugate pairs through a shared helper, preserving the public half-contour convention. Reject unsupported manual operation of the high-level parallel wrapper; normalize polynomial coefficients at the high-level boundary.

**Tech stack:** Julia, LinearAlgebra, Krylov, MPI, Test.

## Checklist

- [x] Add regression tests for partial-subspace Hermitian solves (dense, sparse, banded, MPI; direct and iterative), manual parallel rejection, and abstract polynomial storage.
- [x] Run new tests against the unchanged implementation and confirm the expected failures.
- [x] Add the shared contour-completion helper in `src/core/feast_tools.jl`; use it in `feast_hrci!`, the Hermitian banded driver, and the Hermitian MPI driver. Accumulate full-contour weights once and budget all RCI jobs.
- [x] Reject `auto_rci=false` before allocating workspaces in `src/parallel/feast_parallel_rci.jl`, directing callers to `pfeast_srci!`.
- [x] Convert coefficient matrices to `Matrix{Complex{T}}` in `src/interfaces/feast_interfaces.jl` before `feast_pep!` dispatch.
- [x] Reuse one GMRES workspace per RHS block in `solve_dense_shifted!` to preserve the existing banded allocation budget after adding the lower contour half.
- [x] Run focused tests, the complete local package suite, and two-rank MPI tests; inspect the final diff and document test limitations.

## Validation

Use Julia 1.11.1 from the installed Juliaup version directory. Focused serial command: `julia --project=. -e 'push!(LOAD_PATH,abspath("docs")); using Krylov; include("test/test_hermitian_review.jl")'`. Full suite: enable `FEAST_RUN_LONG_TESTS`, `FEAST_RUN_PARALLEL_TESTS`, and `FEASTKIT_TEST_PARALLEL`, with two Julia threads, then run `Pkg.test()`. MPI tests live in `test/test_mpi_review.jl`, called by `test/test_parallel_backends.jl` under two ranks with Krylov and MPI loaded.

The Hermitian regression must fail before the change and converge with a partial subspace afterward. Check counts, eigenvalues, unit vector norms, independently recomputed generalized residuals, both factor-cache settings, and registered half contours. Keep repository changes local; this request does not require publishing or committing.

## Results

- Before implementation: 39 failed assertions and three coefficient-storage dispatch errors in the new local tests; 12 failed assertions on each MPI rank.
- Final focused regressions: 75 passed.
- Final full local suite: 1,355 passed, two existing marked-broken checks; exit 0. The RCI test count depends on the number of jobs, which falls with faster Hermitian convergence.
- Final MPI backend suite: 198 passed per rank on two ranks, two Julia threads; exit 0.
- Banded iterative allocation fixture: 4,223,488 bytes after workspace reuse, below the unchanged 20 MB limit (23,667,168 bytes before reuse).
- Documentation build: exit 0, with three pre-existing missing-docstring warnings. No deployment performed.
- `git diff --check`: clean. Windows and separate Distributed-worker execution were not rerun locally.
