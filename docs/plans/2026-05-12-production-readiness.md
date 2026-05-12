# FeastKit Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current FEAST parity work into a releaseable, reviewed production branch and then close the remaining IFEAST/PFEAST feature gaps in controlled slices.

**Architecture:** Stabilize and commit the current implemented surface first, then add missing capabilities in dependency order: iterative RCI kernels, polynomial IFEAST wrappers, and finally complex/general parallel and MPI families. Each feature slice must add failing tests before production code and update the parity/API documentation in the same change.

**Tech Stack:** Julia 1.10, LinearAlgebra/LAPACK, SparseArrays, Krylov.jl, Distributed, MPI.jl, Documenter.jl, existing FeastKit RCI/dense/sparse/banded kernels.

### Task 1: Stabilize Current Worktree

**Files:**
- Modify: `docs/feast_parity_report.md`
- Modify: `docs/src/api_reference.md`
- Review: `.github/workflows/ci.yml`
- Review: `README.md`
- Review: `src/**/*.jl`
- Review: `test/**/*.jl`

**Step 1: Inspect the dirty tree**

Run:
```bash
git status --short
git diff --stat
git diff --check
```

Expected: no whitespace errors, and all modified files map to the current production-readiness work.

**Step 2: Run the full serial test entry point**

Run:
```bash
JULIA_DEPOT_PATH=/private/tmp/julia_depot_backend:/Users/subha/.julia /Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/julia --project test/runtests.jl
```

Expected: all serial, matrix-free, allocation, backend API, and production gate testsets pass.

**Step 3: Run distributed and MPI backend gates**

Run:
```bash
JULIA_DEPOT_PATH=/private/tmp/julia_depot_backend:/Users/subha/.julia FEASTKIT_TEST_DISTRIBUTED=true /Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/julia -p 2 --project test/test_parallel_backends.jl
```

Expected: distributed backend tests pass.

Run:
```bash
JULIA_DEPOT_PATH=/private/tmp/julia_depot_backend:/Users/subha/.julia FEASTKIT_TEST_MPI=true FEASTKIT_ENABLE_MPI=true DYLD_FALLBACK_LIBRARY_PATH=/Applications/Julia-1.10.0.app/Contents/Resources/julia/lib/julia:/Users/subha/.julia/artifacts/b820a0a437e8501d06a17439abd84feaa5b6cca3/lib:/Users/subha/.julia/artifacts/0a4714a5cb9f46e0867e7c6c3a26521447f8fae0/lib:/Users/subha/.julia/artifacts/a5796fc0c827dab12e5ca4ac2ff5b8fd48e26c1b/lib:/Users/subha/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/lib:/Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/../lib/julia:/Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/../lib:/Users/subha/lib:/usr/local/lib:/lib:/usr/lib /Users/subha/.julia/artifacts/f822b53e59145a4dfdceef194142de78ce8e510b/bin/mpiexec -n 2 /Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/julia --project test/test_parallel_backends.jl
```

Expected: MPI backend tests pass on each rank.

**Step 4: Commit the stable checkpoint**

Run:
```bash
git add .github README.md docs src test
git commit -m "feat: stabilize FEAST production surface"
```

Expected: clean commit containing the current tested surface.

### Task 2: Iterative RCI Kernels

**Files:**
- Modify: `src/kernel/feast_kernel.jl`
- Modify: `src/FeastKit.jl`
- Test: `test/runtests.jl`
- Docs: `docs/feast_parity_report.md`
- Docs: `docs/src/api_reference.md`

**Step 1: Write failing API existence tests**

Add tests asserting that `ifeast_srci!`, `ifeast_hrci!`, and `ifeast_grci!` are exported or at least available under `FeastKit`.

Run:
```bash
JULIA_DEPOT_PATH=/private/tmp/julia_depot_backend:/Users/subha/.julia /Applications/Julia-1.10.0.app/Contents/Resources/julia/bin/julia --project test/runtests.jl
```

Expected: fails because the iterative RCI entry points are not defined.

**Step 2: Add minimal iterative RCI wrappers**

Implement wrappers that reuse existing RCI state machines but make solver dispatch explicit for iterative shifted solves. Keep direct and iterative RCI behavior separated so external users can drive `ijob` without hidden allocations.

**Step 3: Add behavior tests**

Add small diagonal real symmetric, Hermitian, and general problems that exercise the new iterative RCI wrappers end-to-end through the existing work/workc/Aq/Sq buffers.

Expected: results match existing non-iterative RCI kernels for eigenvalues inside the contour.

**Step 4: Update docs and run full verification**

Update parity notes from "iterative variants absent" to list implemented iterative RCI kernels and remaining limitations. Run serial tests, distributed backend tests, and MPI tests.

**Step 5: Commit**

Run:
```bash
git add src test docs
git commit -m "feat: add iterative RCI kernels"
```

### Task 3: Polynomial IFEAST Variants

**Files:**
- Modify: `src/kernel/feast_kernel.jl`
- Modify: `src/dense/feast_dense.jl`
- Modify: `src/interfaces/feast_precision_aliases.jl`
- Test: `test/runtests.jl`
- Docs: `docs/src/polynomial_problems.md`
- Docs: `docs/feast_parity_report.md`

**Step 1: Write failing precision alias tests**

Add tests for `difeast_srcipev!`, `zifeast_grcipev!`, and their custom-contour `x` variants where the matching generic polynomial kernels already exist.

Expected: fails because the IFEAST polynomial aliases are undefined.

**Step 2: Implement alias layer**

Forward the IFEAST polynomial aliases to the existing polynomial RCI kernels with explicit iterative solver options. Avoid adding new algorithms unless a test proves the existing polynomial kernel cannot represent the case.

**Step 3: Add numerical smoke tests**

Use a small diagonal quadratic polynomial with known roots. Verify returned eigenvalues and residuals.

**Step 4: Update docs and commit**

Run full tests, update parity report, then commit:
```bash
git add src test docs
git commit -m "feat: add polynomial IFEAST aliases"
```

### Task 4: Complex and General MPI/PFEAST Families

**Files:**
- Modify: `src/parallel/feast_mpi.jl`
- Modify: `src/parallel/feast_mpi_interface.jl`
- Modify: `src/parallel/feast_parallel.jl`
- Modify: `src/interfaces/feast_precision_aliases.jl`
- Test: `test/test_parallel_backends.jl`
- Docs: `docs/src/parallel_computing.md`
- Docs: `docs/feast_parity_report.md`

**Step 1: Write failing MPI tests**

Add 2-rank tests for Hermitian sparse generalized and general sparse generalized MPI calls with explicit `comm`.

Expected: fails with the current "real symmetric only" backend guard.

**Step 2: Generalize MPI reductions**

Extend MPI data movement and reductions to complex Hermitian and complex general reduced matrices. Keep real symmetric paths unchanged.

**Step 3: Add PFEAST aliases**

Add `pcfeast_*`, `pzfeast_*`, `pcifeast_*`, and `pzifeast_*` aliases only for paths with passing MPI or distributed tests.

**Step 4: Update backend selection**

Update `_parallel_backend_supported` and `feast_parallel_capabilities` to advertise only verified problem families.

**Step 5: Run MPI/distributed verification and commit**

Run serial tests, distributed gate, MPI gate, and docs. Commit:
```bash
git add src test docs README.md
git commit -m "feat: extend PFEAST MPI coverage"
```

### Task 5: Release Review

**Files:**
- Review: `.github/workflows/ci.yml`
- Review: `docs/feast_parity_report.md`
- Review: `docs/src/testing.md`
- Review: `README.md`

**Step 1: Run all local gates**

Run serial, distributed, MPI, and docs builds.

**Step 2: Check final parity wording**

Ensure all remaining unsupported FEAST families are explicitly documented and fail with clear errors.

**Step 3: Inspect git history**

Run:
```bash
git log --oneline -5
git status --short
```

Expected: clean worktree and reviewable commits.

**Step 4: Tag readiness**

Do not tag automatically. Report the verification evidence and remaining explicitly scoped limitations for human release approval.
