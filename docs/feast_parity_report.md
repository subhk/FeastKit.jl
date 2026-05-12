# FeastKit vs. Fortran FEAST Parity Report

This document tracks what remains to reach feature parity between `FeastKit.jl`
and the reference Fortran FEAST implementation shipped under `FEAST/`.

## Coverage Summary

- **Dense:** Standard/generalized/polynomial real and complex FEAST families are covered, with precision-prefixed aliases (`sfeast_*`, `dfeast_*`, `cfeast_*`, `zfeast_*`) forwarding to the type-generic Julia implementations. GMRES-backed iterative variants exist for real symmetric (`difeast_sygv/syev`), complex Hermitian (`zifeast_heev/hegv`), general non-Hermitian (`zifeast_gegv/geev`), and dense complex-symmetric wrappers, which use a dedicated transpose-bilinear projection.
- **Sparse:** Real, Hermitian, complex-symmetric, and general CSC/CSR-style families are covered, including custom-contour `x` aliases and precision-prefixed FEAST names. GMRES-backed variants cover real CSR (`difeast_scsrgv/x`) and complex Hermitian, complex-symmetric, and general/non-Hermitian (`zifeast_hcsrev/hcsrgv/scsrev/scsrgv/gcsrgv/gcsrev`).
- **Banded:** Real, complex Hermitian, complex-symmetric, and non-Hermitian FEAST families are present, including precision-prefixed aliases and custom-contour variants. Direct real symmetric, complex Hermitian, complex-symmetric, and fully general non-Hermitian banded solves use LAPACK banded storage; GMRES-backed iterative banded solves use banded matvecs without dense matrix conversion.
- **RCI:** Base kernels (`feast_srci!`, `feast_hrci!`, `feast_grci!`) plus the polynomial kernels (`feast_srcipev!`, `feast_grcipev!`) are implemented; iterative variants are still absent.
- **Utilities:** Contour generators, rational helpers, parameter initialization, custom contour registry, and distribution helpers are present.
- **Parallel:** High-level threaded and distributed backends support sparse real symmetric standard/generalized problems; MPI supports real symmetric standard/generalized problems through `mpi_feast` and `backend=:mpi` with an explicit communicator. Real symmetric PFEAST-compatible aliases (`psfeast_*`, `pdfeast_*`) cover dense/sparse standard and generalized paths plus the parallel RCI entry point.
- **Precision:** Double- and single-precision (Float64/ComplexF64 and Float32/ComplexF32) workflows are supported end-to-end.

## Missing Feature High-Priority Items

1. **Iterative FEAST (IFEAST) Variants**
   - Dense real/complex Hermitian, general, and banded wrappers (`difeast_sygv/syev/sbgv/sbev`, `zifeast_heev/hegv/hbev/hbgv`, `zifeast_gegv/geev/gbgv/gbev`) plus sparse real CSR and sparse complex Hermitian/complex-symmetric/general families are supported via GMRES. Remaining work covers polynomial variants across the `difeast_*` / `zifeast_*` families.

2. **MPI/Parallel Families**
   - High-level `backend=:mpi` and `mpi_feast` cover the real symmetric FEAST path.
   - Real symmetric precision-prefixed PFEAST aliases (`psfeast_syev/sygv/scsrev/scsrgv/srci` and `pdfeast_syev/sygv/scsrev/scsrgv/srci`) forward to the threaded/distributed kernels, or to MPI when `comm=` is provided.
   - Remaining parity work is the complex/general and iterative PFEAST surface (`pcfeast_*`, `pzfeast_*`, `psifeast_*`, `pdifeast_*`, `pcifeast_*`, `pzifeast_*`) and the corresponding complex/general MPI kernels.

3. **Comprehensive Example Ports**
   - `examples/feast/run_feast_examples.jl` covers main cases, but additional Fortran demos (`PFEAST-L*`) still pending translation (especially MPI ones).

## Secondary Gaps

- **Banded Storage Fidelity:** Direct real symmetric, complex Hermitian, complex-symmetric, and fully general non-Hermitian banded paths use banded storage. GMRES-backed iterative banded paths also preserve banded storage by applying shifted systems through banded matvecs.
- **Matrix-Free Interfaces:** Fortran RCI offers matrix-free operations; existing Julia matrix-free helper is basic and needs alignment with FEAST’s API (tolerances, restarts).
- **Diagnostic/Tracing Utilities:** Fortran’s extensive diagnostics (trace utilities, error flags) not fully replicated.

## Suggested Next Steps

1. Implement iterative RCI kernels (new module, reuse Krylov.jl or custom GMRES).
2. Expand MPI layer to complex/general and iterative precision-prefixed PFEAST variants.
3. Complete example suite (translate remaining FEAST/PFEAST sample programs, include README guidance).
