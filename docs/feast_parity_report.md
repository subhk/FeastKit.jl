# FeastKit vs. Fortran FEAST Parity Report

This document tracks what remains to reach feature parity between `FeastKit.jl`
and the reference Fortran FEAST implementation shipped under `FEAST/`.

## Coverage Summary

- **Dense:** Standard/generalized/polynomial real & complex covered; GMRES-backed iterative variants exist for real symmetric (`difeast_sygv/syev`), complex Hermitian (`zifeast_heev/hegv`), and general non-Hermitian (`zifeast_gegv/geev`). Complex-symmetric (`zfeast_sy*`) still relies on general solvers.
- **Sparse:** Real/complex standard plus Hermitian generalized (`zfeast_hcsrgv/x`) routines implemented; iterative (`difeast_scsrgv/x`) variants now available for symmetric real CSR problems, while complex-symmetric specializations remain outstanding.
- **Banded:** Real and complex Hermitian/non-Hermitian wrappers present; GMRES-backed iterative options available via conversions to dense solvers (`difeast_sbgv/sbev`, `zifeast_hbev/hbgv`, `zifeast_gbgv/gbev`).
- **RCI:** Base kernels (`feast_srci!`, `feast_hrci!`, `feast_grci!`) done; polynomial and custom-contour wrappers added; iterative variants absent.
- **Utilities:** Contour generators, rational helpers, parameter init, distribution helper present; parallel distribution helpers beyond CSR classification not yet ported.
- **Parallel:** Threaded helpers exist; MPI (`pd*`, `pz*`) and full distributed support not implemented.
- **Precision:** Double- and single-precision (Float64/ComplexF64 and Float32/ComplexF32) workflows are supported end-to-end.

## Missing Feature High-Priority Items

1. **Iterative FEAST (IFEAST) Variants**
   - Dense real/complex Hermitian, general, and banded wrappers (`difeast_sygv/syev/sbgv/sbev`, `zifeast_heev/hegv/hbev/hbgv`, `zifeast_gegv/geev/gbgv/gbev`) plus sparse real CSR (`difeast_scsrgv/x`) are supported via GMRES. Remaining work covers sparse complex/non-Hermitian families and polynomial variants across the `difeast_*` / `zifeast_*` families.

2. **MPI/Parallel Families**
   - `pdfeast_*`, `pzfeast_*`, `pdifeast_*`, `pzifeast_*`.
   - Would need MPI.jl integration, contour distribution, collective reductions.

3. **Complex-Symmetric Specialized Paths**
   - Fortran includes `zfeast_sy*` dense/sparse/banded routines.
   - Julia uses general solvers; specialized symmetric storage not supported.

4. **Advanced Polynomial/Custom Contour RCI**
   - `feast_srcipev`/`feast_grcipev` wrappers exist, but direct RCI entry points for polynomial problems (`feast_grcipev!` etc.) still rely on generalized wrappers.
   - Need direct polynomial linearization for banded/sparse cases without dense expansion.

5. **Comprehensive Example Ports**
   - `examples/feast/run_feast_examples.jl` covers main cases, but additional Fortran demos (`PFEAST-L*`) still pending translation (especially MPI ones).

## Secondary Gaps

- **Banded Storage Fidelity:** Current implementation converts banded matrices to dense; a banded solver to avoid conversion would better match Fortran.
- **Matrix-Free Interfaces:** Fortran RCI offers matrix-free operations; existing Julia matrix-free helper is basic and needs alignment with FEAST’s API (tolerances, restarts).
- **Diagnostic/Tracing Utilities:** Fortran’s extensive diagnostics (trace utilities, error flags) not fully replicated.

## Suggested Next Steps

1. Implement iterative RCI kernels (new module, reuse Krylov.jl or custom GMRES).
2. Add single-precision support via type parametrization and `Float32` testing.
3. Expand MPI layer using MPI.jl to replicate `pdfeast_*` (start with sparse symmetric generalized).
4. Enhance banded solvers to operate without dense conversion (banded LU/solve).
5. Complete example suite (translate remaining Fortran sample programs, include README guidance).
