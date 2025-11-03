# FeastKit vs. Fortran FEAST Parity Report

This document tracks what remains to reach feature parity between `FeastKit.jl`
and the reference Fortran FEAST implementation shipped under `FEAST/`.

## Coverage Summary

- **Dense:** Standard/generalized/polynomial real & complex are covered; complex-symmetric (`zfeast_sy*` family) still relies on general solvers.
- **Sparse:** Real/complex standard and generalized routines implemented; missing iterative (`*ifeast_*`) and complex-symmetric specializations.
- **Banded:** Real and complex Hermitian/non-Hermitian wrappers present, implemented via dense conversions.
- **RCI:** Base kernels (`feast_srci!`, `feast_hrci!`, `feast_grci!`) done; polynomial and custom-contour wrappers added; iterative variants absent.
- **Utilities:** Contour generators, rational helpers, parameter init, distribution helper present; parallel distribution helpers beyond CSR classification not yet ported.
- **Parallel:** Threaded helpers exist; MPI (`pd*`, `pz*`) and full distributed support not implemented.
- **Precision:** Only double-precision supported; single-precision (`sfeast_*`, `cfeast_*`) missing.

## Missing Feature High-Priority Items

1. **Iterative FEAST (IFEAST) Variants**
   - Fortran provides `difeast_*`, `zifeast_*` (dense/sparse/banded).
   - Requires iterative linear solvers inside RCI loop (GMRES/BiCGSTAB support, convergence tolerances).

2. **MPI/Parallel Families**
   - `pdfeast_*`, `pzfeast_*`, `pdifeast_*`, `pzifeast_*`.
   - Would need MPI.jl integration, contour distribution, collective reductions.

3. **Complex-Symmetric Specialized Paths**
   - Fortran includes `zfeast_sy*` dense/sparse/banded routines.
   - Julia uses general solvers; specialized symmetric storage not supported.

4. **Single Precision Support**
   - Provide `Float32`/`ComplexF32` workflows matching `sfeast_*` and `cfeast_*`.

5. **Advanced Polynomial/Custom Contour RCI**
   - `feast_srcipev`/`feast_grcipev` wrappers exist, but direct RCI entry points for polynomial problems (`feast_grcipev!` etc.) still rely on generalized wrappers.
   - Need direct polynomial linearization for banded/sparse cases without dense expansion.

6. **Comprehensive Example Ports**
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

