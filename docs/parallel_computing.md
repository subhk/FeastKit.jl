# Parallel Computing

```@id parallel-computing
```

FeastKit supports multiple parallel backends:

- Threads: `parallel = :threads`
- Distributed workers: `parallel = :distributed`
- MPI (if available): `parallel = :mpi`

Quick check and usage:

```julia
using FeastKit
cap = feast_parallel_capabilities()
@info cap
res = feast(A, (Emin, Emax), M0=20, parallel=:threads)
```

Note: MPI requires MPI.jl and a working MPI installation.
