# Parallel Computing

FEAST can solve shifted systems at different contour nodes concurrently. This
guide contains complete local threading, distributed-worker, MPI, and hybrid
examples. Their assertions check the eigenvalues, not a promised speedup.

```@contents
Pages = ["parallel_computing.md"]
Depth = 2
```

## Overview

| Backend | Supported high-level problems | Required setup |
|:--|:--|:--|
| `:threads` | Real symmetric dense/sparse interval problems | Multiple Julia threads |
| `:distributed` | Real symmetric sparse interval problems | Julia workers with FeastKit available |
| `:mpi` | Real symmetric intervals and complex Hermitian/general dense/sparse problems | MPI.jl, initialized communicator, multiple ranks |
| `feast_hybrid` | Real symmetric MPI problems with threaded contour solves | MPI plus threads on each rank |

Explicit unavailable backends raise an error. `backend=:auto` permits fallback
and does not guarantee that a parallel backend will be chosen. Assembled serial
solves support direct and GMRES inner solvers. Threaded/distributed and real MPI
paths use direct solves; complex MPI also supports GMRES after loading Krylov.

All examples use a project environment containing FeastKit. The MPI examples
also need MPI; the iterative MPI alias needs Krylov. Scripts marked with
`# docs-test:` are executed by `docs/check_parallel_examples.jl` in CI.

## Checking Capabilities

The report reflects the current session:

```@example parallel_capabilities
using FeastKit
capabilities = feast_parallel_capabilities()
@assert all(k -> haskey(capabilities, k), (:threads, :distributed, :mpi))
@assert capabilities[:threads] == (Threads.nthreads() > 1)
feast_parallel_info()  # Output reflects the current session, not fixed hardware.
```

## Threading (Shared Memory)

### Setup

Start Julia with at least two threads:

```sh
julia --project --threads=2
```

`--threads=auto` can select the count from your machine. Julia thread count is
separate from BLAS thread count; extra BLAS threads can oversubscribe the CPU.

### Usage

Save this as `feast_threads.jl` and run it with the setup above. The explicit
threaded request requires more than one Julia thread:

```julia
# docs-test: threads
using FeastKit, LinearAlgebra, SparseArrays
@assert Threads.nthreads() > 1 "Start Julia with --threads=2 or more"
n = 200
A = sparse(SymTridiagonal(2.0*ones(n), -ones(n-1)))
B = spdiagm(0 => ones(n))
expected = [2-2cos(k*π/(n+1)) for k in 1:10]
interval = (0.0, (expected[end] + 2-2cos(11π/(n+1)))/2)
result = feast(A, B, interval; subspace_size=12, backend=:threads)
@assert result.converged && result.M == 10
@assert isapprox(result.values, expected; atol=1e-9)
result_auto = feast(A, B, interval; subspace_size=12, backend=:auto)
@assert result_auto.converged && result_auto.M == 10
println("Found $(result.M) eigenvalues with $(Threads.nthreads()) threads")
```

### Direct RCI Interface

Manual callers own the state and buffers and service every requested product.
The iteration bound below catches an incomplete RCI loop instead of hanging:

```@example parallel_manual_rci
using FeastKit, LinearAlgebra, Random
N, M0 = 12, 4
A = Matrix(Diagonal(collect(1.0:N)))
B = Matrix{Float64}(I, N, N)
Emin, Emax = 0.5, 2.5
fpm = feastinit().fpm
fpm[2] = 8
state = ParallelFeastState{Float64}(fpm[2], M0, true, true)
work = randn(MersenneTwister(42), N, M0)
workc = zeros(ComplexF64, N, M0)
Aq, Sq = zeros(M0, M0), zeros(M0, M0)
lambda, q, res = zeros(M0), zeros(N, M0), zeros(M0)
for request in 1:1000
    pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)
    if state.ijob == Int(FeastKit.Feast_RCI_PARALLEL_SOLVE)
        pfeast_compute_all_contour_points!(state, A, B, work, M0)
    elseif state.ijob == Int(Feast_RCI_MULT_A)
        work[:, 1:state.mode] .= A * q[:, 1:state.mode]
    elseif state.ijob == Int(Feast_RCI_MULT_B)
        work[:, 1:state.mode] .= B * q[:, 1:state.mode]
    elseif state.ijob == Int(Feast_RCI_DONE)
        break
    end
end
@assert state.ijob == Int(Feast_RCI_DONE) && state.info == 0
@assert state.mode == 2
@assert isapprox(lambda[1:state.mode], [1.0, 2.0]; atol=1e-9)
lambda[1:state.mode]
```

```@docs
feast_parallel
pfeast_srci!
```

## Distributed Computing

### Setup and Usage

This example starts two local workers in the active project and removes only
those workers when it finishes. Save it as `feast_distributed.jl` and run
`julia --project feast_distributed.jl` from an environment containing FeastKit.

```julia
# docs-test: distributed
using Distributed, FeastKit, LinearAlgebra, SparseArrays
pids = addprocs(2; exeflags=`--project=$(Base.active_project())`)
try
    n = 200
    A = sparse(SymTridiagonal(2.0*ones(n), -ones(n-1)))
    B = spdiagm(0 => ones(n))
    expected = [2-2cos(k*π/(n+1)) for k in 1:10]
    interval = (0.0, (expected[end] + 2-2cos(11π/(n+1)))/2)
    result = feast(A, B, interval; subspace_size=12, backend=:distributed)
    @assert result.converged && result.M == 10
    @assert isapprox(result.values, expected; atol=1e-9)

    # Match the quadrature task count to the workers; trapezoidal counts are flexible.
    fpm = feastinit().fpm
    fpm[16] = 1
    fpm[2] = max(2*nworkers(), 8)
    tuned = feast(A, B, interval; subspace_size=12, fpm=fpm, backend=:distributed)
    @assert tuned.converged && tuned.M == 10
    pfeast_show_distribution(fpm[2]; use_threads=false)
finally
    rmprocs(pids)
end
```

For remote workers, replace the `addprocs` call with your reachable SSH hosts
and remote Julia/project paths. This is a configuration template, not a
runnable local example:

```julia
pids = addprocs([("node1", 2), ("node2", 2)];
               exeflags=`--project=/path/to/remote/project`)
```

### How It Works

FeastKit loads itself on the selected workers and builds shifted factorizations
there. Factors are reused while right-hand sides and projected results move
between processes. Remote workers must have compatible Julia and package
environments. The high-level distributed path currently requires sparse real
symmetric inputs. The following command reports the current task distribution:

```@example parallel_distribution
using FeastKit
pfeast_show_distribution(16; use_threads=false)
```

## MPI Parallelization

### Prerequisites and Setup

Install optional dependencies in the same environment used to launch scripts:

```sh
julia --project -e 'using Pkg; Pkg.add(["MPI", "Krylov"])'
```

Load MPI, initialize it on every rank, and pass `comm` explicitly to FEAST.
`FEASTKIT_ENABLE_MPI=true` optionally enables automatic availability detection;
it is not required when an initialized communicator is supplied explicitly.
Every rank must provide the same matrices, search region, and solver settings.

### Usage

Save this complete script as `feast_mpi.jl`:

```julia
# docs-test: mpi
using MPI, FeastKit, LinearAlgebra, SparseArrays
MPI.Init()
try
    comm = MPI.COMM_WORLD
    n = 200
    A = sparse(SymTridiagonal(2.0*ones(n), -ones(n-1)))
    B = spdiagm(0 => ones(n))
    expected = [2-2cos(k*π/(n+1)) for k in 1:10]
    interval = (0.0, (expected[end] + 2-2cos(11π/(n+1)))/2)
    result = feast(A, B, interval; subspace_size=12, backend=:mpi, comm=comm)
    @assert result.converged && result.M == 10
    @assert isapprox(result.values, expected; atol=1e-9)

    Az = sparse(Diagonal(ComplexF64[0.5+0.1im, 1.0+0.2im, 2.0-0.1im]))
    Bz = spdiagm(0 => ones(ComplexF64, 3))
    general = feast_general(Az, Bz, 1.0+0.1im, 1.5;
                            subspace_size=3, backend=:mpi, comm=comm)
    @assert general.converged && general.M == 3
    @assert isapprox(sort(general.values; by=real), diag(Az); atol=1e-9)
    MPI.Comm_rank(comm) == 0 && println(result.values)
finally
    MPI.Finalize()
end
```

Launch with MPI.jl's configured executable, which matches its MPI library:

```sh
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 julia --project feast_mpi.jl`)'
```

### MPI-Specific Functions

Save the following as `feast_mpi_aliases.jl` and launch it the same way, in a
fresh process. It exercises direct MPI, real/complex PFEAST aliases, and a
GMRES-backed `pzifeast_*` alias. MPI cannot be reinitialized after finalization
in the same Julia process.

```julia
# docs-test: mpi_aliases
using MPI, Krylov, FeastKit, LinearAlgebra, SparseArrays
MPI.Init()
try
    comm = MPI.COMM_WORLD
    A = spdiagm(0 => [0.5, 1.0, 2.0])
    B = spdiagm(0 => ones(3))
    interval, M0 = (0.0, 2.5), 3
    fpm = feastinit().fpm
    direct = mpi_feast(A, B, interval; M0=M0, comm=comm, fpm=copy(fpm))
    real_alias = pdfeast_scsrgv!(A, B, interval..., M0, copy(fpm); comm=comm)

    Ahd = Matrix(Diagonal(ComplexF64[0.5, 1.0, 2.0]))
    Bhd = Matrix{ComplexF64}(I, 3, 3)
    dense_h = pzfeast_hegv!(Ahd, Bhd, interval..., M0, copy(fpm); comm=comm)
    sparse_h = pzfeast_hcsrgv!(sparse(Ahd), sparse(Bhd), interval..., M0,
                              copy(fpm); comm=comm)
    for result in (direct, real_alias, dense_h, sparse_h)
        @assert result.converged && result.M == 3
        @assert isapprox(result.values, [0.5, 1.0, 2.0]; atol=1e-9)
    end
    Az = sparse(Diagonal(ComplexF64[0.5+0.1im, 1.0+0.2im, 2.0-0.1im]))
    general = pzifeast_gcsrgv!(Az, sparse(Bhd), 1.0+0.1im, 1.5, M0,
                              copy(fpm); comm=comm, solver_tol=1e-12)
    @assert general.converged && general.M == 3
    @assert isapprox(sort(general.values; by=real), diag(Az); atol=1e-9)
finally
    MPI.Finalize()
end
```

MPI broadcasts custom contour nodes, weights, and optional boundary vertices
from the communicator root. General-problem membership uses that geometry.
Rank-local factorization or direct-solve failures return `Feast_ERROR_LAPACK`
collectively; iterative projection failures return
`Feast_ERROR_NO_CONVERGENCE`. Product, projection, eigensolve, and residual
failures are synchronized so ranks do not diverge across collectives.

The default inner MPI GMRES tolerance is one percent of the precision-aware
outer tolerance, subject to a machine-epsilon floor. Use high-level
`solver_opts=(rtol=...,)` or low-level `solver_tol` to override it. Real and
complex `mpi_feast` calls accept a `FeastParameters` wrapper or its raw `.fpm`.

`fpm[10]=1` caches each rank's assigned factors and permits threaded reuse.
`fpm[10]=0` factors shifts one at a time on each sweep, using serial contour
solves within a hybrid rank to limit retained storage.

The real symmetric PFEAST aliases use the parallel Julia kernels without
`comm`, and the supported MPI kernels when `comm` is supplied. Complex
`pcfeast_*`/`pzfeast_*` and iterative `pcifeast_*`/`pzifeast_*` aliases cover
supported dense/sparse Hermitian and general MPI problems; see the
[API reference](api_reference.md) for the precision families.

## Hybrid Parallelization

Hybrid execution combines MPI ranks with threaded solves of contour systems
within each rank. MPI collectives run on the calling thread. It currently
supports real symmetric problems through `feast_hybrid`, not a `:hybrid`
backend keyword.

### Setup

Save the script below as `feast_hybrid.jl`, then run two ranks with two threads
each:

```sh
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 julia --project --threads=2 feast_hybrid.jl`)'
```

### Usage

```julia
# docs-test: hybrid
using MPI, FeastKit, LinearAlgebra, SparseArrays
@assert Threads.nthreads() > 1 "Start each rank with --threads=2 or more"
MPI.Init()
try
    comm = MPI.COMM_WORLD
    A = spdiagm(0 => collect(1.0:12.0))
    B = spdiagm(0 => ones(12))
    result = feast_hybrid(A, B, (0.5, 2.5); M0=4, comm=comm,
                          use_threads_per_rank=true)
    @assert result.converged && result.M == 2
    @assert isapprox(result.values, [1.0, 2.0]; atol=1e-9)
    MPI.Comm_rank(comm) == 0 && println(result.values)
finally
    MPI.Finalize()
end
```

## Performance Tuning

### Choosing the Right Backend

Measure convergence, time, and peak memory for your problem. A supported
backend with more workers can still be slower because of communication,
compilation, or oversubscription. Sparse fill-in often dominates factor costs.

### Integration Points vs Workers

The distributed example also demonstrates setting `fpm[2]` to at least twice
the worker count. It uses trapezoidal integration to allow those counts; Gauss
and Zolotarev have restricted supported counts. More nodes change accuracy and
work as well as task parallelism.

### Benchmarking

This small example checks the target count before reporting timings. Use
warm-up runs and representative matrices for a performance study:

```@example parallel_benchmark
using FeastKit, SparseArrays, LinearAlgebra
n = 200
A = sparse(SymTridiagonal(2.0*ones(n), -ones(n-1)))
B = spdiagm(0 => ones(n))
interval = (0.0, 1.05*(2-2cos(10π/(n+1))))
checked = feast(A, B, interval; subspace_size=12)
@assert checked.converged && checked.M == 10
# M0 is positional. Reports reflect this session; timings include compilation.
feast_parallel_comparison(A, B, interval, 12)
FeastKit.pfeast_rci_benchmark(A, B, interval, 12; compare_serial=true)
```

### Memory Considerations

Threads share input matrices but need per-shift factors and work buffers.
Distributed and MPI processes hold their own matrix data and assigned factors.
There is no fixed total-memory multiplier: fill-in, caching, and subspace size
matter. Reduce subspace width only while retaining room for all target roots.

## Troubleshooting

### Threading Not Working

```@example parallel_threads
# Check thread count
println(Threads.nthreads())  # Should be > 1

# Solution: Restart Julia with threads
# julia --threads=8
```

### Distributed Workers Not Found

```@example parallel_workers
using Distributed
println("Additional workers available: ", nprocs() > 1)
# If false, use the complete Setup and Usage example above to add local workers.
```

### MPI Initialization Fails

```bash
# Install MPI in the same project used to launch the script.
julia --project -e 'using Pkg; Pkg.add("MPI")'
# Run the complete example through the launcher selected by MPI.jl.
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 julia --project feast_mpi.jl`)'
```

### Performance Not Scaling

1. **Check linear solver time**: If solving `(z*B - A)*Y = X` dominates, parallel overhead may be significant
2. **Increase problem size**: Small problems have too much communication overhead
3. **Use matrix-free**: For very large problems, matrix-free with parallel linear solvers scales better

Needs a multi-threaded session.

```@example parallel_efficiency
using FeastKit, LinearAlgebra, SparseArrays
A = spdiagm(0 => collect(1.0:40.0))
B = spdiagm(0 => ones(40))
interval = (0.5, 3.5)
if Threads.nthreads() > 1
    # Warm up each path before timing; this small example makes no speedup claim.
    feast(A, B, interval; subspace_size=6, backend=:serial)
    feast(A, B, interval; subspace_size=6, backend=:threads)
    t_serial = @elapsed result_serial = feast(A, B, interval; subspace_size=6, backend=:serial)
    t_parallel = @elapsed result_parallel = feast(A, B, interval; subspace_size=6, backend=:threads)
    @assert result_serial.converged && result_parallel.converged
    @assert result_parallel.M == result_serial.M == 3
    @assert isapprox(result_parallel.values, result_serial.values; atol=1e-9)
    println("Speedup: ", t_serial/t_parallel)
else
    println("Start Julia with --threads=2 or more to compare these paths.")
end
```

---

## API Reference

### High-Level Functions

Signature catalogue -- `A`, `B`, `interval`, `M0` and the RCI buffers are yours:

```julia
# Automatic backend selection
feast(A, B, interval; backend=:auto)
feast(A, B, interval; backend=:threads)
feast(A, B, interval; backend=:distributed)
feast(A, B, interval; backend=:mpi, comm=MPI.COMM_WORLD)

# Direct parallel interface
feast_parallel(A, B, interval; use_threads=true)
mpi_feast(A, B, interval; comm=comm)
feast_hybrid(A, B, interval; comm=comm, use_threads_per_rank=true)
```

### State Management

```julia
# Parallel RCI state
state = ParallelFeastState{Float64}(ne, M0, use_parallel, use_threads)

# RCI functions
pfeast_srci!(state, N, work, workc, Aq, Sq, fpm, Emin, Emax, M0, lambda, q, res)
pfeast_compute_all_contour_points!(state, A, B, work, M0)
```

### Utilities

```julia
# Check capabilities
feast_parallel_capabilities()
feast_parallel_info()
mpi_available()

# Distribution helpers
pfeast_show_distribution(ne; use_threads=true)
determine_parallel_backend(parallel, comm)
```

---

**Scale your eigenvalue computations!**

[Performance Tips](performance.md) · [Examples](examples.md) · [API Reference](api_reference.md)
