# Performance Guide

Measure the solve needed by your application, including its convergence and
memory requirements. Timing depends on matrix structure, the search region,
precision, and hardware; there is no fixed speedup or problem-size cutoff that
selects the best solver.

## Performance Overview

A refinement sweep includes shifted solves, projection/orthogonalization, and
a reduced eigensolve. For dense direct solves, factorization is cubic in `N`
per distinct shift, while solving each right-hand side is quadratic. The reduced
problem scales cubically in the subspace width. Sparse factorization costs
and storage depend strongly on fill-in. Iterative costs depend on products,
preconditioning, restart size, and the iterations required for each shift.

## Memory Optimization

### Memory Usage Patterns

| Storage | Main costs beyond the input |
|:--|:--|
| Dense direct | Several `N*M0` buffers, projected matrices, complex shifted matrices and LU factors |
| Sparse direct | Subspace buffers plus sparse shifted factors, including fill-in |
| Matrix-free iterative | Subspace buffers, Krylov basis, preconditioner, and callback-owned storage |

A single Float64 matrix uses `8N^2` bytes; a ComplexF64 matrix uses `16N^2`.
These are individual-array sizes, not total solver-memory estimates.
`feast_memory_estimate(N, M0, Float64)` estimates a subset of core workspaces;
it does not include all matrices, factorization caches, backend copies, or
iterative-solver storage.

`fpm[10]=1` retains direct factorizations between sweeps. `fpm[10]=0` recomputes
shifts and can reduce retained factor storage on supported drivers. Matrix-free
methods still allocate CPU workspaces. Reducing `M0` is safe only when there
is enough room for every enclosed eigenvalue, including multiplicity.

### Automatic Subspace Sizing

Use `subspace_size=:auto` when the enclosed eigenvalue count is unknown.
For assembled matrices, a stochastic trace estimate selects the initial width
with oversampling. The estimator performs an additional direct contour sweep,
even if the subsequent solve uses GMRES; its factors are not retained for the
solve. Matrix-free operators start at a small heuristic width without this
factorization step. Both paths grow when the subspace saturates, or when
nonconvergence leaves too little oversampling above the returned count.

Set `max_subspace_size` to bound the width (default: the matrix dimension).
The returned status remains unsuccessful if that limit prevents convergence.
The count estimate and convergence checks are numerical safeguards, not a
certified count, particularly for nonnormal problems or boundary clusters.
Automatic sizing is available on `feast`, `feast_general`, and their contour
wrappers; it cannot be combined with `M0`.

```@example adaptive_subspace
using FeastKit, LinearAlgebra
A = Matrix{Float64}(I, 4, 4)
limited = feast(A, (0.5, 1.5); subspace_size=:auto, max_subspace_size=1)
@assert !limited.converged && limited.info == Int(Feast_ERROR_M0)
result = feast(A, (0.5, 1.5); subspace_size=:auto, max_subspace_size=4)
@assert result.converged && result.M == 4
result
```

### Warm Starts for Related Problems

Pass previous eigenvectors as `initial_subspace`. The solver copies and
normalizes the columns, adds independent directions up to the requested
width, and retains its completeness checks. A partial seed is allowed.
The seed must be a finite `N × k` matrix with nonzero columns and
`1 ≤ k ≤ subspace_size`; real interval solves require real vectors.
If the width is omitted, it defaults to `min(N, max(10, k+2))`.

A seed can reduce refinement work, but completeness probing and numerical
rank decisions can also make a seeded solve slower. Compare it with a cold
solve on your problem sequence.

Warm starts are supported by serial dense/sparse and matrix-free solves,
including generalized problems and contour wrappers. An explicit parallel
backend with a seed raises an error; `backend=:auto` permits serial fallback.
Banded convenience APIs do not currently expose this option.

```@example warm_start
using FeastKit, LinearAlgebra
A = Matrix(SymTridiagonal(fill(2.0, 40), fill(-1.0, 39)))
first = feast(A, (0.9, 1.1); subspace_size=6)
@assert first.converged
next_A = A + 1e-4I
next = feast(next_A, (0.9, 1.1); subspace_size=6,
             initial_subspace=first.vectors)
@assert next.converged && next.M == first.M
@assert isapprox(next.values, first.values .+ 1e-4; atol=1e-10)
next.values
```

## Computational Efficiency

### Integration Method Selection

Benchmark valid quadrature counts on the same target region. This small
fixture holds three eigenvalues. Warm up each configuration before comparing
steady-state timings in a separate performance study.

```@example integration_benchmark
using FeastKit, LinearAlgebra

function benchmark_integration_methods(A, interval)
    methods = [
        (0, "Gauss-Legendre"),
        (1, "Trapezoidal"), 
        (2, "Zolotarev")
    ]
    
    println("Integration Method Comparison")
    println("="^40)
    
    results = []
    
    for (method_id, method_name) in methods
        println("\n$method_name:")
        
        # Test different numbers of integration points
        for ne in [8, 12, 16, 24]
            fpm = feastinit().fpm
            fpm[2] = ne
            fpm[16] = method_id
            
            time = @elapsed begin
                result = feast(A, interval, M0=10, fpm=fpm)
            end
            
            result.converged || error(result.message)
            push!(results, (method_name, ne, time, result.M, result.info))
            println("  ne=$ne: $(time) s, found $(result.M), status $(result.info)")
        end
    end
    
    return results
end

# Run benchmark
A = Matrix(Diagonal(collect(1.0:40.0)))
results = benchmark_integration_methods(A, (0.5, 3.5))
@assert all(row -> row[4] == 3 && row[5] == 0, results)
```

General and polynomial solves use the full-contour count `fpm[8]`; interval
solves use `fpm[2]`. Named `quadrature_points` selects the right entry for
`feast` and `feast_general`. Contour objects carry their own node counts.

### Linear Solver Optimization

Assembled matrices default to `solver=:direct`. Use `solver=:gmres` with
`using Krylov` for iterative shifted solves. High-level options are
`solver_opts=(rtol=..., maxiter=..., restart=...)`; low-level drivers use
`solver_tol`, `solver_maxiter`, and `solver_restart`.

When `rtol` is omitted, iterative drivers loosen early shifted solves and
tighten them as the outer residual falls (initial target `1e-3`). This applies
to serial dense/sparse/banded drivers, complex MPI drivers, and built-in
matrix-free solvers. Explicit `solver_opts.rtol` is kept fixed, and custom
callbacks control their own accuracy. Each owning iterative driver reuses a
Krylov workspace across nodes and refinement sweeps. Separate solves/ranks
own separate workspaces; no shared global Krylov buffer is used.

Matrix-free calls also support `:bicgstab` and a custom callback. There is no
BiCGSTAB `l` option, and CG is not valid for the complex shifted systems.
The matrix-free `preconditioner` option must implement inverse action through
`mul!` on complex vectors. Passing `Pl` or an unadapted factorization is not
supported. See [Matrix-Free Interface](matrix_free_interface.md).

For example, an explicit diagonal inverse action can be supplied as follows:

```@example preconditioned_solve
using FeastKit, Krylov, LinearAlgebra
entries = [1.0, 2.0, 3.0, 4.0]
A_op = LinearOperator{Float64}((y, x) -> (y .= entries .* x),
                              (4, 4); issymmetric=true)
P = Diagonal(ComplexF64.(inv.(entries)))
result = feast(A_op, (0.5, 2.5); subspace_size=3, tol=1e-9,
               solver_opts=(rtol=1e-13, maxiter=200, restart=8, preconditioner=P))
@assert result.converged result.message
@assert isapprox(result.values, [1.0, 2.0]; atol=1e-8)
result.values
```

This illustrates the interface, not a generally optimal preconditioner for
all shifts. Assess performance and accuracy for the actual shifted family.

### Mixed Precision

`mixed_precision=true` enables residual inverse iteration for **serial dense
Float64 or ComplexF64** problems with `solver=:direct`. The matrices, trial
vectors, residuals, and reduced eigensolve stay in Float64/ComplexF64.
Normalized shifted LU factors and residual correction solves use ComplexF32.
The solver explicitly checks each correction in Float64 and caches a Float64
factor for shifts where low precision is unreliable. Outer convergence still
uses the requested tolerance and full-precision eigenpair residuals.

This option supports real symmetric, complex Hermitian, and general pencils,
standard/generalized problems, warm starts, and custom contours. Sparse,
banded, matrix-free, Float32-input, and explicit parallel requests are not
supported; `backend=:auto` can select the serial dense path. With the legacy
parameter API, set `fpm[42]=1`; the default is now `0` (full precision).
Previously this parameter did not select a mixed-precision algorithm.

```@example mixed_precision
using FeastKit, LinearAlgebra
A = Matrix(SymTridiagonal(fill(2.0, 40), fill(-1.0, 39)))
result = feast(A, (0.9, 1.1); subspace_size=6, tol=1e-11,
               mixed_precision=true)
expected = filter(x -> 0.9 <= x <= 1.1, eigvals(Symmetric(A)))
@assert result.converged && result.M == length(expected)
@assert isapprox(result.values, expected; atol=1e-10)
@assert norm(A*result.vectors-result.vectors*Diagonal(result.values)) < 1e-9
result.values
```

Mixed precision reduces retained LU storage when no fallback is needed, but
adds correction buffers and full-precision matrix products. It can be slower
for small matrices or frequent fallback. Benchmark representative matrices
before enabling it by default in an application. The residual inverse form is
based on the linear specialization of
[Gavin, Miedlar, and Polizzi (2018)](https://arxiv.org/abs/1801.09794).

## Parallel Computing

Start Julia with `--threads=N` to set the Julia thread count. Calling
`BLAS.set_num_threads` only changes BLAS threading; it does not change how many
Julia contour tasks can run. Avoid oversubscribing cores with both layers.

| Execution | Public API and setup |
|:--|:--|
| Threads | `backend=:threads`; multiple Julia threads, real symmetric inputs |
| Julia workers | `backend=:distributed`; active workers, sparse real symmetric inputs |
| MPI | `backend=:mpi, comm=comm`; load and initialize MPI on all ranks |
| Hybrid MPI/threads | `feast_hybrid(A, B, interval; comm=comm, use_threads_per_rank=true)` |

`:hybrid` is not a valid value for `backend` or the legacy `parallel` keyword.
Adding Julia workers does not initialize MPI. Explicit unsupported requests
raise errors; `backend=:auto` permits fallback.

Each MPI rank and distributed worker owns matrix/factorization data for its
work. Threaded paths share some input storage but still need per-shift factors
and work buffers. These costs do not reduce to a constant multiplier of serial
memory. Benchmark the actual run and retain room for the selected subspace.
See [Parallel Computing](parallel_computing.md) for complete setup examples.

## Problem-Specific Optimizations

### Eigenvalue Distribution Optimization

The function below illustrates different settings, not automatic density
inference. Its subspace widths are suitable for the small fixture. For another
problem, choose widths from the enclosed count and inspect the returned status.

```@example distribution_options
using FeastKit, LinearAlgebra
function optimize_for_distribution(A, interval, eigenvalue_density="uniform")
    if eigenvalue_density == "clustered"
        # Use more integration points and Zolotarev method
        fpm = feastinit().fpm
        fpm[2] = 24      # More integration points
        fpm[16] = 2      # Zolotarev integration
        fpm[3] = 14      # Higher precision
        
        result = feast(A, interval, M0=30, fpm=fpm)
        
    elseif eigenvalue_density == "sparse"
        # Use fewer integration points, lower precision
        fpm = feastinit().fpm
        fpm[2] = 8      # Half-contour integration points
        fpm[3] = 8      # Tolerance: 10^(-fpm[3]) = 1e-8
        result = feast(A, interval; M0=10, fpm=fpm)
        
    else  # uniform
        # Default settings work well
        result = feast(A, interval, M0=20)
    end
    
    return result
end

A = Matrix(Diagonal(collect(1.0:40.0)))
results = [optimize_for_distribution(A, (0.5, 3.5), kind)
           for kind in ("clustered", "sparse", "uniform")]
@assert all(r -> r.converged && r.M == 3, results)
[r.values for r in results]
```

## Benchmarking and Profiling

Time the call itself; `FeastResult` does not have a timing field. Warm up first
and check that compared methods found the same eigenvalues to the requested
accuracy. Allocated bytes measure cumulative allocation, not peak live memory.

```@example allocation_measurement
using FeastKit, LinearAlgebra, SparseArrays
A = spdiagm(0 => collect(1.0:40.0))
solve() = feast(A, (0.5, 3.5); subspace_size=6)
@assert solve().converged
measurement = @timed solve()
@assert measurement.value.converged
(time=measurement.time, allocated_bytes=measurement.bytes)
```

For profiling, `Profile` is a Julia standard library. Interactive viewers such
as ProfileView are separate dependencies. Run larger benchmarks in dedicated
processes with fixed Julia/BLAS thread counts and reproducible input data.
The tiny timings generated in this manual demonstrate usage; they are not
performance claims for other machines or matrices.

The repository's `benchmark/solver_performance.jl` compares converged solves
at fixed counts and explicit residual accuracy. Run it with
`julia --project=docs --threads=2 benchmark/solver_performance.jl`.
