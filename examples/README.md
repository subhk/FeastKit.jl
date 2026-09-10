# Running the examples

From the repository root, prepare the project once:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then run:

```sh
julia --project=. examples/custom_contour_integration.jl
julia --project=. examples/feast/run_feast_examples.jl
```

For the matrix-free examples, create a separate environment that develops this
checkout and includes the optional Krylov dependency:

```julia
using Pkg
Pkg.activate("/tmp/feast-examples")
Pkg.develop(path=pwd()) # run from the repository root
Pkg.add("Krylov")
```

```sh
julia --project=/tmp/feast-examples examples/matrix_free_examples.jl
```

The matrix-free script runs five problems at their original sizes, including a
10,000-dimensional tridiagonal operator and a 5,000-dimensional sparse matrix.
It needs no plotting package. Contour examples perform and validate real solves,
not just node construction. Polygon weights include `dz/(2πim)`; the helper
corrects clockwise orientation and uses midpoint quadrature on each edge.

## Reference-style fixtures

`feast/data/` contains **synthetic fixtures created for these Julia examples**,
not the original FEAST benchmark matrices. The driver names identify the API
families being demonstrated; the bundled results are not original benchmark
results. Every matrix has dimension 50:

| Fixture | Construction / known spectrum |
|---|---|
| `system1`, `system1B` | Diagonal `0.1i+0.003`, identity mass |
| `system2` | Hermitian diagonal `-1+0.05i+0.003` |
| `system3`, `system3B` | Upper-bidiagonal, diagonal `0.1i+0.003`, superdiagonal `0.01`, identity mass |
| `system4` | Complex diagonal `i+0.03-0.1im` |
| `system5A0/A1/A2` | Diagonal quadratic `(λ-rᵢ)(λ-5)`, with `rᵢ=-2+0.019i` |

Here `i=1,…,50` denotes the row index. The quadratic coefficient files are in
ascending power order. These small matrices exercise dense, sparse, banded,
generalized, polynomial, and custom-contour APIs with known answers.

To run external reference data instead, set `FEAST_EXAMPLE_DATA_DIR` to its
directory before starting Julia. Supply all nine named `.mtx` files. The reader
expects the compact FEAST format: an `N N nnz` header followed by `row col value`
(real) or `row col real imag` (complex), with 1-based indices and explicit
entries. It does not parse general MatrixMarket banners or infer missing
triangles. Retune the driver intervals and `M0` if using different problems.

The lowest-eigenvalue demonstration uses a dense reference spectrum to choose
an interval for the five smallest eigenvalues. It is a small-fixture teaching
example, not a scalable automatic spectral-bound estimator; `fpm[40]` is not
used as a substitute for interval selection.
