# FeastKit.jl Documentation

The website is built with Documenter.jl from `docs/src/`. Navigation and build
settings live in `docs/make.jl`; dependencies are listed in `docs/Project.toml`.
Generated HTML is written to `docs/build/` and is not committed.

## Build locally

Run from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs --threads=2 docs/make.jl
```

Open `docs/build/index.html`. Local builds do not publish. The docs environment
loads Krylov so iterative examples can execute; ordinary direct FEAST solves
do not require it.

`@example` blocks execute during the build, and doctests are enabled. Broken
examples and internal references fail the build. Exported docstrings omitted
from the manual also fail the build, so add them to the API reference. Plain
`julia` blocks are signatures, templates, or examples requiring the stated
external setup; they are not automatically executed by Documenter.

## Verify parallel examples

The standalone scripts in the parallel guide are executed directly from the
Markdown source. From the repository root:

```sh
julia --project=docs --threads=2 docs/check_parallel_examples.jl threads
julia --project=docs --threads=2 docs/check_parallel_examples.jl distributed
```

The distributed example creates and removes two local workers. For MPI, use an
environment containing this checkout, MPI, and Krylov. Each mode must run in
fresh processes; the MPI library cannot be reinitialized after finalization:

```sh
julia --project -e 'using MPI; for mode in ("mpi", "mpi_aliases", "hybrid"); run(`$(MPI.mpiexec()) -n 2 julia --project --threads=2 docs/check_parallel_examples.jl $mode`); end'
```

CI runs these examples with two threads, two local workers, or two MPI ranks.
Remote SSH host configuration and application-specific matrix assembly are
explicit templates; they require the reader's infrastructure or model.
Package installation commands and API signatures are not solver examples.

## Publishing

[The documentation workflow](../.github/workflows/pages.yml) develops FeastKit
from the checked-out repository, builds the site, and enables `deploydocs`
with `FEASTKIT_DOCS_DEPLOY=true`. Documenter publishes to the `gh-pages` branch.
GitHub Pages should serve that branch's root, rather than `main/docs`.

Main-branch builds update the development documentation; version tags produce
versioned documentation and the stable alias. Pull-request preview publishing
requires the workflow's configured credentials and permissions. A local edit
or build does not change the live site.

- [Development documentation](https://subhk.github.io/FeastKit.jl/dev/)
- [Stable documentation](https://subhk.github.io/FeastKit.jl/stable/)

## Contributing

Edit the relevant page in `docs/src/`, and add new pages to `docs/make.jl`.
Use small, deterministic `@example` blocks with assertions for solver behavior.
Check `result.converged` and verify the expected eigenvalues instead of only
printing a result. State when code needs multiple processes or an optional
package. Keep the public signatures and numerical conventions aligned with
`src/` and the API tests.

Run the local build before submitting changes. The package test suite also
executes selected documentation snippets in `test/docs/examples.jl`
and checks the named and checked APIs in their dedicated regression files.
