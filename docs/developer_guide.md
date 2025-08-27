# Developer Guide

This section outlines the project layout and development tips.

- Source: `src/`
- Tests: `test/`
- Docs: `docs/`

Build and test locally:

```sh
julia --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.test()'
```

Docs (MkDocs): see `docs/README.md` for commands to serve and deploy.

