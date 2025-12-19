# Testing

Run the package tests:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Tips:

- Use `JULIA_DEPOT_PATH=$PWD/.julia` to keep artifacts local to the repo
- Prefer small, focused tests that run quickly

