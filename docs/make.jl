#!/usr/bin/env julia

using Documenter
using FeastKit
# Loading Krylov activates FeastKitKrylovExt so the iterative `@example` blocks
# in the guides run rather than erroring.
using Krylov

makedocs(
    sitename = "FeastKit.jl",
    modules = [FeastKit],
    checkdocs = :exports,
    # `@example` blocks are executed at build time, so a doc example that stops
    # working now fails the build instead of silently rotting on the site.
    warnonly = false,
    authors = "FeastKit.jl Contributors",
    repo = Remotes.GitHub("subhk", "FeastKit.jl"),
    doctest = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "") == "true",
        canonical = "https://subhk.github.io/FeastKit.jl/stable/",
        assets = String[],
        size_threshold = 500 * 1024,  # 500 KB
        # The API reference page renders every exported docstring (~107 KiB).
        size_threshold_warn = 200 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Problem Setup" => "problem_setup.md",
        "Zero to FeastKit" => "zero_to_feast.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Examples" => "examples.md",
            "Matrix-Free Interface" => "matrix_free_interface.md",
            "Performance Tips" => "performance.md",
            "Custom Contours" => "custom_contours.md",
            "Complex Eigenvalues" => "complex_eigenvalues.md",
            "Polynomial Problems" => "polynomial_problems.md",
            "Parallel Computing" => "parallel_computing.md",
        ],
        "API Reference" => "api_reference.md",
        "Project" => [
            "Contributing" => "contributing.md",
            "Developer Guide" => "developer_guide.md",
            "Testing" => "testing.md",
            "License" => "license.md",
            "Changelog" => "changelog.md",
            "Bibliography" => "bibliography.md",
        ],
    ],
)

# Local builds only generate docs/build. The publishing workflow opts in.
if get(ENV, "FEASTKIT_DOCS_DEPLOY", "false") == "true"
    deploydocs(
        repo = "github.com/subhk/FeastKit.jl.git",
        devbranch = "main",
        push_preview = true,
    )
end
