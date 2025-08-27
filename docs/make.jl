#!/usr/bin/env julia

using Documenter
using FEAST

# Enable pretty URLs on CI (GitHub Actions) and plain URLs locally
const PRETTY_URLS = get(ENV, "CI", "") == "true"

makedocs(
    sitename = "FEAST.jl",
    modules = [FEAST],
    authors = "FEAST.jl Contributors",
    source = @__DIR__,
    build = joinpath(@__DIR__, "build"),
    clean = true,
    doctest = false,
    format = Documenter.HTML(
        prettyurls = PRETTY_URLS,
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Zero to FEAST" => "zero_to_feast.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => Any[
            "Examples" => "examples.md",
            "Matrix-Free Interface" => "matrix_free_interface.md",
            "Performance Tips" => "performance.md",
            "Custom Contours" => "custom_contours.md",
            "Complex Eigenvalues" => "complex_eigenvalues.md",
            "Polynomial Problems" => "polynomial_problems.md",
            "Parallel Computing" => "parallel_computing.md",
        ],
        "API Reference" => "api_reference.md",
        "Project" => Any[
            "Contributing" => "contributing.md",
            "Developer Guide" => "developer_guide.md",
            "Testing" => "testing.md",
            "License" => "license.md",
            "Changelog" => "changelog.md",
            "Bibliography" => "bibliography.md",
        ],
    ],
)

# Note: For GitHub Pages via Documenter (gh-pages branch), uncomment deploydocs
# and add an SSH key secret (DOCUMENTER_KEY) to the repository.
#
# deploydocs(
#     repo = "github.com/subhk/FEAST.jl.git",
#     devbranch = "main",
#     target = "build", # build directory from makedocs
# )

println("\n✔ FEAST.jl docs built at: ", joinpath(@__DIR__, "build", "index.html"))
