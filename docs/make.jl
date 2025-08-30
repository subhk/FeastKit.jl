#!/usr/bin/env julia

using Documenter

# Make the package available when building from the repository without installing
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using FeastKit

# Enable pretty URLs on CI (GitHub Actions) and plain URLs locally
const PRETTY_URLS = get(ENV, "CI", "") == "true"

makedocs(
    sitename = "FeastKit.jl",
    modules = [FeastKit],
    authors = "FeastKit.jl Contributors",
    source = @__DIR__,
    build = joinpath(@__DIR__, "..", "docs-build"),
    clean = true,
    doctest = false,
    format = Documenter.HTML(
        prettyurls = PRETTY_URLS,
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Zero to FeastKit" => "zero_to_feast.md",
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
#     repo = "github.com/subhk/FeastKit.jl.git",
#     devbranch = "main",
#     target = "build", # build directory from makedocs
# )

println("\n FeastKit.jl docs built at: ", joinpath(@__DIR__, "..", "docs-build", "index.html"))
