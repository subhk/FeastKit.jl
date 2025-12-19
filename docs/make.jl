#!/usr/bin/env julia

using Documenter
using FeastKit

makedocs(
    sitename = "FeastKit.jl",
    modules = [FeastKit],
    authors = "FeastKit.jl Contributors",
    repo = Remotes.GitHub("subhk", "FeastKit.jl"),
    doctest = false,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "") == "true",
        canonical = "https://subhk.github.io/FeastKit.jl/stable/",
        assets = String[],
        size_threshold = 500 * 1024,  # 500 KB
    ),
    pages = [
        "Home" => "index.md",
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

deploydocs(
    repo = "github.com/subhk/FeastKit.jl.git",
    devbranch = "main",
    push_preview = true,
)
