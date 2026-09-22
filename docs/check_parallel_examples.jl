#!/usr/bin/env julia

# Run the exact standalone scripts printed in the parallel guide. MPI modes
# must each run in a fresh MPI-launched process; MPI cannot be reinitialized.
using Markdown

length(ARGS) == 1 || error("Usage: julia docs/check_parallel_examples.jl threads|distributed|mpi|mpi_aliases|hybrid")
mode = only(ARGS)
mode in ("threads", "distributed", "mpi", "mpi_aliases", "hybrid") ||
    error("Unknown documentation example: $mode")

source = joinpath(@__DIR__, "src", "parallel_computing.md")
page = Markdown.parse(read(source, String))
examples = filter(page.content) do block
    block isa Markdown.Code && startswith(block.code, "# docs-test: $mode\n")
end
length(examples) == 1 || error("Expected one documented script for $mode, found $(length(examples))")
Base.include_string(Main, only(examples).code, source * " ($mode)")
println("Documentation example passed: $mode")
