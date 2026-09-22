include("support/setup.jl")

@testset "FeastKit.jl" begin
    @testset "Core" begin
        include("core/parameters.jl")
        include("core/utilities.jl")
        include("core/allocations.jl")
        include("core/coverage_gates.jl")
    end
    @testset "Contours" begin
        include("contours/generation.jl")
        include("contours/shapes.jl")
    end
    @testset "Public API" begin
        include("api/dispatch.jl")
        include("api/precision_aliases.jl")
        include("api/backends.jl")
        include("api/compatibility.jl")
        include("api/named_options.jl")
        include("api/preparation.jl")
        include("api/solver_controls.jl")
        include("api/checked_wrappers.jl")
    end
    @testset "Storage" begin
        include("storage/dense.jl")
        include("storage/sparse.jl")
        include("storage/banded.jl")
    end
    @testset "RCI" begin
        include("rci/kernels.jl")
    end
    @testset "Matrix-free" begin
        include("matrixfree/operators_and_solvers.jl")
    end
    @testset "Numerical correctness" begin
        include("numerics/eigenpairs.jl")
        include("numerics/mixed_precision.jl")
        include("numerics/generalized.jl")
        include("numerics/hermitian.jl")
        include("numerics/serial_drivers.jl")
        include("numerics/solver_failures.jl")
        include("numerics/projection_scaling.jl")
        include("numerics/projectors_and_scaling.jl")
    end
    @testset "Parallel backends" begin
        include("backends/threaded.jl")
        include("backends/precision_aliases.jl")
        include("backends/contours.jl")
        include("backends/execution.jl")
    end
    @testset "Documentation" begin
        include("docs/examples.jl")
    end
end
