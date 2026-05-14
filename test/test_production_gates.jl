using Test

@testset "Production verification gates" begin
    # CI should continue running the opt-in distributed and MPI jobs. These tests
    # catch workflow edits that would silently drop production backend coverage.
    ci_workflow = read(joinpath(@__DIR__, "..", ".github", "workflows", "ci.yml"), String)

    @test occursin("FEASTKIT_TEST_DISTRIBUTED", ci_workflow)
    @test occursin("FEASTKIT_TEST_MPI", ci_workflow)
    @test occursin("MPI.mpiexec", ci_workflow)
end
