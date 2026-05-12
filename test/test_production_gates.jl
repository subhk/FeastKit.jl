using Test

@testset "Production verification gates" begin
    ci_workflow = read(joinpath(@__DIR__, "..", ".github", "workflows", "ci.yml"), String)

    @test occursin("FEASTKIT_TEST_DISTRIBUTED", ci_workflow)
    @test occursin("FEASTKIT_TEST_MPI", ci_workflow)
    @test occursin("MPI.mpiexec", ci_workflow)
end
