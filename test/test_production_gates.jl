using Test

@testset "Production verification gates" begin
    # CI must actually execute the opt-in suites, not merely mention them.
    # These tests catch workflow edits that would silently drop coverage.
    ci_workflow = read(joinpath(@__DIR__, "..", ".github", "workflows", "ci.yml"), String)

    # Opt-in backend jobs still exist.
    @test occursin("FEASTKIT_TEST_DISTRIBUTED", ci_workflow)
    @test occursin("FEASTKIT_TEST_MPI", ci_workflow)
    @test occursin("MPI.mpiexec", ci_workflow)

    # The env-gated blocks inside runtests.jl are switched on. Checking for the
    # name alone would pass even when the variable is only named in a comment or
    # a skip message, which is how these suites stayed dormant.
    for flag in ("FEAST_RUN_LONG_TESTS", "FEAST_RUN_PARALLEL_TESTS", "FEASTKIT_TEST_PARALLEL")
        @test occursin(Regex("$(flag):\\s*'true'"), ci_workflow)
    end

    # Every env gate used by the suite must be enabled somewhere in the workflow,
    # so a newly added gate cannot default to "skipped forever".
    runtests_source = read(joinpath(@__DIR__, "runtests.jl"), String)
    gates = Set(m.captures[1] for m in eachmatch(r"get\(ENV,\s*\"([A-Z_]+)\"", runtests_source))
    for gate in gates
        @test occursin(gate, ci_workflow)
    end
end
