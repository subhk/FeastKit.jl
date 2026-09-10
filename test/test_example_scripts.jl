using Test, FeastKit, LinearAlgebra

@testset "Example script regressions" begin
    sandbox = Module(:ReferenceExamples)
    Core.eval(sandbox, :(include(path) = Base.include(@__MODULE__,path)))
    Base.include(sandbox, joinpath(@__DIR__,"..","examples","feast","run_feast_examples.jl"))
    @test isdefined(sandbox,:read_mm_dense_real)
    utils = getfield(sandbox,:FeastExampleUtils)
    @test isdir(utils.DATA_DIR)
    for vertices in (ComplexF64[-1-im,1-im,1+im,-1+im],
                     ComplexF64[-1+im,1+im,1-im,-1-im])
        c = utils.build_polygonal_contour(vertices,fill(32,4))
        @test abs(sum(c.Wne ./ c.Zne)-1) < 1e-3
        @test abs(sum(c.Wne ./ (c.Zne .- 3))) < 1e-3
    end
    @test_throws ArgumentError utils.build_polygonal_contour(ComplexF64[0,1,2],[1,1,1])
    @test_throws ArgumentError utils.build_polygonal_contour(ComplexF64[0,1,im],[1,0,1])
    band,kl,ku = utils.read_banded_real("system3")
    @test banded_to_full(band,max(kl,ku),50) == utils.read_mm_dense_real("system3")
    real_values = [0.1i+0.003 for i in 2:9]
    herm_values = [-1+0.05i+0.003 for i in 13:24]
    poly_values = [-2+0.019i for i in 22:26]
    cases = [(:dense_real_sygv,real_values),(:dense_complex_heev,herm_values),
        (:dense_real_gegv,real_values),(:dense_real_pep,poly_values),
        (:dense_complex_syev,ComplexF64[i+.03-.1im for i in 1:6]),
        (:dense_complex_syevx,ComplexF64[i+.03-.1im for i in 1:4]),
        (:sparse_real_scsrgv,real_values),(:sparse_real_scsrgv_lowest,[.1i+.003 for i in 1:5]),
        (:sparse_real_gcsrgv,real_values),(:sparse_real_scsrpev,poly_values),
        (:sparse_complex_hcsrev,herm_values),
        (:sparse_complex_scsrev,ComplexF64[i+.03-.1im for i in 1:6]),
        (:sparse_complex_scsrevx,ComplexF64[i+.03-.1im for i in 1:4]),
        (:banded_real_sbgv,real_values),(:banded_real_gbgv,real_values),
        (:banded_complex_hbev,herm_values),
        (:banded_complex_sbev,ComplexF64[i+.03-.1im for i in 1:6]),
        (:banded_complex_sbevx,ComplexF64[i+.03-.1im for i in 1:4])]
    for (name,expected) in cases
        @testset "$name" begin
            result = Base.invokelatest(getfield(sandbox,name))
            @test result.info == 0
            @test result.M == length(expected)
            @test sort(result.lambda;by=real) ≈ expected atol=1e-8
        end
    end
    contour_sandbox = Module(:ContourExamples)
    Core.eval(contour_sandbox, :(include(path) = Base.include(@__MODULE__,path)))
    Base.include(contour_sandbox,joinpath(@__DIR__,"..","examples","custom_contour_integration.jl"))
    c = Base.invokelatest(contour_sandbox.example_custom_contour)
    @test abs(sum(c.Wne ./ c.Zne)-1) < 1e-3
    results = Base.invokelatest(contour_sandbox.example_eigenvalue_problem)
    @test all(r -> r.info==0 && r.M==4,results)
end
