using Test, FeastKit, LinearAlgebra

@testset "Standard contour shapes" begin
    for name in (:feast_circle, :feast_ellipse, :feast_rectangle)
        @test isdefined(FeastKit, name)
    end
    if all(name -> isdefined(FeastKit, name), (:feast_circle, :feast_ellipse, :feast_rectangle))
        contours = (feast_circle(0, 1; n=64),
                    feast_ellipse(0, 2, 1; n=64, rotation=0.2),
                    feast_rectangle(-1, 1, -1, 1; points_per_edge=32))
        for c in contours
            @test sum(c.Wne ./ c.Zne) ≈ 1 atol=1e-3
            @test sum(c.Wne ./ (c.Zne .- 5)) ≈ 0 atol=1e-3
            # The third value is inside the nominal radius but outside each shape.
            A = Matrix(Diagonal(ComplexF64[-0.3+0.2im, 0.4-0.1im, 2.5]))
            for B in (nothing, Matrix{ComplexF64}(I, 3, 3))
                fpm = feastinit().fpm
                fpm[16] = 1
                r = FeastKit.with_custom_contour(fpm, c) do
                    B === nothing ? feast_general(A, 0im, 3.0; M0=3, fpm=fpm, backend=:serial) :
                        feast_general(A, B, 0im, 3.0; M0=3, fpm=fpm, backend=:serial)
                end
                @test r.info == 0
                @test r.M == 2
                @test sort(r.lambda; by=real) ≈ [-0.3+0.2im, 0.4-0.1im] atol=1e-8
            end
        end
        @test length(contours[1].Zne) == 64
        @test length(contours[3].Zne) == 128
        @test eltype(feast_circle(0f0, 1f0).Zne) == ComplexF32
        @test eltype(feast_rectangle(-1f0, 1f0, -1f0, 1f0).Zne) == ComplexF32
        @test_throws ArgumentError feast_circle(0, -1)
        @test_throws ArgumentError feast_circle(Inf, 1)
        @test_throws ArgumentError feast_circle(0, 1; n=2)
        @test_throws ArgumentError feast_ellipse(0, 1, 0)
        @test_throws ArgumentError feast_ellipse(0, 1, 1; rotation=NaN)
        @test_throws ArgumentError feast_rectangle(1, -1, -1, 1)
        @test_throws ArgumentError feast_rectangle(-1, 1, 0, Inf)
        @test_throws ArgumentError feast_rectangle(-1, 1, -1, 1; points_per_edge=0)
        @testset "Published shape example" begin
            source = read(joinpath(@__DIR__, "..", "docs", "src", "custom_contours.md"), String)
            block = match(r"(?ms)^```@example standard_shapes\r?\n(.*?)^```", source)
            @test block !== nothing
            sandbox = Module(gensym(:ShapeDocs))
            Base.include_string(sandbox, block.captures[1])
            @test all(r -> r.info == 0 && r.M == 2, getfield(sandbox, :results))
        end
    end
end
