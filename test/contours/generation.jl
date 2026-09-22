include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Contour generation" begin
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    feastdefault!(fpm)  # Apply defaults before using contour functions

    # Test elliptical contour for real interval (symmetric/Hermitian)
    # Uses fpm[2] for half-contour (default: 8 points)
    Emin, Emax = 0.0, 10.0
    contour = feast_contour(Emin, Emax, fpm)
    @test length(contour.Zne) == fpm[2]
    @test length(contour.Wne) == fpm[2]

    # Test elliptical contour for complex region (non-Hermitian/general)
    # Uses fpm[8] for full-contour (default: 16 points) - matches Fortran FEAST
    Emid = 5.0 + 2.0im
    r = 3.0
    contour_g = feast_gcontour(Emid, r, fpm)
    @test length(contour_g.Zne) == fpm[8]  # Full contour uses fpm[8]
    @test length(contour_g.Wne) == fpm[8]
end
