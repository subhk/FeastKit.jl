include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Utility functions" begin
    # Test feast_name function
    code = 241500  # Example Feast code
    name = feast_name(code)
    @test isa(name, String)
    @test length(name) > 0

    # Test eigenvalue filtering
    lambda = [0.5, 1.5, 2.5, 3.5]
    @test feast_inside_contour(1.0, 0.0, 2.0) == true
    @test feast_inside_contour(3.0, 0.0, 2.0) == false

    # Test complex contour
    @test feast_inside_gcontour(1.0+1.0im, 1.0+1.0im, 2.0) == true
    @test feast_inside_gcontour(5.0+5.0im, 1.0+1.0im, 2.0) == false
    @test feast_inside_gcontour(1.0, 1.0+0.0im, 2.0) == true
end

@testset "Memory estimation" begin
    # Test memory estimation
    N, M0 = 100, 10
    mem_size = feast_memory_estimate(N, M0, Float64)
    @test mem_size > 0
end

@testset "Error handling" begin
    # Test error enum values
    @test Feast_SUCCESS.value == 0
    @test Feast_ERROR_N.value == 1
    @test Feast_ERROR_M0.value == 2

    # Test parameter validation on the normal initialized-parameter path
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    feastdefault!(fpm)
    @test fpm[1] == 0   # Default print level (off)
    @test fpm[2] == 8   # Default integration points
    @test fpm[3] == 12  # Default tolerance exponent
    @test fpm[4] == 20  # Default max loops

    # Test that invalid values outside valid range throw errors
    fpm_bad = zeros(Int, 64)
    feastinit!(fpm_bad)
    fpm_bad[1] = 5  # Invalid print level (must be 0 or 1)
    @test_throws ArgumentError feastdefault!(fpm_bad)
end

@testset "Performance utilities" begin
    # Test memory estimation
    N, M0 = 50, 8
    mem_size = feast_memory_estimate(N, M0, Float64)
    @test mem_size > 0

    # Test interval validation
    A = diagm(0 => [1.0, 2.0, 3.0, 4.0])
    bounds = feast_validate_interval(A, (1.5, 3.5))
    @test bounds[1] <= bounds[2]  # min <= max

    # Test result summary (should not crash)
    lambda = [1.0, 2.0]
    q = [1.0 0.0; 0.0 1.0]
    res = [1e-12, 1e-12]
    result = FeastResult{Float64, Float64}(lambda, q, 2, res, 0, 1e-12, 3)

    # Capture output to avoid cluttering test results
    buf = IOBuffer()
    feast_summary(buf, result)
    output_str = String(take!(buf))
    @test length(output_str) > 0  # Should produce some output
end
