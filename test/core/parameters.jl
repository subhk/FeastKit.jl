include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Parameter initialization" begin
    # Test feastinit - now uses -111 sentinel value (matches Fortran)
    fpm = feastinit()
    @test length(fpm.fpm) == 64
    @test all(fpm.fpm .== -111)  # All initialized to sentinel value

    # Test direct fpm array initialization
    fpm_array = zeros(Int, 64)
    feastinit!(fpm_array)
    @test all(fpm_array .== -111)  # All initialized to sentinel value

    # Test that feastdefault! sets defaults for -111 values
    feastdefault!(fpm_array)
    @test fpm_array[1] == 0   # Default print level (off)
    @test fpm_array[2] == 8   # Default integration points
    @test fpm_array[3] == 12  # Default tolerance
    @test fpm_array[42] == 0  # Mixed precision is opt-in
    @test fpm_array[4] == 20  # Default max loops

    # Test that user values are preserved
    fpm_user = zeros(Int, 64)
    feastinit!(fpm_user)
    fpm_user[2] = 16  # User sets 16 contour points
    fpm_user[3] = 8   # User sets tolerance 10^-8
    feastdefault!(fpm_user)
    @test fpm_user[2] == 16  # User value preserved
    @test fpm_user[3] == 8   # User value preserved
    @test fpm_user[4] == 20  # Default applied to unset parameters
end

@testset "Input validation" begin
    # Test parameter validation
    @test_throws ArgumentError check_feast_srci_input(0, 10, 0.0, 1.0, zeros(Int, 64))
    @test_throws ArgumentError check_feast_srci_input(10, 0, 0.0, 1.0, zeros(Int, 64))
    @test_throws ArgumentError check_feast_srci_input(10, 15, 0.0, 1.0, zeros(Int, 64))
    @test_throws ArgumentError check_feast_srci_input(10, 5, 1.0, 0.0, zeros(Int, 64))
    @test_throws ArgumentError check_feast_srci_input(10, 5, 0.0, 1.0, zeros(Int, 32))

    # Should pass for valid inputs
    @test check_feast_srci_input(10, 5, 0.0, 1.0, zeros(Int, 64)) == true
end
