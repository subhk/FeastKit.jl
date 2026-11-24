@testset "Exterior polynomial Ritz pairs do not establish an empty contour" begin
    center = 0.0 + 0.0im
    radius = 1.0
    fpm = feastinit().fpm
    fpm[8] = 4
    b = 1.02 * first(feast_gcontour(center, radius, fpm).Zne)
    # The nearby exterior root dominates the quadrature even though zero is
    # inside. The settled exterior Ritz pairs do not reveal that missed root.
    coeffs = [Matrix(Diagonal(ComplexF64[0, 6])),
              Matrix(Diagonal(ComplexF64[-b, -5])),
              Matrix{ComplexF64}(I, 2, 2)]

    result = feast_grcipev!(coeffs, 2, center, radius, 2, copy(fpm))
    if result.converged
        @test result.lambda ≈ ComplexF64[0] atol=1e-10
    else
        @test result.info == Int(FeastKit.Feast_ERROR_NO_CONVERGENCE)
    end

    empty_result = feast_grcipev!(coeffs, 2, 100.0 + 0.0im, radius, 2, feastinit().fpm)
    @test empty_result.converged
    @test isempty(empty_result.lambda)
end

@testset "GMRES warm starts preserve uniform pencil scaling" begin
    n = 8
    interval = (0.5, 2.5)
    expected = [1.0, 2.0]
    # These powers keep the pencil finite while its squared norm would
    # underflow or overflow, causing the old warm start to be skipped.
    for scale in (1e-170, 1e155)
        A = Matrix(Diagonal(scale .* collect(1.0:n)))
        B = scale * Matrix{Float64}(I, n, n)
        multiply_A! = (y, x) -> mul!(y, A, x)
        multiply_B! = (y, x) -> mul!(y, B, x)
        A_op = LinearOperator{Float64}(multiply_A!, (n, n); issymmetric=true)
        B_op = LinearOperator{Float64}(multiply_B!, (n, n); issymmetric=true)

        for result in (feast(A, B, interval; M0=4, solver=:gmres),
                       feast(sparse(A), sparse(B), interval; M0=4, solver=:gmres),
                       feast(A_op, B_op, interval; M0=4),
                       feast_matvec(multiply_A!, multiply_B!, n, interval; M0=4))
            @test result.converged
            @test result.lambda ≈ expected atol=1e-10
        end
    end
end
