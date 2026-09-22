include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "High-level feast dispatch" begin
    n = 3
    # Real symmetric generalized problem
    @info "Dispatch: real symmetric generalized"
    A_sym = Symmetric(diagm(0 => 2.0 .* ones(n), 1 => -1.0 .* ones(n-1), -1 => -1.0 .* ones(n-1)))
    B_real = Matrix{Float64}(I, n, n)
    fpm = zeros(Int, 64)
    feastinit!(fpm)
    fpm[1] = 0
    result_real = feast(A_sym, B_real, (0.5, 3.5), M0=n, fpm=copy(fpm), parallel=:serial)
    @test result_real.info == 0
    @test result_real.M == n
    @test isapprox(sort(result_real.lambda), sort(eigvals(Matrix(A_sym))); atol=1e-10)

    # Ensure non-symmetric input errors out
    A_nonsym = [1.0 2.0 0.0; 0.0 3.0 1.0; 0.5 0.0 4.0]
    @test_throws ArgumentError feast(A_nonsym, B_real, (0.5, 3.5), M0=n, fpm=copy(fpm), parallel=:serial)

    # Dense complex Hermitian standard problem
    @info "Dispatch: dense Hermitian standard"
    A_herm = Hermitian([2.5 + 0im 0.2 + 0.1im 0.0;
                        0.2 - 0.1im 3.5 + 0im 0.3 - 0.2im;
                        0.0 0.3 + 0.2im 4.0 + 0im])
    result_complex = feast(A_herm, (2.0, 5.0), M0=n, fpm=copy(fpm), parallel=:serial)
    @test result_complex.info == 0
    @test result_complex.M == n
    hermitian_eigs = sort(eigvals(Matrix(A_herm)))
    @test isapprox(sort(result_complex.lambda), hermitian_eigs; atol=1e-9)

    # Complex non-Hermitian input should throw
    A_nonherm = [1.0 + 0im 2.0 + 1.0im; 3.0 - 1.0im 4.0 + 0im]
    B_complex = Matrix{ComplexF64}(I, 2, 2)
    @test_throws ArgumentError feast(A_nonherm, B_complex, (0.0, 5.0), M0=2, fpm=copy(fpm), parallel=:serial)

    # Sparse complex Hermitian standard problem
    @info "Dispatch: sparse Hermitian standard"
    v = ComplexF64[0.1 + 0.2im, -0.05 + 0.1im]
    diag_vals = ComplexF64[2.0, 3.0, 4.0]
    A_sparse = spdiagm(-1 => conj.(v), 0 => diag_vals, 1 => v)
    result_sparse = feast(A_sparse, (1.5, 4.5), M0=size(A_sparse, 1), fpm=copy(fpm), parallel=:serial)
    sparse_eigs = sort(real.(eigvals(Matrix(A_sparse))))
    @test result_sparse.info == 0
    @test result_sparse.M == size(A_sparse, 1)
    @test isapprox(sort(result_sparse.lambda), sparse_eigs; atol=1e-9)
end
