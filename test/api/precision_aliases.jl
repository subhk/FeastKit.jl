include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "FEAST-compatible precision aliases" begin
    required_aliases = [
        :sfeast_syev!, :dfeast_syev!, :dfeast_syevx!,
        :cfeast_heev!, :zfeast_heev!, :zfeast_geev!,
        :dfeast_scsrev!, :zfeast_hcsrev!,
        :dfeast_sbev!, :zfeast_hbev!,
        :cfeast_sbev!, :zfeast_sbev!
    ]
    @test all(name -> isdefined(FeastKit, name), required_aliases)

    n = 5
    A64 = Matrix{Float64}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    fpm64 = zeros(Int, 64)
    feastinit!(fpm64)
    fpm64[1] = 0

    generic_dense = feast_syev!(copy(A64), 0.1, 3.9, n, copy(fpm64))
    alias_dense = FeastKit.dfeast_syev!(copy(A64), 0.1, 3.9, n, copy(fpm64))
    @test alias_dense.info == generic_dense.info
    @test alias_dense.M == generic_dense.M
    @test isapprox(sort(alias_dense.lambda), sort(generic_dense.lambda); atol=1e-10)

    contour = feast_contour(0.1, 3.9, copy(fpm64))
    alias_dense_x = FeastKit.dfeast_syevx!(copy(A64), 0.1, 3.9, n, copy(fpm64),
                                           contour.Zne, contour.Wne)
    @test alias_dense_x.info == generic_dense.info
    @test isapprox(sort(alias_dense_x.lambda), sort(generic_dense.lambda); atol=1e-10)

    A32 = Matrix{Float32}(A64)
    fpm32 = zeros(Int, 64)
    feastinit!(fpm32)
    fpm32[1] = 0
    alias_single = FeastKit.sfeast_syev!(copy(A32), 0.1f0, 3.9f0, n, copy(fpm32))
    @test alias_single.info == 0
    @test alias_single.M == n

    A_complex = Matrix(Diagonal(ComplexF64[-1.0, 0.25, 1.0, 2.0, 3.0]))
    generic_h = feast_heev!(copy(A_complex), -1.5, 2.5, n, copy(fpm64))
    alias_h = FeastKit.zfeast_heev!(copy(A_complex), -1.5, 2.5, n, copy(fpm64))
    @test alias_h.info == generic_h.info
    @test alias_h.M == generic_h.M
    @test isapprox(sort(alias_h.lambda), sort(generic_h.lambda); atol=1e-10)

    A_complex32 = ComplexF32.(A_complex)
    alias_c = FeastKit.cfeast_heev!(copy(A_complex32), -1.5f0, 2.5f0, n, copy(fpm32))
    @test alias_c.info == 0
    @test alias_c.M == 4

    A_general = Matrix(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 - 0.2im, 2.0, 3.0 + 0.3im, 4.0]))
    generic_g = feast_geev!(copy(A_general), 2.0 + 0.0im, 3.0, n, copy(fpm64))
    alias_g = FeastKit.zfeast_geev!(copy(A_general), 2.0 + 0.0im, 3.0, n, copy(fpm64))
    @test alias_g.info == generic_g.info
    @test alias_g.M == generic_g.M
    @test isapprox(sort(real.(alias_g.lambda)), sort(real.(generic_g.lambda)); atol=1e-10)

    A_sparse = sparse(A64)
    generic_sparse = feast_scsrev!(copy(A_sparse), 0.1, 3.9, n, copy(fpm64))
    alias_sparse = FeastKit.dfeast_scsrev!(copy(A_sparse), 0.1, 3.9, n, copy(fpm64))
    @test alias_sparse.info == generic_sparse.info
    @test alias_sparse.M == generic_sparse.M
    @test isapprox(sort(alias_sparse.lambda), sort(generic_sparse.lambda); atol=1e-10)

    A_sparse_h = sparse(A_complex)
    generic_sparse_h = feast_hcsrev!(copy(A_sparse_h), -1.5, 2.5, n, copy(fpm64))
    alias_sparse_h = FeastKit.zfeast_hcsrev!(copy(A_sparse_h), -1.5, 2.5, n, copy(fpm64))
    @test alias_sparse_h.info == generic_sparse_h.info
    @test alias_sparse_h.M == generic_sparse_h.M
    @test isapprox(sort(alias_sparse_h.lambda), sort(generic_sparse_h.lambda); atol=1e-10)

    A_band = full_to_banded(A64, 1)
    generic_band = feast_sbev!(copy(A_band), 1, 0.1, 3.9, n, copy(fpm64))
    alias_band = FeastKit.dfeast_sbev!(copy(A_band), 1, 0.1, 3.9, n, copy(fpm64))
    @test alias_band.info == generic_band.info
    @test alias_band.M == generic_band.M
    @test isapprox(sort(alias_band.lambda), sort(generic_band.lambda); atol=1e-10)

    A_band_h = full_to_banded(A_complex, 0)
    generic_band_h = feast_hbev!(copy(A_band_h), 0, -1.5, 2.5, n, copy(fpm64))
    alias_band_h = FeastKit.zfeast_hbev!(copy(A_band_h), 0, -1.5, 2.5, n, copy(fpm64))
    @test alias_band_h.info == generic_band_h.info
    @test alias_band_h.M == generic_band_h.M
    @test isapprox(sort(alias_band_h.lambda), sort(generic_band_h.lambda); atol=1e-10)

    A_band_cs = full_to_banded(Matrix(Diagonal(ComplexF64[0.5 + 0.1im, 1.0 - 0.1im,
                                                          1.5 + 0.2im, 2.0, 3.0])), 0)
    generic_band_cs = feast_sbev_complex!(copy(A_band_cs), 0, 1.2 + 0.0im, 1.2, n, copy(fpm64))
    alias_band_cs = FeastKit.zfeast_sbev!(copy(A_band_cs), 0, 1.2 + 0.0im, 1.2, n, copy(fpm64))
    @test alias_band_cs.info == generic_band_cs.info
    @test alias_band_cs.M == generic_band_cs.M
    @test isapprox(sort(real.(alias_band_cs.lambda)), sort(real.(generic_band_cs.lambda)); atol=1e-10)
end

@testset "Polynomial IFEAST precision aliases" begin
    required_poly_ifeast_aliases = [
        :sifeast_srcipev!, :difeast_srcipev!,
        :sifeast_srcipevx!, :difeast_srcipevx!,
        :cifeast_grcipev!, :zifeast_grcipev!,
        :cifeast_grcipevx!, :zifeast_grcipevx!,
        :sifeast_scsrpev!, :difeast_scsrpev!,
        :sifeast_scsrpevx!, :difeast_scsrpevx!,
        :cifeast_hcsrpev!, :zifeast_hcsrpev!,
        :cifeast_hcsrpevx!, :zifeast_hcsrpevx!,
        :cifeast_gcsrpev!, :zifeast_gcsrpev!,
        :cifeast_gcsrpevx!, :zifeast_gcsrpevx!
    ]
    aliases_available = all(name -> isdefined(FeastKit, name), required_poly_ifeast_aliases)
    @test aliases_available

    if aliases_available
        n = 3
        fpm_poly = zeros(Int, 64)
        feastinit!(fpm_poly)
        fpm_poly[1] = 0
        fpm_poly[4] = 8
        fpm_poly[8] = 16

        # P(λ) = λ²I - diag(1, 4, 9), so the spectrum is ±1, ±2, ±3.
        A0 = Matrix(Diagonal([-1.0, -4.0, -9.0]))
        A1 = zeros(n, n)
        A2 = Matrix{Float64}(I, n, n)
        coeffs_real = [A0, A1, A2]
        coeffs_complex = [ComplexF64.(A) for A in coeffs_real]
        center = 0.0 + 0.0im
        radius = 4.0

        # The moment-based RCI kernel needs a contour that is not symmetric
        # about the origin. Enclosing ±k in equal measure makes the residues
        # of the two cancel, A0 vanishes, and the reduced pencil is singular
        # -- every routine then returns M = 0 and an alias-vs-generic
        # comparison holds trivially. Target 2 and 3 instead, and assert the
        # values, so this block fails if the kernel regresses.
        rci_fpm = copy(fpm_poly)
        rci_fpm[8] = 32
        rci_fpm[16] = 1     # trapezoidal, the accurate rule on a circle
        rci_center, rci_radius = 2.5 + 0.0im, 1.0
        rci_expected = [2.0, 3.0]

        generic_real = feast_srcipev!(coeffs_real, 2, rci_center, rci_radius, n, copy(rci_fpm))
        @test generic_real.info == 0
        @test generic_real.M == 2
        @test isapprox(sort(real.(generic_real.lambda)), rci_expected; atol=1e-8)

        alias_real = FeastKit.difeast_srcipev!(coeffs_real, 2, rci_center, rci_radius, n, copy(rci_fpm))
        @test alias_real.info == generic_real.info
        @test alias_real.M == generic_real.M
        @test isapprox(sort(real.(alias_real.lambda)), sort(real.(generic_real.lambda)); atol=1e-10)

        contour = feast_gcontour(rci_center, rci_radius, copy(rci_fpm))
        alias_real_x = FeastKit.difeast_srcipevx!(coeffs_real, 2, rci_center, rci_radius,
                                                  n, copy(rci_fpm),
                                                  contour.Zne, contour.Wne)
        @test alias_real_x.info == generic_real.info
        @test isapprox(sort(real.(alias_real_x.lambda)), sort(real.(generic_real.lambda)); atol=1e-10)

        generic_complex = feast_grcipev!(coeffs_complex, 2, rci_center, rci_radius, n, copy(rci_fpm))
        @test generic_complex.M == 2
        @test isapprox(sort(real.(generic_complex.lambda)), rci_expected; atol=1e-8)

        alias_complex = FeastKit.zifeast_grcipev!(coeffs_complex, 2, rci_center, rci_radius,
                                                  n, copy(rci_fpm))
        @test alias_complex.info == generic_complex.info
        @test alias_complex.M == generic_complex.M
        @test isapprox(sort(real.(alias_complex.lambda)), sort(real.(generic_complex.lambda)); atol=1e-10)

        sparse_real = sparse.(coeffs_real)
        generic_sparse = feast_scsrpev!(sparse_real, 2, center, radius, n, copy(fpm_poly))
        alias_sparse = FeastKit.difeast_scsrpev!(sparse_real, 2, center, radius, n, copy(fpm_poly))
        @test alias_sparse.info == generic_sparse.info
        @test alias_sparse.M == generic_sparse.M

        sparse_complex = sparse.(coeffs_complex)
        generic_h_sparse = feast_hcsrpev!(sparse_complex, 2, center, radius, n, copy(fpm_poly))
        alias_h_sparse = FeastKit.zifeast_hcsrpev!(sparse_complex, 2, center, radius, n, copy(fpm_poly))
        @test alias_h_sparse.info == generic_h_sparse.info
        @test alias_h_sparse.M == generic_h_sparse.M
    end
end
