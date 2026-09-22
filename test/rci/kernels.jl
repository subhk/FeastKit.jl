include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "Iterative RCI kernels" begin
    @test isdefined(FeastKit, :ifeast_srci!)
    @test isdefined(FeastKit, :ifeast_hrci!)
    @test isdefined(FeastKit, :ifeast_grci!)

    n = 4
    m0 = 3
    fpm_rci = zeros(Int, 64)
    feastinit!(fpm_rci)
    fpm_rci[1] = 0

    ijob = Ref(-1)
    ze = Ref(0.0 + 0.0im)
    epsout = Ref(0.0)
    loop = Ref(0)
    mode = Ref(0)
    info = Ref(0)
    work = Matrix{Float64}(undef, n, m0)
    workc = Matrix{ComplexF64}(undef, n, m0)
    Aq = Matrix{Float64}(undef, m0, m0)
    Sq = Matrix{Float64}(undef, m0, m0)
    lambda = Vector{Float64}(undef, m0)
    q = Matrix{Float64}(undef, n, m0)
    res = Vector{Float64}(undef, m0)
    FeastKit.ifeast_srci!(ijob, n, ze, work, workc, Aq, Sq, copy(fpm_rci),
                          epsout, loop, 0.0, 2.0, m0, lambda, q, mode, res, info)
    @test info[] == Int(Feast_SUCCESS)
    @test ijob[] == Int(Feast_RCI_FACTORIZE)

    ijob[] = -1
    zAq = Matrix{ComplexF64}(undef, m0, m0)
    zSq = Matrix{ComplexF64}(undef, m0, m0)
    qh = Matrix{ComplexF64}(undef, n, m0)
    FeastKit.ifeast_hrci!(ijob, n, ze, work, workc, zAq, zSq, copy(fpm_rci),
                          epsout, loop, 0.0, 2.0, m0, lambda, qh, mode, res, info)
    @test info[] == Int(Feast_SUCCESS)
    @test ijob[] == Int(Feast_RCI_FACTORIZE)

    ijob[] = -1
    lambdag = Vector{ComplexF64}(undef, m0)
    qg = Matrix{ComplexF64}(undef, n, m0)
    FeastKit.ifeast_grci!(ijob, n, ze, work, workc, zAq, zSq, copy(fpm_rci),
                          epsout, loop, 1.0 + 0.0im, 2.0, m0, lambdag, qg,
                          mode, res, info)
    @test info[] == Int(Feast_SUCCESS)
    @test ijob[] == Int(Feast_RCI_FACTORIZE)
end
