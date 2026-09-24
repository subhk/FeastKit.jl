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
    # Before the first sweep the kernel measures the pencil's spectral scale
    # with one MULT_A/MULT_B pair on the probe columns of q.
    @test ijob[] == Int(Feast_RCI_MULT_A)
    @test mode[] == m0

    ijob[] = -1
    zAq = Matrix{ComplexF64}(undef, m0, m0)
    zSq = Matrix{ComplexF64}(undef, m0, m0)
    qh = Matrix{ComplexF64}(undef, n, m0)
    FeastKit.ifeast_hrci!(ijob, n, ze, work, workc, zAq, zSq, copy(fpm_rci),
                          epsout, loop, 0.0, 2.0, m0, lambda, qh, mode, res, info)
    @test info[] == Int(Feast_SUCCESS)
    @test ijob[] == Int(Feast_RCI_MULT_A)
    @test mode[] == m0

    ijob[] = -1
    lambdag = Vector{ComplexF64}(undef, m0)
    qg = Matrix{ComplexF64}(undef, n, m0)
    FeastKit.ifeast_grci!(ijob, n, ze, work, workc, zAq, zSq, copy(fpm_rci),
                          epsout, loop, 1.0 + 0.0im, 2.0, m0, lambdag, qg,
                          mode, res, info)
    @test info[] == Int(Feast_SUCCESS)
    @test ijob[] == Int(Feast_RCI_MULT_A)
    @test mode[] == m0
end

@testset "RCI startup measures the spectral scale" begin
    # A caller that answers MULT_A/MULT_B generically needs no changes: the
    # extra pair arrives before the first FACTORIZE.
    n, m0 = 12, 6
    A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    loops = Int[]
    for s in (1e-9, 1.0, 1e9)
        As = s * A
        fpm = feastinit().fpm
        ijob = Ref(-1); ze = Ref(0.0 + 0.0im); epsout = Ref(0.0); loop = Ref(0)
        mode = Ref(0); info = Ref(0)
        work = zeros(n, m0); workc = zeros(ComplexF64, n, m0)
        Aq = zeros(m0, m0); Sq = zeros(m0, m0)
        lambda = zeros(m0); q = zeros(n, m0); res = zeros(m0)
        state = FeastSRCIState{Float64}()
        jobs = Int[]
        F = nothing
        while true
            feast_srci!(ijob, n, ze, work, workc, Aq, Sq, fpm, epsout, loop,
                        0.5s, 1.5s, m0, lambda, q, mode, res, info; state=state)
            push!(jobs, ijob[])
            ijob[] == Int(Feast_RCI_DONE) && break
            if ijob[] == Int(Feast_RCI_FACTORIZE)
                F = lu(ze[] * I - As)
            elseif ijob[] == Int(Feast_RCI_SOLVE)
                workc .= F \ ComplexF64.(work)
            elseif ijob[] == Int(Feast_RCI_MULT_A)
                work[:, 1:mode[]] .= As * q[:, 1:mode[]]
            elseif ijob[] == Int(Feast_RCI_MULT_B)
                work[:, 1:mode[]] .= q[:, 1:mode[]]
            end
        end
        @test jobs[1:3] == [Int(Feast_RCI_MULT_A), Int(Feast_RCI_MULT_B), Int(Feast_RCI_FACTORIZE)]
        expected = filter(x -> 0.5 < x < 1.5, eigvals(Symmetric(A)))
        @test info[] == 0
        @test mode[] == length(expected)
        @test sort(lambda[1:mode[]]) ./ s ≈ expected atol=1e-10
        push!(loops, loop[])
    end
    # The residual floor tracks the pencil, so a change of units cannot change
    # how many refinement loops convergence takes.
    @test allequal(loops)
end
