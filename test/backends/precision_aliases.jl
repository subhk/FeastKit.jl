include(joinpath(@__DIR__, "..", "support", "setup.jl"))

@testset "PFEAST-compatible precision aliases" begin
    required_parallel_aliases = [
        :psfeast_syev!, :pdfeast_syev!,
        :psfeast_sygv!, :pdfeast_sygv!,
        :psfeast_scsrev!, :pdfeast_scsrev!,
        :psfeast_scsrgv!, :pdfeast_scsrgv!,
        :psfeast_srci!, :pdfeast_srci!
    ]
    @test all(name -> isdefined(FeastKit, name), required_parallel_aliases)

    n = 4
    A64 = Matrix(Diagonal([0.5, 1.0, 1.5, 3.0]))
    B64 = Matrix{Float64}(I, n, n)
    fpm64 = zeros(Int, 64)
    feastinit!(fpm64)
    fpm64[1] = 0
    fpm64[2] = 8
    fpm64[4] = 12

    dense_standard = if nworkers() == 1
        @test_logs (:warn, "No worker processes available, falling back to serial computation") begin
            FeastKit.pdfeast_syev!(copy(A64), 0.4, 1.6, n, copy(fpm64);
                                   use_threads=false)
        end
    else
        FeastKit.pdfeast_syev!(copy(A64), 0.4, 1.6, n, copy(fpm64);
                               use_threads=false)
    end
    @test dense_standard.info == 0
    @test dense_standard.M == 3
    @test isapprox(sort(dense_standard.lambda), [0.5, 1.0, 1.5]; atol=1e-8)

    dense_generalized = if nworkers() == 1
        @test_logs (:warn, "No worker processes available, falling back to serial computation") begin
            FeastKit.pdfeast_sygv!(copy(A64), copy(B64), 0.4, 1.6, n, copy(fpm64);
                                   use_threads=false)
        end
    else
        FeastKit.pdfeast_sygv!(copy(A64), copy(B64), 0.4, 1.6, n, copy(fpm64);
                               use_threads=false)
    end
    @test dense_generalized.info == dense_standard.info
    @test dense_generalized.M == dense_standard.M
    @test isapprox(sort(dense_generalized.lambda), sort(dense_standard.lambda); atol=1e-8)

    A_sparse = sparse(A64)
    B_sparse = sparse(B64)
    sparse_standard = FeastKit.pdfeast_scsrev!(copy(A_sparse), 0.4, 1.6, n, copy(fpm64);
                                               use_threads=false)
    @test sparse_standard.info == 0
    @test sparse_standard.M == 3
    @test isapprox(sort(sparse_standard.lambda), [0.5, 1.0, 1.5]; atol=1e-8)

    sparse_generalized = FeastKit.pdfeast_scsrgv!(copy(A_sparse), copy(B_sparse),
                                                  0.4, 1.6, n, copy(fpm64);
                                                  use_threads=false)
    @test sparse_generalized.info == sparse_standard.info
    @test sparse_generalized.M == sparse_standard.M
    @test isapprox(sort(sparse_generalized.lambda), sort(sparse_standard.lambda); atol=1e-8)

    A_sparse32 = sparse(Float32.(A64))
    B_sparse32 = spdiagm(0 => ones(Float32, n))
    fpm32 = copy(fpm64)
    Random.seed!(1234)
    randn(Float64, 64)
    sparse_single_first = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                                   0.4f0, 1.6f0, n, copy(fpm32);
                                                   use_threads=false)
    Random.seed!(5678)
    randn(Float64, 64)
    sparse_single_second = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                                    0.4f0, 1.6f0, n, copy(fpm32);
                                                    use_threads=false)
    @test sparse_single_first.info == sparse_single_second.info
    @test sparse_single_first.M == sparse_single_second.M
    @test sort(Float64.(sparse_single_first.lambda)) == sort(Float64.(sparse_single_second.lambda))

    sparse_single = FeastKit.psfeast_scsrgv!(copy(A_sparse32), copy(B_sparse32),
                                             0.4f0, 1.6f0, n, copy(fpm32);
                                             use_threads=false)
    @test sparse_single.info == 0
    @test sparse_single.M == 3
    @test isapprox(sort(Float64.(sparse_single.lambda)), [0.5, 1.0, 1.5]; atol=1e-5)

    state = ParallelFeastState{Float64}(fpm64[2], n, false, false)
    work = Matrix{Float64}(undef, n, n)
    workc = Matrix{ComplexF64}(undef, n, n)
    Aq = Matrix{Float64}(undef, n, n)
    Sq = Matrix{Float64}(undef, n, n)
    lambda = Vector{Float64}(undef, n)
    q = Matrix{Float64}(undef, n, n)
    res = Vector{Float64}(undef, n)
    FeastKit.pdfeast_srci!(state, n, work, workc, Aq, Sq, copy(fpm64),
                           0.4, 1.6, n, lambda, q, res)
    @test state.info == Int(Feast_SUCCESS)
    @test state.ijob == Int(Feast_RCI_FACTORIZE)
end
