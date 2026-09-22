include(joinpath(@__DIR__, "..", "support", "setup.jl"))

if get(ENV, "FEAST_RUN_PARALLEL_TESTS", "false") == "true"
    @testset "Parallel support" begin
        # Test parallel state creation
        state = ParallelFeastState{Float64}(8, 10, true, true)
        @test state.use_parallel == true
        @test state.use_threads == true
        @test state.total_points == 8
        @test length(state.moment_contributions) == 8

        # Test contour point distribution
        ne = 16
        nw = 4
        chunks = distribute_contour_points(ne, nw)
        @test length(chunks) == nw
        @test sum(length(chunk) for chunk in chunks) == ne

        # Test backend determination (basic test only)
        backend = determine_parallel_backend(:auto, nothing)
        @test backend in [:serial, :threads, :distributed, :mpi]

        # Test parallel capabilities check
        caps = feast_parallel_capabilities()
        @test isa(caps, Dict)
        @test haskey(caps, :threads)
        @test haskey(caps, :distributed)
        @test haskey(caps, :mpi)

        if get(ENV, "FEASTKIT_TEST_PARALLEL", "false") == "true"
            @info "Parallel execution tests enabled"
            n = 10
            A = diagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))
            B = Matrix{Float64}(I, n, n)

            if Threads.nthreads() > 1
                reference = eigvals(Symmetric(A))
                inside = filter(lambda -> 0.5 <= lambda <= 2.5, reference)
                result = feast(A, B, (0.5, 2.5), M0=5, parallel=:threads)
                @test isa(result, FeastResult)
                @test result.M == length(inside)
                @test isapprox(sort(result.lambda[1:result.M]), sort(inside); atol=1e-7)
            end
        else
            @info "Parallel execution tests disabled (set FEASTKIT_TEST_PARALLEL=true to enable)"
        end
    end
else
    @info "Skipping parallel support tests (set FEAST_RUN_PARALLEL_TESTS=true to enable)"
end

if get(ENV, "FEAST_RUN_PARALLEL_TESTS", "false") == "true"
    @testset "Threaded vs Serial comparison" begin
        if get(ENV, "FEASTKIT_TEST_PARALLEL", "false") == "true"
            @info "Threaded vs Serial comparison enabled"
            if Threads.nthreads() > 1
                n = 20
                A = diagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))

                fpm = zeros(Int, 64)
                feastinit!(fpm)
                fpm[1] = 0
                fpm[2] = 4

                # The threaded backend must match serial, and both must
                # match a dense reference. `M >= 0` asserted nothing.
                reference = eigvals(Symmetric(A))
                inside = filter(lambda -> 0.5 <= lambda <= 1.5, reference)
                result_serial = feast(A, (0.5, 1.5), M0=6, fpm=copy(fpm), parallel=:serial)
                result_parallel = feast(A, (0.5, 1.5), M0=6, fpm=copy(fpm), parallel=:threads)
                @test result_serial.M == length(inside)
                @test result_parallel.M == result_serial.M
                @test isapprox(sort(result_serial.lambda[1:result_serial.M]),
                               sort(inside); atol=1e-7)
                @test isapprox(sort(result_parallel.lambda[1:result_parallel.M]),
                               sort(result_serial.lambda[1:result_serial.M]); atol=1e-7)
            else
                @info "Skipping (only 1 thread available)"
            end
        else
            @info "Threaded vs Serial comparison disabled (set FEASTKIT_TEST_PARALLEL=true to enable)"
        end
    end
else
    @info "Skipping threaded vs serial comparison (set FEAST_RUN_PARALLEL_TESTS=true to enable)"
end

@testset "Parallel backend selection" begin
    # Test parallel backend determination logic

    # Test explicit backend selection
    @test_throws ArgumentError determine_parallel_backend(:invalid, nothing)

    # Test auto selection logic
    backend = determine_parallel_backend(:auto, nothing)
    @test backend in [:serial, :threads, :distributed, :mpi]

    # Test with different thread/worker configurations
    if Threads.nthreads() > 1
        backend_threads = determine_parallel_backend(:threads, nothing)
        @test backend_threads == :threads
    else
        backend_threads = determine_parallel_backend(:threads, nothing)
        @test backend_threads == :serial
    end

    if Distributed.nworkers() > 1
        backend_dist = determine_parallel_backend(:distributed, nothing)
        @test backend_dist == :distributed
    else
        backend_dist = determine_parallel_backend(:distributed, nothing)
        @test backend_dist == :serial
    end
end
