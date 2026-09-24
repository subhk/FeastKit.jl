using Test, FeastKit, LinearAlgebra, SparseArrays, Random

@testset "Checked convenience wrappers" begin
    @testset "Successful $T $storage generalized=$generalized" for T in (Float64, ComplexF64),
            storage in (Matrix, sparse), generalized in (false, true)
        mass = generalized ? [2.0, 3.0, 4.0, 5.0] : ones(4)
        A = storage(Matrix(Diagonal(T.(mass .* [1.0, 2.0, 3.0, 4.0]))))
        B = storage(Matrix(Diagonal(T.(mass))))
        args = generalized ? (A, B, (0.5, 2.5)) : (A, (0.5, 2.5))
        for check in (true, false)
            values = eigvals_feast(args...; subspace_size=3, tol=1e-10, check=check)
            decomposition = eigen_feast(args...; subspace_size=3, tol=1e-10, check=check)
            @test values isa Vector
            @test values ≈ [1.0, 2.0] atol=1e-9
            @test decomposition isa Eigen
            @test decomposition.values ≈ values atol=1e-9
            @test norm(A * decomposition.vectors - B * decomposition.vectors * Diagonal(values)) < 1e-8
        end
    end

    @testset "Saturation rejects incomplete eigenpairs $T generalized=$generalized" for
            T in (Float64, ComplexF64), generalized in (false, true)
        B = Matrix{T}(I, 4, 4) .* (generalized ? 2 : 1)
        A = copy(B)
        args = generalized ? (A, B, (0.5, 1.5)) : (A, (0.5, 1.5))
        result = feast(args...; subspace_size=1)
        @test result.info == Int(Feast_ERROR_M0)
        @test maximum(result.res) < 1e-10
        for wrapper in (eigvals_feast, eigen_feast)
            err = try
                wrapper(args...; subspace_size=1, check=true)
            catch e
                e
            end
            @test err isa ErrorException
            if err isa ErrorException
                message = sprint(showerror, err)
                @test occursin("info=2", message)
                @test occursin("subspace_size", message)
                @test occursin("feast(...)", message)
            end
            # Both unchecked forms preserve the previous return types and
            # partial values, and warn that the values are unverified.
            for keywords in ((;), (; check=false))
                partial = @test_logs (:warn, r"did not converge \(info=2\)") wrapper(
                    args...; subspace_size=1, keywords...)
                @test partial isa (wrapper === eigvals_feast ? Vector : Eigen)
                values = wrapper === eigvals_feast ? partial : partial.values
                @test values ≈ [1.0] atol=1e-10
            end
        end
    end

    @testset "Iteration limits and existing exceptions" begin
        A = Matrix(Diagonal(collect(1.0:12.0)))
        B = Matrix{Float64}(I, 12, 12)
        options = (; subspace_size=4, quadrature_points=3, maxiter=1, tol=1e-16)
        for args in ((A, (0.5, 2.5)), (A, B, (0.5, 2.5)))
            Random.seed!(42)
            @test feast(args...; options...).info == Int(Feast_ERROR_NO_CONVERGENCE)
            for wrapper in (eigvals_feast, eigen_feast)
                Random.seed!(42)
                err = try
                    wrapper(args...; options..., check=true)
                catch e
                    e
                end
                @test err isa ErrorException
                if err isa ErrorException
                    @test occursin("info=5", sprint(showerror, err))
                    @test occursin("maxiter", sprint(showerror, err))
                end
                for check in (false, true)
                    @test_throws ArgumentError wrapper(args...; subspace_size=0, check=check)
                end
            end
        end
    end

    @testset "Published example" begin
        source = read(joinpath(@__DIR__, "..", "..", "docs", "src", "api_reference.md"), String)
        block = match(r"(?ms)^```@example checked_wrappers\r?\n(.*?)^```", source)
        @test block !== nothing
        sandbox = Module(gensym(:CheckedWrappersDocs))
        Base.include_string(sandbox, block.captures[1])
        @test getfield(sandbox, :values) ≈ [1.0, 2.0]
        @test getfield(sandbox, :decomposition) isa Eigen
    end
end
