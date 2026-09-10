using Test, FeastKit, LinearAlgebra, SparseArrays, Logging

@testset "Serial review regressions" begin
    @testset "Scale-invariant custom contour $T scale=$scale" for T in (Float32, Float64),
            scale in (T(1e-8), T(1e-16), one(T), T(1e8))
        f = feastinit().fpm
        center, radius = Complex{T}(T(1.5)*scale), T(0.75)*scale
        contour = feast_gcontour(center, radius, f)
        @test FeastKit._feast_inside_polygon(Complex{T}(scale), contour.Zne)
        @test !FeastKit._feast_inside_polygon(Complex{T}(T(2.4)*scale), contour.Zne)
        @test FeastKit._feast_inside_polygon(contour.Zne[1], contour.Zne)
        # The scale of the eigenproblem and the contour must not change which
        # eigenvalues are selected. Suppress the legacy near-node warning for
        # very small (but distinct) quadrature nodes.
        for storage in (Matrix, sparse)
            A = storage(Matrix(Diagonal(Complex{T}.(scale .* T[1,2,2.4]))))
            result = with_logger(NullLogger()) do
                FeastKit.with_custom_contour(f, contour) do
                    feast_general(A, center, radius; M0=3, fpm=f)
                end
            end
            @test result.info == 0
            @test result.M == 2
            @test result.M == 2 && isapprox(sort(real.(result.lambda ./ scale)), T[1,2]; rtol=1e-5)
        end
    end

    @testset "Translated and extreme-scale polygon geometry" begin
        for scale in (1e-200, 1.0, 1e200)
            nodes = scale .* ComplexF64[-1-im, 1-im, 1+im, -1+im]
            @test FeastKit._feast_inside_polygon(0.0+0im, nodes)
            @test !FeastKit._feast_inside_polygon(2scale+0im, nodes)
        end
        nodes = ComplexF64[1e8-im, 1e8+2-im, 1e8+2+im, 1e8+im]
        @test FeastKit._feast_inside_polygon(1e8+1+0im, nodes)
        @test !FeastKit._feast_inside_polygon(1e8+1+2im, nodes)
    end

    @testset "Matrix-free low-level default tolerance $T" for T in (Float32, Float64)
        A = Matrix(Diagonal(T[1,2,3]))
        B = Matrix{T}(I,3,3)
        ao = LinearOperator{T}((y,x)->mul!(y,A,x),(3,3); issymmetric=true)
        bo = LinearOperator{T}((y,x)->copyto!(y,x),(3,3); issymmetric=true)
        callback = (Y,z,X)->copyto!(Y,(z*B-A)\X)
        result = try
            feast_matfree_srci!(ao,bo,(T(0.5),T(2.5)),3;linear_solver=callback)
        catch err
            err
        end
        @test result isa FeastResult
        if result isa FeastResult
            @test result.info == 0
            @test result.lambda ≈ T[1,2] rtol=1e-5
        end
        az = LinearOperator{Complex{T}}((y,x)->mul!(y,A,x),(3,3))
        bz = LinearOperator{Complex{T}}((y,x)->copyto!(y,x),(3,3))
        result = try
            feast_matfree_grci!(az,bz,Complex{T}(1.5),T(0.75),3;linear_solver=callback)
        catch err
            err
        end
        @test result isa FeastGeneralResult
        if result isa FeastGeneralResult
            @test result.info == 0
            @test sort(real.(result.lambda)) ≈ T[1,2] rtol=1e-5
        end
    end

    @testset "Dense general no-storage policy refactorizes across sweeps" begin
        n = 30
        A = Matrix(Diagonal(ComplexF64.(1:n)))
        function solve(cache)
            f = feastinit().fpm
            f[10] = cache; f[3] = 16; f[4] = 3; f[8] = 4; f[16] = 1
            return feast_geev!(A,2.0+0im,1.5,5,f)
        end
        cached, uncached = solve(1), solve(0) # warm compilation for both modes
        @test cached.loop > 0
        @test uncached.loop == cached.loop
        @test uncached.info == cached.info
        @test uncached.lambda ≈ cached.lambda
        cached_bytes = @allocated solve(1)
        uncached_bytes = @allocated solve(0)
        # Discarding factors requires fresh LU storage on later sweeps, unlike
        # retaining one LU per node. Keep the bound below one full extra sweep.
        @test uncached_bytes > cached_bytes + n*n*sizeof(ComplexF64)
    end
end
