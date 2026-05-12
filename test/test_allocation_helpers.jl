using Test
using FeastKit
using LinearAlgebra

function _repeat_reorder!(lambda, vectors, lambda_src, vectors_src,
                          perm, lambda_tmp, vector_tmp, Emin, Emax, nrepeat)
    for _ in 1:nrepeat
        copyto!(lambda, lambda_src)
        copyto!(vectors, vectors_src)
        FeastKit._feast_reorder_by_interval!(lambda, vectors, perm,
                                              lambda_tmp, vector_tmp,
                                              Emin, Emax, length(lambda))
    end
    return lambda, vectors
end

function _repeat_copy_real!(dest, src, nrepeat)
    for _ in 1:nrepeat
        FeastKit._feast_copy_real!(dest, src)
    end
    return dest
end

@testset "Allocation helper regressions" begin
    @testset "In-place interval reorder" begin
        lambda_src = [4.0, 2.0, 1.0, 3.0]
        vectors_src = ComplexF64[
            11 12 13 14
            21 22 23 24
            31 32 33 34
        ]
        lambda = copy(lambda_src)
        vectors = copy(vectors_src)
        perm = Vector{Int}(undef, length(lambda))
        lambda_tmp = similar(lambda)
        vector_tmp = similar(vectors)

        m = FeastKit._feast_reorder_by_interval!(lambda, vectors, perm,
                                                 lambda_tmp, vector_tmp,
                                                 1.5, 3.5, length(lambda))

        @test m == 2
        @test lambda == [2.0, 3.0, 4.0, 1.0]
        @test vectors == vectors_src[:, [2, 4, 1, 3]]

        _repeat_reorder!(lambda, vectors, lambda_src, vectors_src, perm,
                         lambda_tmp, vector_tmp, 1.5, 3.5, 1)
        bytes = @allocated _repeat_reorder!(lambda, vectors, lambda_src, vectors_src,
                                            perm, lambda_tmp, vector_tmp, 1.5, 3.5,
                                            1_000)
        @test bytes < 1024
    end

    @testset "In-place real copy" begin
        src = ComplexF64[
            1 + 2im 3 + 4im
            5 + 6im 7 + 8im
        ]
        dest = zeros(Float64, size(src))

        FeastKit._feast_copy_real!(dest, src)
        @test dest == real.(src)

        bytes = @allocated _repeat_copy_real!(dest, src, 1_000)
        @test bytes < 1024
    end

    @testset "Matrix-free complex workspace has RHS scratch" begin
        workspace = allocate_matfree_workspace(ComplexF64, 8, 3)
        @test hasproperty(workspace, :rhs)
        @test size(workspace.rhs) == (8, 3)
        @test eltype(workspace.rhs) === ComplexF64
    end
end
