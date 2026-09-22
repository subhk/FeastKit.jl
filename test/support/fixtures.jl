# Shared deterministic fixtures for API and worker/MPI tests.
module FeastTestFixtures
using FeastKit, LinearAlgebra, SparseArrays

function backend_problem(n; storage=Matrix)
    A = storage(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    B = storage(Matrix{Float64}(I, n, n))
    fpm = feastinit().fpm
    fpm[1] = 0
    fpm[2] = 8
    fpm[4] = 20
    return A, B, fpm
end
end
