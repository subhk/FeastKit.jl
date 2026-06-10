# Quick allocation sanity check for the optimized hot paths.
# Run: julia --project=. -t 4 benchmark/alloc_check.jl
using FeastKit, LinearAlgebra, SparseArrays

function dense_problem(N)
    A = SymTridiagonal(2.0 * ones(N), -1.0 * ones(N - 1))
    return Matrix(A), Matrix{Float64}(I, N, N)
end

function sparse_problem(N)
    A = spdiagm(-1 => -ones(N - 1), 0 => 2.0 * ones(N), 1 => -ones(N - 1))
    B = sparse(1.0I, N, N)
    return A, B
end

function run_case(label, f)
    f()  # warm up / compile
    GC.gc()
    stats = @timed f()
    println(rpad(label, 34), " time=", round(stats.time; digits=3),
            "s  alloc=", round(stats.bytes / 1024^2; digits=1), " MiB")
    return stats.bytes
end

N = 600
M0 = 12
λref = eigvals(dense_problem(N)[1])
Emin, Emax = 0.0, λref[10] + 1e-8   # capture ~10 eigenvalues

Ad, Bd = dense_problem(N)
As, Bs = sparse_problem(N)

println("threads = ", Threads.nthreads())

run_case("pfeast_sygv! (dense threaded)", () -> begin
    fpm = zeros(Int, 64); feastinit!(fpm)
    pfeast_sygv!(copy(Ad), copy(Bd), Emin, Emax, M0, fpm)
end)

run_case("pfeast_scsrgv! (sparse threaded)", () -> begin
    fpm = zeros(Int, 64); feastinit!(fpm)
    pfeast_scsrgv!(As, Bs, Emin, Emax, M0, fpm)
end)

run_case("feast (dense serial)", () -> begin
    feast(Ad, Bd, (Emin, Emax); M0 = M0)
end)

run_case("feast (sparse serial)", () -> begin
    feast(As, Bs, (Emin, Emax); M0 = M0)
end)
