# Run in an environment containing FeastKit and Krylov, e.g. --project=docs.
# Timings exclude compilation. Compare rows only at the same residual/count.
using FeastKit, Krylov, LinearAlgebra, SparseArrays, Statistics, Printf

BLAS.set_num_threads(1)
function measure_case(name, solve, A, B, expected; samples=3)
    solve() # compilation/warmup
    times, bytes = Float64[], Int[]
    result = nothing
    for _ in 1:samples
        GC.gc()
        measured = @timed solve()
        result = measured.value
        push!(times, measured.time)
        push!(bytes, measured.bytes)
    end
    @assert result.converged "$name did not converge"
    @assert result.M == length(expected) "$name missed eigenvalues"
    @assert sort(real.(result.values)) ≈ expected atol=1e-8
    actual = maximum((norm(A * result.vectors[:,j] - result.values[j] * (B * result.vectors[:,j])) /
                      norm(B * result.vectors[:,j]) / max(abs(result.values[j]), 1)
                      for j in 1:result.M); init=0.0)
    @assert actual < 1e-9 "$name residual too large"
    @printf("%s\t%.6f\t%d\t%d\t%d\t%.3e\n", name, median(times), Int(median(bytes)),
            result.M, result.loop, actual)
    return result
end

println("# Julia ", VERSION, "; Julia threads=", Threads.nthreads(), "; BLAS threads=", BLAS.get_num_threads())
println("case\tmedian_seconds\tallocated_bytes\teigenvalues\trefinements\tactual_residual")
for (name, n, storage, complex_problem, general) in
    (("dense_symmetric_gmres", 160, Matrix, false, false),
     ("sparse_symmetric_gmres", 400, sparse, false, false),
     ("dense_hermitian_gmres", 160, Matrix, true, false),
     ("sparse_general_gmres", 160, sparse, true, true),
     ("dense_symmetric_direct", 300, Matrix, false, false))
    # Similarity transforms preserve a known spectrum while exercising
    # nontrivial Hermitian and nonsymmetric eigenvectors.
    L = SymTridiagonal(fill(2.0,n), fill(-1.0,n-1))
    phase = complex_problem ? cis.(range(0, 2; length=n)) : ones(n)
    general && (phase .*= exp.(range(0, 1; length=n)))
    a = Diagonal(phase) * Matrix(L) / Diagonal(phase)
    !general && complex_problem && (a = Matrix(Hermitian(a)))
    A = storage(a)
    B = storage(Matrix{eltype(A)}(I,n,n))
    interval = (0.9, 1.1)
    expected = filter(x -> interval[1] <= x <= interval[2], eigvals(L))
    opts = (; subspace_size=length(expected)+6, tol=1e-10, maxiter=30,
            quadrature_points=8, solver=endswith(name,"direct") ? :direct : :gmres,
            solver_opts=endswith(name,"direct") ? (;) : (restart=n, maxiter=2n))
    solve = general ? () -> feast_general(A, 1.0+0im, 0.1; opts...) :
                      () -> feast(A, interval; opts...)
    measure_case(name, solve, A, B, expected)
end

# Opt-in controls have costs as well as benefits. Exclude these rows when
# running this same script against a checkout without the new keywords.
if get(ENV,"FEAST_BENCH_CONTROLS","true") == "true"
    n = 600
    A = Matrix(SymTridiagonal(fill(2.0,n),fill(-1.0,n-1)))
    interval = (0.9,1.1)
    expected = filter(x -> interval[1] <= x <= interval[2], eigvals(Symmetric(A)))
    opts = (; subspace_size=length(expected)+8, tol=1e-10, maxiter=30)
    cold = measure_case("controls_dense_direct",()->feast(A,interval;opts...),A,I,expected)
    measure_case("controls_dense_mixed",()->feast(A,interval;opts...,mixed_precision=true),A,I,expected)
    next_A = A + 1e-4I
    next_expected = expected .+ 1e-4
    measure_case("controls_related_cold",()->feast(next_A,interval;opts...),next_A,I,next_expected)
    measure_case("controls_related_warm",()->feast(next_A,interval;opts...,initial_subspace=cold.vectors),next_A,I,next_expected)
    measure_case("controls_auto",()->feast(A,interval;subspace_size=:auto,tol=1e-10,maxiter=30),A,I,expected)
end
