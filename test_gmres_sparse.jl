#!/usr/bin/env julia

# Test script for the GMRES implementation in feast_sparse_matvec!
using Pkg
Pkg.activate(".")

using FeastKit
using LinearAlgebra
using SparseArrays

println("Testing GMRES implementation in feast_sparse_matvec!")

# Create a simple test problem
n = 10
println("Creating test matrices of size $n x $n")

# Create sparse matrices A and B
A = spdiagm(0 => 2.0*ones(n), 1 => -1.0*ones(n-1), -1 => -1.0*ones(n-1))
B = sparse(I, n, n)  # Identity matrix

println("Matrix A non-zeros: ", nnz(A))
println("Matrix B non-zeros: ", nnz(B))

# Display matrix info
feast_sparse_info(A)

# Set up FEAST parameters
fpm = zeros(Int, 64)
feastinit!(fpm)
fpm[1] = 1  # Enable output
fpm[2] = 8  # Integration points

# Search interval [0.5, 3.5] should contain some eigenvalues
Emin, Emax = 0.5, 3.5
M0 = 6  # Number of eigenvalues to search for

println("\nTesting matrix-free GMRES interface...")

# Define matrix-vector multiplication functions
function A_matvec!(y::AbstractVector{Float64}, x::AbstractVector{Float64})
    mul!(y, A, x)
end

function B_matvec!(y::AbstractVector{Float64}, x::AbstractVector{Float64})  
    mul!(y, B, x)
end

try
    println("Calling feast_sparse_matvec! with matrix-free operators...")
    result = feast_sparse_matvec!(A_matvec!, B_matvec!, n, Emin, Emax, M0, fpm;
                                 gmres_rtol=1e-8, gmres_atol=1e-12,
                                 gmres_restart=20, gmres_maxiter=100)
    
    println("FEAST result:")
    println("  Found eigenvalues: ", result.M)
    println("  Info code: ", result.info)
    println("  Convergence tolerance: ", result.epsout)
    println("  FEAST iterations: ", result.loop)
    
    if result.M > 0
        println("  Eigenvalues: ", result.lambda[1:result.M])
        println("  Residuals: ", result.res[1:result.M])
    end
    
    println("\nTesting sparse matrix wrapper...")
    result2 = feast_sparse_matvec!(A, B, Emin, Emax, M0, fpm;
                                  gmres_rtol=1e-8, gmres_atol=1e-12)
    
    println("FEAST result (sparse wrapper):")
    println("  Found eigenvalues: ", result2.M)
    println("  Info code: ", result2.info)
    println("  Eigenvalues: ", result2.M > 0 ? result2.lambda[1:result2.M] : "none")
    
    # Compare with direct eigenvalue computation for validation
    println("\nComparing with direct eigenvalue computation...")
    lambda_true = eigvals(Matrix(A), Matrix(B))
    lambda_in_interval = filter(λ -> Emin <= λ <= Emax, lambda_true)
    println("True eigenvalues in [$Emin, $Emax]: ", lambda_in_interval)
    
catch e
    println("Error during FEAST computation:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\nTest completed!")