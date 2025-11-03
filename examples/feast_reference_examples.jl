#!/usr/bin/env julia

# Reference FEAST examples translated to FeastKit.jl
# Mirrors the canonical dense, sparse, and banded driver programs
# shipped with the original Fortran FEAST distribution.

using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
import FeastKit

const FEAST_DATA_DIR = joinpath(@__DIR__, "..", "FEAST", "example", "FEAST")

function read_real_matrix(path::AbstractString; dense::Bool)
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        row = Vector{Int}(undef, nnz)
        col = Vector{Int}(undef, nnz)
        val = Vector{Float64}(undef, nnz)
        k = 1
        for line in eachline(io)
            parts = split(strip(line))
            row[k] = parse(Int, parts[1])
            col[k] = parse(Int, parts[2])
            val[k] = parse(Float64, parts[3])
            k += 1
        end
        if dense
            A = zeros(Float64, n, n)
            for idx in 1:nnz
                A[row[idx], col[idx]] = val[idx]
            end
            return A
        else
            return sparse(row, col, val, n, n)
        end
    end
end

function read_complex_matrix(path::AbstractString; dense::Bool)
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        row = Vector{Int}(undef, nnz)
        col = Vector{Int}(undef, nnz)
        val = Vector{ComplexF64}(undef, nnz)
        k = 1
        for line in eachline(io)
            parts = split(strip(line))
            row[k] = parse(Int, parts[1])
            col[k] = parse(Int, parts[2])
            real_part = parse(Float64, parts[3])
            imag_part = parse(Float64, parts[4])
            val[k] = complex(real_part, imag_part)
            k += 1
        end
        if dense
            A = zeros(ComplexF64, n, n)
            for idx in 1:nnz
                A[row[idx], col[idx]] = val[idx]
            end
            return A
        else
            return sparse(row, col, val, n, n)
        end
    end
end

function read_banded_real(path::AbstractString)
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        offsets = Int[]
        entries = Float64[]
        rows = Int[]
        cols = Int[]
        for line in eachline(io)
            parts = split(strip(line))
            i = parse(Int, parts[0])
            j = parse(Int, parts[1])
            v = parse(Float64, parts[2])
            push!(rows, i)
            push!(cols, j)
            push!(entries, v)
            push!(offsets, abs(i - j))
        end
        k = maximum(offsets)
        band = zeros(Float64, k + 1 + k, n)
        for (i, j, v) in zip(rows, cols, entries)
            band[k + 1 + i - j, j] = v
        end
        return band, k
    end
end

function run_dense_real_generalized()
    A = read_real_matrix(joinpath(FEAST_DATA_DIR, "system1.mtx"); dense=true)
    B = read_real_matrix(joinpath(FEAST_DATA_DIR, "system1B.mtx"); dense=true)
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_sygv!(copy(A), copy(B), Emin, Emax, M0, fpm)
    println("Dense real symmetric generalized example (system1):")
    println("  info = ", result.info, ", eigenpairs found = ", result.M)
    println("  sample eigenvalues: ", result.lambda[1:min(result.M, 5)])
end

function run_dense_complex_hermitian()
    A = read_complex_matrix(joinpath(FEAST_DATA_DIR, "system2.mtx"); dense=true)
    Emin, Emax = -0.35, 0.23
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_heev!(copy(A), Emin, Emax, M0, fpm)
    println("Dense complex Hermitian example (system2):")
    println("  info = ", result.info, ", eigenpairs found = ", result.M)
    println("  sample eigenvalues: ", result.lambda[1:min(result.M, 5)])
end

function run_sparse_real_generalized()
    A = read_real_matrix(joinpath(FEAST_DATA_DIR, "system1.mtx"); dense=false)
    B = read_real_matrix(joinpath(FEAST_DATA_DIR, "system1B.mtx"); dense=false)
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, fpm)
    println("Sparse real symmetric generalized example (system1):")
    println("  info = ", result.info, ", eigenpairs found = ", result.M)
    println("  sample eigenvalues: ", result.lambda[1:min(result.M, 5)])
end

function run_banded_real_generalized()
    A_band, ka = read_banded_real(joinpath(FEAST_DATA_DIR, "system1.mtx"))
    B_band, kb = read_banded_real(joinpath(FEAST_DATA_DIR, "system1B.mtx"))
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_sbgv!(copy(A_band), copy(B_band), ka, kb, Emin, Emax, M0, fpm)
    println("Banded real symmetric generalized example (system1):")
    println("  info = ", result.info, ", eigenpairs found = ", result.M)
    println("  sample eigenvalues: ", result.lambda[1:min(result.M, 5)])
end

function main()
    run_dense_real_generalized()
    println()
    run_dense_complex_hermitian()
    println()
    run_sparse_real_generalized()
    println()
    run_banded_real_generalized()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
