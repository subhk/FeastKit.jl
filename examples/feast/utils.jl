module FeastExampleUtils

using LinearAlgebra
using SparseArrays
using FeastKit

const DATA_DIR = joinpath(@__DIR__, "..", "FEAST", "example", "FEAST")

feast_data_path(parts...) = joinpath(DATA_DIR, parts...)

function read_mm_dense_real(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        A = zeros(Float64, n, n)
        for _ in 1:nnz
            parts = split(strip(readline(io)))
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            val = parse(Float64, parts[3])
            A[i, j] = val
        end
        return A
    end
end

function read_mm_dense_complex(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        A = zeros(ComplexF64, n, n)
        for _ in 1:nnz
            parts = split(strip(readline(io)))
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            re = parse(Float64, parts[3])
            im = parse(Float64, parts[4])
            A[i, j] = complex(re, im)
        end
        return A
    end
end

function read_mm_sparse_real(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        row = Vector{Int}(undef, nnz)
        col = Vector{Int}(undef, nnz)
        val = Vector{Float64}(undef, nnz)
        for k in 1:nnz
            parts = split(strip(readline(io)))
            row[k] = parse(Int, parts[1])
            col[k] = parse(Int, parts[2])
            val[k] = parse(Float64, parts[3])
        end
        return sparse(row, col, val, n, n)
    end
end

function read_mm_sparse_complex(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        row = Vector{Int}(undef, nnz)
        col = Vector{Int}(undef, nnz)
        val = Vector{ComplexF64}(undef, nnz)
        for k in 1:nnz
            parts = split(strip(readline(io)))
            row[k] = parse(Int, parts[1])
            col[k] = parse(Int, parts[2])
            re = parse(Float64, parts[3])
            im = parse(Float64, parts[4])
            val[k] = complex(re, im)
        end
        return sparse(row, col, val, n, n)
    end
end

function read_banded_real(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    n = 0
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        resize!(rows, nnz)
        resize!(cols, nnz)
        resize!(vals, nnz)
        for k in 1:nnz
            parts = split(strip(readline(io)))
            rows[k] = parse(Int, parts[1])
            cols[k] = parse(Int, parts[2])
            vals[k] = parse(Float64, parts[3])
        end
    end
    k_lower = maximum(max(0, rows[i] - cols[i]) for i in eachindex(rows))
    k_upper = maximum(max(0, cols[i] - rows[i]) for i in eachindex(rows))
    band = zeros(Float64, k_lower + k_upper + 1, n)
    for (r, c, v) in zip(rows, cols, vals)
        band[k_upper + 1 + r - c, c] = v
    end
    return band, k_lower, k_upper
end

function read_banded_complex(name::AbstractString)
    path = feast_data_path("$(name).mtx")
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    n = 0
    open(path, "r") do io
        header = split(strip(readline(io)))
        n = parse(Int, header[1])
        nnz = parse(Int, header[3])
        resize!(rows, nnz)
        resize!(cols, nnz)
        resize!(vals, nnz)
        for k in 1:nnz
            parts = split(strip(readline(io)))
            rows[k] = parse(Int, parts[1])
            cols[k] = parse(Int, parts[2])
            re = parse(Float64, parts[3])
            im = parse(Float64, parts[4])
            vals[k] = complex(re, im)
        end
    end
    k_lower = maximum(max(0, rows[i] - cols[i]) for i in eachindex(rows))
    k_upper = maximum(max(0, cols[i] - rows[i]) for i in eachindex(rows))
    band = zeros(ComplexF64, k_lower + k_upper + 1, n)
    for (r, c, v) in zip(rows, cols, vals)
        band[k_upper + 1 + r - c, c] = v
    end
    return band, k_lower, k_upper
end

function read_polynomial_dense_real(prefix::AbstractString)
    matrices = Vector{Matrix{Float64}}(undef, 3)
    matrices[1] = read_mm_dense_real(prefix * "A0")
    matrices[2] = read_mm_dense_real(prefix * "A1")
    matrices[3] = read_mm_dense_real(prefix * "A2")
    return matrices
end

function read_polynomial_sparse_real(prefix::AbstractString)
    matrices = Vector{SparseMatrixCSC{Float64, Int}}(undef, 3)
    matrices[1] = read_mm_sparse_real(prefix * "A0")
    matrices[2] = read_mm_sparse_real(prefix * "A1")
    matrices[3] = read_mm_sparse_real(prefix * "A2")
    return matrices
end

function to_complex_sparse(A::SparseMatrixCSC{Float64, Int})
    return SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), ComplexF64.(A.nzval))
end

function build_polygonal_contour(zedge::Vector{ComplexF64}, nedge::Vector{Int})
    nodes = ComplexF64[]
    ne = length(zedge)
    @assert ne == length(nedge)
    for idx in 1:ne
        start = zedge[idx]
        stop = zedge[mod(idx, ne) + 1]
        steps = nedge[idx]
        for k in 0:steps-1
            t = k / steps
            push!(nodes, start + t * (stop - start))
        end
    end
    weights = zeros(ComplexF64, length(nodes))
    contour = FeastKit.feast_contour_custom_weights!(nodes, weights)
    return contour
end

function print_summary(label::AbstractString, result; max_values::Int=5)
    println(label)
    println("  info = ", result.info, ", loops = ", result.loop, ", epsout = ", result.epsout)
    println("  eigenpairs found = ", result.M)
    if result.M > 0
        count = min(result.M, max_values)
        λ = result.lambda[1:count]
        if eltype(λ) <: Complex
            println("  eigenvalues: ", round.(λ; digits=6))
        else
            println("  eigenvalues: ", round.(λ; digits=6))
        end
        println("  residuals : ", round.(result.res[1:count]; digits=6))
    end
    println()
end

end # module
