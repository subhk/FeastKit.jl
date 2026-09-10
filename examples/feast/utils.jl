# Utilities shared by the FEAST reference examples. The fixture files use the
# compact MatrixMarket-like format from the original Fortran examples.
module FeastExampleUtils

using LinearAlgebra
using SparseArrays
using FeastKit

export read_mm_dense_real, read_mm_dense_complex, read_mm_sparse_real,
       read_mm_sparse_complex, read_banded_real, read_banded_complex,
       read_polynomial_dense_real, read_polynomial_sparse_real,
       to_complex_sparse, build_polygonal_contour, print_summary

# Bundled synthetic fixtures are not the original Fortran benchmark matrices.
# Set FEAST_EXAMPLE_DATA_DIR to use an external compatible fixture collection.
const DATA_DIR = get(ENV, "FEAST_EXAMPLE_DATA_DIR", joinpath(@__DIR__, "data"))

# Keep paths centralized so example functions can name FEAST systems rather than
# hard-code fixture directories.
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
    # Pad to FeastKit's equal-bandwidth storage: diagonal row k + 1,
    # with k = max(k_lower, k_upper).
    k_lower = maximum(max(0, rows[i] - cols[i]) for i in eachindex(rows))
    k_upper = maximum(max(0, cols[i] - rows[i]) for i in eachindex(rows))
    k = max(k_lower,k_upper)
    band = zeros(Float64, 2k + 1, n)
    for (r, c, v) in zip(rows, cols, vals)
        band[k + 1 + r - c, c] = v
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
    # Preserve both lower and upper bandwidth so examples can pass the correct
    # FEAST band dimensions to symmetric and general wrappers.
    k_lower = maximum(max(0, rows[i] - cols[i]) for i in eachindex(rows))
    k_upper = maximum(max(0, cols[i] - rows[i]) for i in eachindex(rows))
    k = max(k_lower,k_upper)
    band = zeros(ComplexF64, 2k + 1, n)
    for (r, c, v) in zip(rows, cols, vals)
        band[k + 1 + r - c, c] = v
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
    # FEAST custom-contour examples specify polygon edges plus the number of
    # quadrature nodes per edge. Supply normalized midpoint weights ourselves;
    # feast_contour_custom_weights! only copies them.
    nodes = ComplexF64[]
    weights = ComplexF64[]
    ne = length(zedge)
    ne >= 3 && ne == length(nedge) || throw(ArgumentError("Need matching polygon vertices and edge counts"))
    all(>(0),nedge) || throw(ArgumentError("Edge counts must be positive"))
    area = sum(imag(conj(zedge[i])*zedge[mod1(i+1,ne)]) for i in 1:ne)
    isfinite(area) && area != 0 || throw(ArgumentError("Polygon must have nonzero finite area"))
    for idx in 1:ne
        start = zedge[idx]
        stop = zedge[mod(idx, ne) + 1]
        steps = nedge[idx]
        for k in 0:steps-1
            t = (k + 0.5) / steps
            push!(nodes, start + t * (stop - start))
            push!(weights, sign(area) * (stop-start) / (steps * 2π * im))
        end
    end
    contour = FeastKit.feast_contour_custom_weights!(nodes, weights)
    return contour
end

function print_summary(label::AbstractString, result; max_values::Int=5)
    # Keep output close to the original reference programs while limiting long
    # eigenvalue lists for quick example runs.
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
    result.info == 0 || error("$label failed with info=$(result.info)")
    return result
end

end # module
