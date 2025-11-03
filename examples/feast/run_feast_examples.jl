#!/usr/bin/env julia

# Julia translations of the FEAST Fortran reference examples.

using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using FeastKit

include(joinpath(@__DIR__, "utils.jl"))
using .FeastExampleUtils

const TRI_ZEDGE = ComplexF64[(0.10 + 0.410im), (4.2 + 0.41im), (4.2 - 8.3im)]
const TRI_NEDGE = [6, 6, 18]

function dense_real_sygv()
    A = read_mm_dense_real("system1")
    B = read_mm_dense_real("system1B")
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_sygv!(copy(A), copy(B), Emin, Emax, M0, fpm)
    print_summary("F90dense_dfeast_sygv", result)
end

function dense_complex_heev()
    A = read_mm_dense_complex("system2")
    Emin, Emax = -0.35, 0.23
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_heev!(copy(A), Emin, Emax, M0, fpm)
    print_summary("F90dense_zfeast_heev", result)
end

function dense_real_gegv()
    A = read_mm_dense_real("system3")
    B = read_mm_dense_real("system3B")
    Emid = complex(0.590, 0.0)
    r = 0.410
    M0 = 30
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_gegv!(complex.(A), complex.(B), Emid, r, M0, fpm)
    print_summary("F90dense_dfeast_gegv", result)
end

function dense_real_pep()
    coeffs = read_polynomial_dense_real("system5")
    Emid = complex(-1.55, 0.0)
    r = 0.05
    M0 = 30
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[18] = Int(round(100 * (0.0035 / r)))
    result = FeastKit.feast_sypev!(coeffs, 2, Emid, r, M0, fpm)
    print_summary("F90dense_dfeast_sypev", result)
end

function dense_complex_syev()
    A = read_mm_dense_complex("system4")
    Emid = complex(4.0, 0.0)
    r = 3.0
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_geev!(copy(A), Emid, r, M0, fpm)
    print_summary("F90dense_zfeast_syev (general solver)", result)
end

function dense_complex_syevx()
    A = read_mm_dense_complex("system4")
    contour = build_polygonal_contour(TRI_ZEDGE, TRI_NEDGE)
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[8] = length(contour.Zne)
    Emid = complex(4.0, 0.0)
    r = 3.0
    result = FeastKit.feast_geevx!(copy(A), Emid, r, M0, fpm,
                                   contour.Zne, contour.Wne)
    print_summary("F90dense_zfeast_syevx (general solver)", result)
end

function sparse_real_scsrgv()
    A = read_mm_sparse_real("system1")
    B = read_mm_sparse_real("system1B")
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, fpm)
    print_summary("F90sparse_dfeast_scsrgv", result)
end

function sparse_real_scsrgv_lowest()
    A = read_mm_sparse_real("system1")
    B = read_mm_sparse_real("system1B")
    Emin, Emax = 0.18, 1.0
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[40] = -1
    result = FeastKit.feast_scsrgv!(copy(A), copy(B), Emin, Emax, M0, fpm)
    print_summary("F90sparse_dfeast_scsrgv_lowest", result)
end

function sparse_real_gcsrgv()
    A_real = read_mm_sparse_real("system3")
    B_real = read_mm_sparse_real("system3B")
    A = to_complex_sparse(A_real)
    B = to_complex_sparse(B_real)
    Emid = complex(0.590, 0.0)
    r = 0.410
    M0 = 30
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_gcsrgv!(copy(A), copy(B), Emid, r, M0, fpm)
    print_summary("F90sparse_dfeast_gcsrgv", result)
end

function sparse_real_scsrpev()
    coeffs = read_polynomial_sparse_real("system5")
    Emid = complex(-1.55, 0.0)
    r = 0.05
    M0 = 30
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[18] = Int(round(100 * (0.0035 / r)))
    result = FeastKit.feast_scsrpev!(coeffs, 2, Emid, r, M0, fpm)
    print_summary("F90sparse_dfeast_scsrpev", result)
end

function sparse_complex_hcsrev()
    A = read_mm_sparse_complex("system2")
    Emin, Emax = -0.35, 0.23
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_hcsrev!(copy(A), Emin, Emax, M0, fpm)
    print_summary("F90sparse_zfeast_hcsrev", result)
end

function sparse_complex_scsrev()
    A = read_mm_sparse_complex("system4")
    Emid = complex(4.0, 0.0)
    r = 3.0
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_gcsrev!(copy(A), Emid, r, M0, fpm)
    print_summary("F90sparse_zfeast_scsrev", result)
end

function sparse_complex_scsrevx()
    A = read_mm_sparse_complex("system4")
    contour = build_polygonal_contour(TRI_ZEDGE, TRI_NEDGE)
    Emid = complex(4.0, 0.0)
    r = 3.0
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[8] = length(contour.Zne)
    fpm[42] = 0
    result = FeastKit.feast_gcsrevx!(copy(A), Emid, r, M0,
                                     fpm, contour.Zne, contour.Wne)
    print_summary("F90sparse_zfeast_scsrevx", result)
end

function banded_real_sbgv()
    A_band, kl_a, ku_a = read_banded_real("system1")
    B_band, kl_b, ku_b = read_banded_real("system1B")
    ka = max(kl_a, ku_a)
    kb = max(kl_b, ku_b)
    Emin, Emax = 0.18, 1.0
    M0 = 25
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    result = FeastKit.feast_sbgv!(copy(A_band), copy(B_band), ka, kb, Emin, Emax, M0, fpm)
    print_summary("F90banded_dfeast_sbgv", result)
end

function banded_real_gbgv()
    A_band, kl_a, ku_a = read_banded_real("system3")
    B_band, kl_b, ku_b = read_banded_real("system3B")
    Emid = complex(0.590, 0.0)
    r = 0.410
    M0 = 30
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    ka = max(kl_a, ku_a)
    kb = max(kl_b, ku_b)
    result = FeastKit.feast_gbgv!(complex.(A_band), complex.(B_band),
                                  ka, kb, Emid, r, M0, fpm)
    print_summary("F90banded_dfeast_gbgv", result)
end

function banded_complex_hbev()
    A_band, kl_a, ku_a = read_banded_complex("system2")
    Emin, Emax = -0.35, 0.23
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    ka = max(kl_a, ku_a)
    result = FeastKit.feast_hbev!(copy(A_band), ka, Emin, Emax, M0, fpm)
    print_summary("F90banded_zfeast_hbev", result)
end

function banded_complex_sbev()
    A_band, kl_a, ku_a = read_banded_complex("system4")
    Emid = complex(4.0, 0.0)
    r = 3.0
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    ka = max(kl_a, ku_a)
    result = FeastKit.feast_gbev!(copy(A_band), ka, Emid, r, M0, fpm)
    print_summary("F90banded_zfeast_sbev (general solver)", result)
end

function banded_complex_sbevx()
    A_band, kl_a, ku_a = read_banded_complex("system4")
    contour = build_polygonal_contour(TRI_ZEDGE, TRI_NEDGE)
    Emin = real(4.0 - 3.0)
    Emax = real(4.0 + 3.0)
    M0 = 40
    fpm = zeros(Int, 64)
    FeastKit.feastinit!(fpm)
    fpm[1] = 1
    fpm[8] = length(contour.Zne)
    ka = max(kl_a, ku_a)
    Emid = complex(4.0, 0.0)
    r = 3.0
    result = FeastKit.feast_gbevx!(copy(A_band), ka, Emid, r, M0,
                                   fpm, contour.Zne, contour.Wne)
    print_summary("F90banded_zfeast_sbevx (general solver)", result)
end

function main()
    dense_real_sygv()
    dense_complex_heev()
    dense_real_gegv()
    dense_real_pep()
    dense_complex_syev()
    dense_complex_syevx()
    sparse_real_scsrgv()
    sparse_real_scsrgv_lowest()
    sparse_real_gcsrgv()
    sparse_real_scsrpev()
    sparse_complex_hcsrev()
    sparse_complex_scsrev()
    sparse_complex_scsrevx()
    banded_real_sbgv()
    banded_real_gbgv()
    banded_complex_hbev()
    banded_complex_sbev()
    banded_complex_sbevx()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
