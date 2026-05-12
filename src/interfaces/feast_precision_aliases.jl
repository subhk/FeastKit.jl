# FEAST-compatible precision-prefixed entry points.
#
# The core Julia implementation uses element types for dispatch (`Float32`,
# `Float64`, `ComplexF32`, `ComplexF64`). These aliases expose the familiar
# FEAST naming convention while forwarding to the tested generic solvers.

for (prefix, RT) in ((:s, Float32), (:d, Float64))
    @eval begin
        $(Symbol(prefix, "feast_syev!"))(A::Matrix{$RT},
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_syev!(A, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_sygv!"))(A::Matrix{$RT}, B::Matrix{$RT},
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_sygv!(A, B, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_syevx!"))(A::Matrix{$RT},
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_syevx!(A, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sygvx!"))(A::Matrix{$RT}, B::Matrix{$RT},
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sygvx!(A, B, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sypev!"))(A::Vector{Matrix{$RT}}, d::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int}) =
            feast_sypev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_sypevx!"))(A::Vector{Matrix{$RT}}, d::Int,
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int},
                                           Zne::AbstractVector{Complex{TZ}},
                                           Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sypevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_srcipev!"))(A::Vector{Matrix{$RT}}, d::Int,
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int}) =
            feast_srcipev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_srcipevx!"))(A::Vector{Matrix{$RT}}, d::Int,
                                             Emid::Complex{$RT}, r::$RT,
                                             M0::Int, fpm::Vector{Int},
                                             Zne::AbstractVector{Complex{TZ}},
                                             Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_srcipevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_scsrev!"))(A::SparseMatrixCSC{$RT,Int},
                                           Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int}) =
            feast_scsrev!(A, Emin, Emax, M0, fpm)

        $(Symbol(prefix, "feast_scsrgv!"))(A::SparseMatrixCSC{$RT,Int},
                                           B::SparseMatrixCSC{$RT,Int},
                                           Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                           kwargs...) =
            feast_scsrgv!(A, B, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_scsrevx!"))(A::SparseMatrixCSC{$RT,Int},
                                            Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_scsrevx!(A, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_scsrgvx!"))(A::SparseMatrixCSC{$RT,Int},
                                            B::SparseMatrixCSC{$RT,Int},
                                            Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_scsrgvx!(A, B, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_scsrpev!"))(A::Vector{SparseMatrixCSC{$RT,Int}}, d::Int,
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int}) =
            feast_scsrpev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_scsrpevx!"))(A::Vector{SparseMatrixCSC{$RT,Int}}, d::Int,
                                             Emid::Complex{$RT}, r::$RT,
                                             M0::Int, fpm::Vector{Int},
                                             Zne::AbstractVector{Complex{TZ}},
                                             Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_scsrpevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sbev!"))(A::Matrix{$RT}, ka::Int,
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_sbev!(A, ka, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_sbgv!"))(A::Matrix{$RT}, B::Matrix{$RT},
                                         ka::Int, kb::Int,
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_sbgv!(A, B, ka, kb, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_sbevx!"))(A::Matrix{$RT}, ka::Int,
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sbevx!(A, ka, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sbgvx!"))(A::Matrix{$RT}, B::Matrix{$RT},
                                          ka::Int, kb::Int,
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sbgvx!(A, B, ka, kb, Emin, Emax, M0, fpm, Zne, Wne)
    end
end

for (prefix, RT) in ((:c, Float32), (:z, Float64))
    @eval begin
        $(Symbol(prefix, "feast_heev!"))(A::Matrix{Complex{$RT}},
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_heev!(A, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_hegv!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_hegv!(A, B, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_heevx!"))(A::Matrix{Complex{$RT}},
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_heevx!(A, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hegvx!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hegvx!(A, B, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_geev!"))(A::Matrix{Complex{$RT}},
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_geev!(A, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_gegv!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_gegv!(A, B, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_geevx!"))(A::Matrix{Complex{$RT}},
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_geevx!(A, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gegvx!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gegvx!(A, B, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gepev!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int}) =
            feast_gepev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_gepevx!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int},
                                           Zne::AbstractVector{Complex{TZ}},
                                           Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gepevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hepev!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int}) =
            feast_hepev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_hepevx!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int},
                                           Zne::AbstractVector{Complex{TZ}},
                                           Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hepevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_grcipev!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int}) =
            feast_grcipev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_grcipevx!"))(A::Vector{Matrix{Complex{$RT}}}, d::Int,
                                             Emid::Complex{$RT}, r::$RT,
                                             M0::Int, fpm::Vector{Int},
                                             Zne::AbstractVector{Complex{TZ}},
                                             Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_grcipevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hcsrev!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                           kwargs...) =
            feast_hcsrev!(A, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_hcsrgv!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           B::SparseMatrixCSC{Complex{$RT},Int},
                                           Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                           kwargs...) =
            feast_hcsrgv!(A, B, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_hcsrevx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hcsrevx!(A, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hcsrgvx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            B::SparseMatrixCSC{Complex{$RT},Int},
                                            Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hcsrgvx!(A, B, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_scsrev!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_scsrev_complex!(A, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_scsrgv!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           B::SparseMatrixCSC{Complex{$RT},Int},
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_scsrgv_complex!(A, B, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_scsrevx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_scsrevx_complex!(A, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_scsrgvx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            B::SparseMatrixCSC{Complex{$RT},Int},
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_scsrgvx_complex!(A, B, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gcsrev!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_gcsrev!(A, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_gcsrgv!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                           B::SparseMatrixCSC{Complex{$RT},Int},
                                           Emid::Complex{$RT}, r::$RT,
                                           M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_gcsrgv!(A, B, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_gcsrevx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gcsrevx!(A, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gcsrgvx!"))(A::SparseMatrixCSC{Complex{$RT},Int},
                                            B::SparseMatrixCSC{Complex{$RT},Int},
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int},
                                            Zne::AbstractVector{Complex{TZ}},
                                            Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gcsrgvx!(A, B, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hcsrpev!"))(A::Vector{SparseMatrixCSC{Complex{$RT},Int}}, d::Int,
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int}) =
            feast_hcsrpev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_hcsrpevx!"))(A::Vector{SparseMatrixCSC{Complex{$RT},Int}}, d::Int,
                                             Emid::Complex{$RT}, r::$RT,
                                             M0::Int, fpm::Vector{Int},
                                             Zne::AbstractVector{Complex{TZ}},
                                             Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hcsrpevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gcsrpev!"))(A::Vector{SparseMatrixCSC{Complex{$RT},Int}}, d::Int,
                                            Emid::Complex{$RT}, r::$RT,
                                            M0::Int, fpm::Vector{Int}) =
            feast_gcsrpev!(A, d, Emid, r, M0, fpm)

        $(Symbol(prefix, "feast_gcsrpevx!"))(A::Vector{SparseMatrixCSC{Complex{$RT},Int}}, d::Int,
                                             Emid::Complex{$RT}, r::$RT,
                                             M0::Int, fpm::Vector{Int},
                                             Zne::AbstractVector{Complex{TZ}},
                                             Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gcsrpevx!(A, d, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hbev!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_hbev!(A, ka, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_hbgv!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                         ka::Int, kb::Int,
                                         Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                         kwargs...) =
            feast_hbgv!(A, B, ka, kb, Emin, Emax, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_hbevx!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hbevx!(A, ka, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_hbgvx!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                          ka::Int, kb::Int,
                                          Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_hbgvx!(A, B, ka, kb, Emin, Emax, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sbev!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_sbev_complex!(A, ka, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_sbgv!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                         ka::Int, kb::Int,
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_sbgv_complex!(A, B, ka, kb, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_sbevx!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sbevx_complex!(A, ka, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_sbgvx!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                          ka::Int, kb::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_sbgvx_complex!(A, B, ka, kb, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gbev!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_gbev!(A, ka, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_gbgv!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                         ka::Int, kb::Int,
                                         Emid::Complex{$RT}, r::$RT,
                                         M0::Int, fpm::Vector{Int}; kwargs...) =
            feast_gbgv!(A, B, ka, kb, Emid, r, M0, fpm; kwargs...)

        $(Symbol(prefix, "feast_gbevx!"))(A::Matrix{Complex{$RT}}, ka::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gbevx!(A, ka, Emid, r, M0, fpm, Zne, Wne)

        $(Symbol(prefix, "feast_gbgvx!"))(A::Matrix{Complex{$RT}}, B::Matrix{Complex{$RT}},
                                          ka::Int, kb::Int,
                                          Emid::Complex{$RT}, r::$RT,
                                          M0::Int, fpm::Vector{Int},
                                          Zne::AbstractVector{Complex{TZ}},
                                          Wne::AbstractVector{Complex{TW}}) where {TZ<:Real,TW<:Real} =
            feast_gbgvx!(A, B, ka, kb, Emid, r, M0, fpm, Zne, Wne)
    end
end

function _pfeast_mpi_unavailable(name::Symbol)
    throw(ArgumentError("$(name) with comm= requires MPI support to be loaded. Use the threaded/distributed path without comm=, or load FeastKit with MPI available."))
end

for (prefix, RT) in ((:ps, Float32), (:pd, Float64))
    @eval begin
        function $(Symbol(prefix, "feast_sygv!"))(A::Matrix{$RT}, B::Matrix{$RT},
                                                  Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                                  use_threads::Bool=true,
                                                  verbose::Bool=false,
                                                  comm=nothing,
                                                  root::Int=0)
            if comm === nothing
                return pfeast_sygv!(A, B, Emin, Emax, M0, fpm;
                                    use_threads=use_threads, verbose=verbose)
            end
            isdefined(@__MODULE__, :mpi_feast_sygv!) ||
                _pfeast_mpi_unavailable($(QuoteNode(Symbol(prefix, "feast_sygv!"))))
            return mpi_feast_sygv!(A, B, Emin, Emax, M0, fpm; comm=comm, root=root)
        end

        function $(Symbol(prefix, "feast_syev!"))(A::Matrix{$RT},
                                                  Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                                  kwargs...)
            B = Matrix{$RT}(I, size(A, 1), size(A, 1))
            return $(Symbol(prefix, "feast_sygv!"))(A, B, Emin, Emax, M0, fpm; kwargs...)
        end

        function $(Symbol(prefix, "feast_scsrgv!"))(A::SparseMatrixCSC{$RT,Int},
                                                    B::SparseMatrixCSC{$RT,Int},
                                                    Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                                    use_threads::Bool=true,
                                                    verbose::Bool=false,
                                                    comm=nothing,
                                                    root::Int=0)
            if comm === nothing
                return pfeast_scsrgv!(A, B, Emin, Emax, M0, fpm;
                                      use_threads=use_threads, verbose=verbose)
            end
            isdefined(@__MODULE__, :mpi_feast_scsrgv!) ||
                _pfeast_mpi_unavailable($(QuoteNode(Symbol(prefix, "feast_scsrgv!"))))
            return mpi_feast_scsrgv!(A, B, Emin, Emax, M0, fpm; comm=comm, root=root)
        end

        function $(Symbol(prefix, "feast_scsrev!"))(A::SparseMatrixCSC{$RT,Int},
                                                    Emin::$RT, Emax::$RT, M0::Int, fpm::Vector{Int};
                                                    kwargs...)
            B = spdiagm(0 => ones($RT, size(A, 1)))
            return $(Symbol(prefix, "feast_scsrgv!"))(A, B, Emin, Emax, M0, fpm; kwargs...)
        end

        function $(Symbol(prefix, "feast_srci!"))(state,
                                                  N::Int,
                                                  work::Matrix{$RT},
                                                  workc::Matrix{Complex{$RT}},
                                                  Aq::Matrix{$RT},
                                                  Sq::Matrix{$RT},
                                                  fpm::Vector{Int},
                                                  Emin::$RT,
                                                  Emax::$RT,
                                                  M0::Int,
                                                  lambda::Vector{$RT},
                                                  q::Matrix{$RT},
                                                  res::Vector{$RT})
            return pfeast_srci!(state, N, work, workc, Aq, Sq, fpm,
                                Emin, Emax, M0, lambda, q, res)
        end
    end
end
