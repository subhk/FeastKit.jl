using FeastKit
using Test
using LinearAlgebra
using SparseArrays
using Distributed
using Random
# Krylov is a weak dependency: loading it here activates FeastKitKrylovExt so
# the iterative (IFEAST / solver=:gmres) paths below are exercised.
using Krylov
