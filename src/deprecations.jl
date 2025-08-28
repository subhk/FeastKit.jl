# Deprecated aliases for legacy FEAST_* constants
# These map old all-caps names to the new Feast_* enums/constants

# Export legacy names so `using Feast` continues to bring them into scope
export FEAST_SUCCESS, FEAST_ERROR_N, FEAST_ERROR_M0, FEAST_ERROR_EMIN_EMAX,
       FEAST_ERROR_EMID_R, FEAST_ERROR_NO_CONVERGENCE, FEAST_ERROR_MEMORY,
       FEAST_ERROR_INTERNAL, FEAST_ERROR_LAPACK, FEAST_ERROR_FPM
export FEAST_RCI_INIT, FEAST_RCI_DONE, FEAST_RCI_FACTORIZE, FEAST_RCI_SOLVE,
       FEAST_RCI_SOLVE_TRANSPOSE, FEAST_RCI_MULT_A, FEAST_RCI_MULT_B

# Error codes
Base.@deprecate_binding FEAST_SUCCESS Feast_SUCCESS
Base.@deprecate_binding FEAST_ERROR_N Feast_ERROR_N
Base.@deprecate_binding FEAST_ERROR_M0 Feast_ERROR_M0
Base.@deprecate_binding FEAST_ERROR_EMIN_EMAX Feast_ERROR_EMIN_EMAX
Base.@deprecate_binding FEAST_ERROR_EMID_R Feast_ERROR_EMID_R
Base.@deprecate_binding FEAST_ERROR_NO_CONVERGENCE Feast_ERROR_NO_CONVERGENCE
Base.@deprecate_binding FEAST_ERROR_MEMORY Feast_ERROR_MEMORY
Base.@deprecate_binding FEAST_ERROR_INTERNAL Feast_ERROR_INTERNAL
Base.@deprecate_binding FEAST_ERROR_LAPACK Feast_ERROR_LAPACK
Base.@deprecate_binding FEAST_ERROR_FPM Feast_ERROR_FPM

# RCI job identifiers
Base.@deprecate_binding FEAST_RCI_INIT Feast_RCI_INIT
Base.@deprecate_binding FEAST_RCI_DONE Feast_RCI_DONE
Base.@deprecate_binding FEAST_RCI_FACTORIZE Feast_RCI_FACTORIZE
Base.@deprecate_binding FEAST_RCI_SOLVE Feast_RCI_SOLVE
Base.@deprecate_binding FEAST_RCI_SOLVE_TRANSPOSE Feast_RCI_SOLVE_TRANSPOSE
Base.@deprecate_binding FEAST_RCI_MULT_A Feast_RCI_MULT_A
Base.@deprecate_binding FEAST_RCI_MULT_B Feast_RCI_MULT_B

