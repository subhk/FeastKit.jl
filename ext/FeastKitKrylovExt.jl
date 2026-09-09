# Krylov.jl package extension.
#
# FeastKit's iterative (IFEAST) paths need a shifted-system Krylov solver, but
# most users only ever run the direct paths. Keeping Krylov weak means the
# solver is loaded only when someone actually asks for `solver=:gmres`.
#
# The parent module declares `_feast_gmres` and friends and calls them through
# that seam, so no Krylov type ever appears in FeastKit itself.
module FeastKitKrylovExt

using FeastKit
using Krylov

import FeastKit: _feast_gmres, _feast_gmres_workspace, _feast_gmres!,
                 _feast_gmres_solution, _feast_bicgstab, FEAST_KRYLOV_AVAILABLE

function __init__()
    FEAST_KRYLOV_AVAILABLE[] = true
    return nothing
end

# Each wrapper returns `(solution, converged::Bool)` so call sites never touch
# Krylov's stats objects.
function _feast_gmres(op, b; restart::Bool = true, memory::Int = 20,
                      rtol = 1e-8, atol = 1e-8, itmax::Int = 200)
    x, stats = Krylov.gmres(op, b; restart = restart, memory = memory,
                            rtol = rtol, atol = atol, itmax = itmax)
    return x, stats.solved
end

function _feast_gmres(op, b, x0; restart::Bool = true, memory::Int = 20,
                      rtol = 1e-8, atol = 1e-8, itmax::Int = 200)
    x, stats = Krylov.gmres(op, b, x0; restart = restart, memory = memory,
                            rtol = rtol, atol = atol, itmax = itmax)
    return x, stats.solved
end

# Reusable workspace so a driver that solves many right-hand sides at the same
# shift does not reallocate the Krylov basis per column.
_feast_gmres_workspace(N::Int, ::Type{CT}; memory::Int = 20) where {CT} =
    Krylov.GmresWorkspace(N, N, Vector{CT}; memory = memory)

function _feast_gmres!(workspace, op, b; restart::Bool = true,
                       rtol = 1e-8, atol = 1e-8, itmax::Int = 200)
    Krylov.gmres!(workspace, op, b; restart = restart, rtol = rtol,
                  atol = atol, itmax = itmax)
    return workspace.stats.solved
end

_feast_gmres_solution(workspace) = workspace.x

function _feast_bicgstab(op, b; rtol = 1e-8, atol = 1e-8, itmax::Int = 200)
    x, stats = Krylov.bicgstab(op, b; rtol = rtol, atol = atol, itmax = itmax)
    return x, stats.solved
end

end # module
