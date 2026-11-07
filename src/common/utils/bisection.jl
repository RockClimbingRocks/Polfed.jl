"""
    bisection(f, x1, x2; tol=1e-8, max_iter=500) -> Real

Find a root of the scalar function `f` in `[x1, x2]` by bisection.

The method requires a sign change across the bracket: `f(x1)` and `f(x2)` must have
opposite signs. It repeatedly halves the interval, keeping the half that still
brackets the root, until the half-width drops below `tol` (or `max_iter` is reached),
then returns the interval midpoint.

# Arguments
- `f`: scalar function with opposite signs at the two endpoints.
- `x1`, `x2`: bracketing interval endpoints.

# Keyword Arguments
- `tol::Real=1e-8`: absolute half-interval tolerance (stopping criterion).
- `max_iter::Int=500`: maximum number of bisection steps.

# Returns
- The approximate root (interval midpoint at termination).

# Throws
- `ErrorException`: if `f(x1)` and `f(x2)` do not have opposite signs (no bracket).

# Notes
This is a POLFED-agnostic numerical utility. It backs the spectral-transform
resolutions (bounds from DoS mass, polynomial order from cutoff), where the
monotone bisection targets are built in `Polfed/Transform/`.
"""
function bisection(f, x1::Real, x2::Real; tol::Real=1e-8, max_iter::Int=500)
    if f(x1) * f(x2) > 0
        error("bisection: f must have opposite signs at the bracket endpoints x1=$x1, x2=$x2")
    end

    iter = 0
    while (x2 - x1) / 2 >= tol && iter < max_iter
        x = (x1 + x2) / 2
        fc = f(x)

        if fc == 0 || (x2 - x1) / 2 <= tol
            return x
        elseif f(x1) * fc < 0
            x2 = x
        else
            x1 = x
        end

        iter += 1
    end

    return (x1 + x2) / 2
end
