
"""
    bisection(f::Function, x1::Real, x2::Real; tol=1e-8, max_iter=500) -> Real

Find a root of `f` in `[x1, x2]` using the bisection method.

# Arguments
- `f`: Scalar function with opposite signs at the interval endpoints.
- `x1`, `x2`: Bracketing interval endpoints.

# Keyword Arguments
- `tol`: Absolute half-interval tolerance.
- `max_iter`: Maximum number of bisection iterations.

# Throws
- `ErrorException`: If `f(x1)` and `f(x2)` do not have opposite signs.
"""
function bisection(f::Function, x1::Real, x2::Real; tol=1e-8, max_iter=500)
    if f(x1) * f(x2) > 0
        error("The function must have opposite signs at a and b")
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
    
    return (x1 + x2) / 2  # Return the midpoint as the approximate root
end
