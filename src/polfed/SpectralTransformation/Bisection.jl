
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
