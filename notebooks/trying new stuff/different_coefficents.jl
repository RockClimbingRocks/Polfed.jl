using QuadGK
using PythonPlot

"""
    chebyshev_expand_plot(funcs, orders; npts=400)

Compute Chebyshev expansions of functions `funcs` (Vector of functions)
for multiple polynomial orders in `orders` (Vector of Int).
Plots each function with approximations for all given orders.

Returns: Dict mapping (function_index, order) => coefficients.
"""
function chebyshev_expand_plot(funcs::Vector, orders::Vector{Int}; npts=400)
    coeffs_dict = Dict{Tuple{Int,Int}, Vector{Float64}}()
    xs = range(-1, 1; length=npts)

    fig, axs = subplots(length(funcs), figsize=(6*length(funcs), 4))
    if length(funcs) == 1
        axs = [axs]  # make iterable
    end

    for (i, f) in enumerate(funcs)
        # Exact curve
        exact = [f(x) for x in xs]
        axs[i-1].plot(xs, exact, label="f$i(x)", linewidth=2)

        # Each order
        for N in orders
            coeffs = zeros(Float64, N+1)
            for n in 0:N
                integrand(θ) = f(cos(θ)) * cos(n*θ)
                val, _ = quadgk(integrand, 0, π)
                coeffs[n+1] = (2 - (n == 0)) / π * val
            end
            coeffs_dict[(i, N)] = coeffs

            approx = [sum(coeffs[n+1]*cos(n*acos(x)) for n=0:N) for x in xs]
            axs[i-1].plot(xs, approx, "--", label="N=$N")
        end

        axs[i-1].set_title("Function $i")
        axs[i-1].set_xlabel("x")
        axs[i-1].set_ylabel("y")
        axs[i-1].legend()
        axs[i-1].grid(true)
    end

    fig.tight_layout()
    display(fig)
    return coeffs_dict
end



f1(x) = x==0 ? 1.0 : 0.0
f2(x) = 1 - abs(x)
funs = [f2, f2, f2]
orders = [10, 25, 50]
coeffs = chebyshev_expand_plot(funs, orders)