

using DrWatson
@quickactivate("notebooks")

# chebyshev_pythonplot_subplots.jl

using QuadGK
using PythonPlot

# Alias PythonPlot to plt for convenience
const plt = PythonPlot

# --- 1. Define the Functions (No changes) ---

function box_function(x, a=-0.5, b=0.5)
    return a <= x <= b ? 1.0 : 0.0
end

function chebyshev_T(n, x)
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        T_nm2 = 1.0
        T_nm1 = x
        for _ in 2:n
            T_n = 2*x*T_nm1 - T_nm2
            T_nm2 = T_nm1
            T_nm1 = T_n
        end
        return T_nm1
    end
end

# --- 2. Calculate the Expansion Coefficients (No changes) ---

function calculate_chebyshev_coeffs(f, max_order::Int)
    coeffs = zeros(max_order + 1)
    w(x) = 1.0 / sqrt(1.0 - x^2)

    integrand_0(x) = f(x) * chebyshev_T(0, x) * w(x)
    integral_0, _ = quadgk(integrand_0, -1.0, 1.0)
    coeffs[1] = (1.0 / π) * integral_0

    for n in 1:max_order
        integrand_n(x) = f(x) * chebyshev_T(n, x) * w(x)
        integral_n, _ = quadgk(integrand_n, -1.0, 1.0)
        coeffs[n+1] = (2.0 / π) * integral_n
    end
    
    return coeffs
end

# --- 3. Construct the Approximation from Coefficients (No changes) ---

function evaluate_chebyshev_series(x, coeffs)
    N = length(coeffs) - 1
    sum_val = 0.0
    for n in 0:N
        sum_val += coeffs[n+1] * chebyshev_T(n, x)
    end
    return sum_val
end

# --- [NEW] Function to apply the Lanczos filter ---

function apply_lanczos_filter!(coeffs)
    N = length(coeffs) - 1
    if N == 0 return end

    for n in 1:N
        x = n / N
        sinc_val = sin(π * x) / (π * x + eps())
        coeffs[n+1] *= sinc_val
    end
end

# --- 4. Main Execution and Plotting (CORRECTED) ---

function main()
    println("Calculating Chebyshev coefficients for the box function...")
    max_order = 201
    f = x -> box_function(x, -0.5, 0.5)
    coeffs = calculate_chebyshev_coeffs(f, max_order)
    
    println("Coefficients calculated. Generating plots...")

    x_range = -0.95:0.0001:0.95
    
    # --- Plotting with PythonPlot using subplots ---
    fig, axs = plt.subplots(1, 4, figsize=(21, 6), sharey=true)

    # Subplot 1: Original Box Function (CORRECTED: using axs[1])
    axs[1].plot(x_range, f.(x_range), "k--", linewidth=2, label="Original Box Function")
    axs[1].set_title("Original Box Function")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("f(x)")
    axs[1].grid(true, linestyle="--", alpha=0.6)
    axs[1].legend()
    
    # Subplot 2: Standard Chebyshev Approximation (CORRECTED: using axs[2])
    axs[2].set_title("Standard Approximation (Gibbs Oscillations)")
    orders_to_plot = [50, 100, 200]
    for N in orders_to_plot
        approx_values = [evaluate_chebyshev_series(x, coeffs[1:N+1]) for x in x_range]
        axs[2].plot(x_range, approx_values, linewidth=2, label="N = $N terms")
    end
    axs[2].set_xlabel("x")
    axs[2].grid(true, linestyle="--", alpha=0.6)
    axs[2].legend()

    # Subplot 3: Smoothed Chebyshev Approximation (CORRECTED: using axs[3])
    axs[3].set_title("Smoothed Approximation (Lanczos Filter)")
    for N in orders_to_plot
        coeffs_subset = copy(coeffs[1:N+1])
        apply_lanczos_filter!(coeffs_subset)
        
        approx_values_filtered = [evaluate_chebyshev_series(x, coeffs_subset) for x in x_range]
        axs[3].plot(x_range, approx_values_filtered, linewidth=2, label="N = $N terms (filtered)")
    end
    axs[3].set_xlabel("x")
    axs[3].grid(true, linestyle="--", alpha=0.6)
    axs[3].legend()

    # Set common y-limits for all subplots
    for ax in axs
        ax.set_ylim(-0.4, 1.4)
        ax.axhline(0, color="black", linewidth=0.5)
    end
    
    fig.tight_layout()

    # Show the plot (CORRECTED: using display(fig))
    display(fig)
    println("Plot displayed. Compare the middle and right plots.")
end

# Run the main function
main()