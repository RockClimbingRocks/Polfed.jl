

"""
    struct SpectralTransformReport

Holds parameters and statistics for the spectral transformation phase of the computation.

# Constructor
    SpectralTransformReport(transform_plan,
                            mapping_plan,
                            fact::FactorizationReport)

# Fields
- `target::Real`: The target eigenvalue (if any).
- `left::Real`: Left endpoint of the spectral interval.
- `right::Real`: Right endpoint of the spectral interval.
- `polynomialtype::String`: The polynomial type used (e.g., `"Chebyshev"`).
- `order::Integer`: Order of the polynomial.
- `order_safety_factor::Real`: Safety multiplier applied to the order.
- `parallel_strategy::Parallelization`: Parallelization strategy used.
- `howmany::Integer`: Number of requested eigenpairs.
- `howmany_ininterval::Integer`: Number of eigenvalues found in the target interval.
- `matrixvec_muls::Integer`: Number of matrix–vector multiplications performed.
- `clenshaw_recurrence::Bool`: Whether Clenshaw recurrence optimization was used.

# Notes
The field `howmany_ininterval` may be updated later, after results are known.
"""
mutable struct SpectralTransformReport
    target::Real
    left::Real
    right::Real
    polynomialtype::String
    order::Integer
    order_safety_factor::Real
    parallel_strategy::Parallelization
    howmany::Integer
    howmany_ininterval::Integer
    matrixvec_muls::Integer
    clenshaw_recurrence::Bool

    """Build `SpectralTransformReport` from transform/mapping plans and factorization report."""
    function SpectralTransformReport(transform_plan, mapping_plan, fact::FactorizationReport)
        target = isnothing(transform_plan.target) ? NaN : transform_plan.target
        left   = isnothing(transform_plan.left)   ? NaN : transform_plan.left
        right  = isnothing(transform_plan.right)  ? NaN : transform_plan.right
        polynomialtype = String(transform_plan.polynomialtype)
        order  = isnothing(transform_plan.order) ? 0 : transform_plan.order
        howmany = transform_plan.howmany
        blocksize = fact.blocksize
        num_vecmuls = order * blocksize * fact.itersneeded + howmany

        new(
            target,
            left,
            right,
            polynomialtype,
            order,
            transform_plan.order_safety_factor,
            mapping_plan.parallel_strategy,
            howmany,
            0,   # howmany_ininterval → to be updated later
            num_vecmuls, 
            !isnothing(mapping_plan.clenshaw_recurrence)
        )
    end
end





"""
    display_spectral_report(report::SpectralTransformReport; use_colors::Bool=true)

Pretty-prints a [`SpectralTransformReport`](@ref) summarizing key parameters and statistics.

# Keyword Arguments
- `use_colors::Bool=true`: Enable or disable ANSI color formatting.

# Output
Displays:
- Target energy and number of eigenpairs requested.
- Spectral interval and width.
- Polynomial type and order.
- Number of matrix–vector multiplications performed.
- Parallelization strategy and Clenshaw recurrence status.
"""
function display_spectral_report(report::SpectralTransformReport, use_colors::Bool)
    f = Formatter(use_colors)

    format_with_underscores(n::Integer) = reverse(join(Iterators.partition(reverse(string(n)), 3), "_"))

    target   = cyan(f, @sprintf("%.6f", report.target))
    left     = cyan(f, @sprintf("%.6f", report.left))
    right    = cyan(f, @sprintf("%.6f", report.right))
    width    = cyan(f, @sprintf("δ = %.5f", report.right - report.left))
    order    = cyan(f, @sprintf("K = %d", report.order))
    osf      = cyan(f, @sprintf("%.2f", report.order_safety_factor))
    howmany  = cyan(f, string(report.howmany))
    num_mul  = cyan(f, format_with_underscores(report.matrixvec_muls))
    clenshaw = report.clenshaw_recurrence ? cyan(f, "enabled") : cyan(f, "disabled")

    # Parallelization strategy formatting
    parallel = if report.parallel_strategy isa NoParallel
        cyan(f, split(string(report.parallel_strategy), ".")[end])
    elseif report.parallel_strategy isa MulColsParallel
        cyan(f, split(string(report.parallel_strategy), ".")[end])
    elseif report.parallel_strategy isa TwoLevelParallel
        cyan(f, "TwoLevelParallel($(report.parallel_strategy.nt_per_col))")
    else
        throw(ArgumentError("Unknown parallelization strategy"))
    end

    println(blue(f, "Spectral Transformation Report:"))
    println("- Targeted $howmany eigenpairs at rescaled energy $target")
    println("- Exposing ev's in the rescaled interval [$left, $right], with width $width")
    println("- Performing '$(report.polynomialtype)' spectral transformation of order $order (and order safety factor $osf)")
    println("- Matrix multiplication performed $num_mul times! With parallelization strategy: $parallel")
    println("- Optimization with Clenshaw recurrence is $clenshaw")
end
