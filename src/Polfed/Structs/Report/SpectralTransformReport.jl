

"""
    struct SpectralTransformReport

Holds parameters and statistics for the spectral transformation phase of the computation.

# Constructor
    SpectralTransformReport(transform_plan,
                            mapping_plan,
                            fact::FactorizationReport)

# Fields
- `target::Real`: The target eigenvalue (if any).
- `target_strategy::String`: The user-facing target strategy used to choose the target.
- `left::Real`: Left endpoint of the spectral interval.
- `right::Real`: Right endpoint of the spectral interval.
- `polynomialtype::String`: The polynomial type used (e.g., `"Chebyshev"`).
- `order::Integer`: Order of the polynomial.
- `order_safety_factor::Real`: Safety multiplier applied to the order.
- `parallel_strategy::Parallelization`: Parallelization strategy used.
- `howmany::Integer`: Number of requested eigenpairs.
- `howmany_ininterval::Integer`: Number of eigenvalues found in the target interval.
- `matrixvec_muls::Integer`: Number of matrix–vector multiplications performed.
- `optimize_mapping::Bool`: Whether automatic mapping optimization was enabled.

# Notes
The field `howmany_ininterval` may be updated later, after results are known.
"""
mutable struct SpectralTransformReport
    target::Real
    target_strategy::String
    left::Real
    right::Real
    polynomialtype::String
    order::Integer
    order_safety_factor::Real
    parallel_strategy::Parallelization
    howmany::Integer
    howmany_ininterval::Integer
    matrixvec_muls::Integer
    optimize_mapping::Bool

    """Build `SpectralTransformReport` from transform/mapping plans and factorization report."""
    function SpectralTransformReport(transform_plan, mapping_plan, fact::FactorizationReport)
        target = isnothing(transform_plan.target) ? NaN : transform_plan.target
        target_strategy = format_target_strategy(transform_plan.target_spec)
        left   = isnothing(transform_plan.left)   ? NaN : transform_plan.left
        right  = isnothing(transform_plan.right)  ? NaN : transform_plan.right
        polynomialtype = String(transform_plan.polynomialtype)
        order  = isnothing(transform_plan.order) ? 0 : transform_plan.order
        howmany = transform_plan.howmany
        blocksize = fact.blocksize
        num_vecmuls = order * blocksize * fact.itersneeded + howmany

        new(
            target,
            target_strategy,
            left,
            right,
            polynomialtype,
            order,
            transform_plan.order_safety_factor,
            mapping_plan.parallel_strategy,
            howmany,
            0,   # howmany_ininterval → to be updated later
            num_vecmuls, 
            mapping_plan !== nothing && hasproperty(mapping_plan, :optimize_mapping) ? mapping_plan.optimize_mapping : false
        )
    end
end

@inline format_target_strategy(::TargetMaxDoS) = ":maxdos"
@inline format_target_strategy(::TargetMiddle) = ":middle"
@inline format_target_strategy(spec::TargetOffset) = @sprintf("(:offset, %.6g)", spec.frac)
@inline format_target_strategy(spec::TargetAbsolute) = @sprintf("(:unrescaled, %.6g)", spec.value)
@inline format_target_strategy(spec::TargetRescaled) = @sprintf("(:rescaled, %.6g)", spec.value)





"""
    display_spectral_report(report::SpectralTransformReport; use_colors::Bool=true)

Pretty-prints a [`SpectralTransformReport`](@ref) summarizing key parameters and statistics.

# Keyword Arguments
- `use_colors::Bool=true`: Enable or disable ANSI color formatting.

# Output
Displays:
- Target strategy, target energy, and number of eigenpairs requested.
- Spectral interval and width.
- Polynomial type and order.
- Number of matrix–vector multiplications performed.
- Parallelization strategy and automatic optimization status.
"""
function display_spectral_report(report::SpectralTransformReport, use_colors::Bool)
    f = Formatter(use_colors)

    format_with_underscores(n::Integer) = reverse(join(Iterators.partition(reverse(string(n)), 3), "_"))

    target   = cyan(f, @sprintf("%.6f", report.target))
    target_strategy = cyan(f, report.target_strategy)
    left     = cyan(f, @sprintf("%.2e", report.left))
    right    = cyan(f, @sprintf("%.2e", report.right))
    width    = cyan(f, @sprintf("δ = %.2e", report.right - report.left))
    order    = cyan(f, @sprintf("K = %d", report.order))
    osf      = cyan(f, @sprintf("%.2f", report.order_safety_factor))
    howmany  = cyan(f, string(report.howmany))
    num_mul  = cyan(f, format_with_underscores(report.matrixvec_muls))
    optimize_mapping = report.optimize_mapping ? green(f, "on") : yellow(f, "off")

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
    println("- Targeted $howmany eigenpairs with strategy $target_strategy at rescaled energy $target")
    println("- Exposing ev's in the rescaled interval [$left, $right], with width $width")
    println("- Performing '$(report.polynomialtype)' spectral transformation of order $order (and order safety factor $osf)")
    println("- Matrix multiplication performed $num_mul times! With parallelization strategy: $parallel")
    println("- Automatic optimization $optimize_mapping")
end
