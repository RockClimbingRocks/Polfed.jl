
mutable struct SpectralTransformReport
    target::Real
    left::Real
    right::Real
    polynomialtype::String
    order::Integer
    order_safety_factor::Real
    parallelization::Parallelization
    howmany::Integer
    howmany_ininterval::Integer
    matrixvec_muls::Integer
    clenshaw_recurrence::Bool

    function SpectralTransformReport(config::SpectralTransformConfigFull, fact::FactorizationReport)
        target = isnothing(config.target) ? NaN : config.target
        left   = isnothing(config.left)   ? NaN : config.left
        right  = isnothing(config.right)  ? NaN : config.right
        polynomialtype = String(config.polynomialtype)
        order  = isnothing(config.order) ? 0 : config.order
        howmany = config.howmany
        blocksize = fact.blocksize
        num_vecmuls = order * blocksize * fact.itersneeded + howmany

        new(
            target,
            left,
            right,
            polynomialtype,
            order,
            config.order_safety_factor,
            config.parallelization,
            howmany,
            0,   # howmany_ininterval → to be updated later
            num_vecmuls, 
            !isnothing(config.clenshaw_recurrence)
        )
    end
end



"""
    display_spectral_report(report::SpectralTransformReport; use_colors=true)

Pretty-prints a Spectral Transformation Report with optional ANSI colors.
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
    parallel = if report.parallelization isa NoParallel
        cyan(f, split(string(report.parallelization), ".")[end])
    elseif report.parallelization isa MulColsParallel
        cyan(f, split(string(report.parallelization), ".")[end])
    elseif report.parallelization isa TwoLevelParallel
        cyan(f, "TwoLevelParallel($(report.parallelization.nt_per_col))")
    else
        throw(ArgumentError("Unknown parallelization strategy"))
    end

    println(blue(f, "Spectral Transformation Report:"))
    println("- Targeted $howmany eigenpairs at energy $target")
    println("- Exposing ev's in the interval [$left, $right], with width $width")
    println("- Performing '$(report.polynomialtype)' spectral transformation of order $order (and order safety factor $osf)")
    println("- Matrix multiplication performed $num_mul times! With parallelization strategy: $parallel")
    println("- Optimization with Clenshaw recurrence is $clenshaw")
end

