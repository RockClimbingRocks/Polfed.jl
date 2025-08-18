
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

    function SpectralTransformReport(config::SpectralTransformConfigFull)
        target = isnothing(config.target) ? NaN : config.target
        left   = isnothing(config.left)   ? NaN : config.left
        right  = isnothing(config.right)  ? NaN : config.right
        polynomialtype = String(config.polynomialtype)
        order  = isnothing(config.order) ? 0 : config.order
        howmany = config.howmany

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
            0,   # matrixvec_muls     → to be updated later
            !isnothing(config.clenshaw_recurrence)
        )
    end
end



function display_report(report::SpectralTransformReport)
    format_with_underscores(n::Integer) = reverse(join(Iterators.partition(reverse(string(n)), 3), "_"))

    target   = @sprintf("\e[1;36m%.6f\e[0m", report.target)
    left     = @sprintf("\e[1;36m%.6f\e[0m", report.left)
    right    = @sprintf("\e[1;36m%.6f\e[0m", report.right)
    width        = @sprintf("\e[1;36mδ = %.5f\e[0m", report.right - report.left)
    order        = @sprintf("\e[1;36m K = %d \e[0m", report.order)
    howmany  = @sprintf("\e[1;36m %d \e[0m", report.howmany)
    num_mul  = @sprintf("\e[1;36m %s \e[0m", format_with_underscores(report.matrixvec_muls))
    clenshaw = report.clenshaw_recurrence ? "enabled" : "disabled"
    parall = String(report.parallelization) 


    println("\e[1;34mSpectral Transformation Report:\e[0m")
    println("- Targeted $howmany eigenpairs at energy $target")
    println("- Exposing ev's in the interval [$left, $right], with width $width")
    println("- Performing '$(report.polynomialtype)' spectral transformation of order $order")
    println("- Matrix multiplication performed $num_mul times! With parallelization strategy: $parall")
    println("- Optimization with Clenshaw recurrence is $clenshaw")
end




