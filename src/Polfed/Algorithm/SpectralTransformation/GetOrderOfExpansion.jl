
"""
    analytical_orderofexpansion(λ::Real, δ::Real) -> Real

Heuristic estimate for Chebyshev expansion order as a function of interval
center `λ` and half-width `δ`.
"""
function analytical_orderofexpansion(λ::Real, δ::Real)
    return 2.65499*√(1-λ^2)/δ
end



"""
    getorderofexpansion!(transform_plan::TransformPlan) -> nothing

Resolve polynomial order if it was not explicitly set.
"""
function getorderofexpansion!(transform_plan::TransformPlan)
    is_order_set = !isnothing(transform_plan.order)

    !is_order_set && (getorderofexpansion_in_interval!(transform_plan))
end


# NOTE: If we reduce K via the safety factor, we must recompute bounds to keep the interval consistent.
"""
    getorderofexpansion_in_interval!(transform_plan::TransformPlan) -> nothing

Compute polynomial order from interval bounds and cutoff by bisection over
order, then apply `order_safety_factor`.

When `order_safety_factor < 1`, bounds are recomputed so the final interval is
consistent with the reduced order.
"""
function getorderofexpansion_in_interval!(transform_plan::TransformPlan)
    @unpack coefficients, polynomialtype, normalization, cutoff, left, right, order_safety_factor = transform_plan
    target = (left+right)/2
    halfwidth = (right-left)/2
    T = typeof(float(real(target)))

    bisection_fun(order) = begin
        order = floor(Int, order)
        transform = Clenshaw(polynomialtype, n -> coefficients(target,n), order, (y,x)->*(y,x), 1, T)
        normalization_ = transform(target)
        p(x::Real) = transform(x)/normalization_ * normalization

        p_a = p(left)
        p_b = p(right)

        max(p_a, p_b) >= cutoff && (return -1)
        max(p_a, p_b) <  cutoff && (return +1)
    end

    Kmax = ceil(Int, analytical_orderofexpansion(target, halfwidth)*2)
    Kmin = 10
    K = bisection(bisection_fun, Kmin, Kmax; tol=1)

    transform_plan.order = floor(Int64, K*order_safety_factor)

    (order_safety_factor < 1.) && (getbounds_from_K!(transform_plan, :mean))

    PolfedDefaults.polfed_log(
        PolfedDefaults.POLFED_INFO_LEVEL,
        "Polynomial order resolved.",
        order=transform_plan.order,
        order_safety_factor=order_safety_factor,
    )

end
