
get_bounds_fun = Dict{Symbol, Vector{Function}}(
    :mean => [(λ̃,δ) -> λ̃ - δ , (λ̃,δ) -> λ̃ + δ ],
    :left => [(λ̃,δ) -> λ̃     , (λ̃,δ) -> λ̃ + 2δ],
    :right=> [(λ̃,δ) -> λ̃ - 2δ, (λ̃,δ) -> λ̃     ],
)


"""
    getbounds!(transform_plan::TransformPlan, mapping_plan::MappingPlan, ρ::Function, target_position::Symbol) -> nothing

Resolve the rescaled spectral interval `[left, right]` in `transform_plan`.

Behavior depends on which fields are already set:
- explicit bounds: kept as-is (unless order is also set, which is invalid),
- explicit order only: bounds are derived from polynomial cutoff,
- neither: bounds are derived from DoS mass around target.
"""
function getbounds!(
    transform_plan::TransformPlan,
    mapping_plan::MappingPlan,
    ρ::Function,
    target_position::Symbol,
)
    are_bounds_set = !isnothing(transform_plan.left) && !isnothing(transform_plan.right)
    is_target_set = !isnothing(transform_plan.target)
    is_order_set = !isnothing(transform_plan.order)

    !is_target_set && resolve_target!(transform_plan, mapping_plan, ρ)

    if are_bounds_set && is_order_set
        throw(error("Both interval bounds and polynomial order are specified. Please set only one or neither."))
    elseif are_bounds_set
        nothing
    elseif is_order_set
        getbounds_from_K!(transform_plan, target_position)
    else
        getbounds_from_dos!(transform_plan, ρ, target_position)
    end

    PolfedDefaults.polfed_log(
        PolfedDefaults.POLFED_INFO_LEVEL,
        "Spectral bounds resolved.",
        left=transform_plan.left,
        right=transform_plan.right,
        target=transform_plan.target,
    )
end


"""
    getbounds_from_dos!(transform_plan::TransformPlan, ρ::Function, target_position::Symbol) -> nothing

Set bounds by integrating the DoS function `ρ` and choosing an interval that
contains at least `transform_plan.howmany` states.
"""
function getbounds_from_dos!(
    transform_plan::TransformPlan,
    ρ::Function,
    target_position::Symbol,
)
    @unpack target, howmany = transform_plan
    â, b̂ = get_bounds_fun[target_position]

    bisection_fun(δ) = begin
        a = max(â(target,δ), -1.)
        b = min(b̂(target,δ),  1.)

        howmany_dos, _ = quadgk(ρ, a, b; atol=1e-12, rtol=1e-12)
        howmany_dos >= howmany && (return -1)
        howmany_dos <  howmany && (return +1)
    end

    δmin = 0.
    δmax = 1.
    δ = bisection(bisection_fun, δmin, δmax; tol=1e-8)

    transform_plan.left = max(â(target,δ), -1.)
    transform_plan.right = min(b̂(target,δ),  1.)

end


"""
    getbounds_from_K!(transform_plan::TransformPlan, target_position::Symbol) -> nothing

Set bounds for a fixed polynomial order by solving for interval width where
the transformed filter reaches the configured cutoff at an endpoint.
"""
function getbounds_from_K!(
    transform_plan::TransformPlan,
    target_position::Symbol,
)
    @unpack order, polynomialtype, normalization, coefficients, cutoff, target = transform_plan
    â, b̂ = get_bounds_fun[target_position]
    T = typeof(float(real(target)))

    bisection_fun(δ) = begin
        a = max(â(target,δ), -1.)
        b = min(b̂(target,δ),  1.)
        transform = Clenshaw(polynomialtype, n -> coefficients(target,n), order, (y,x)->*(y,x), 1, T)
        normalization_ = transform(target)
        p(x::Real) = transform(x)/normalization_ * normalization

        p_a = p(a)
        p_b = p(b)

        max(p_a, p_b) >= cutoff && (return -1)
        max(p_a, p_b) <  cutoff && (return +1)
    end

    δmin = 0.
    δmax = 1.
    δ = bisection(bisection_fun, δmin, δmax; tol=1e-8)

    transform_plan.left = max(â(target,δ), -1.)
    transform_plan.right = min(b̂(target,δ), 1.)
end
