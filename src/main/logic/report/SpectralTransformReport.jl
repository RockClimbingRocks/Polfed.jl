# Spectral-transformation section of the POLFED report: the target/window/filter/order the
# run resolved, and the headline cost (number of H·v multiplications). A plain-data snapshot
# derived from the resolved Interval + filter + mapping plan.

"""
    SpectralTransformReport

Immutable snapshot of POLFED's **spectral-transformation** stage for the report: where the
target sits, the rescaled window `[left, right]` exposed (and its true-energy image), the
filter/order chosen, and the total number of `H·v` multiplications performed.

Built by [`SpectralTransformReport`](@ref)`(interval, filter, mp, spec, blocksize, itersneeded,
howmany)` and rendered as the "Spectral Transformation" section by [`Report`](@ref)'s `show`.

# Fields
- `target_energy::Float64`, `target_rescaled::Float64`: the target center in true energy and in
  rescaled `[-1, 1]` coordinates.
- `target_strategy::String`: the user-facing target strategy (`":maxdos"`, `":middle"`, …).
- `left::Float64`, `right::Float64`: rescaled window endpoints.
- `energy_left::Float64`, `energy_right::Float64`: the same window mapped back to true energies.
- `filter_name::String`, `polynomial_type::String`: the filter type and its polynomial family.
- `order::Int`, `cutoff::Float64`: the resolved polynomial order `K` and the filter's suppression
  cutoff (`NaN` for a filter without one, e.g. an `EdgeFilter`).
- `matvecs::Int`: number of `H·v` multiplications `= K · blocksize · itersneeded + howmany` (the
  transform is POLFED's bottleneck, so this is the headline cost).
"""
struct SpectralTransformReport
    target_energy::Float64
    target_rescaled::Float64
    target_strategy::String
    left::Float64
    right::Float64
    energy_left::Float64
    energy_right::Float64
    filter_name::String
    polynomial_type::String
    order::Int
    cutoff::Float64
    matvecs::Int
end

"""
    format_target_strategy(spec::TargetSpec) -> String

The user-facing string for a resolved [`TargetSpec`](@ref) — e.g. `":maxdos"`,
`"(:unrescaled, 0.13)"` — for the report's target line.
"""
format_target_strategy(::TargetMaxDoS)      = ":maxdos"
format_target_strategy(::TargetMiddle)      = ":middle"
format_target_strategy(spec::TargetOffset)   = @sprintf("(:offset, %.6g)", spec.frac)
format_target_strategy(spec::TargetAbsolute) = @sprintf("(:unrescaled, %.6g)", spec.value)
format_target_strategy(spec::TargetRescaled) = @sprintf("(:rescaled, %.6g)", spec.value)
format_target_strategy(spec::TargetRescaledWindow) = @sprintf("(:rescaled, (%.6g, %.6g))", spec.left, spec.right)
format_target_strategy(spec::TargetAbsoluteWindow) = @sprintf("(:unrescaled, (%.6g, %.6g))", spec.left, spec.right)

"""
    SpectralTransformReport(interval, filter, mp, spec, blocksize, itersneeded, howmany)

Build the [`SpectralTransformReport`](@ref) from the resolved window, filter, mapping plan and
target spec. The rescaled target center is read from `filter.target`; its true-energy image and
the window's true-energy endpoints are recovered through the mapping (`E = spread·x̃ + center`).
The `H·v` cost is `polynomial_order(filter) · blocksize · itersneeded + howmany`.
"""
function SpectralTransformReport(interval::Interval, filter::AbstractFilter, mp::MappingPlan,
                                 spec::TargetSpec, blocksize::Int, itersneeded::Int, howmany::Int)
    x̃ = Float64(filter.target)
    order = polynomial_order(filter)
    matvecs = order * blocksize * itersneeded + howmany
    return SpectralTransformReport(
        mp.a * x̃ + mp.b, x̃, format_target_strategy(spec),
        interval.left, interval.right,
        mp.a * interval.left + mp.b, mp.a * interval.right + mp.b,
        string(nameof(typeof(filter))), string(polynomial_type(filter)),
        order, cutoff(filter), matvecs,
    )
end

"""
    render_transform(io::IO, f::Formatter, st::SpectralTransformReport) -> Nothing

Print the "Spectral Transformation" section of the report to `io`, coloring values via `f`.
"""
function render_transform(io::IO, f::Formatter, st::SpectralTransformReport)
    _section_header(io, f, "Spectral Transformation")
    println(io, "- target: ", cyan(f, st.target_strategy))
    println(io, "- interval: resc. [", cyan(f, @sprintf("%+.2e", st.left)), ",",
            cyan(f, @sprintf("%+.2e", st.right)), "], unresc. [",
            cyan(f, @sprintf("%+.2e", st.energy_left)), ",",
            cyan(f, @sprintf("%+.2e", st.energy_right)), "]")
    cutoff_str = isnan(st.cutoff) ? "" : string(" · cutoff ", cyan(f, @sprintf("%.2f", st.cutoff)))
    println(io, "- filter: ", cyan(f, st.filter_name),
            " · order K = ", cyan(f, string(st.order)), cutoff_str)
    println(io, "- mat-vec muls: ", cyan(f, _underscore(st.matvecs)))
    return nothing
end
