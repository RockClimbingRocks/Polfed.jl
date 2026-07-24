
function _window_reshape(λ::Real, position_of_λ::Symbol, δ::Real)
    position_of_λ === :center  && return (λ - δ, λ + δ)
    position_of_λ === :left  && return (λ, λ + 2δ)
    position_of_λ === :right && return (λ - 2δ, λ)
    throw(ArgumentError("Unknown window placement :$position_of_λ (expected :center, :left, or :right)"))
end


"""`_clamp_window((a, b))` — clamp a window to the rescaled spectrum `[-1, 1]`."""
_clamp_window((a, b)) = (max(Float64(a), -1.0), min(Float64(b), 1.0))


"""
    _size_window_from_dos(center, howmany, moments, kernel) -> (Float64, Float64)

Grow a symmetric window `[center-δ, center+δ]` (clamped to `[-1, 1]`) until it holds at
least `howmany` states, by bisecting the half-width `δ ∈ [0, 1]`.

The state count in `[a, b]` is `(1/π)·`[`integrate_chebyshev_weighted`](@ref) of the
kernel-damped Chebyshev coefficients `gₖ·μₖ·(k==0 ? 1 : 2)` — the same reconstruction as
`kpm_density`, integrated exactly instead of sampled. If the widest admissible window
(`δ = 1`) still holds fewer than `howmany` states, that window is returned with a warning.
"""
function _size_window_from_dos(target::Float64, position_of_target::Symbol, howmany::Int, dos::DoSResult)

    coeffs = dos.coefficients
    state_count(a, b) = integrate_chebyshev_weighted(coeffs, a, b) / π
    win_reshaped(δ) = _window_reshape(target, position_of_target, δ)
    win(δ) = _clamp_window(win_reshaped(δ))

    available = state_count(win(1.0)...)
    if available < howmany
        polfed_log(POLFED_WARN_LEVEL,
                   "The widest window around the target cannot hold the requested number of " *
                   "states; using the widest clamped window.",
                   howmany = howmany, available = available)
        return win(1.0)
    end
    # bfun(0) = +1 (empty window), bfun(1) = -1 (holds ≥ howmany): brackets the crossing.
    bfun(δ) = state_count(win(δ)...) >= howmany ? -1.0 : 1.0
    δ = bisection(bfun, 0.0, 1.0; tol=1e-8)
    return win(δ)
end


function resolve_interval(target::Target, dos::DoSResult)
    left, right = _size_window_from_dos(
        target.rescaled_energy,
        target.rescaled_energy_position,
        target.howmany,
        dos,
    )
    polfed_log(POLFED_INFO_LEVEL, "Target interval resolved from the DoS (rescaled coordinates).",
               left = left, right = right, width = right - left,
               center = target.rescaled_energy, howmany = target.howmany)
    return Interval(left, right)
end


function resolve_patched_intervals(target::Target, dos::DoSResult)
    
    left, right = _size_window_from_dos(
        target.rescaled_energy,
        target.rescaled_energy_position,
        target.howmany,
        dos,
    )
    return Interval(left, right)
end


"""
    howmany_in_interval(dos::DoSResult, interval::Interval) -> Int

Estimate how many eigenstates the rescaled `interval` holds, from the DoS.

The count is `(1/π)·`[`integrate_chebyshev_weighted`](@ref) of the kernel-damped Chebyshev
coefficients over `[left, right]` — the same density reconstruction as the DoS, integrated exactly
rather than sampled.
"""
function howmany_in_interval(dos::DoSResult, interval::Interval)
    count = integrate_chebyshev_weighted(dos.coefficients, interval.left, interval.right) / π
    return max(1, ceil(Int, count))
end
