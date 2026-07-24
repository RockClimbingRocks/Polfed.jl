
"""
    Interval(left, right)

A closed window `[left, right] ⊆ [-1, 1]` of the **rescaled** spectrum that POLFED must
expose. Produced by [`resolve_target`](@ref) and consumed by the filter stage
(`build_filter`), which chooses the concrete filter from where the window sits and how
wide it is.
"""
struct Interval
    left::Float64
    right::Float64
end




include("resolve_interval.jl")






# """`_clamp_window((a, b))` — clamp a window to the rescaled spectrum `[-1, 1]`."""
# _clamp_window((a, b)) = (max(Float64(a), -1.0), min(Float64(b), 1.0))


# """
#     _size_window_from_dos(center, howmany, moments, kernel) -> (Float64, Float64)

# Grow a symmetric window `[center-δ, center+δ]` (clamped to `[-1, 1]`) until it holds at
# least `howmany` states, by bisecting the half-width `δ ∈ [0, 1]`.

# The state count in `[a, b]` is `(1/π)·`[`integrate_chebyshev_weighted`](@ref) of the
# kernel-damped Chebyshev coefficients `gₖ·μₖ·(k==0 ? 1 : 2)` — the same reconstruction as
# `kpm_density`, integrated exactly instead of sampled. If the widest admissible window
# (`δ = 1`) still holds fewer than `howmany` states, that window is returned with a warning.
# """
# function _size_window_from_dos(center::Float64, howmany::Int,
#                                moments::AbstractVector{<:Real}, kernel)
#     N = length(moments)
#     g = kernel isa Function ? kernel : get_kernel(kernel)
#     coeffs = [g(k, N) * moments[k+1] * (k == 0 ? 1 : 2) for k in 0:N-1]
#     state_count(a, b) = integrate_chebyshev_weighted(coeffs, a, b) / π
#     win(δ) = _clamp_window((center - δ, center + δ))

#     available = state_count(win(1.0)...)
#     if available < howmany
#         @warn "The widest window around the target cannot hold the requested number of \
#                states; using the widest clamped window." howmany available
#         return win(1.0)
#     end
#     # bfun(0) = +1 (empty window), bfun(1) = -1 (holds ≥ howmany): brackets the crossing.
#     bfun(δ) = state_count(win(δ)...) >= howmany ? -1.0 : 1.0
#     δ = bisection(bfun, 0.0, 1.0; tol=1e-8)
#     return win(δ)
# end