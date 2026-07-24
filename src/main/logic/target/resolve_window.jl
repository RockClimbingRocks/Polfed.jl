
# Window target resolution: turn a window TargetSpec directly into the rescaled Interval POLFED
# exposes, and estimate its state count for the Krylov budget. Unlike the point path
# (resolve_target → resolve_interval, which GROWS a center to hold `howmany`), a window names the
# interval outright — so this only rescales (unrescaled windows) and validates/clamps to [-1, 1].

"""
    resolve_window(spec::TargetWindowSpec, a::Real, b::Real) -> Interval

The rescaled [`Interval`](@ref) a window spec names directly — no DoS growth:

- [`TargetRescaledWindow`](@ref): bounds are already rescaled; each is validated in `[-1, 1]`
  ([`_check_rescaled`](@ref)), erroring if outside (the caller chose rescaled coordinates).
- [`TargetAbsoluteWindow`](@ref): unrescaled physical bounds `E`, rescaled as `(E - b)/a`. A
  partially overhanging window is **clamped** to `[-1, 1]` with a `@warn`; a window entirely
  outside the estimated spectrum throws without first emitting a redundant warning.

`a`, `b` are the mapping spread/center (`a = (Emax-Emin)/2`, `b = (Emax+Emin)/2`).
"""
function resolve_window(spec::TargetRescaledWindow, ::Real, ::Real)
    left  = _check_rescaled(Float64(spec.left))
    right = _check_rescaled(Float64(spec.right))
    return Interval(left, right)
end

function resolve_window(spec::TargetAbsoluteWindow, a::Real, b::Real)
    l̃ = (Float64(spec.left)  - Float64(b)) / Float64(a)
    r̃ = (Float64(spec.right) - Float64(b)) / Float64(a)
    lo, hi = _clamp_window((l̃, r̃))
    lo < hi || throw(ArgumentError("Unrescaled window [$(spec.left), $(spec.right)] is empty after clamping to the spectrum [Emin, Emax]."))
    if l̃ < -1 || r̃ > 1
        polfed_log(POLFED_WARN_LEVEL,
                   "Unrescaled window exceeds the estimated spectrum [Emin, Emax]; clamping to [-1, 1].",
                   window = (spec.left, spec.right), rescaled = (l̃, r̃))
    end
    return Interval(lo, hi)
end
