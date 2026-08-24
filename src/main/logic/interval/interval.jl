
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


"""
    _fmt_iv(left, right) / _fmt_iv(iv::Interval) / _fmt_iv(iv, a, b) -> String

Compact `"[left, right]"` rendering of a window for log messages — the three-argument form maps
the rescaled window through the mapping first (`E = a·x̃ + b`), so a log line can show the same
window in both coordinate systems. Log-only; the report has its own formatting.
"""
_fmt_iv(left::Real, right::Real) = @sprintf("[%.6g, %.6g]", left, right)
_fmt_iv(iv::Interval) = _fmt_iv(iv.left, iv.right)
_fmt_iv(iv::Interval, a::Real, b::Real) = _fmt_iv(a * iv.left + b, a * iv.right + b)




include("resolve_interval.jl")
