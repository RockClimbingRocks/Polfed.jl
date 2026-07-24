
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
