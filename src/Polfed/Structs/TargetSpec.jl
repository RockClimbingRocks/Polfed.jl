abstract type TargetSpec end

struct TargetMaxDoS <: TargetSpec end
struct TargetMiddle <: TargetSpec end
struct TargetOffset <: TargetSpec
    frac::Real
end
struct TargetAbsolute{T<:Real} <: TargetSpec
    value::T
end
struct TargetRescaled{T<:Real} <: TargetSpec
    value::T
end

"""
Normalize user input into a TargetSpec.

Accepted forms:
- `:maxdos`
- `:middle`
- `(:offset, eta)` with `eta ∈ [-1, 1]`
- `(:unrescaled, value)`
- `(:rescaled, value)`
- `value::Real`
"""
function normalize_target_spec(target)
    if target isa TargetSpec
        return target
    end

    if target isa Symbol
        target === :maxdos && return TargetMaxDoS()
        target === :middle && return TargetMiddle()
        throw(ArgumentError("Unknown target symbol: $target"))
    end

    if target isa Tuple && length(target) == 2 && target[1] isa Symbol
        tag = target[1]
        val = target[2]
        val isa Real || throw(ArgumentError("Target tuple value must be Real, got $(typeof(val))"))
        tag === :absolute && throw(ArgumentError("`(:absolute, value)` is no longer supported. Use `(:unrescaled, value)` instead."))
        tag === :quantile && throw(ArgumentError("`(:quantile, q)` is no longer supported."))
        if tag === :offset
            (-1 <= val <= 1) || throw(ArgumentError("Offset target requires `eta ∈ [-1, 1]`, got $val"))
            return TargetOffset(val)
        end
        tag === :unrescaled && return TargetAbsolute(val)
        tag === :rescaled && return TargetRescaled(val)
        throw(ArgumentError("Unknown target tuple tag: $tag"))
    end

    if target isa Real
        # Plain numeric targets are interpreted as unrescaled energies.
        return TargetAbsolute(target)
    end

    throw(ArgumentError("Unsupported target specification: $target"))
end
