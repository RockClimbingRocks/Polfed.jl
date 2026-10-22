# Target input vocabulary: the ways a user can name the energy POLFED should aim at,
# and the parser that normalizes loose user input into one of them. This is purely the
# *input* side — turning a spec into an actual rescaled window is `resolve_target`.

"""
    TargetSpec

Supertype for the ways a POLFED target energy can be specified. User input to the
entrypoint is normalized into one of these by [`normalize_target_spec`](@ref); the
target *center* it denotes is then grown into a rescaled window ([`Interval`](@ref))
by [`resolve_target`](@ref).

Concrete specs come in two families. **Point** specs name a single center energy that is
then grown into a window holding `howmany` states:
- [`TargetMaxDoS`](@ref) — peak of the density of states.
- [`TargetMiddle`](@ref) — center of the (rescaled) spectrum.
- [`TargetOffset`](@ref) — a fractional offset into the rescaled spectrum.
- [`TargetAbsolute`](@ref) — an unrescaled physical energy.
- [`TargetRescaled`](@ref) — an already-rescaled value in `[-1, 1]`.

**Window** specs ([`TargetWindowSpec`](@ref)) name the target interval *directly* — no DoS
growth, and `howmany` is derived from the DoS count over the window instead of supplied:
- [`TargetRescaledWindow`](@ref) — an already-rescaled window `⊆ [-1, 1]`.
- [`TargetAbsoluteWindow`](@ref) — an unrescaled physical-energy window.
"""
abstract type TargetSpec end

"""
    TargetWindowSpec <: TargetSpec

Supertype for the specs that name the target *interval* directly (rather than a center to
grow). The pipeline branches on this: a window spec resolves straight to the rescaled
[`Interval`](@ref) and derives `howmany` from the DoS, whereas a point [`TargetSpec`](@ref)
grows a center via [`resolve_target`](@ref)/[`resolve_interval`](@ref). Concrete windows:
[`TargetRescaledWindow`](@ref), [`TargetAbsoluteWindow`](@ref).
"""
abstract type TargetWindowSpec <: TargetSpec end

"""Target the energy where the density of states is maximal."""
struct TargetMaxDoS <: TargetSpec end

"""Target the center of the rescaled spectrum (`λ̃ = 0`)."""
struct TargetMiddle <: TargetSpec end

"""Target a fractional offset `frac ∈ [-1, 1]` from max DoS toward a spectral edge.
`frac = 0` targets max DoS, `frac = 1` targets `+1`, and `frac = -1` targets `-1`.
Values in between linearly interpolate on the corresponding side."""
struct TargetOffset{T<:Real} <: TargetSpec
    frac::T
end

"""Target an unrescaled physical energy `value`, rescaled via the mapping."""
struct TargetAbsolute{T<:Real} <: TargetSpec
    value::T
end

"""Target an already-rescaled `value ∈ [-1, 1]`."""
struct TargetRescaled{T<:Real} <: TargetSpec
    value::T
end

"""
    TargetRescaledWindow(left, right)

Target an already-rescaled window `[left, right] ⊆ [-1, 1]` directly. Both bounds are in
rescaled coordinates; `left < right` is enforced at construction, the `⊆ [-1, 1]` check
happens at resolution. `howmany` is derived from the DoS count over the window.
"""
struct TargetRescaledWindow{T<:Real} <: TargetWindowSpec
    left::T
    right::T
    function TargetRescaledWindow(left::Real, right::Real)
        left < right || throw(ArgumentError("TargetRescaledWindow requires left < right, got ($left, $right)"))
        l, r = promote(left, right)
        new{typeof(l)}(l, r)
    end
end

"""
    TargetAbsoluteWindow(left, right)

Target an unrescaled physical-energy window `[left, right]`, rescaled into `[-1, 1]` via the
mapping at resolution time. `left < right` is enforced at construction. This is the family a
bare `(lo, hi)` tuple normalizes to.
"""
struct TargetAbsoluteWindow{T<:Real} <: TargetWindowSpec
    left::T
    right::T
    function TargetAbsoluteWindow(left::Real, right::Real)
        left < right || throw(ArgumentError("TargetAbsoluteWindow requires left < right, got ($left, $right)"))
        l, r = promote(left, right)
        new{typeof(l)}(l, r)
    end
end

"""
    normalize_target_spec(target) -> TargetSpec

Normalize user target input into a [`TargetSpec`](@ref).

# Accepted forms
- an existing `TargetSpec` (returned unchanged),
- `:maxdos` → [`TargetMaxDoS`](@ref),
- `:middle` → [`TargetMiddle`](@ref),
- `(:offset, η)` with `η ∈ [-1, 1]` → [`TargetOffset`](@ref),
- `(:unrescaled, E)` → [`TargetAbsolute`](@ref),
- `(:rescaled, v)` → [`TargetRescaled`](@ref),
- a plain `value::Real` → [`TargetAbsolute`](@ref) (interpreted as an unrescaled energy),
- `(:rescaled, (lo, hi))` → [`TargetRescaledWindow`](@ref),
- `(:unrescaled, (lo, hi))` → [`TargetAbsoluteWindow`](@ref),
- a bare `(lo, hi)` of two `Real`s → [`TargetAbsoluteWindow`](@ref) (unrescaled by default).

The second element of a tagged tuple disambiguates point vs window: a `Real` gives the point
spec, a `(lo, hi)` tuple gives the window spec.

# Throws
- `ArgumentError`: on an unknown symbol/tuple tag, a non-`Real`/`(lo,hi)` tuple value, an
  out-of-range offset, or an unsupported target type.
"""
function normalize_target_spec(target)
    if target isa TargetSpec
        return target
    end

    if target isa Symbol
        target === :maxdos && return TargetMaxDoS()
        target === :middle && return TargetMiddle()
        throw(ArgumentError("Unknown target symbol: :$target (expected :maxdos or :middle)"))
    end

    if target isa Tuple && length(target) == 2 && target[1] isa Symbol
        tag = target[1]
        val = target[2]

        # Window forms: a (lo, hi) second element selects the window spec over the point spec.
        if val isa Tuple
            (length(val) == 2 && val[1] isa Real && val[2] isa Real) ||
                throw(ArgumentError("Window target must be (lo, hi) of two Reals, got $val"))
            tag === :rescaled   && return TargetRescaledWindow(val[1], val[2])
            tag === :unrescaled && return TargetAbsoluteWindow(val[1], val[2])
            throw(ArgumentError("Unknown window target tag: :$tag (expected :rescaled or :unrescaled)"))
        end

        val isa Real || throw(ArgumentError("Target tuple value must be Real or (lo, hi), got $(typeof(val))"))
        if tag === :offset
            (-1 <= val <= 1) || throw(ArgumentError("Offset target requires frac ∈ [-1, 1], got $val"))
            return TargetOffset(val)
        end
        tag === :unrescaled && return TargetAbsolute(val)
        
        if tag === :rescaled 
            (-1 <= val <= 1) || throw(ArgumentError("Rescaled target requires value ∈ [-1, 1], got $val"))
            return TargetRescaled(val)
        end
        throw(ArgumentError("Unknown target tuple tag: :$tag (expected :offset, :unrescaled, or :rescaled)"))
    end

    # Bare (lo, hi) of two Reals → unrescaled window (the untagged default).
    if target isa Tuple && length(target) == 2 && target[1] isa Real && target[2] isa Real
        return TargetAbsoluteWindow(target[1], target[2])
    end

    if target isa Real
        return TargetAbsolute(target)
    end

    throw(ArgumentError("Unsupported target specification: $target"))
end
