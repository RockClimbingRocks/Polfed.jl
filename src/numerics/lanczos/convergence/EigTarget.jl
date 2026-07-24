# ═══════════════════════════════════════════════════════════════════════════════
# TYPES ONLY. No logic lives here — see `end_spec.jl` for the per-spec rules and
# `converged_run.jl` for the walker. Keeping this file logic-free is deliberate:
# in the current module `EigTarget.jl` is 326 lines of interleaved types, index
# arithmetic, and window brackets, which is most of why the module is hard to read.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EndSpec

How far to converge at **one** spectral end. Three concrete specs, distinguished by
what their stopping rule is allowed to look at:

| spec                        | stop rule reads | needs eigenvectors? |
|:----------------------------|:----------------|:--------------------|
| [`NumEigs`](@ref)           | run length      | no                  |
| [`ToValue`](@ref)           | `p` only        | no                  |
| [`Window`](@ref)            | `E`, optionally a `p` boundary | yes |

See the module README for the meaning of the `p` / `E` coordinate spaces.
"""
abstract type EndSpec end

"""
    NumEigs(n)

Converge a fixed number `n` of eigenpairs at this end.
"""
struct NumEigs <: EndSpec
    n::Int
end

"""
    ToValue(value)

Converge **all** eigenpairs from this end out to `value`, stated in `p` (transformed)
units. Completeness is confirmed when the contiguous converged run has stepped *past*
`value` — the pair beyond the boundary is the completeness guard.

The count is dynamic (determined by how many `p` land inside the boundary).
"""
struct ToValue{T<:Real} <: EndSpec
    value::T
end

"""
    Window(low, high)
    Window(low, high, edge_value; return_brackets=false)

Converge eigenpairs whose **mapped** value `E` lies in `[low, high]`.

`low`/`high` are in `E` (mapped) units — the units `mapvals` produces — so the caller must
pass a `mapvals` in the same space.

The two-argument constructor preserves the live API. It walks inward from the targeted
transformed extreme and stops after its converged run straddles the mapped window. It returns
the in-window pairs plus one *converged bracket candidate* on each finite side. With an
oscillatory filter tail those candidates are not guaranteed to be immediate mapped-energy
neighbours; `p = P(E)` order and `E` order are different outside the passband.

The three-argument constructor is the convergence-safe form for patching. `edge_value` is a
transformed-space boundary (normally the validated filter cutoff). The run must converge one
pair past that boundary before `[low, high]` is selected. Adjacent patches should pass
overlapping mapped intervals (normally overlapping filter passbands) and use
`return_brackets=false`, so their shared eigenpairs form a contiguous suffix/prefix. This form
keeps the old `Window(low, high, edge_value)` call shape source-compatible while the live
two-argument constructor remains valid.

An infinite bound (`±Inf`) means the window runs past the end of the spectrum (an edge
window): no eigenpair can exist beyond it, so that side's straddle is waived.

`return_brackets=true` asks selection to retain one bracket candidate on each finite side.
It defaults to `true` for the legacy two-argument constructor and `false` for the certified
three-argument constructor.
"""
struct Window{T<:Real} <: EndSpec
    low::T
    high::T
    edge_value::Union{Nothing,T}
    return_brackets::Bool

    function Window(low::Real, high::Real, edge_value::Union{Nothing,Real},
                    return_brackets::Bool)
        low < high || throw(ArgumentError("Window requires low < high, got ($low, $high)"))
        if edge_value === nothing
            l, h = promote(low, high)
            return new{typeof(l)}(l, h, nothing, return_brackets)
        end

        isfinite(edge_value) || throw(ArgumentError(
            "Window edge_value must be finite, got $edge_value",
        ))
        l, h, edge = promote(low, high, edge_value)
        return new{typeof(l)}(l, h, edge, return_brackets)
    end
end

Window(low::Real, high::Real) = Window(low, high, nothing, true)
Window(low::Real, high::Real, edge_value::Real; return_brackets::Bool=false) =
    Window(low, high, edge_value, return_brackets)

"""
    EigTarget

Which spectral end(s) an [`EndSpec`](@ref) is placed at: [`Lowest`](@ref),
[`Highest`](@ref), or [`BothEnds`](@ref).
"""
abstract type EigTarget end

"""    Lowest(spec::EndSpec) — target the low (smallest `p`) end."""
struct Lowest{S<:EndSpec} <: EigTarget
    spec::S
end

"""    Highest(spec::EndSpec) — target the high (largest `p`) end."""
struct Highest{S<:EndSpec} <: EigTarget
    spec::S
end

"""
    BothEnds(low::EndSpec, high::EndSpec)

Target both extremes in one Krylov space, with an independent spec per end. Each end is
walked and decided independently; the returned set is the union of the two contiguous
runs (the middle of the spectrum is legitimately absent — that is not a hole).
"""
struct BothEnds{L<:EndSpec, H<:EndSpec} <: EigTarget
    low::L
    high::H
end

# --- convenience aliases -------------------------------------------------------
"""    LowestEigs(n) ≡ `Lowest(NumEigs(n))`."""
LowestEigs(n::Integer) = Lowest(NumEigs(Int(n)))
"""    HighestEigs(n) ≡ `Highest(NumEigs(n))`."""
HighestEigs(n::Integer) = Highest(NumEigs(Int(n)))
"""    BothEnds(n_low, n_high) ≡ `BothEnds(NumEigs(n_low), NumEigs(n_high))`."""
BothEnds(n_low::Integer, n_high::Integer) = BothEnds(NumEigs(Int(n_low)), NumEigs(Int(n_high)))

# --- ends of a target, as (spec, islow) pairs ----------------------------------
# Everything downstream is written once against a single end and reused, so `Lowest`,
# `Highest` and `BothEnds` need no separate code paths beyond this one function.

"""
    ends(target::EigTarget) -> Tuple of (spec::EndSpec, islow::Bool)

The spectral ends a target covers. One entry for [`Lowest`](@ref)/[`Highest`](@ref),
two for [`BothEnds`](@ref). `islow` selects the walk direction (see `end_walk`).
"""
ends(t::Lowest)   = ((t.spec, true),)
ends(t::Highest)  = ((t.spec, false),)
ends(t::BothEnds) = ((t.low, true), (t.high, false))

"""
    requested_count(target::EigTarget) -> Int

Number of requested eigenpairs for count-based targets, or `-1` when dynamic. Used only
to schedule the first check and for reporting.
"""
requested_count(t::Lowest{NumEigs})           = t.spec.n
requested_count(t::Highest{NumEigs})          = t.spec.n
requested_count(t::BothEnds{NumEigs,NumEigs}) = t.low.n + t.high.n
requested_count(::EigTarget)                  = -1
