# Target resolution: turn a user target + requested state count + the DoS into the
# rescaled window POLFED must expose. This is the *logic/output* side of the target
# stage — the input vocabulary lives in target_spec.jl, the DoS adapter in calc_dos.jl.



function resolve_target(target, howmany::Integer, dos::DoSResult, a::Real, b::Real)
    spec = normalize_target_spec(target)
    center = _resolve_center(spec, dos, Float64(a), Float64(b))
    
    return Target(center, Int(howmany), :center)
end

# ── target center per spec (rescaled λ̃ ∈ [-1, 1]) ─────────────────────────────

"""
    _resolve_center(spec::TargetSpec, dos::DoSResult, a, b) -> Float64

The rescaled center `λ̃` the window is grown around, per [`TargetSpec`](@ref):

- [`TargetRescaled`](@ref)`(v)` → `v` (validated in `[-1, 1]`).
- [`TargetAbsolute`](@ref)`(E)` → `(E - b) / a` using the rescaling spread `a`
  and center `b`,
  validated in `[-1, 1]`.
- [`TargetMiddle`](@ref) → `0.0`.
    - [`TargetOffset`](@ref)`(η)` → linearly interpolates from the max-DoS energy
      to the nearest edge selected by the sign of `η`.
- [`TargetMaxDoS`](@ref) → `argmax` of the density `dos.ρ` over the open interior.

Only [`TargetMaxDoS`](@ref) inspects the DoS; the rest are coordinate conversions.

Returns
- `Float64`: the rescaled center `λ̃` in `[-1, 1]`.
"""
_resolve_center(spec::TargetRescaled, ::DoSResult, ::Float64, ::Float64) =
    _check_rescaled(Float64(spec.value))

function _resolve_center(spec::TargetAbsolute, ::DoSResult, a::Float64, b::Float64)
    λ̃ = (Float64(spec.value) - b) / a
    return _check_rescaled(λ̃; energy=spec.value)
end

_resolve_center(::TargetMiddle, ::DoSResult, ::Float64, ::Float64) = 0.0

function _offset_from_maxdos(εmaxdos::Float64, η::Float64)
    if η >= 0
        return εmaxdos + η * (1 - εmaxdos)
    else
        return εmaxdos + η * (1 + εmaxdos)
    end
end

function _resolve_center(spec::TargetOffset, dos::DoSResult, a::Float64, b::Float64)
    εmaxdos = _resolve_center(TargetMaxDoS(), dos, a, b)
    return _offset_from_maxdos(εmaxdos, Float64(spec.frac))
end

function _refined_argmax(f; left::Float64, right::Float64, npts::Int, refinements::Int)
    npts >= 3 || throw(ArgumentError("Max-DoS search needs at least 3 grid points, got $npts."))
    refinements >= 0 || throw(ArgumentError("Max-DoS search refinements must be >= 0, got $refinements."))

    best = left
    lo, hi = left, right

    for _ in 0:refinements
        xs = range(lo, hi; length = npts)
        imax = argmax(f(x) for x in xs)
        best = Float64(xs[imax])

        ilo = max(firstindex(xs), imax - 1)
        ihi = min(lastindex(xs), imax + 1)
        lo = Float64(xs[ilo])
        hi = Float64(xs[ihi])
    end

    return best
end

function _resolve_center(::TargetMaxDoS, dos::DoSResult, ::Float64, ::Float64;
                         npts::Int=2000, refinements::Int=2,
                         edge_margin::Float64=Defaults.polfed.maxdos_edge_margin)
    # ρ(x) = expansion(x)/(π√(1-x²)) carries the Chebyshev Jacobian, which DIVERGES as x → ±1.
    # For a bounded spectrum the true DoS vanishes at the edges, so the numerator should cancel the
    # √(1-x²) — but at finite KPM order + kernel damping that cancellation is imperfect, leaving a
    # large numerical artifact in a thin shell near ±1. Scanning all the way to the edge therefore
    # makes `argmax` snap to the endpoint regardless of where the real peak is. Keep a physical
    # `edge_margin` away from ±1 (matching the legacy `[-0.99, 0.99]` search) so the bulk mode wins.
    return _refined_argmax(dos.ρ; left = -1.0 + edge_margin, right = 1.0 - edge_margin,
                           npts = npts, refinements = refinements)
end




"""
    _check_rescaled(λ̃; energy=nothing, atol=1e-8) -> Float64

Validate that a rescaled target `λ̃` lies in `[-1, 1]`, throwing an informative
`ArgumentError` otherwise. `energy`, if given, is the original unrescaled value,
included in the message.

A target within `atol` outside is **snapped** onto the boundary rather than rejected: the
mapping's `Emin`/`Emax` are Ritz estimates that converge to the extremal eigenvalues from
*inside* the spectrum, so targeting an extremal energy exactly (e.g. the ground-state energy
from `eigvals`) legitimately rescales to `∓(1 + ε)` with `ε` at rounding scale. Beyond `atol`
the target is genuinely outside the estimated spectrum and the error stands.
"""
function _check_rescaled(λ̃::Float64; energy=nothing, atol::Float64=1e-8)
    -1 <= λ̃ <= 1 && return λ̃
    -1 - atol <= λ̃ <= 1 + atol && return clamp(λ̃, -1.0, 1.0)
    detail = isnothing(energy) ? "" : " (from unrescaled energy $energy)"
    throw(ArgumentError("Rescaled target λ̃ = $λ̃$detail is outside [-1, 1]; " *
                        "the target energy lies outside the estimated spectrum [Emin, Emax]."))
end
