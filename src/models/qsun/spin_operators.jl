# On-site spin-`S` operators on a bit-packed computational basis.
#
# A single site holds one of `d = 2S + 1` magnetization states, stored in `bits`
# consecutive bits of a `UInt64`. Site `site` occupies bits `[site*bits, site*bits + bits)`
# (0-based site index). The operators below act on one site of such a packed
# state and return the resulting amplitude(s) and target state(s); the
# many-body Hamiltonian builders chain them across sites.

"""
    SpinBasis(S::Real)

Local (single-site) basis descriptor for a spin-`S` degree of freedom packed into
a `UInt64` computational state.

# Fields

- `S::Float64`:
  On-site spin. Must be integer or half-integer.
- `d::Int`:
  Local Hilbert-space dimension, `d = 2S + 1`.
- `bits::Int`:
  Number of bits used to store one site, `bits = ⌈log₂ d⌉`. This is the stride
  between consecutive sites in a packed `UInt64`.
- `mask::UInt64`:
  Bit mask `(1 << bits) - 1` selecting a single site's payload.

# Arguments

- `S::Real`:
  On-site spin passed to the constructor. Only the integer/half-integer values
  for which `2S` is an integer are physically meaningful; the constructor itself
  does not validate this (the caller [`qsun_hamiltonian`](@ref) does).

# Returns

- A `SpinBasis` with `d`, `bits`, and `mask` derived from `S`.

# Notes

- The magnetization stored at a site is an integer index `si ∈ 0:(d-1)`; the
  physical projection is recovered as `m = si - S` (see [`get_spin`](@ref)).
- For spin-1/2 (`S = 0.5`): `d = 2`, `bits = 1`, `mask = 0x1` — one bit per site.
- When `d` is not a power of two (e.g. spin-1, `d = 3` ⇒ `bits = 2`) some bit
  patterns are unused; the Hilbert-space enumeration in [`build_hilbert`](@ref)
  only ever produces valid `si ∈ 0:(d-1)` payloads.
"""
struct SpinBasis
    S::Float64
    d::Int
    bits::Int
    mask::UInt64
end

function SpinBasis(S::Real)
    d = Int(2S + 1)
    bits = ceil(Int, log2(d))
    mask = (UInt64(1) << bits) - 1
    return SpinBasis(Float64(S), d, bits, mask)
end

"""
    get_spin(state::UInt64, site::Int, B::SpinBasis) -> Float64

Read the magnetization projection `m` of `site` in the packed `state`.

# Arguments

- `state::UInt64`:
  Bit-packed many-body computational state.
- `site::Int`:
  Zero-based site index. Its payload lives in bits `[site*B.bits, site*B.bits + B.bits)`.
- `B::SpinBasis`:
  Local basis descriptor providing the stride `bits`, mask, and spin `S`.

# Returns

- `Float64`: the projection `m = si - S`, where `si ∈ 0:(d-1)` is the stored index.
  Ranges over `-S, -S+1, …, S`.

# Notes

- Inverse of [`set_spin`](@ref): `get_spin(set_spin(state, site, B, m), site, B) == m`.
"""
@inline function get_spin(state::UInt64, site::Int, B::SpinBasis)
    shift = site * B.bits
    si = (state >> shift) & B.mask
    return Float64(si) - B.S
end

"""
    set_spin(state::UInt64, site::Int, B::SpinBasis, m_new) -> UInt64

Return a copy of `state` with `site` set to magnetization projection `m_new`.

# Arguments

- `state::UInt64`:
  Bit-packed many-body computational state.
- `site::Int`:
  Zero-based site index to overwrite.
- `B::SpinBasis`:
  Local basis descriptor providing the stride and spin `S`.
- `m_new`:
  Target magnetization projection, in `-S:S`. Stored as index `si = round(Int, m_new + S)`.

# Returns

- `UInt64`: the state with the site's bits replaced; all other sites are unchanged.

# Notes

- The site's old payload is cleared with `~(mask << shift)` before the new index
  is written, so repeated calls are idempotent for a fixed `m_new`.
- Inverse of [`get_spin`](@ref).
"""
@inline function set_spin(state::UInt64, site::Int, B::SpinBasis, m_new)
    shift = site * B.bits
    state &= ~(B.mask << shift)
    state |= UInt64(round(Int, m_new + B.S)) << shift
    return state
end

"""
    sigma_0(state::UInt64, site::Int, B::SpinBasis) -> Tuple{Float64,UInt64}

Identity operator on `site`: returns amplitude `1.0` and the unchanged `state`.

# Returns

- `(1.0, state)`.

# Notes

- Provided for symmetry with the other on-site operators; useful when assembling
  operator products that may include identity factors.
"""
@inline sigma_0(state::UInt64, site::Int, B::SpinBasis) = (1.0, state)

"""
    sigma_z(state::UInt64, site::Int, B::SpinBasis) -> Tuple{Float64,UInt64}

Apply `Sᶻ` to `site`. Since the computational basis diagonalizes `Sᶻ`, this is a
multiplication by the projection `m`.

# Returns

- `(m, state)` where `m = get_spin(state, site, B)` and `state` is unchanged.

# Notes

- `Sᶻ` is diagonal, so the target state equals the input state.
"""
@inline function sigma_z(state::UInt64, site::Int, B::SpinBasis)
    m = get_spin(state, site, B)
    return (m, state)
end

"""
    sigma_x(state::UInt64, site::Int, B::SpinBasis)
        -> (has_up, c_up, s_up, has_down, c_down, s_down)

Apply `Sˣ = (S⁺ + S⁻)/2` to `site`, returning its (at most two) nonzero matrix
elements as a flat tuple.

# Returns

A 6-tuple describing the raising and lowering branches:

- `has_up::Bool`: whether `m < S`, i.e. the raised state `m+1` exists.
- `c_up::Float64`: amplitude `½√(S(S+1) − m(m+1))` of the raised branch (`0.0` if `!has_up`).
- `s_up::UInt64`: the state with `site` raised to `m+1` (`UInt64(0)` if `!has_up`).
- `has_down::Bool`: whether `m > -S`, i.e. the lowered state `m−1` exists.
- `c_down::Float64`: amplitude `½√(S(S+1) − m(m−1))` of the lowered branch (`0.0` if `!has_down`).
- `s_down::UInt64`: the state with `site` lowered to `m−1` (`UInt64(0)` if `!has_down`).

# Notes

- Coefficients follow the standard ladder relations `S⁺|m⟩ = √(S(S+1) − m(m+1))|m+1⟩`
  and `S⁻|m⟩ = √(S(S+1) − m(m−1))|m−1⟩`, each halved because `Sˣ = (S⁺+S⁻)/2`.
- For spin-1/2 the single nonzero amplitude equals `0.5` (so `Sˣ = ½σˣ`).
- The `has_*` flags must be checked before using the corresponding state/amplitude;
  the placeholder `UInt64(0)` is *not* a valid target when the branch is absent.
"""
@inline function sigma_x(state::UInt64, site::Int, B::SpinBasis)
    m = get_spin(state, site, B)
    S = B.S

    has_up = m < S
    has_down = m > -S

    c_up = has_up ? 0.5 * sqrt(S * (S + 1) - m * (m + 1)) : 0.0
    c_down = has_down ? 0.5 * sqrt(S * (S + 1) - m * (m - 1)) : 0.0

    s_up = has_up ? set_spin(state, site, B, m + 1) : UInt64(0)
    s_down = has_down ? set_spin(state, site, B, m - 1) : UInt64(0)

    return (has_up, c_up, s_up, has_down, c_down, s_down)
end

"""
    sigma_y(state::UInt64, site::Int, B::SpinBasis)
        -> (has_up, c_up, s_up, has_down, c_down, s_down)

Apply `Sʸ = (S⁺ − S⁻)/(2i)` to `site`, returning its (at most two) nonzero matrix
elements as a flat tuple, mirroring [`sigma_x`](@ref).

# Returns

A 6-tuple `(has_up, c_up, s_up, has_down, c_down, s_down)` where the amplitudes are
complex:

- `c_up = (1/2i)·√(S(S+1) − m(m+1))` for the raised branch (`0.0 + 0im` if `!has_up`).
- `c_down = −(1/2i)·√(S(S+1) − m(m−1))` for the lowered branch (`0.0 + 0im` if `!has_down`).

The `has_*` flags and target states `s_up`/`s_down` follow the same convention as
[`sigma_x`](@ref).

# Notes

- The relative sign between the branches (`+` raising, `−` lowering) is what makes
  `Sʸ` Hermitian and reproduces the spin algebra `[Sˣ, Sʸ] = i Sᶻ`.
- Used by the U(1)-conserving Quantum Sun builder, where the `SˣSˣ + SʸSʸ`
  combination is the magnetization-conserving (XY) coupling.
"""
@inline function sigma_y(state::UInt64, site::Int, B::SpinBasis)
    m = get_spin(state, site, B)
    S = B.S

    has_up = m < S
    has_down = m > -S

    c_up = has_up ? (1 / (2im)) * sqrt(S * (S + 1) - m * (m + 1)) : 0.0 + 0im
    c_down = has_down ? -(1 / (2im)) * sqrt(S * (S + 1) - m * (m - 1)) : 0.0 + 0im

    s_up = has_up ? set_spin(state, site, B, m + 1) : UInt64(0)
    s_down = has_down ? set_spin(state, site, B, m - 1) : UInt64(0)

    return (has_up, c_up, s_up, has_down, c_down, s_down)
end
