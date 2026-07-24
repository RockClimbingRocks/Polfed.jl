module SpinOperators

export SpinBasis, sigma_0, sigma_x, sigma_y, sigma_z, get_spin, set_spin

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

@inline function get_spin(state::UInt64, site::Int, B::SpinBasis)
    shift = site * B.bits
    si = (state >> shift) & B.mask
    return Float64(si) - B.S
end

@inline function set_spin(state::UInt64, site::Int, B::SpinBasis, m_new)
    shift = site * B.bits
    state &= ~(B.mask << shift)
    state |= UInt64(round(Int, m_new + B.S)) << shift
    return state
end

@inline sigma_0(state::UInt64, site::Int, B::SpinBasis) = (1.0, state)

@inline function sigma_z(state::UInt64, site::Int, B::SpinBasis)
    m = get_spin(state, site, B)
    return (m, state)
end

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

end
