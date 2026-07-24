include("SpinOperators.jl")

using .SpinOperators

struct SpinHilbert
    basis::SpinBasis
    L::Int
    states::Vector{UInt64}
end

function build_hilbert(L::Int, basis::SpinBasis)
    states = UInt64[]

    function build_recursive(site::Int, current::UInt64)
        if site == L
            push!(states, current)
            return
        end

        shift = site * basis.bits
        for si in 0:(basis.d - 1)
            new_state = current | (UInt64(si) << shift)
            build_recursive(site + 1, new_state)
        end
    end

    build_recursive(0, UInt64(0))
    sort!(states)

    return SpinHilbert(basis, L, states)
end

@inline function state_index(H::SpinHilbert, state::UInt64)
    idx = searchsortedfirst(H.states, state)

    if idx <= length(H.states) && H.states[idx] == state
        return idx
    end

    return 0
end
