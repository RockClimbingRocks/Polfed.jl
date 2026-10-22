# Enumeration of the many-body computational basis and fast state → index lookup.
# `spin_operators.jl` (included before this file in QSun.jl) provides `SpinBasis`
# and the bit-packing convention these states use.

"""
    SpinHilbert

The full computational basis of an `L`-site spin chain with a fixed local basis.

# Fields

- `basis::SpinBasis`:
  Local single-site descriptor (spin `S`, bits per site, mask).
- `L::Int`:
  Number of sites.
- `states::Vector{UInt64}`:
  All `d^L` bit-packed basis states, sorted in ascending integer order so that
  [`state_index`](@ref) can binary-search them.

# Notes

- Construct with [`build_hilbert`](@ref) rather than directly, so that `states`
  is guaranteed enumerated and sorted.
- For spin-1/2 (`d = 2`) the states are exactly `0 : 2^L - 1`.
"""
struct SpinHilbert
    basis::SpinBasis
    L::Int
    states::Vector{UInt64}
end

"""
    build_hilbert(L::Int, basis::SpinBasis) -> SpinHilbert

Enumerate the full `d^L`-dimensional computational basis of an `L`-site chain.

# Arguments

- `L::Int`:
  Number of sites.
- `basis::SpinBasis`:
  Local basis descriptor (sets `d` and the per-site bit stride).

# Returns

- A [`SpinHilbert`](@ref) whose `states` field holds all `d^L` packed states,
  sorted ascending.

# Notes

- States are generated recursively, assigning each site one of its `d` local
  indices `0:(d-1)`; the result is then sorted so that membership and indexing use
  a binary search (see [`state_index`](@ref)).
- This enumerates the *entire* Hilbert space; the U(1)-conserving builder
  ([`build_hamiltonian_cons`](@ref)) restricts to a fixed-magnetization sector
  afterwards rather than enumerating it directly.
- Cost and memory grow as `d^L`; this is the exponential wall that POLFED's
  sparse, matrix-free machinery is built to push back.
"""
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

"""
    state_index(H::SpinHilbert, state::UInt64) -> Int

Look up the 1-based position of `state` in the enumerated basis `H.states`.

# Arguments

- `H::SpinHilbert`:
  The enumerated, sorted basis to search.
- `state::UInt64`:
  A bit-packed many-body state.

# Returns

- `Int`: the index `k` such that `H.states[k] == state`, or `0` if `state` is not
  present in the basis.

# Notes

- Uses `searchsortedfirst` on the sorted `states`, so lookups are `O(log d^L)`.
- The `0` sentinel for "not found" is how the Hamiltonian builders skip
  off-sector transitions (a flipped state that leaves the enumerated basis).
"""
@inline function state_index(H::SpinHilbert, state::UInt64)
    idx = searchsortedfirst(H.states, state)

    if idx <= length(H.states) && H.states[idx] == state
        return idx
    end

    return 0
end
