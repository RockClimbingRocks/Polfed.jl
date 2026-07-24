module QSun

# Quantum Sun model: a chaotic GOE "grain" of `L_grain` spins coupled to `L_loc`
# localized spins through long-range couplings. Submodule of `Models`.
#
# Files are plain `include`d text sharing this module's namespace (no nested
# modules — the old `SpinOperators` submodule was flattened in the restructure).
# Include order is dependency-first:
#   spin_operators        SpinBasis + on-site Sˣ/Sʸ/Sᶻ and bit packing
#   random_utils          disorder, GOE grain, attachment sites, couplings
#   hilbert_space         basis enumeration + state→index lookup (needs SpinBasis)
#   quantum_sun           full-space builder        (needs all of the above)
#   quantum_sun_u1_…      fixed-magnetization builder (needs all of the above)

using SparseArrays, LinearAlgebra, Random

include("spin_operators.jl")
include("random_utils.jl")
include("hilbert_space.jl")
include("quantum_sun.jl")
include("quantum_sun_u1_conserving.jl")

export SpinBasis, SpinHilbert
export build_hilbert, state_index
export get_spin, set_spin, sigma_0, sigma_x, sigma_y, sigma_z
export uniform_array, goe_matrix, random_neighbors, long_range_couplings
export build_hamiltonian, build_hamiltonian_cons, qsun_hamiltonian

"""
    qsun_hamiltonian(
        L_loc, L_grain, g0, α;
        γ=1, w=0.5, hz=1, ζ=0.2, S=0.5,
        rng=Random.default_rng(), use_U1=false, S_z=0.0, use_sparse=true,
    )

Construct a Quantum Sun Hamiltonian, dispatching on the `use_U1` flag.

By default (`use_U1=false`) this builds the full-Hilbert-space model via
[`build_hamiltonian`](@ref). With `use_U1=true` it builds the magnetization-
conserving model restricted to the total-`Sᶻ = S_z` sector via
[`build_hamiltonian_cons`](@ref).

# Positional Arguments

- `L_loc`:
  Number of localized ("ray") spins coupled to the grain.
- `L_grain`:
  Number of spins inside the ergodic grain.
- `g0`:
  Overall prefactor of the grain-to-ray couplings.
- `α`:
  Decay base of the long-range coupling profile `α^{u_j}`.

# Keyword Arguments

- `γ=1`:
  Energy scale of the random grain block.
- `w=0.5`:
  Half-width of the uniform disorder window for the local `Sᶻ` fields
  (sampled from `[hz - w, hz + w]`).
- `hz=1`:
  Center of the local-field disorder window.
- `ζ=0.2`:
  Randomness half-width of the effective distance exponent `u_j = (j-1) + η_j`.
- `S=0.5`:
  On-site spin. Must be integer or half-integer (`2S ∈ ℤ`), else an error is thrown.
- `rng=Random.default_rng()`:
  Random-number generator used for disorder, couplings, and the grain matrix.
- `use_U1=false`:
  If `false`, build on the full Hilbert space; if `true`, build the U(1)-symmetric
  model in the fixed `S_z` sector.
- `S_z=0.0`:
  Total magnetization sector used when `use_U1=true` (ignored otherwise).
- `use_sparse=true`:
  Return a `SparseMatrixCSC` if `true`, a dense `Matrix` if `false`.

# Returns

- `SparseMatrixCSC{Float64,Int}` when `use_sparse=true`, otherwise `Matrix{Float64}`.
  The dimension is `d^{L_loc+L_grain}` in the full case, or the size of the `S_z`
  sector in the U(1) case (`d = 2S + 1`).

# Notes

- The only validation performed here is that `S` is integer or half-integer; sector
  validity for `use_U1=true` is checked inside [`build_hamiltonian_cons`](@ref).
- `Models.qsun_hamiltonian` is a thin keyword-typed wrapper around this method; use
  whichever entry point is convenient.
"""
function qsun_hamiltonian(
    L_loc,
    L_grain,
    g0,
    α;
    γ=1,
    w=0.5,
    hz=1,
    ζ=0.2,
    S=0.5,
    rng=Random.default_rng(),
    use_U1=false,
    S_z=0.0,
    use_sparse=true,
)
    isinteger(2S) || error("Wrong value of spin. Only integer spins or half-integer spins are possible. You chose S = $S")

    if use_U1
        return build_hamiltonian_cons(
            L_loc,
            L_grain,
            g0,
            α;
            γ=γ,
            w=w,
            hz=hz,
            ζ=ζ,
            S=S,
            S_z=S_z,
            rng=rng,
            use_sparse=use_sparse,
        )
    end

    return build_hamiltonian(
        L_loc,
        L_grain,
        g0,
        α;
        γ=γ,
        w=w,
        hz=hz,
        ζ=ζ,
        S=S,
        rng=rng,
        use_sparse=use_sparse,
    )
end

end # module QSun
