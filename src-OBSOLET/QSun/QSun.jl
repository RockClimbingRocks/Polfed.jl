module QSun

using SparseArrays, LinearAlgebra, Random

include("RandomUtils.jl")
include("HilbertSpace.jl")
include("QuantumSun.jl")
include("QuantumSun_U1_conserving.jl")

export SpinBasis, SpinHilbert
export build_hilbert, state_index
export get_spin, set_spin, sigma_0, sigma_x, sigma_y, sigma_z
export uniform_array, goe_matrix, random_neighbors, long_range_couplings
export build_hamiltonian, build_hamiltonian_cons, qsun_hamiltonian

"""
    qsun_hamiltonian(
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

Construct a Quantum Sun Hamiltonian. By default this builds the full Hilbert
space; pass `use_U1=true` and `S_z=...` to build the fixed-magnetization U(1)
sector.
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

end
