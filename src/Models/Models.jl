module Models

using LinearAlgebra
using Random
using SparseArrays

using ..QSun

include("SpinHalfChains.jl")

"""
    qsun_hamiltonian(
        L_loc::Integer,
        L_grain::Integer,
        g0::Real,
        α::Real;
        γ::Real=1,
        w::Real=0.5,
        hz::Real=1,
        ζ::Real=0.2,
        S::Real=0.5,
        rng::AbstractRNG=Random.default_rng(),
        use_U1::Bool=false,
        S_z::Real=0.0,
        use_sparse::Bool=true,
    )

Construct a Quantum Sun Hamiltonian through the public `Polfed.Models`
namespace.

By default this builds the full Hilbert-space Quantum Sun model. With
`use_U1=true`, it instead builds the U(1)-symmetric version in a fixed total
magnetization sector.

# Positional Arguments

- `L_loc::Integer`:
  Number of outer localized spins ("rays") coupled to the ergodic grain.
- `L_grain::Integer`:
  Number of spins inside the ergodic grain.
- `g0::Real`:
  Overall prefactor multiplying the grain-to-ray couplings.
- `α::Real`:
  Decay parameter controlling the long-range coupling profile `α^{u_j}`.
  Smaller `α` means faster decay away from the grain.

# Keyword Arguments

- `γ::Real=1`:
  Overall energy scale of the random grain Hamiltonian.
- `w::Real=0.5`:
  Half-width of the uniform disorder window for the local `S^z` fields.
  The fields are sampled from `[hz - w, hz + w]`.
- `hz::Real=1`:
  Center of the disorder window for the local `S^z` fields.
- `ζ::Real=0.2`:
  Randomness in the effective distance exponent `u_j = (j - 1) + η_j`,
  where `η_j ∈ [-ζ, ζ]`.
- `S::Real=0.5`:
  On-site spin. It must be integer or half-integer.
- `rng::AbstractRNG=Random.default_rng()`:
  Random-number generator used for disorder, couplings, and the grain matrix.
- `use_U1::Bool=false`:
  If `false`, build the conventional Quantum Sun Hamiltonian in the full
  Hilbert space. If `true`, build the U(1)-symmetric version.
- `S_z::Real=0.0`:
  Total magnetization sector used when `use_U1=true`.
- `use_sparse::Bool=true`:
  If `true`, return a sparse matrix. If `false`, return a dense matrix.

# Returns

- `SparseMatrixCSC{Float64,Int}` when `use_sparse=true`.
- `Matrix{Float64}` when `use_sparse=false`.

# Notes

- This is the public constructor you should use in examples and workflows:
  `using Polfed.Models: qsun_hamiltonian`.
- The `use_U1=true` path reduces the Hilbert-space dimension by keeping only
  states in the requested total-`S_z` sector.
- This is a forwarding wrapper around the internal QSun implementation,
  exposed alongside [`xxz_hamiltonian`](@ref) and [`j1j2_hamiltonian`](@ref).
"""
function qsun_hamiltonian(
    L_loc::Integer,
    L_grain::Integer,
    g0::Real,
    α::Real;
    γ::Real=1,
    w::Real=0.5,
    hz::Real=1,
    ζ::Real=0.2,
    S::Real=0.5,
    rng::AbstractRNG=Random.default_rng(),
    use_U1::Bool=false,
    S_z::Real=0.0,
    use_sparse::Bool=true,
)
    return QSun.qsun_hamiltonian(
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
        use_U1=use_U1,
        S_z=S_z,
        use_sparse=use_sparse,
    )
end

export qsun_hamiltonian, xxz_hamiltonian, j1j2_hamiltonian

end
