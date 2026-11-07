# Random ingredients of the Quantum Sun model: the disordered local fields, the
# GOE grain matrix, the random grain attachment sites, and the long-range
# coupling profile. All accept an explicit `rng` so that a whole Hamiltonian
# realization is reproducible from a single seed.

"""
    uniform_array(n::Int, a, b; rng=Random.default_rng()) -> Vector{Float64}

Draw `n` i.i.d. samples from the uniform distribution on `[a, b)`.

# Arguments

- `n::Int`:
  Number of samples.
- `a`, `b`:
  Inclusive lower / exclusive upper bound of the sampling window. `b ≥ a` is
  assumed (not checked).
- `rng`:
  Random-number generator (keyword). Defaults to the task-local default RNG.

# Returns

- `Vector{Float64}` of length `n` with entries in `[a, b)`.

# Notes

- Used for the on-site disorder fields, sampled from `[hz - w, hz + w]`.
"""
function uniform_array(n::Int, a, b; rng=Random.default_rng())
    return rand(rng, n) .* (b - a) .+ a
end

"""
    goe_matrix(n::Int; rng=Random.default_rng()) -> Matrix{Float64}

Sample an `n × n` matrix from the Gaussian Orthogonal Ensemble (GOE), used as the
chaotic "grain" block of the Quantum Sun Hamiltonian.

# Arguments

- `n::Int`:
  Matrix dimension.
- `rng`:
  Random-number generator (keyword).

# Returns

- A real symmetric `Matrix{Float64}` `M = (X + Xᵀ)/√2`, where `X` has i.i.d.
  standard-normal entries.

# Notes

- The result is symmetric by construction (`M == Mᵀ`), hence the grain block is
  Hermitian.
- The `/√2` symmetrization fixes the off-diagonal variance to the conventional GOE
  scaling. The overall energy scale of the grain is set afterwards by the caller
  (via `γ` and a dimension-dependent normalization in the Hamiltonian builders).
"""
function goe_matrix(n::Int; rng=Random.default_rng())
    X = randn(rng, n, n)
    return (X + X') / sqrt(2)
end

"""
    random_neighbors(L_loc::Int, L_grain::Int; rng=Random.default_rng()) -> Vector{Int}

For each of the `L_loc` localized spins, pick a random grain site to couple to.

# Arguments

- `L_loc::Int`:
  Number of localized ("ray") spins outside the grain.
- `L_grain::Int`:
  Number of spins inside the grain.
- `rng`:
  Random-number generator (keyword).

# Returns

- `Vector{Int}` of length `L_loc`, each entry a zero-based grain-site index in
  `0:(L_grain-1)`.

# Notes

- Sampling is with replacement: several localized spins may attach to the same
  grain site.
- The returned indices are zero-based to match the bit-packed site convention used
  by [`get_spin`](@ref) / [`sigma_x`](@ref).
"""
function random_neighbors(L_loc::Int, L_grain::Int; rng=Random.default_rng())
    return rand(rng, 0:(L_grain - 1), L_loc)
end

"""
    long_range_couplings(L_loc::Int, α, ζ; rng=Random.default_rng()) -> Vector{Float64}

Build the long-range coupling profile `α^{u_j}` of the localized spins to the
grain, with randomized effective distances.

# Arguments

- `L_loc::Int`:
  Number of localized spins.
- `α`:
  Decay base of the coupling profile. `0 < α < 1` gives couplings that decay with
  distance; `α ≤ 0` disables the couplings entirely (all zeros).
- `ζ`:
  Half-width of the uniform randomness `η_j ∈ [-ζ, ζ]` added to the integer
  distance, so the effective exponent is `u_j = (j - 1) + η_j`.
- `rng`:
  Random-number generator (keyword).

# Returns

- `Vector{Float64}` of length `L_loc`. With `α > 0`: `couplings[1] = 1` and
  `couplings[j] = α^{(j-1) + η_j}` for `j ≥ 2`. With `α ≤ 0`: all zeros.

# Notes

- The first localized spin (`j = 1`) is, by convention, coupled with unit strength
  and no randomness — it sits adjacent to the grain.
- The randomized exponent `u_j` is the mechanism that makes each disorder
  realization distinct even at fixed `α`.
"""
function long_range_couplings(L_loc::Int, α, ζ; rng=Random.default_rng())
    couplings = zeros(Float64, L_loc)

    if α > 0
        couplings[1] = 1
        for j in 2:L_loc
            u = j - 1 + rand(rng) * 2ζ - ζ
            couplings[j] = α^u
        end
    end

    return couplings
end
