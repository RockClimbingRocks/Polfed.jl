# Full-Hilbert-space Quantum Sun Hamiltonian. The magnetization-conserving variant
# lives in quantum_sun_u1_conserving.jl; the public entry point that dispatches
# between them is `qsun_hamiltonian` in QSun.jl.

"""
    build_hamiltonian(
        L_loc, L_grain, g0, α;
        γ=1, w=0.5, hz=1, ζ=0.2, S=0.5,
        rng=Random.default_rng(), use_sparse=true,
    )

Assemble the Quantum Sun Hamiltonian on the **full** `d^{L_loc+L_grain}`-dimensional
Hilbert space (no symmetry sector imposed).

The Quantum Sun model couples a small chaotic "grain" of `L_grain` spins (a GOE
block) to `L_loc` localized spins, where localized spin `j` attaches to a random
grain site with a long-range coupling `g0 · α^{u_j}`:

```math
H = H_\\text{grain} \\;+\\; \\sum_{j=1}^{L_\\text{loc}} h_j\\, S^z_{n_j}
       \\;+\\; g_0 \\sum_{j=1}^{L_\\text{loc}} α^{u_j}\\, S^x_{\\text{loc}(j)} S^x_{\\text{grain}(j)} .
```

# Positional Arguments

- `L_loc`:
  Number of localized ("ray") spins outside the grain.
- `L_grain`:
  Number of spins inside the chaotic grain.
- `g0`:
  Overall prefactor of the grain-to-ray couplings.
- `α`:
  Decay base of the long-range coupling profile `α^{u_j}` (smaller ⇒ faster decay).

# Keyword Arguments

- `γ=1`:
  Energy scale of the random grain block.
- `w=0.5`:
  Half-width of the uniform disorder window for the local `Sᶻ` fields, sampled
  from `[hz - w, hz + w]`.
- `hz=1`:
  Center of the local-field disorder window.
- `ζ=0.2`:
  Randomness half-width of the effective distance exponent `u_j = (j-1) + η_j`,
  `η_j ∈ [-ζ, ζ]`.
- `S=0.5`:
  On-site spin (integer or half-integer).
- `rng=Random.default_rng()`:
  Random-number generator used for the disorder, couplings, attachment sites, and
  grain matrix.
- `use_sparse=true`:
  Return a `SparseMatrixCSC` if `true`, a dense `Matrix` if `false`.

# Returns

- `SparseMatrixCSC{Float64,Int}` (or `Matrix{Float64}` when `use_sparse=false`) of
  dimension `d^{L_loc+L_grain} × d^{L_loc+L_grain}`, where `d = 2S + 1`.

# Notes

- The grain block is `H_grain = γ · GOE(d^{L_grain}) / √(d^{L_grain} + 1)`, embedded
  via `I_{loc} ⊗ H_grain` so it acts only on the grain sites.
- The grain–ray coupling uses `Sˣ Sˣ` only (not `SʸSʸ`), so this Hamiltonian does
  **not** conserve total magnetization — that is intentional for the full-space
  model. The fixed-sector version [`build_hamiltonian_cons`](@ref) uses the
  magnetization-conserving `SˣSˣ + SʸSʸ` coupling instead.
- The grain sites occupy the low site indices `0:(L_grain-1)`; localized spin `j`
  sits at site `L_grain + j - 1`.
- Validation of `S` (integer/half-integer) happens in the public wrapper
  [`qsun_hamiltonian`](@ref); this builder assumes a valid `S`.
"""
function build_hamiltonian(
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
    use_sparse=true,
)
    B = SpinBasis(S)
    L = L_loc + L_grain

    Hspace = build_hilbert(L, B)
    states = Hspace.states
    dim = length(states)

    disorder = uniform_array(L_loc, hz - w, hz + w; rng=rng)
    neighbors = random_neighbors(L_loc, L_grain; rng=rng)
    couplings = long_range_couplings(L_loc, α, ζ; rng=rng)

    d = Int(2S + 1)
    dim_loc = d^L_loc
    dim_grain = d^L_grain

    H_grain = γ * goe_matrix(dim_grain; rng=rng)
    H_grain ./= sqrt(dim_grain + 1)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (k, state) in enumerate(states)
        for j in 1:L_loc
            site = L_grain + j - 1
            val_z, _ = sigma_z(state, site, B)

            push!(rows, k)
            push!(cols, k)
            push!(vals, disorder[j] * val_z)
        end

        for j in 1:L_loc
            site_loc = L_grain + j - 1
            site_grain = neighbors[j]

            has1, c1, s1, has2, c2, s2 = sigma_x(state, site_loc, B)

            if has1
                hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_x(s1, site_grain, B)

                if hasg1
                    new_idx = state_index(Hspace, sg1)
                    if new_idx != 0
                        push!(rows, new_idx)
                        push!(cols, k)
                        push!(vals, g0 * couplings[j] * c1 * cg1)
                    end
                end

                if hasg2
                    new_idx = state_index(Hspace, sg2)
                    if new_idx != 0
                        push!(rows, new_idx)
                        push!(cols, k)
                        push!(vals, g0 * couplings[j] * c1 * cg2)
                    end
                end
            end

            if has2
                hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_x(s2, site_grain, B)

                if hasg1
                    new_idx = state_index(Hspace, sg1)
                    if new_idx != 0
                        push!(rows, new_idx)
                        push!(cols, k)
                        push!(vals, g0 * couplings[j] * c2 * cg1)
                    end
                end

                if hasg2
                    new_idx = state_index(Hspace, sg2)
                    if new_idx != 0
                        push!(rows, new_idx)
                        push!(cols, k)
                        push!(vals, g0 * couplings[j] * c2 * cg2)
                    end
                end
            end
        end
    end

    H = sparse(rows, cols, vals, dim, dim)
    H += kron(sparse(I, dim_loc, dim_loc), H_grain)

    return use_sparse ? H : Matrix(H)
end
