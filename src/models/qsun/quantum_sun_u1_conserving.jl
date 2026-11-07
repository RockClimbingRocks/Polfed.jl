# U(1)-symmetric Quantum Sun Hamiltonian, restricted to a fixed total-magnetization
# sector. The full-Hilbert-space variant lives in quantum_sun.jl; `qsun_hamiltonian`
# in QSun.jl dispatches between them on the `use_U1` flag.

"""
    build_hamiltonian_cons(
        L_loc, L_grain, g0, α;
        γ=1, w=0.5, hz=1, ζ=0.2, S=0.5, S_z=0.0,
        rng=Random.default_rng(), use_sparse=true,
    )

Assemble the **U(1)-symmetric** Quantum Sun Hamiltonian in the fixed
total-magnetization sector `S_z`.

This is the magnetization-conserving counterpart of [`build_hamiltonian`](@ref):
the grain–ray coupling uses the XY form `SˣSˣ + SʸSʸ` (which conserves total `Sᶻ`),
and the GOE grain block is projected onto the conserving subspace before the chain
is restricted to the requested sector.

# Positional Arguments

- `L_loc`:
  Number of localized spins outside the grain.
- `L_grain`:
  Number of spins inside the grain.
- `g0`:
  Overall prefactor of the grain-to-ray couplings.
- `α`:
  Decay base of the long-range coupling profile `α^{u_j}`.

# Keyword Arguments

- `γ=1`:
  Energy scale of the random grain block (here applied to the sector-projected,
  trace-normalized grain operator).
- `w=0.5`:
  Half-width of the uniform disorder window for the local `Sᶻ` fields,
  sampled from `[hz - w, hz + w]`.
- `hz=1`:
  Center of the local-field disorder window.
- `ζ=0.2`:
  Randomness half-width of the effective distance exponent `u_j = (j-1) + η_j`.
- `S=0.5`:
  On-site spin (integer or half-integer).
- `S_z=0.0`:
  Total magnetization of the retained sector. Must satisfy `|S_z| ≤ L·S` and make
  `S_z + L·S` an integer (`L = L_loc + L_grain`); otherwise an error is thrown.
- `rng=Random.default_rng()`:
  Random-number generator used for the disorder, couplings, attachment sites, and
  grain matrix.
- `use_sparse=true`:
  Return a `SparseMatrixCSC` if `true`, a dense `Matrix` if `false`.

# Returns

- `SparseMatrixCSC{Float64,Int}` (or `Matrix{Float64}` when `use_sparse=false`) of
  dimension `d_sec × d_sec`, where `d_sec` is the number of full-space states whose
  total magnetization equals `S_z`.

# Notes

- The grain block is restricted to magnetization-block-diagonal entries, embedded
  as `I_{loc} ⊗ H_grain`, projected onto the `S_z` sector, and then trace-normalized
  to `γ · R / √(Tr R² / d_sec)`. This differs from the full-space normalization in
  [`build_hamiltonian`](@ref) (`γ · GOE / √(d_grain + 1)`).
- Off-sector hops (a flipped state whose magnetization differs from `S_z`, or which
  leaves the enumerated basis) are dropped via the `state_index == 0` /
  `mag == S_z` guards.
- Both `SˣSˣ` and `SʸSʸ` contributions are accumulated; together they form the
  magnetization-conserving hopping that defines the U(1) symmetry of this variant.
- Validation of `S` (integer/half-integer) happens in the public wrapper
  [`qsun_hamiltonian`](@ref); the `S_z` sector validity is checked here.
"""
function build_hamiltonian_cons(
    L_loc,
    L_grain,
    g0,
    α;
    γ=1,
    w=0.5,
    hz=1,
    ζ=0.2,
    S=0.5,
    S_z=0.0,
    rng=Random.default_rng(),
    use_sparse=true,
)
    L = L_loc + L_grain
    sector_index = S_z + L * S
    if abs(S_z) > L * S || !isinteger(sector_index)
        error("Wrong total value of spin to given spin-S particles. You chose S = $S and S_z = $S_z.")
    end

    B = SpinBasis(S)

    Hspace = build_hilbert(L, B)
    states = Hspace.states

    disorder = uniform_array(L_loc, hz - w, hz + w; rng=rng)
    neighbors = random_neighbors(L_loc, L_grain; rng=rng)
    couplings = long_range_couplings(L_loc, α, ζ; rng=rng)

    d = Int(2S + 1)
    dim_loc = d^L_loc
    dim_grain = d^L_grain

    H_grain = goe_matrix(dim_grain; rng=rng)

    for II in 1:dim_grain
        for JJ in 1:dim_grain
            summ_I = 0
            summ_J = 0
            for j in 1:L_grain
                summ_I += get_spin(states[II], j - 1, B)
                summ_J += get_spin(states[JJ], j - 1, B)
            end
            if summ_J != summ_I
                H_grain[II, JJ] = 0
            end
        end
    end

    mag = zeros(Float64, length(states))
    for k in eachindex(states)
        summ_mag = 0.0
        for j in 1:L
            val_z, _ = sigma_z(states[k], j - 1, B)
            summ_mag += val_z
        end
        mag[k] = summ_mag
    end

    rows_mag = zeros(Int, length(states))
    counter = 0
    for k in eachindex(states)
        if mag[k] == S_z
            counter += 1
            rows_mag[k] = counter
        end
    end

    dim_red = counter

    R = kron(sparse(I, dim_loc, dim_loc), H_grain)

    rows_cons = Int[]
    cols_cons = Int[]
    vals_cons = Float64[]

    I_idx, J_idx, V = findnz(R)
    for idx in eachindex(I_idx)
        r = I_idx[idx]
        c = J_idx[idx]

        if mag[r] == S_z && mag[c] == S_z
            push!(rows_cons, rows_mag[r])
            push!(cols_cons, rows_mag[c])
            push!(vals_cons, V[idx])
        end
    end

    R_cons = sparse(rows_cons, cols_cons, vals_cons, dim_red, dim_red)
    R_cons = γ * R_cons / sqrt(tr(R_cons * R_cons) / dim_red)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for k in eachindex(states)
        if mag[k] == S_z
            for j in 1:L_loc
                site = L_grain + j - 1
                val_z, _ = sigma_z(states[k], site, B)

                push!(rows, rows_mag[k])
                push!(cols, rows_mag[k])
                push!(vals, disorder[j] * val_z)
            end
        end

        if mag[k] == S_z
            for j in 1:L_loc
                site_loc = L_grain + j - 1
                site_grain = neighbors[j]

                # Only the magnetization-conserving cross terms survive in this
                # fixed-Sᶻ sector: a local raise pairs with a grain lower (S⁺S⁻) and
                # a local lower with a grain raise (S⁻S⁺). Same-direction flips shift
                # total Sᶻ by ±2 and leave the sector, so they are not built.
                # (SˣSˣ + SʸSʸ = ½(S⁺S⁻ + S⁻S⁺), the XY hopping of the U(1) symmetry.)
                has1, c1, s1, has2, c2, s2 = sigma_x(states[k], site_loc, B)

                if has1                                  # local raise → grain lower
                    _, _, _, hasg2, cg2, sg2 = sigma_x(s1, site_grain, B)
                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg2)
                        end
                    end
                end

                if has2                                  # local lower → grain raise
                    hasg1, cg1, sg1, _, _, _ = sigma_x(s2, site_grain, B)
                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg1)
                        end
                    end
                end

                has1, c1, s1, has2, c2, s2 = sigma_y(states[k], site_loc, B)

                if has1                                  # local raise → grain lower
                    _, _, _, hasg2, cg2, sg2 = sigma_y(s1, site_grain, B)
                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg2)
                        end
                    end
                end

                if has2                                  # local lower → grain raise
                    hasg1, cg1, sg1, _, _, _ = sigma_y(s2, site_grain, B)
                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg1)
                        end
                    end
                end
            end
        end
    end

    H = sparse(rows, cols, vals, dim_red, dim_red)
    H += R_cons

    return use_sparse ? H : Matrix(H)
end
