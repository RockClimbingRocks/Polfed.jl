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

                has1, c1, s1, has2, c2, s2 = sigma_x(states[k], site_loc, B)

                if has1
                    hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_x(s1, site_grain, B)

                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg1)
                        end
                    end

                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg2)
                        end
                    end
                end

                if has2
                    hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_x(s2, site_grain, B)

                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg1)
                        end
                    end

                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg2)
                        end
                    end
                end

                has1, c1, s1, has2, c2, s2 = sigma_y(states[k], site_loc, B)

                if has1
                    hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_y(s1, site_grain, B)

                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg1)
                        end
                    end

                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c1 * cg2)
                        end
                    end
                end

                if has2
                    hasg1, cg1, sg1, hasg2, cg2, sg2 = sigma_y(s2, site_grain, B)

                    if hasg1
                        new_idx = state_index(Hspace, sg1)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg1)
                        end
                    end

                    if hasg2
                        new_idx = state_index(Hspace, sg2)
                        if new_idx != 0 && mag[new_idx] == S_z
                            push!(rows, rows_mag[new_idx])
                            push!(cols, rows_mag[k])
                            push!(vals, g0 * couplings[j] * c2 * cg2)
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
