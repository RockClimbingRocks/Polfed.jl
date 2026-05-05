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
