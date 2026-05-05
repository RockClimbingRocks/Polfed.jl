function uniform_array(n::Int, a, b; rng=Random.default_rng())
    return rand(rng, n) .* (b - a) .+ a
end

function goe_matrix(n::Int; rng=Random.default_rng())
    X = randn(rng, n, n)
    return (X + X') / sqrt(2)
end

function random_neighbors(L_loc::Int, L_grain::Int; rng=Random.default_rng())
    return rand(rng, 0:(L_grain - 1), L_loc)
end

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
