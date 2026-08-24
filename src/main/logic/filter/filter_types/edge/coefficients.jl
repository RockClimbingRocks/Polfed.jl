# EdgeFilter coefficients: the Taylor (monomial) table of the mode's closed-form polynomial.
# Unlike the delta filters there is NO normalization step — every mode peaks at exactly ±1 at its
# exposed edge(s) by construction.

"""
    coefficients(filter::EdgeFilter) -> Function

The filter's Taylor (monomial) coefficients `n -> cₙ`, exact by construction — no normalization
step is needed because every mode peaks at exactly `±1` at its exposed edge(s):

- `:highest` — `((1+x)/2)^p = Σₙ C(p,n)/2^p · xⁿ`, so `cₙ = C(p,n)/2^p`.
- `:lowest` — `−((1−x)/2)^p`, the mirror image: `cₙ = (−1)^(n+1) C(p,n)/2^p`.
- `:both` — `x^p`: the Kronecker delta `cₙ = [n == p]`.

The scaled binomials `C(p,n)/2^p ∈ (0, 1]` are precomputed by cumulative product (never forming
`C(p,n)` alone, which overflows `Int` already at `p ≥ 62`), so any realistic order is stable; the
returned closure just indexes the table.
"""
function coefficients(filter::EdgeFilter)
    p = filter.order
    filter.exposed_end === :both && return n -> (n == p ? 1.0 : 0.0)

    c = Vector{Float64}(undef, p + 1)                 # c[n+1] = C(p,n)/2^p, built multiplicatively
    c[1] = 0.5^p
    for n in 1:p
        c[n + 1] = c[n] * (p - n + 1) / n
    end
    if filter.exposed_end === :lowest                 # mirror: cₙ ← (−1)^(n+1)·cₙ
        for n in 0:2:p
            c[n + 1] = -c[n + 1]
        end
    end
    return n -> (0 <= n <= p ? c[n + 1] : 0.0)
end
