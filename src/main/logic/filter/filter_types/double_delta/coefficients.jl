# DoubleDeltaFilter coefficients: two delta-peaked Chebyshev expansions combined so the polynomial
# peaks at exactly +1 at target1 (highest end) and -1 at target2 (lowest end). Because the two
# peaks cross-talk, a single overall scale can only pin one of them; the two mixing coefficients
# `a`, `b` are solved from a 2×2 system so BOTH peaks land exactly.

"""    _coefficients_nonnormalized(::DoubleDeltaFilter, n, target) -> Float64
Raw Chebyshev coefficient `cₙ = (2 - [n==0])·cos(n·acos target)` for a single delta peak at
`target` (same as the single delta filters)."""
_coefficients_nonnormalized(_::DoubleDeltaFilter, n::Int, target::Float64) =
    (2 - (n == 0)) * cos(n * acos(target))

"""
    coefficients(filter::DoubleDeltaFilter) -> Function

Normalized coefficients `n -> cₙ` of the combined double-delta polynomial `P = a·U + b·D`, where
`U`/`D` are the (unnormalized) delta expansions peaked at `target1`/`target2` to `order1`/`order2`.
The mixing coefficients `a`, `b` are the solution of the 2×2 system enforcing both peaks at once:

    ┌ U(t1)  D(t1) ┐ ┌a┐   ┌+1┐
    └ U(t2)  D(t2) ┘ └b┘ = └−1┘

so `P(target1) = +1` and `P(target2) = −1` **exactly** (the off-diagonal cross-talk is handled by
the solve). Requires resolved `order1`/`order2`.
"""
function coefficients(filter::DoubleDeltaFilter)
    order1, order2 = filter.order1, filter.order2
    t1, t2         = filter.target1, filter.target2

    # Unnormalized sub-polynomials: U peaks at t1 (→ +1), D peaks at t2 (→ −1).
    P1 = ClenshawExpansion(polynomial_type(filter), n -> _coefficients_nonnormalized(filter, n, t1), order1)
    P2 = ClenshawExpansion(polynomial_type(filter), n -> _coefficients_nonnormalized(filter, n, t2), order2)

    # Solve [U(t1) D(t1); U(t2) D(t2)] · [a; b] = [+1; −1] (Cramer's rule).
    P1_t1, P1_t2 = P1(t1), P1(t2)
    P2_t1, P2_t2 = P2(t1), P2(t2)
    det = P1_t1 * P2_t2 - P2_t1 * P1_t2
    a =  (P2_t1 + P2_t2) / det
    b = -(P1_t1 + P1_t2) / det

    return n ->
        a * _coefficients_nonnormalized(filter, n, t1) * Int(n <= order1) +
        b * _coefficients_nonnormalized(filter, n, t2) * Int(n <= order2)
end
