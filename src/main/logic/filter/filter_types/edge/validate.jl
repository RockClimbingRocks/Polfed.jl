# EdgeFilter validation. A monotone polynomial cannot leak — there are no side lobes to rise
# back over any threshold — so there is no leakage geometry to scan (contrast the delta filters'
# passband/tail scans). What remains is a cheap sanity guard on the polynomial's designed shape.

"""
    validate_filter(filter::EdgeFilter) -> Bool

`true` when the filter polynomial has its designed shape. A monotone polynomial cannot leak —
there are no side lobes to rise back over any threshold — so unlike the delta filters there is no
leakage geometry to scan; this is a cheap sanity guard (against a future coefficient regression)
that `P` is finite and monotone nondecreasing on a `[-1, 1]` grid and hits its exact endpoint
values: `P(∓1) = ∓1` at each exposed edge and `P = 0` at an unexposed one.
"""
function validate_filter(filter::EdgeFilter)
    P  = _scalar_filter(filter)
    xs = range(-1.0, 1.0; length = Defaults.filters.edge.npts_validate)
    prev = -Inf
    for x in xs
        v = P(x)
        isfinite(v) || return _reject(filter, @sprintf("P(%.4g) is not finite", x))
        v >= prev - 1e-12 ||                          # monotone nondecreasing (fp slack)
            return _reject(filter, @sprintf("P is not monotone at x = %.4g", x))
        prev = v
    end

    lo_exposed = filter.exposed_end in (:lowest, :both)
    hi_exposed = filter.exposed_end in (:highest, :both)
    isapprox(P(-1.0), lo_exposed ? -1.0 : 0.0; atol = 1e-9) ||
        return _reject(filter, "P(-1) is off its designed endpoint value")
    isapprox(P(1.0), hi_exposed ? 1.0 : 0.0; atol = 1e-9) ||
        return _reject(filter, "P(+1) is off its designed endpoint value")
    return true
end
