const _DEFAULT_EDGE_DISTANCE = 0.2
const _EDGE_RESOLUTION_ATOL = 1e-9

"""
    _run_filtered_patch(mp, filter, eig_target, budget, mapvals, tol,
                        eigentol_rescaled, overestimate_iters, layout, timer;
                        label, algorithm_defaults)

Build the filtered operator and run one Lanczos patch. `budget` controls the expected
eigenpair count, while `overestimate_iters` controls the maximum Krylov dimension. The
returned vectors remain transient and should be copied into a result buffer immediately.
"""
function _run_filtered_patch(
    mp::MappingPlan,
    filter,
    eig_target,
    budget::Integer,
    mapvals,
    tol::Real,
    eigentol_rescaled::Real,
    overestimate_iters::Real,
    layout::Layout,
    timer::StageTimer;
    label::AbstractString,
    algorithm_defaults = Defaults.patched_polfed,
)
    n = max(1, Int(budget))
    transformed = @stage timer :transform build_transform(filter, mp)
    expected_dim = expected_krylov_dim(filter, n, mp.sys.block_size, 1.0)
    maxdim = expected_krylov_dim(
        filter, n, mp.sys.block_size, max(2.5, Float64(overestimate_iters)),
    )
    result = @stage timer :lanczos lanczos(transformed, mp.sys.x0, eig_target;
        rot = algorithm_defaults.rot,
        basistype = algorithm_defaults.basistype,
        maxdim = maxdim,
        tol = tol,
        eigentol = eigentol_rescaled,
        layout = layout,
        mapvals = mapvals,
        next_check = algorithm_defaults.conv_checking(expected_dim),
        warn_incomplete = false,
    )
    polfed_log(POLFED_DEBUG_LEVEL, "$label finished.";
        complete = result.report.complete,
        converged = result.converged,
        iterations = result.report.itersneeded,
        maxdim = maxdim,
    )
    return result
end

"""
    _edge_target(filter, left_active, right_active, left_cut, right_cut)

Construct the Lanczos target for the active spectral edge bands. Both active edges are
represented by one `BothEnds` target so they can be resolved in a single Lanczos run.
"""
function _edge_target(filter, left_active::Bool, right_active::Bool, left_cut, right_cut)
    polynomial = _scalar_filter(filter)
    low = Window(-Inf, left_cut, polynomial(left_cut); return_brackets = false)
    high = Window(right_cut, Inf, polynomial(right_cut); return_brackets = false)
    left_active && right_active && return BothEnds(low, high)
    left_active && return Lowest(low)
    return Highest(high)
end

"""
    _edge_seam(vals, indices, side, cut) -> Float64

Return the second eigenvalue inward from a resolved edge. This seam extends the adjacent
interior solve far enough to overlap the edge result; `cut` is used when fewer than two
usable edge eigenvalues are available.
"""
function _edge_seam(vals, indices, side::Symbol, cut::Float64)
    length(indices) >= 2 || begin
        polfed_log(POLFED_WARN_LEVEL,
            "An edge returned fewer than two eigenpairs; using the edge-band cut as its seam.",
            side = side, converged = length(indices), seam = cut)
        return cut
    end
    seam = side === :left ? vals[indices[end - 1]] : vals[indices[2]]
    return isfinite(seam) && -1.0 + _EDGE_RESOLUTION_ATOL < seam <
           1.0 - _EDGE_RESOLUTION_ATOL ? Float64(seam) : cut
end

"""
    _resolve_edges(mp, dos, request, buffer, records, mapvals, ...; edge_distance)
        -> (left_edge, right_edge)

Always inspect the request for edge-band work. A request entering the lower/upper band
resolves that complete band from the true spectral end, then emits only the requested
part. When both bands are active, one `EdgeFilter(:both)` and one `BothEnds` Lanczos run
resolve them together. Returned edge metadata contains the second-inward overlap seam;
inactive sides return `nothing`.
"""
function _resolve_edges(
    mp::MappingPlan,
    dos::DoSResult,
    request::Interval,
    buffer,
    records::Vector{PatchReport},
    mapvals,
    tol::Real,
    eigentol_rescaled::Real,
    overestimate_iters::Real,
    layout::Layout,
    timer::StageTimer;
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    algorithm_defaults = Defaults.patched_polfed,
)
    _check_interval(request)
    distance = Float64(edge_distance)
    0.0 < distance < 1.0 ||
        throw(ArgumentError("edge_distance must satisfy 0 < edge_distance < 1; got $edge_distance."))
    max_retries >= 0 || throw(ArgumentError("max_retries must be non-negative."))
    retry_growth > 1 || throw(ArgumentError("retry_growth must be greater than one."))

    left_cut, right_cut = -1.0 + distance, 1.0 - distance
    left_active = request.left < left_cut
    right_active = request.right > right_cut
    (left_active || right_active) || return nothing, nothing

    left_band = Interval(-1.0, left_cut)
    right_band = Interval(right_cut, 1.0)
    filter = @stage timer :filter begin
        left_active && right_active ?
            build_filter(EdgeFilter, left_band, right_band) :
            build_filter(EdgeFilter, left_active ? left_band : right_band)
    end
    eig_target = _edge_target(filter, left_active, right_active, left_cut, right_cut)
    left_budget = left_active ? howmany_in_interval(dos, left_band) : 0
    right_budget = right_active ? howmany_in_interval(dos, right_band) : 0
    krylov_budget = left_budget + right_budget

    result = nothing
    retries = 0
    inflation = max(2.5, Float64(overestimate_iters))
    while true
        result = _run_filtered_patch(
            mp, filter, eig_target, krylov_budget, mapvals, tol, eigentol_rescaled,
            inflation, layout, timer; label = left_active && right_active ? "Both edges" : "Edge",
            algorithm_defaults,
        )
        (result.report.complete || retries == max_retries) && break
        retries += 1
        inflation *= retry_growth
    end

    vals, vecs = Float64.(collect(result.vals)), result.vecs
    left_indices = left_active ?
        (1:searchsortedlast(vals, left_cut + _EDGE_RESOLUTION_ATOL)) : (1:0)
    right_start = right_active ?
        searchsortedfirst(vals, right_cut - _EDGE_RESOLUTION_ATOL) : length(vals) + 1
    right_indices = right_start:length(vals)

    if left_active
        lower = request.left <= -1.0 + _EDGE_RESOLUTION_ATOL ? -Inf : request.left
        upper = min(request.right, left_cut)
        lower <= upper &&
            _store_interval!(buffer, vals, vecs, left_indices, lower, upper, layout)
    end
    if right_active
        lower = max(request.left, right_cut)
        upper = request.right >= 1.0 - _EDGE_RESOLUTION_ATOL ? Inf : request.right
        lower <= upper &&
            _store_interval!(buffer, vals, vecs, right_indices, lower, upper, layout)
    end

    kind = left_active && right_active ? :edge_both :
           left_active ? :edge_lowest : :edge_highest
    report_interval = left_active && right_active ? Interval(-1.0, 1.0) :
                      left_active ? left_band : right_band
    push!(records, PatchReport(
        kind, string(nameof(typeof(filter))), polynomial_order(filter), report_interval,
        left_budget + right_budget, result.report, result.report.complete, retries,
    ))
    result.report.complete || error(
        "Edge resolution exhausted $retries retries before completeness was certified ($kind).",
    )

    left_edge = left_active ? (
        seam = _edge_seam(vals, left_indices, :left, left_cut),
        band = left_band,
        count = length(left_indices),
        budget = left_budget,
        complete = result.report.complete,
    ) : nothing
    right_edge = right_active ? (
        seam = _edge_seam(vals, right_indices, :right, right_cut),
        band = right_band,
        count = length(right_indices),
        budget = right_budget,
        complete = result.report.complete,
    ) : nothing
    return left_edge, right_edge
end
