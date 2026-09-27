"""
    _interior_target(filter, interval)

Construct the one-sided Lanczos target matching the exposed end of an interior filter.
"""
function _interior_target(filter, interval::Interval)
    band = passband(filter)
    isfinite(band.left) && isfinite(band.right) ||
        error("Interior patches require a finite validated filter passband.")
    polynomial = _scalar_filter(filter)
    exposed = exposed_end(filter)
    boundary = exposed === :highest ?
        min(polynomial(band.left), polynomial(band.right)) :
        max(polynomial(band.left), polynomial(band.right))
    window = Window(interval.left, interval.right, boundary; return_brackets = false)
    exposed === :highest && return Highest(window)
    exposed === :lowest && return Lowest(window)
    error("Interior patch unexpectedly resolved a :both filter.")
end

"""
    _interior_solve_interval(padded, patch_index, patch_count,
                             left_edge, right_edge, left_cut, right_cut)

Choose the actual solve interval for one interior patch. Boundary patches extend to the
second-inward edge eigenvalue when that edge was resolved; other boundaries use the padded
patch clamped to the interior band.
"""
function _interior_solve_interval(
    padded::Interval,
    patch_index::Int,
    patch_count::Int,
    left_edge,
    right_edge,
    left_cut::Float64,
    right_cut::Float64,
)
    left = patch_index == 1 && left_edge !== nothing ?
        left_edge.seam : max(padded.left, left_cut)
    right = patch_index == patch_count && right_edge !== nothing ?
        right_edge.seam : min(padded.right, right_cut)
    left < right || error("Interior overlap collapsed to [$left, $right].")
    return Interval(left, right)
end

"""
    _resolve_interior(mp, dos, request, left_edge, right_edge, buffer, records, ...)

Resolve everything not owned by the edge bands. The first/last solve windows extend to the
second-inward edge eigenvalues. Overlapping candidates are written straight to the result
buffer, which removes repeated copies by eigenvalue proximity and eigenvector-subspace overlap.
"""
function _resolve_interior(
    mp::MappingPlan,
    dos::DoSResult,
    request::Interval,
    left_edge,
    right_edge,
    buffer,
    records::Vector{PatchReport},
    howmany_per_patch::Integer,
    mapvals,
    tol::Real,
    eigentol_rescaled::Real,
    overestimate_iters::Real,
    layout::Layout,
    timer::StageTimer;
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    algorithm_defaults = Defaults.patched_polfed,
)
    distance = Float64(edge_distance)
    left_cut, right_cut = -1.0 + distance, 1.0 - distance
    left, right = max(request.left, left_cut), min(request.right, right_cut)
    left < right || return nothing

    patches = @stage timer :interval intervalize_interval_overlapping(
        dos, Interval(left, right), howmany_per_patch, howmany_pad,
    )
    patch_count = length(patches.unpadded)
    if should_log(POLFED_INFO_LEVEL)
        polfed_log(POLFED_INFO_LEVEL,
                   "Interior decomposed into $patch_count patches over $(_fmt_iv(Interval(left, right))).",
                   howmany_per_patch = Int(howmany_per_patch), howmany_pad = Int(howmany_pad))
        for i in eachindex(patches.unpadded)
            polfed_log(POLFED_DEBUG_LEVEL, "Patch $i/$patch_count.",
                       owns = _fmt_iv(patches.unpadded[i]), padded = _fmt_iv(patches.padded[i]))
        end
    end
    for i in eachindex(patches.unpadded)
        own = patches.unpadded[i]
        solve = _interior_solve_interval(
            patches.padded[i], i, patch_count, left_edge, right_edge, left_cut, right_cut,
        )
        inflation = max(2.5, Float64(overestimate_iters))
        retries = 0
        result = filter = nothing
        budget = 0

        while true
            filter = @stage timer :filter resolve_filter(solve)
            budget = howmany_in_interval(dos, solve)
            result = _run_filtered_patch(
                mp, filter, _interior_target(filter, solve), budget, mapvals, tol,
                eigentol_rescaled, inflation, layout, timer;
                label = "Interior $i/$patch_count",
                algorithm_defaults,
            )
            (result.report.complete || retries == max_retries) && break
            result = nothing      # its vecs view its Krylov basis: drop it before the retry builds one
            retries += 1
            inflation *= retry_growth
            polfed_log(POLFED_DEBUG_LEVEL,
                       "Interior patch $i/$patch_count did not bracket its interval; widening " *
                       "the overlap and the Krylov budget.",
                       retry = retries, of = max_retries, overestimate_iters = inflation,
                       howmany_pad = Int(howmany_pad) * (retries + 1))
            wider = _pad_interval(
                dos, own, Int(howmany_pad) * (retries + 1),
            )
            solve = _interior_solve_interval(
                wider, i, patch_count, left_edge, right_edge, left_cut, right_cut,
            )
        end

        push!(records, PatchReport(
            :interior, string(nameof(typeof(filter))), polynomial_order(filter), own,
            budget, result.report, result.report.complete, retries,
        ))
        result.report.complete || error(
            "Interior patch $i/$patch_count exhausted $retries retries before completeness.",
        )

        vals = Float64.(collect(result.vals))
        _store_interval!(
            buffer, vals, result.vecs, eachindex(vals), request.left, request.right, layout,
        )
    end
    return nothing
end

"""
    _edge_budget(left_edge, right_edge) -> Int

Return the larger density-of-states budget used by the resolved edge bands, or zero when
neither edge was active.
"""
_edge_budget(left_edge, right_edge) = max(
    left_edge === nothing ? 0 : left_edge.budget,
    right_edge === nothing ? 0 : right_edge.budget,
)
