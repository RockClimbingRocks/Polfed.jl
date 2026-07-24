# ── Shared core for the patched POLFED pipelines (window + count target) ───────────────────────
#
# Both pipelines reduce to the same problem: given a rescaled window `full_iv`, converge and glue
# its spectrum. This file is that shared machinery; the two pipelines differ only in how `full_iv`
# is obtained (an explicit window vs. a target+count) and how the glued result is post-processed.
#
# EDGE-FIRST strategy. The spectrum EDGES (±1) are awkward for the interior machinery: the extreme
# eigenvalues sit ON ±1 (where `Emin`/`Emax` are only estimates), and delta filters cannot be placed
# that close to an edge. So when `full_iv` touches an edge, that edge is peeled FIRST by a dedicated
# `EdgeFilter` run with a COUNT target (`LowestEigs`/`HighestEigs`), converging the extremal
# `howmany_edge` eigenpairs from the true extreme inward — robust to exactly where ±1 falls, no
# window bracket, no ±1 ownership subtlety. The remaining INTERIOR (strictly inside the spectrum,
# below the peels' inner boundaries) is then tiled with the usual delta-filter Window patches.
# Ownership stitches edge peels and interior patches with no gaps and no duplicates.

"""
    _gate_to_passband(vals_i, vecs_i, filter, layout; atol=1e-9) -> (vals, vecs, ndrop)

Keep only the eigenpairs whose (rescaled) eigenvalue lies within `filter`'s [`passband`](@ref)
(widened by `atol` for rounding), dropping the rest — values and eigenvectors together — and report
how many were dropped. The kept eigenvectors are a lazy view (column- or row-selected per `layout`).
A monotone [`EdgeFilter`](@ref) has an unbounded passband side, so its extremes always survive.
"""
function _gate_to_passband(vals_i::AbstractVector, vecs_i::AbstractMatrix, filter, layout::Layout;
                           atol::Real = 1e-9)
    pb = passband(filter)
    lo, hi = pb.left - atol, pb.right + atol
    mask = (vals_i .>= lo) .& (vals_i .<= hi)
    all(mask) && return vals_i, vecs_i, 0
    vals_k = vals_i[mask]
    vecs_k = layout isa RowLayout ? @view(vecs_i[mask, :]) : @view(vecs_i[:, mask])
    return vals_k, vecs_k, count(!, mask)
end

"""
    _run_patch(mp, filter, eig_target, budget, mapvals, tol, eigentol_rescaled, overestimate_iters,
               layout, timer; label) -> (vals_i, vecs_i, report_i)

Build the transform for `filter` and run one filtered Lanczos with `eig_target`; `budget` (a state
count) sizes the Krylov space via [`expected_krylov_dim`](@ref). Shared by the edge peels (count
target) and the interior patches (Window target). `warn_incomplete=false`: hitting `maxdim` is
expected (see the pipelines), and the callers do their own, more specific checks. Emits a DEBUG
preview of the leading/trailing converged (rescaled) values.
"""
function _run_patch(mp::MappingPlan, filter, eig_target, budget::Integer, mapvals,
                    tol::Real, eigentol_rescaled::Real, overestimate_iters::Real,
                    layout::Layout, timer::StageTimer; label::AbstractString)
    f!_transformed = @stage timer :transform build_transform(filter, mp)
    expected_dim = expected_krylov_dim(filter, Int(budget), mp.sys.block_size, 1.0)
    maxdim       = expected_krylov_dim(filter, Int(budget), mp.sys.block_size, max(2.5, overestimate_iters))

    vals_i, vecs_i, report_i = @stage timer :lanczos lanczos(f!_transformed, mp.sys.x0, eig_target;
        rot        = Defaults.patched_polfed.rot,
        basistype  = Defaults.patched_polfed.basistype,
        maxdim     = maxdim,
        tol        = tol,
        eigentol   = eigentol_rescaled,
        layout     = layout,
        mapvals    = mapvals,
        next_check = Defaults.patched_polfed.conv_checking(expected_dim),
        warn_incomplete = false,
    )

    # SAFETY: discard any pair whose eigenvalue sits OUTSIDE the filter's passband — a `Window`
    # target's brackets and any Chebyshev-tail leakage land there, and outside the passband the
    # filter is not monotone so the value is not a certified result. Values and eigenvectors go
    # together. (All survivors already passed the eigen-residual gate inside `lanczos`.)
    vals_i, vecs_i, ndrop = _gate_to_passband(vals_i, vecs_i, filter, layout)
    ndrop > 0 && polfed_log(POLFED_DEBUG_LEVEL, "$label dropped out-of-passband pairs.",
                            dropped = ndrop, passband = (passband(filter).left, passband(filter).right))

    npreview  = min(length(vals_i), Defaults.patched_polfed.howmany_pad + 2)
    head_vals = isempty(vals_i) ? "[]" : string(round.(vals_i[1:npreview], digits = 6))
    tail_vals = isempty(vals_i) ? "[]" : string(round.(vals_i[end-npreview+1:end], digits = 6))
    polfed_log(POLFED_DEBUG_LEVEL, "$label Lanczos finished.",
               filter = nameof(typeof(filter)), converged = report_i.converged_tol,
               iterations = report_i.itersneeded, worst_residual = report_i.residual,
               maxdim = maxdim, first_vals = head_vals, last_vals = tail_vals)
    return vals_i, vecs_i, report_i
end

"""
    _edge_peel(mp, side, n, mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer)
        -> (vals_e, vecs_e, inner)

Converge the `n` extremal eigenpairs at one spectrum edge with an [`EdgeFilter`](@ref) and a COUNT
target. `side` is `:lowest` (bottom, `-1`) or `:highest` (top, `+1`). Returns the peeled values
(ascending, `Vector{Float64}`, rescaled), their eigenvectors (as views), and the peel's **inner
boundary** — the largest peeled value for `:lowest`, the smallest for `:highest` — the rescaled
energy where the interior tiling begins.
"""
function _edge_peel(mp::MappingPlan, side::Symbol, n::Integer, mapvals,
                    tol::Real, eigentol_rescaled::Real, overestimate_iters::Real,
                    layout::Layout, timer::StageTimer)
    ed = Defaults.patched_polfed.edge_distance
    edge_iv    = side === :lowest ? Interval(-1.0, -1.0 + ed) : Interval(1.0 - ed, 1.0)
    filter     = @stage timer :filter build_filter(EdgeFilter, edge_iv)
    eig_target = side === :lowest ? LowestEigs(Int(n)) : HighestEigs(Int(n))

    polfed_log(POLFED_INFO_LEVEL, "Edge peel ($side): converging the $n extremal eigenpairs.",
               edge = side === :lowest ? -1.0 : 1.0, howmany = Int(n))

    vals_e, vecs_e, report = _run_patch(mp, filter, eig_target, n, mapvals, tol, eigentol_rescaled,
                                        overestimate_iters, layout, timer; label = "Edge($side)")

    report.converged_tol >= n || polfed_log(POLFED_WARN_LEVEL,
        "Edge peel ($side) converged fewer eigenpairs than requested; the spectrum end may be short.",
        requested = Int(n), converged = report.converged_tol)

    patch = PatchReport(side === :lowest ? :edge_lowest : :edge_highest,
                        string(nameof(typeof(filter))), filter.order, edge_iv, Int(n),
                        report, true, 0)   # edge peels use a count target — no coverage check

    if isempty(vals_e)
        return Float64[], AbstractVector{Float64}[], (side === :lowest ? -1.0 : 1.0), patch
    end
    vv    = AbstractVector{Float64}[_eigenvector_view(layout, vecs_e, k) for k in eachindex(vals_e)]
    inner = side === :lowest ? last(vals_e) : first(vals_e)   # vals_e ascending
    return collect(Float64, vals_e), vv, inner, patch
end

"""
    _interior_tile!(vals, vecs, mp, dos, interior_iv, howmany_per_patch, mapvals, tol,
                    eigentol_rescaled, overestimate_iters, layout, timer; open_left, open_right)

Tile the strictly-interior window `interior_iv` with delta-filter [`Window`](@ref) patches and glue
their owned eigenpairs onto `vals`/`vecs` (in place). `open_left`/`open_right` mark a boundary that
an adjacent edge peel already owns, so the abutting patch leaves it open (no double count); the
window's own outer endpoint is closed.
"""
function _interior_tile!(vals::Vector{Float64}, vecs::Vector{<:AbstractVector},
                         records::Vector{PatchReport}, mp::MappingPlan, dos::DoSResult,
                         interior_iv::Interval, howmany_per_patch::Integer, mapvals, tol::Real,
                         eigentol_rescaled::Real, overestimate_iters::Real, layout::Layout,
                         timer::StageTimer; open_left::Bool, open_right::Bool)
    howmany_pad = Defaults.patched_polfed.howmany_pad
    patches = @stage timer :interval intervalize_interval_overlapping(
        dos, interior_iv, howmany_per_patch, howmany_pad;
        edge_distance = Defaults.patched_polfed.edge_distance)
    npatches = length(patches.unpadded)

    for i in 1:npatches
        iv     = patches.unpadded[i]
        iv_ext = patches.padded[i]
        budget = howmany_in_interval(dos, iv_ext)

        covered, filter, vals_i, vecs_i, report_i, retries = _converge_interior_covered(
            mp, dos, iv, iv_ext, mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer,
            "Interior $i/$npatches")

        covered || polfed_log(POLFED_WARN_LEVEL,
            "Interior patch $i/$npatches still did not bracket its owned interval after " *
            "$(Defaults.patched_polfed.max_glue_retries) retries; the glued spectrum may have a gap " *
            "near it. Raise overestimate_iters / howmany_pad / max_glue_retries.",
            owned = (round(iv.left, digits=5), round(iv.right, digits=5)),
            converged_range = isempty(vals_i) ? nothing :
                (round(first(vals_i), digits=5), round(last(vals_i), digits=5)))

        push!(records, PatchReport(:interior, string(nameof(typeof(filter))), filter.order,
                                   iv, budget, report_i, covered, retries))

        # Seam ownership: the first patch leaves its left endpoint open when an edge peel owns it;
        # the last patch closes its right endpoint unless the right edge peel owns it.
        inc_left  = !(i == 1        && open_left)
        inc_right =  (i == npatches) && !open_right
        @stage timer :glue patch_results_owned!(vals_i, vecs_i, iv, vals, vecs, layout;
                                                include_left = inc_left, include_right = inc_right)
    end
    return nothing
end

"""
    _converge_interior_covered(mp, dos, iv, iv_ext, mapvals, tol, eigentol_rescaled,
                               overestimate_iters, layout, timer, label)
        -> (covered, filter, vals_i, vecs_i, report_i, retries)

Converge one interior patch (filter → transform → Lanczos) and check that its run **brackets** the
owned interval `iv` — `first(vals) ≤ iv.left` and `last(vals) ≥ iv.right`, so every owned pair is
inside the converged set. On failure, RETRY up to `Defaults.patched_polfed.max_glue_retries` times,
each retry widening **both** the overlap (`+howmany_pad` states of padding, re-padded via
[`_pad_interval`](@ref)) **and** the Krylov budget (`overestimate_iters × retry_budget_growth`).
These fix different causes of a short run — an owned edge too close to the bracket edge vs. too few
iterations — so both grow. Returns whether coverage was reached, and the final filter/eigenpairs.
"""
function _converge_interior_covered(mp::MappingPlan, dos::DoSResult, iv::Interval, iv_ext::Interval,
                                    mapvals, tol::Real, eigentol_rescaled::Real,
                                    overestimate_iters::Real, layout::Layout, timer::StageTimer,
                                    label::AbstractString)
    pad0    = Defaults.patched_polfed.howmany_pad
    ed      = Defaults.patched_polfed.edge_distance
    retries = Defaults.patched_polfed.max_glue_retries
    ext   = iv_ext
    # `_run_patch` floors the budget at 2.5×; start the retry ladder at that effective value so the
    # first ×`retry_budget_growth` actually raises `maxdim` instead of being swallowed by the floor.
    oi    = max(2.5, float(overestimate_iters))
    extra = 0
    local vals_i, vecs_i, filter, report_i
    for attempt in 0:retries
        filter     = @stage timer :filter resolve_filter(ext)
        eig_target = window_eig_target(filter, ext)
        budget     = howmany_in_interval(dos, ext)

        attempt == 0 && polfed_log(POLFED_INFO_LEVEL, "$label.",
            interval = (round(iv.left, digits=5), round(iv.right, digits=5)),
            extended = (round(ext.left, digits=5), round(ext.right, digits=5)),
            filter = nameof(typeof(filter)), order = filter.order, howmany = budget)

        vals_i, vecs_i, report_i = _run_patch(mp, filter, eig_target, budget, mapvals, tol,
                                              eigentol_rescaled, oi, layout, timer;
                                              label = attempt == 0 ? label : "$label retry $attempt")

        left_ok  = !isempty(vals_i) && first(vals_i) <= iv.left
        right_ok = !isempty(vals_i) && last(vals_i)  >= iv.right
        (left_ok && right_ok) && return (true, filter, vals_i, vecs_i, report_i, attempt)

        attempt == retries && break
        extra += pad0
        oi    *= Defaults.patched_polfed.retry_budget_growth
        ext    = _pad_interval(dos, iv, pad0 + extra, ed)
        polfed_log(POLFED_INFO_LEVEL, "$label coverage short — retrying wider (padding + budget).",
                   attempt = attempt + 1, extra_pad = extra, overestimate_iters = oi,
                   extended = (round(ext.left, digits=5), round(ext.right, digits=5)))
    end
    return (false, filter, vals_i, vecs_i, report_i, retries)
end

"""
    _patched_glued_spectrum(mp, dos, full_iv, howmany_per_patch, tol, eigentol, overestimate_iters,
                            layout, timer) -> (vals, vecs, records)

Converge and glue the spectrum inside the rescaled window `full_iv`, edge-first (see the file
header). Returns `(vals, vecs, records)` in **rescaled** coordinates, ascending — `vals::Vector{Float64}`,
`vecs` a vector of eigenvector views, and `records::Vector{PatchReport}` (one per run: left edge,
interior patches in order, right edge) for the report. The caller un-rescales and post-processes.
"""
function _patched_glued_spectrum(mp::MappingPlan, dos::DoSResult, full_iv::Interval,
                                 howmany_per_patch::Integer, tol::Real, eigentol::Real,
                                 overestimate_iters::Real, layout::Layout, timer::StageTimer)
    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    # `eigentol` is a TRUE-energy tolerance but `mapvals` produces a RESCALED residual
    # (‖H̃·v − λ̃·v‖ = ‖H·v − λ·v‖ / a); divide by `a` so the test still means ‖H·v − λ·v‖ < eigentol.
    eigentol_rescaled = eigentol / mp.a
    howmany_edge = Defaults.patched_polfed.howmany_edge
    total = howmany_in_interval(dos, full_iv)

    left_present  = full_iv.left  <= -1.0 + 1e-9
    right_present = full_iv.right >=  1.0 - 1e-9

    vals = Float64[]
    vecs = AbstractVector{Float64}[]
    right_vals = Float64[]
    right_vecs = AbstractVector{Float64}[]
    records = PatchReport[]          # one per run, in display order (left edge, interior…, right edge)
    right_patch = nothing            # held so the right-edge record lands AFTER the interior ones
    interior_lo, interior_hi = full_iv.left, full_iv.right

    # Peel the present edges first (count targets); the interior shrinks to their inner boundaries.
    if left_present
        n = clamp(howmany_edge, 1, total)
        lv, lvv, inner, patch = _edge_peel(mp, :lowest, n, mapvals, tol, eigentol_rescaled,
                                           overestimate_iters, layout, timer)
        append!(vals, lv); append!(vecs, lvv); push!(records, patch)
        interior_lo = inner
    end
    if right_present
        n = clamp(howmany_edge, 1, total)
        rv, rvv, inner, patch = _edge_peel(mp, :highest, n, mapvals, tol, eigentol_rescaled,
                                           overestimate_iters, layout, timer)
        right_vals, right_vecs, right_patch = rv, rvv, patch
        interior_hi = inner
    end

    # Tile whatever interior remains between the peels (nothing to do if the peels met/overlapped).
    if interior_lo < interior_hi
        _interior_tile!(vals, vecs, records, mp, dos, Interval(interior_lo, interior_hi),
                        howmany_per_patch, mapvals, tol, eigentol_rescaled, overestimate_iters,
                        layout, timer; open_left = left_present, open_right = right_present)
    end

    # Append the right-edge peel last so the whole set stays ascending (left ≤ interior ≤ right).
    append!(vals, right_vals); append!(vecs, right_vecs)
    right_patch !== nothing && push!(records, right_patch)

    # Drop seam duplicates. At each edge/interior seam the boundary eigenvalue is converged by BOTH
    # the edge peel and the abutting interior patch, to slightly different values (independent
    # Lanczos runs), so ownership can't cleanly exclude one by an energy threshold. Sort and remove
    # any value within `dedup_atol` of its predecessor — the only near-duplicates possible, since
    # every genuine level is separated by ≫ the (rescaled) convergence tolerance. `dedup_atol` sits
    # far above the two copies' disagreement (~`eigentol_rescaled`) yet far below the level spacing.
    perm = sortperm(vals)
    vals = vals[perm]; vecs = vecs[perm]
    dedup_atol = 1e3 * eigentol_rescaled
    keep = trues(length(vals))
    for i in 2:length(vals)
        (vals[i] - vals[i-1] <= dedup_atol) && (keep[i] = false)
    end
    return vals[keep], vecs[keep], records
end
