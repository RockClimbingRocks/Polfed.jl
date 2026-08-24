# ── PatchReport / PatchedReport — the report for a PATCHED polfed run ─────────────────────────────
#
# A patched run is not one factorization but MANY: each edge peel and each interior patch is its own
# filtered Lanczos. So it gets its OWN report — one `PatchReport` per run, keeping that run's whole
# `FactorizationReport` (converged counts, residuals, iterations, Krylov usage, internal timings),
# aggregated by `PatchedReport`. Rendering reuses the single-run machinery (`render_mapping`, the
# section headers, the timing split) and adds a Decomposition headline + an aggregate Convergence
# section, with the full per-run table one `:convergence_details` flag away.

"""
    PatchReport

One patched-POLFED run (an edge peel or an interior patch). Fields:
- `kind::Symbol`: `:edge_lowest` / `:edge_highest` / `:interior`.
- `filtertype::String`, `order::Int`: the filter used and its polynomial order.
- `interval::Interval`: the owned (interior) or edge-band interval, in rescaled coordinates.
- `howmany::Int`: this run's Krylov budget (state count).
- `factorization::FactorizationReport`: the full per-run Lanczos report.
- `covered::Bool`: whether the run bracketed its owned interval (`true` for edge peels).
- `retries::Int`: coverage retries used (interior only).
"""
struct PatchReport
    kind::Symbol
    filtertype::String
    order::Int
    interval::Interval
    howmany::Int
    factorization::FactorizationReport
    covered::Bool
    retries::Int
end

"""
    PatchedReport

Aggregate report for a patched-POLFED run: the [`MappingReport`](@ref), the rescaled `window`
covered, the sizing knobs, one [`PatchReport`](@ref) per run, the glued eigenpair count, the
rescaling (`a`, `b`, for true-energy display), and the pipeline [`StageTimer`](@ref). Rendered by
`Base.show(::IO, ::MIME"text/plain", …)` as four sections (Decomposition, Mapping, Convergence,
Timings); pass `:convergence_details` on the IO to append the full per-run table.
"""
struct PatchedReport
    mapping::MappingReport
    window::Interval
    howmany_per_patch::Int
    howmany_edge::Int
    howmany_pad::Int
    patches::Vector{PatchReport}
    total_eigenpairs::Int
    a::Float64
    b::Float64
    timer::StageTimer
end

# ── Decomposition (the patched headline) ──────────────────────────────────────

"""Print the "Spectral Decomposition" section: run counts, sizing, filter histogram, order spread,
and the aggregate spectral-transform cost."""
function render_decomposition(io::IO, f::Formatter, r::PatchedReport)
    _section_header(io, f, "Spectral Decomposition")
    n_edge = count(p -> p.kind !== :interior, r.patches)
    n_int  = count(p -> p.kind === :interior, r.patches)
    El, Er = r.a * r.window.left + r.b, r.a * r.window.right + r.b

    println(io, "- Window: resc. [", cyan(f, @sprintf("%.4f", r.window.left)), ", ",
            cyan(f, @sprintf("%.4f", r.window.right)), "]  (true [",
            cyan(f, @sprintf("%.4g", El)), ", ", cyan(f, @sprintf("%.4g", Er)), "])")
    # An edge PEEL is a run KIND (the count-target extremal grab, only when the window touches ±1);
    # it is distinct from the EdgeFilter TYPE below, which a near-±1 INTERIOR patch can also use.
    println(io, "- Runs: ", cyan(f, string(length(r.patches))), " — ",
            cyan(f, string(n_int)), " interior patches (Window target) + ",
            cyan(f, string(n_edge)), " edge peel(s) (count target)")
    println(io, "- Sizing: howmany_per_patch=", cyan(f, string(r.howmany_per_patch)),
            ", howmany_edge=", cyan(f, string(r.howmany_edge)),
            ", howmany_pad=", cyan(f, string(r.howmany_pad)))

    counts = Dict{String,Int}()
    for p in r.patches
        counts[p.filtertype] = get(counts, p.filtertype, 0) + 1
    end
    hist = join([string(cyan(f, string(v)), "× ", k) for (k, v) in sort(collect(counts))], ", ")
    # EdgeFilters used by interior patches (padded window reached ±1) are NOT edge peels — call it
    # out so the "0 edge peels" vs "N× EdgeFilter" counts don't read as a contradiction.
    n_edge_interior = count(p -> p.kind === :interior && p.filtertype == "EdgeFilter", r.patches)
    edge_note = n_edge_interior > 0 ?
        string("  (", cyan(f, string(n_edge_interior)), " EdgeFilter",
               n_edge_interior == 1 ? " is a" : "s are", " near-±1 interior patch",
               n_edge_interior == 1 ? "" : "es", ", not edge peels)") : ""
    println(io, "- Filters: ", hist, edge_note)

    orders = [p.order for p in r.patches]
    println(io, "- Polynomial order: min ", cyan(f, string(minimum(orders))),
            ", max ", cyan(f, string(maximum(orders))),
            ", mean ", cyan(f, @sprintf("%.1f", sum(orders) / length(orders))), " (max drives the cost)")

    cost = sum(p -> p.order * p.factorization.blocksize * p.factorization.itersneeded, r.patches)
    println(io, "- Cost: H·v = Σ Kₚ·bsₚ·itersₚ = ", cyan(f, string(cost)), " mat-vec muls")
    return nothing
end

# ── Convergence (aggregate over runs, + optional per-run table) ────────────────

"""Print the "Convergence" section aggregated across all runs; with `details=true`, append a
per-run [`PrettyTables`](@ref) table (every patched POLFED run in full)."""
function render_patched_convergence(io::IO, f::Formatter, r::PatchedReport; details::Bool=false)
    np = length(r.patches)
    _section_header(io, f, "Convergence (across $np runs)")

    # `converged`, not either gate count: it is what each run RETURNED. Both gate counts are
    # per-gate tallies that overcount whenever a run was truncated — by `eigentol`, or by the
    # contiguity invariant stranding pairs behind a failure.
    total_conv = sum(p -> p.factorization.converged, r.patches)
    n_short    = count(p -> !p.covered, r.patches)
    # The EIGEN residual (‖H·v − λ·v‖) of each run's RETURNED pairs — the accuracy that matters.
    # (Not `.residual`, the worst Ritz residual in the pool, which includes unconverged brackets.)
    worst_res  = maximum(p -> p.factorization.eigenresidual, r.patches)
    iters      = sum(p -> p.factorization.itersneeded, r.patches)
    reserved   = sum(p -> p.factorization.itersreserved, r.patches)
    retries    = sum(p -> p.retries, r.patches)

    println(io, "- Eigenpairs kept (glued): ", cyan(f, string(r.total_eigenpairs)),
            " (", cyan(f, string(total_conv)), " converged across runs)")
    short_col = n_short == 0 ? green(f, "0") : red(f, string(n_short))
    retry_note = retries > 0 ? string("  (", cyan(f, string(retries)), " coverage retr",
                                       retries == 1 ? "y" : "ies", " used)") : ""
    println(io, "- Runs short of coverage: ", short_col, " / ", cyan(f, string(np)), retry_note)
    println(io, "- Worst eigen residual (‖H·v−λ·v‖) across runs: ", yellow(f, @sprintf("%.2e", worst_res)))
    println(io, "- Krylov iterations: ", bold(f, string(iters)), " total (of ",
            cyan(f, string(reserved)), " reserved)")

    if details
        println(io, "- Per-run detail:")
        data = Matrix{String}(undef, np, 8)
        for (i, p) in enumerate(r.patches)
            data[i, :] = [string(i), string(p.kind), p.filtertype, string(p.order),
                          string(p.factorization.converged, "/", p.howmany),
                          @sprintf("%.1e", p.factorization.eigenresidual),
                          string(p.factorization.itersneeded), string(p.retries)]
        end
        header = ["#", "kind", "filter", "K", "conv/req", "eigen res", "iters", "retries"]
        render_detail_table(io, header, data; bold_header = f.color)
    end
    return nothing
end

# ── Timings (pipeline split + Lanczos-internal summed over runs) ───────────────

"""Print the "Timings" section: total run time, the pipeline Lanczos-vs-preparation split (from the
pipeline timer), and the Lanczos-internal decomposition summed over every run's factorization."""
function render_patched_timings(io::IO, f::Formatter, r::PatchedReport)
    _section_header(io, f, "Timings")
    is_gpu = any(p -> p.factorization.ncols_gpu_needed > 0, r.patches)
    st     = r.timer
    tot_wt = total_wall(st)
    wt_v, wt_u = _fmt_secs(tot_wt)

    if is_gpu
        println(io, "- Total patched run took: ", cyan(f, @sprintf("%.2f %s", wt_v, wt_u)), " (walltime)")
    else
        ct_v, ct_u = _fmt_secs(total_cpu(st))
        println(io, "- Total patched run took: ", cyan(f, @sprintf("%.2f %s", wt_v, wt_u)),
                " (walltime), ", cyan(f, @sprintf("%.2f %s", ct_v, ct_u)), " (CPU time)")
    end

    print(io, "- Walltime distribution: ")
    _pct_line(io, f, [("Lanczos",     _safe_pct(st[:lanczos], tot_wt)),
                      ("preparation", _safe_pct(max(0.0, tot_wt - st[:lanczos]), tot_wt))])

    # Lanczos-internal decomposition, summed by stage NAME over every run's factorization timer.
    lz_total = sum(p -> total_wall(p.factorization.timer), r.patches)
    _sum(key) = sum(p -> p.factorization.timer[key], r.patches)
    if lz_total > 0
        print(io, "- Lan walltime: ")
        _pct_line(io, f, [("matvec",     _safe_pct(_sum(:matvec), lz_total)),
                          ("reorthog.",  _safe_pct(_sum(:reorthog), lz_total)),
                          ("tridiag.",   _safe_pct(_sum(:tridiag) + _sum(:overlap) + _sum(:orthog), lz_total)),
                          ("conv.check", _safe_pct(_sum(:conv_check), lz_total))])
    end
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", r::PatchedReport)

Render the patched report: Decomposition, Mapping, Convergence, Timings. Honors the IO properties
`:color` (auto from the sink) and `:convergence_details` (append the per-run table).
"""
function Base.show(io::IO, ::MIME"text/plain", r::PatchedReport)
    f = Formatter(get(io, :color, false))
    details = get(io, :convergence_details, false)
    render_decomposition(io, f, r)
    render_mapping(io, f, r.mapping)
    render_patched_convergence(io, f, r; details)
    render_patched_timings(io, f, r)
    return nothing
end

Base.show(io::IO, r::PatchedReport) =
    print(io, "PatchedReport(", length(r.patches), " runs, ", r.total_eigenpairs, " eigenpairs)")
