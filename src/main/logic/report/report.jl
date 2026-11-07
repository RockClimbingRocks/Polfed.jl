# Report slice: the structured, human-readable summary POLFED returns alongside the
# eigenpairs. A `Report` aggregates four sections — Mapping, Spectral Transformation,
# Convergence, Timings — and is rendered through `Base.show(::IO, ::MIME"text/plain", …)`,
# so `display(report)` (and typing it at the REPL) just works. Two IO properties tune the
# output, both read straight off the sink:
#   * `:color`                (auto: `true` on a TTY, `false` when redirected to a file)
#   * `:convergence_details`  (default `false`; `true` appends the per-check history table)
#
# The Mapping/Transform sections are plain-data snapshots (MappingReport / SpectralTransform-
# Report); the Convergence/Timings sections read the Lanczos `FactorizationReport` directly.
# Post-processing of the raw Lanczos result into the returned `PolfedResult` lives in
# build_report.jl.

# ── shared rendering helpers ──────────────────────────────────────────────────

"""Total printed width of a section header line (title + trailing rule)."""
const _RULE_WIDTH = 72

"""
    _section_header(io, f::Formatter, title::AbstractString) -> Nothing

Print a section header: the title followed by a rule filling the line out to `_RULE_WIDTH`
(`"Mapping ─────…"`), all in blue when `f` has color enabled. `textwidth` is used rather than
`length` so the rule lands correctly for titles carrying wide/combining characters.
"""
function _section_header(io::IO, f::Formatter, title::AbstractString)
    pad = max(3, _RULE_WIDTH - textwidth(title) - 1)
    println(io, blue(f, string(title, " ", "─"^pad)))
    return nothing
end

"""
    _underscore(n::Integer) -> String

Group a non-negative integer into underscore-separated thousands (`471800 → "471_800"`).
"""
_underscore(n::Integer) = reverse(join(Iterators.partition(reverse(string(n)), 3), "_"))

"""
    _fmt_secs(t::Real) -> (Float64, String)

Scale a duration in seconds to a `(value, unit)` pair (`"seconds"`/`"minutes"`/`"hours"`),
switching units at two minutes and two hours.
"""
_fmt_secs(t::Real) = t <= 120  ? (Float64(t), "seconds") :
                     t <= 7200 ? (t / 60, "minutes") : (t / 3600, "hours")

"""    _safe_pct(x, total) -> Float64 — `x/total·100`, or `0.0` when `total ≤ 0`."""
_safe_pct(x::Real, total::Real) = total > 0 ? x / total * 100 : 0.0

"""
    _pct_line(io, f, pairs; digits::Int=2) -> Nothing

Print `label p% · label p% · …` for a vector of `(label, percent)` pairs, percentages in cyan.
"""
function _pct_line(io::IO, f::Formatter, pairs; digits::Int = 2)
    parts = [string(lab, " ", cyan(f, _fmt_pct(p, digits))) for (lab, p) in pairs]
    println(io, join(parts, " · "))
    return nothing
end

"""
    _fmt_pct(p, digits) -> String

A percentage at a FIXED number of decimals (`12.3 → "12.30%"`, `0 → "0.00%"`). Fixed width matters
here: these render as aligned `·`-separated runs, and `round` would print `0.0%` next to `15.15%`.
"""
_fmt_pct(p::Real, digits::Int) = Printf.format(Printf.Format("%.$(digits)f%%"), p)

"""
    _overestimate_color(f::Formatter, needed::Int, reserved::Int) -> String

The Krylov-budget headroom `(1 − needed/reserved)·100` as a colored percentage, using the same
bands as the Lanczos [`display_factorization_report`](@ref): **red** below 3% (the budget nearly
ran out — the run may be short) or above 25% (badly over-reserved, wasting basis memory),
**yellow** in 3–5%, **green** in the healthy 5–25% range.
"""
function _overestimate_color(f::Formatter, needed::Int, reserved::Int)
    pct = reserved > 0 ? (1 - needed / reserved) * 100 : 0.0
    s = @sprintf("%.2f%%", pct)
    return pct <= 3  ? red(f, s)   :
           pct <= 5  ? yellow(f, s) :
           pct <= 25 ? green(f, s)  : red(f, s)
end

include("MappingReport.jl")
include("SpectralTransformReport.jl")

# ── the aggregated report ─────────────────────────────────────────────────────

"""
    Report

Structured summary of a POLFED run, carried on the returned [`PolfedResult`](@ref) and
rendered as four sections by `show(io, MIME"text/plain"(), report)`:

1. **Mapping** — [`MappingReport`](@ref): operator/kernel, rescaling, layout, parallelization.
2. **Spectral Transformation** — [`SpectralTransformReport`](@ref): target, window, filter,
   order, and the `H·v` cost.
3. **Convergence** — from the Lanczos [`FactorizationReport`](@ref): requested/converged counts,
   residual/eigen tolerance failures, worst residual, Krylov usage (+ optional per-check table).
4. **Timings** — the POLFED-level [`StageTimer`](@ref) plus the Lanczos-internal decomposition
   from the `FactorizationReport`'s own timer.

Display is controlled entirely through IO properties: `:color` (auto from the sink) and
`:convergence_details` (default `false`). See [`display`](@ref) / `show`.

# Fields
- `mapping::MappingReport`, `transform::SpectralTransformReport`, `factorization::FactorizationReport`.
- `timer::StageTimer`: POLFED-level stage timings, addressed by name — `:bounds`, `:dos`,
  `:transform_build`, `:lanczos` — as annotated in the pipelines. Stages the pipeline never ran
  read as zero, so the renderer needs no presence checks.
"""
struct Report
    mapping::MappingReport
    transform::SpectralTransformReport
    factorization::FactorizationReport
    timer::StageTimer
end

# ── Convergence section (reads the Lanczos FactorizationReport) ────────────────

"""
    render_convergence(io, f::Formatter, fr::FactorizationReport; details::Bool=false) -> Nothing

Print the "Convergence" section: requested/converged counts, how many eigenpairs passed the
residual (`tol`) and eigen (`eigentol`) gates, the worst residual of each, and Krylov usage.
With `details=true`, append the per-check convergence-history table (via `PrettyTables`).

Three counts, not one, because they answer different questions: the Ritz count is how many
pairs the Krylov space resolved, the eigen count how many of those are certified eigenpairs of
the ORIGINAL operator, and `converged` the contiguous run those collapse to — the pairs actually
returned. A gap earns an extra "Limited by" line naming which of the two losses occurred; see
[`CONVERGENCE_TABLE_HEADER`](@ref).
"""
function render_convergence(io::IO, f::Formatter, fr::FactorizationReport; details::Bool=false)
    kind = occursin("Block", fr.factorizationtype) ? "block lanczos" : "lanczos"
    _section_header(io, f, "Convergence ($kind)")

    # Counts are green only when the full request converged — the one number worth spotting at a
    # glance. Residuals are yellow throughout: they are always "how close", never pass/fail.
    _count(n) = n == fr.howmany ? green(f, string(n)) : red(f, string(n))
    # The Ritz gate can legitimately reach PAST the request (a window target's brackets, an
    # over-targeted patch), so it is green at "enough", not at "exactly".
    _count_atleast(n) = n >= fr.howmany ? green(f, string(n)) : red(f, string(n))

    println(io, "- Number of converged eigenpairs: ", _count(fr.converged),
            " (", cyan(f, string(fr.howmany)), " requested)")
    println(io, "- Ritz  conv tol=", cyan(f, @sprintf("%.1e", fr.tol)), " passed:  ",
            _count_atleast(fr.converged_tol), " (max res. ", yellow(f, @sprintf("%.2e", fr.residual)), ")")
    # `eigenworst` (over everything examined), not `eigenresidual` (over what was kept): on this
    # line the count includes pairs the run did not return, so pairing it with the kept-set
    # residual would understate by orders of magnitude exactly when it matters.
    println(io, "- Eigen conv tol=", cyan(f, @sprintf("%.1e", fr.eigentol)), " passed:  ",
            _count_atleast(fr.converged_eigentol), " (worst examined ",
            yellow(f, isnan(fr.eigenworst) ? "-" : @sprintf("%.2e", fr.eigenworst)), ")")
    # The one line that separates "needs a bigger budget" from "cannot be reached at this
    # tolerance". They look identical on the residual columns above: when the eigen gate is
    # what truncates, the Ritz residual is at its FLOOR precisely because the Krylov space
    # converged, so a reader with only that number concludes the opposite of the truth.
    if fr.converged_tol > fr.converged
        beyond   = fr.converged_tol - fr.converged            # Ritz-converged but not returned
        rejected = fr.converged_tol - fr.converged_eigentol   # of those, the actual eigen failures
        stranded = beyond - rejected                          # the rest: behind a failure
        println(io, "- Limited by: ", yellow(f, "the eigen tolerance, not the Krylov budget"),
                " — ", cyan(f, string(beyond)),
                " pair(s) beyond the returned run passed the Ritz gate.")
        # Splitting the loss matters because the two halves are very differently sized and only
        # the FIRST is a tolerance problem. A handful of scattered failures can strand hundreds
        # of pairs that passed both gates, which reads as "nothing converged" unless said aloud.
        println(io, "  ", cyan(f, string(rejected)), " fail `eigentol`; ",
                cyan(f, string(stranded)),
                " passed BOTH gates but sit behind one, and the contiguous-run rule drops them.")
        println(io, "  More iterations cannot extend the run; relax `eigentol` or narrow the window.")
    end
    println(io, "- Iterations needed: ", bold(f, string(fr.itersneeded)),
            " (out of ", cyan(f, string(fr.itersreserved)), " reserved, overestimated by ",
            _overestimate_color(f, fr.itersneeded, fr.itersreserved), ")")
    # The criterion's own completeness verdict. Worth a line of its own: an uncertified set is
    # gap-free but may be SHORT at the window edges, which no residual on the line above reveals.
    # Name the actual cause — "budget exhausted" is a lie whenever the eigen gate is what cut
    # the run, and it is the lie that sends the reader off to raise `overestimate_iters`.
    uncertified = fr.converged_tol > fr.converged ?
        "NOT certified — cut by `eigentol`, set is short" :
        "NOT certified — budget exhausted, set may be short"
    println(io, "- Returned set: ", fr.complete ? green(f, "certified complete") :
            yellow(f, uncertified))

    if details && fr.numofchecks > 0
        println(io, "- Convergence check ran ", cyan(f, string(fr.numofchecks)), " times:")
        # Bold (not blue) header: blue is the section-title color here, and reusing it made the
        # table read as a new section rather than as detail hanging off the Convergence one.
        kwargs = (; column_labels = CONVERGENCE_TABLE_HEADER,
                  table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                  # whole table, never cropped to the display (PrettyTables 2 did not crop)
                  fit_table_in_display_horizontally = false,
                  fit_table_in_display_vertically = false)
        if f.color
            pretty_table(io, convergence_table_rows(fr); kwargs...,
                         style = TextTableStyle(first_line_column_label = crayon"white bold"))
        else
            pretty_table(io, convergence_table_rows(fr); kwargs...)
        end
    end
    return nothing
end

# ── Timings section (POLFED-level split + Lanczos-internal decomposition) ──────

"""
    render_timings(io, f::Formatter, r::Report) -> Nothing

Print the "Timings" section: the total run time (wall and CPU), the POLFED-level stage
distribution, and the Lanczos-internal decomposition (`matvec`/`reorthog.`/`tridiag.`/`conv.check`)
in wall **and** CPU time. CPU time is **omitted on GPU runs** (where it is misleading).

Both breakdowns are read by stage NAME off a [`StageTimer`](@ref), so the pipelines choose which
stages exist and in what order — the renderer walks whatever they recorded, in execution order.
"""
function render_timings(io::IO, f::Formatter, r::Report)
    _section_header(io, f, "Timings")

    is_gpu = r.factorization.ncols_gpu_needed > 0
    st     = r.timer
    tot_wt = total_wall(st)
    tot_ct = total_cpu(st)
    wt_v, wt_u = _fmt_secs(tot_wt)
    ct_v, ct_u = _fmt_secs(tot_ct)

    if is_gpu
        println(io, "- Total polfed run took: ", cyan(f, @sprintf("%.2f %s", wt_v, wt_u)), " (walltime)")
    else
        println(io, "- Total polfed run took: ", cyan(f, @sprintf("%.2f %s", wt_v, wt_u)),
                " (walltime), ", cyan(f, @sprintf("%.2f %s", ct_v, ct_u)), " (CPU time)")
    end

    # Two-way split only: Lanczos (the bottleneck) vs everything that set it up. The pipeline
    # records each preparation stage separately (`:bounds`, `:dos`, `:interval`, `:filter`,
    # `:transform`), but at report altitude the per-stage detail is noise — it is one line away
    # for anyone who wants it: `display(result.report.timer)` prints the full tree.
    lanczos_wt = st[:lanczos]
    print(io, "- Walltime distribution: ")
    _pct_line(io, f, [("Lanczos",     _safe_pct(lanczos_wt, tot_wt)),
                      ("preparation", _safe_pct(max(0.0, tot_wt - lanczos_wt), tot_wt))])

    # Lanczos-internal decomposition, read by stage NAME from the factorization's StageTimer
    # (see lanczos Factorization.jl / lanczos_algorithm.jl for where each is annotated). The three
    # cheap per-iteration steps are folded into one "tridiag." bucket — individually they are noise.
    t  = r.factorization.timer
    lz_wt, lz_ct = total_wall(t), total_cpu(t)
    _lanczos_split(tot, key) = [
        ("matvec",     _safe_pct(t[:matvec, key], tot)),
        ("reorthog.",  _safe_pct(t[:reorthog, key], tot)),
        ("tridiag.",   _safe_pct(t[:tridiag, key] + t[:overlap, key] + t[:orthog, key], tot)),
        ("conv.check", _safe_pct(t[:conv_check, key], tot)),
    ]
    if lz_wt > 0
        print(io, "- Lan walltime: ")
        _pct_line(io, f, _lanczos_split(lz_wt, :wall))
    end
    if lz_ct > 0 && !is_gpu
        print(io, "- Lan CPU time: ")
        _pct_line(io, f, _lanczos_split(lz_ct, :cpu))
    end
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", r::Report)

Render the full four-section report to `io`. Honors the IO properties `:color` (auto from the
sink) and `:convergence_details` (default `false`, appends the per-check history table).
"""
function Base.show(io::IO, ::MIME"text/plain", r::Report)
    f = Formatter(get(io, :color, false))
    details = get(io, :convergence_details, false)
    # Transformation first: it is the run's headline (what was targeted, at what order and cost).
    # Mapping follows as the setup it ran on, then the outcome, then the timings.
    render_transform(io, f, r.transform)
    render_mapping(io, f, r.mapping)
    render_convergence(io, f, r.factorization; details)
    render_timings(io, f, r)
    return nothing
end

Base.show(io::IO, ::Report) = print(io, "Report(Mapping/Transform/Convergence/Timings)")

include("PatchedReport.jl")   # aggregate report for the patched variants (reuses the helpers above)
include("PolfedResult.jl")
include("PatchedResult.jl")   # the (vals, vecs, report) object returned by patched_polfed / full_polfed
include("DoublePolfedResult.jl")   # the two-window pair returned by double_polfed
include("display_report.jl")  # the `display_report(…)` print verb over the show methods above
include("build_report.jl")
