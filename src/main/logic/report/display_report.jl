# The user-facing print verb for every report/result type in this slice. It is a thin
# convenience wrapper: the rendering itself lives in the `Base.show(::IO, ::MIME"text/plain",
# …)` methods (report.jl, PatchedReport.jl, PolfedResult.jl, PatchedResult.jl,
# DoublePolfedResult.jl), which stay the single source of truth so `display(report)` and
# typing a report at the REPL keep working. All this file does is turn the two display
# switches — which the renderers read off the IO sink as the `:color` and
# `:convergence_details` properties — into ordinary keyword arguments, so callers can write
# `display_report(report; convergence_details = true)` instead of assembling an `IOContext`
# by hand (`display(report; convergence_details = true)` cannot work: `Base.display` takes
# no keywords).

"""
Anything in the report slice that renders through `show(io, MIME"text/plain"(), x)`: the two
report aggregates and the three result objects the entrypoints return. Kept as an explicit
`Union` so [`display_report`](@ref) fails with a method error — not a fallback one-line
`show` — when handed something else.
"""
const ReportLike = Union{Report, PatchedReport, PolfedResult, PatchedResult, DoublePolfedResult}

"""
    _auto_color(io::IO) -> Bool

Default for `display_report`'s `color` keyword: honor an explicit `:color` property on `io`
if one is set, otherwise enable ANSI colors only when `io` is a color-capable terminal — so
piping to a file or a pager yields plain text without the caller asking.
"""
_auto_color(io::IO) = get(io, :color, Base.have_color === true && io isa Base.TTY)

"""
    display_report(r; io = stdout, convergence_details = false, color = auto) -> Nothing
    display_report(io::IO, r; convergence_details = false, color = auto)      -> Nothing

Print a POLFED report or result object — [`Report`](@ref), [`PatchedReport`](@ref),
[`PolfedResult`](@ref), [`PatchedResult`](@ref), [`DoublePolfedResult`](@ref) — to `io` as
its full multi-section summary.

Equivalent to `show(IOContext(io, :color => color, :convergence_details =>
convergence_details), MIME"text/plain"(), r)`; the sections and their layout are documented
on the type being printed.

# Arguments
- `r::ReportLike`: the report or result to render.
- `io::IO = stdout`: sink, positionally or as a keyword.

# Keywords
- `convergence_details::Bool = false`: append the per-check (single run) or per-run (patched)
  convergence-history table.
- `color::Bool`: ANSI colors. Defaults to `io`'s `:color` property when set, else `true`
  only for a color-capable terminal — see [`_auto_color`](@ref).

# Returns
`nothing`; the report is written to `io`.

# Notes
- This is a convenience verb only. `display(r)`, `show(io, MIME"text/plain"(), r)`, and
  typing `r` at the REPL all still render the same report through the same `Base.show`
  methods; use those (with an `IOContext`) when you need the two switches to travel with an
  IO sink rather than be spelled at the call site.
- Writing a report to a file is `open(f -> display_report(f, r), "run.txt", "w")` — `color`
  then defaults to `false`, so the file holds no escape sequences.

# Examples
```julia
vals, vecs, report = polfed(H, x0, 50, :maxdos)
display_report(report)                              # four sections, colored on a TTY
display_report(report; convergence_details = true)  # + the per-check history table
open(f -> display_report(f, report), "run.txt", "w")
```
"""
function display_report(io::IO, r::ReportLike; convergence_details::Bool = false,
                        color::Bool = _auto_color(io))
    show(IOContext(io, :color => color, :convergence_details => convergence_details),
         MIME"text/plain"(), r)
    println(io)
    return nothing
end

display_report(r::ReportLike; io::IO = stdout, kwargs...) = display_report(io, r; kwargs...)
