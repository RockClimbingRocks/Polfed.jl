# Detail-table rendering for the reports' optional `:convergence_details` sections.
#
# PrettyTables.jl is an OPTIONAL dependency. When the user loads it, the package extension
# `UniPolfedPrettyTablesExt` swaps its renderer in via `set_table_renderer!`; with no extra
# package loaded the plain renderer below draws the same table. Either way the detail tables
# always render — a debugging aid that silently disappears is worse than no aid at all.
#
# Contract shared by both renderers: cells arrive ALREADY STRINGIFIED. Numeric formatting
# (`%d`, `%.2e`, …) is the caller's job, which keeps the two renderers interchangeable and
# means neither needs a formatter DSL.

"""
    plain_table(io, header, rows; bold_header=false) -> Nothing

Draw `rows` under `header` as a rounded box-drawing table, using only `Base` string padding.
The default [`render_detail_table`](@ref) backend.

Column widths are the widest cell (header included) measured in display columns; a column
whose every cell parses as a number is right-aligned, any other column is left-aligned.

# Arguments
- `io::IO`: sink to draw into.
- `header::AbstractVector{<:AbstractString}`: column titles, one per column.
- `rows::AbstractMatrix`: `nrow × ncol` cells; each is passed through `string`.
- `bold_header::Bool=false`: wrap the header row in the ANSI bold escape. Pass the
  formatter's own `color` flag so a redirect to a file stays plain.

# Returns
- `nothing`.

# Notes
Cells must not themselves carry ANSI escapes: widths are measured with `textwidth`, which
counts an escape sequence's bytes as printing characters and would misalign the column. The
header is the one styled element, and it is wrapped as a whole *after* padding.

Emits nothing at all when `rows` has no rows, rather than an empty frame.
"""
function plain_table(io::IO, header::AbstractVector{<:AbstractString},
                     rows::AbstractMatrix; bold_header::Bool=false)
    ncol = length(header)
    size(rows, 2) == ncol || throw(DimensionMismatch(
        "table has $(size(rows, 2)) columns but $(ncol) header entries"))
    size(rows, 1) == 0 && return nothing

    cells = string.(rows)
    width = [max(textwidth(header[j]), maximum(textwidth, view(cells, :, j); init = 0))
             for j in 1:ncol]
    # A column is numeric only if EVERY cell parses as one; mixed columns read better left-aligned.
    right = [all(c -> occursin(r"^[-+]?[\d.]+(?:[eE][-+]?\d+)?$", c), view(cells, :, j))
             for j in 1:ncol]

    pad(s, j) = right[j] ? lpad(s, width[j]) : rpad(s, width[j])
    row(cs)   = string("│", join((" " * pad(cs[j], j) * " " for j in 1:ncol), "│"), "│")
    rule(l, m, r) = println(io, l, join(("─"^(width[j] + 2) for j in 1:ncol), m), r)

    rule("╭", "┬", "╮")
    head = row(header)
    println(io, bold_header ? string("\e[1m", head, "\e[0m") : head)
    rule("├", "┼", "┤")
    for i in axes(cells, 1)
        println(io, row(view(cells, i, :)))
    end
    rule("╰", "┴", "╯")
    return nothing
end

# Abstractly typed BY DESIGN, exactly as the GPU backend hooks in `backend/gpu_dispatch.jl`
# are: written once when an extension loads, read only while rendering a report — never in a
# hot loop, so the dynamic dispatch is free here.
const _TABLE_RENDERER = Ref{Function}(plain_table)

"""
    render_detail_table(io, header, rows; bold_header=false) -> Nothing

Draw a report detail table through the active backend: [`plain_table`](@ref) by default, or
PrettyTables.jl when the user has loaded it (see [`set_table_renderer!`](@ref)).

Arguments match [`plain_table`](@ref); cells must already be strings.
"""
render_detail_table(io::IO, header::AbstractVector{<:AbstractString},
                    rows::AbstractMatrix; bold_header::Bool=false) =
    _TABLE_RENDERER[](io, header, rows; bold_header)

"""
    set_table_renderer!(f) -> Nothing

Install `f` as the [`render_detail_table`](@ref) backend. Called from
`UniPolfedPrettyTablesExt.__init__` when PrettyTables.jl is loaded; `f` must accept the same
`(io, header, rows; bold_header)` signature as [`plain_table`](@ref).
"""
set_table_renderer!(f::Function) = (_TABLE_RENDERER[] = f; nothing)

"""
    reset_table_renderer!() -> Nothing

Restore the built-in [`plain_table`](@ref) backend. Used by the tests to exercise both
renderers in one session.
"""
reset_table_renderer!() = (_TABLE_RENDERER[] = plain_table; nothing)
