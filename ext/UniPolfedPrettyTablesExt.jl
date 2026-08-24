module UniPolfedPrettyTablesExt

# PrettyTables.jl is an OPTIONAL dependency of UniPolfed. Loading it swaps the reports'
# detail-table backend from the built-in `Common.plain_table` to PrettyTables; without it
# the tables still render, just drawn by UniPolfed itself. See `src/common/table.jl`.

using PrettyTables

import UniPolfed

"""
    _pretty_table(io, header, rows; bold_header=false) -> Nothing

Render a report detail table with PrettyTables.jl. Signature-compatible with
`UniPolfed.Common.plain_table`, which it replaces while this extension is loaded.

Cells arrive already stringified (the renderer contract), so no `formatters` are needed here —
the numeric formatting was applied by the caller and is identical under either backend.
"""
function _pretty_table(io::IO, header::AbstractVector{<:AbstractString},
                       rows::AbstractMatrix; bold_header::Bool=false)
    size(rows, 1) == 0 && return nothing
    kwargs = (; header = collect(String, header), tf = tf_unicode_rounded)
    if bold_header
        pretty_table(io, rows; kwargs..., header_crayon = crayon"white bold")
    else
        pretty_table(io, rows; kwargs...)
    end
    return nothing
end

function __init__()
    UniPolfed.Common.set_table_renderer!(_pretty_table)
    return nothing
end

end
