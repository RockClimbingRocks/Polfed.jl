# PolfedResult: the value POLFED returns. Holds the post-processed eigenpairs (already
# filtered to the target interval, sorted ascending by true energy) and the structured
# `Report`. Iterating a `PolfedResult` yields `(val, vec)` pairs; the eigenvector view is
# extracted according to the Krylov `Layout` (columns for ColumnLayout, rows for RowLayout),
# so the caller always sees a length-`dim` vector regardless of storage orientation.

"""
    _result_eigvec(layout::Layout, vecs::AbstractMatrix, i::Int) -> AbstractVector

The `i`-th eigenvector as a length-`dim` view: a column of `vecs` for [`ColumnLayout`](@ref)
(`dim × n`), a row for [`RowLayout`](@ref) (`n × dim`).
"""
_result_eigvec(::ColumnLayout, vecs::AbstractMatrix, i::Int) = view(vecs, :, i)
_result_eigvec(::RowLayout,    vecs::AbstractMatrix, i::Int) = view(vecs, i, :)

"""
    PolfedResult

Result of [`polfed`](@ref): the eigenpairs near the target (already filtered to the target
interval and sorted ascending by true energy of `H`) plus the run's [`Report`](@ref).

Iterate it for `(val, vec)` pairs (`for (λ, v) in result`); index it (`result[i]`) for the same;
`length(result)` is the number of eigenpairs. Displaying it (`display(result)`, or typing it at
the REPL) prints the report. See [`Report`](@ref) for the display-tuning IO properties.

# Fields
- `vals::V`: eigenvalues (true energies of `H`), sorted ascending, within the target interval.
- `vecs::M`: eigenvectors — columns for [`ColumnLayout`](@ref), rows for [`RowLayout`](@ref).
- `residuals::Vector{Float64}`: per-eigenvalue true residual `‖H·vᵢ − λᵢ·vᵢ‖`, aligned with `vals`.
- `converged::Int`: total converged eigenpairs from Lanczos (before interval filtering; the
  in-interval count is `length(vals)`).
- `layout::L`: the Krylov [`Layout`](@ref) the eigenvectors are stored in.
- `report::Report`: the structured Mapping/Transform/Convergence/Timings summary.
"""
struct PolfedResult{V<:AbstractVector, M<:AbstractMatrix, L<:Layout}
    vals::V
    vecs::M
    residuals::Vector{Float64}
    converged::Int
    layout::L
    report::Report
end

Base.length(r::PolfedResult) = length(r.vals)
Base.eltype(::Type{<:PolfedResult{V}}) where {V} = Tuple{eltype(V), AbstractVector}
Base.firstindex(::PolfedResult) = 1
Base.lastindex(r::PolfedResult) = length(r)

"""    getindex(r::PolfedResult, i) -> (val, vec) — the `i`-th eigenpair (eigenvector as a length-`dim` view)."""
Base.getindex(r::PolfedResult, i::Integer) = (r.vals[i], _result_eigvec(r.layout, r.vecs, i))

function Base.iterate(r::PolfedResult, i::Int=1)
    i > length(r) && return nothing
    return (r[i], i + 1)
end

"""
    show(io::IO, ::MIME"text/plain", r::PolfedResult)

Print the `POLFED result` header followed by the full [`Report`](@ref). Honors the report's IO
properties (`:color`, `:convergence_details`).
"""
function Base.show(io::IO, ::MIME"text/plain", r::PolfedResult)
    f = Formatter(get(io, :color, false))
    println(io, blue(f, "POLFED result"))
    println(io, blue(f, "═"^60))
    println(io)
    show(io, MIME"text/plain"(), r.report)
    return nothing
end

Base.show(io::IO, r::PolfedResult) = print(io, "PolfedResult(", length(r), " eigenpairs)")
