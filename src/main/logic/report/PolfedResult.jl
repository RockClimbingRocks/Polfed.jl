# PolfedResult: the value POLFED returns. Holds the post-processed eigenpairs (already
# filtered to the target interval, sorted ascending by true energy) and the structured
# `Report`. It destructures as `(vals, vecs, report)`, matching `LanczosResult` — every
# entrypoint in the package therefore unpacks the same way. `vecs` is ALWAYS column-oriented:
# `vecs[:, i]` is the i-th eigenvector, whatever Krylov layout the run used internally (the
# matrix entrypoints run in RowLayout and return a lazy `transpose` of the rows).

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

It destructures as `(vals, vecs, report)` — the same protocol as [`LanczosResult`](@ref) — so

```julia
vals, vecs, report = polfed(H, x0, howmany, target)
```

works, and matches how `patched_polfed`, `full_polfed` and `lanczos` unpack. The fields are also
available directly (`result.vals`, `result.residuals`, …). `vecs` is column-oriented, so the
`i`-th eigenvector is simply `result.vecs[:, i]`; [`eigenvector`](@ref) does the same as a
non-allocating view. `length(result.vals)` is the count.
Displaying it (`display(result)`, or typing it at the REPL) prints the report. See
[`Report`](@ref) for the display-tuning IO properties.

# Fields
- `vals::V`: eigenvalues (true energies of `H`), sorted ascending, within the target interval.
- `vecs::M`: eigenvectors as **columns** (`dim × n`), independent of the run's Krylov layout.
  A view into the Krylov basis storage (a lazy `transpose` for a row-layout run), so it keeps
  that storage alive; `copy(vecs)` gives an independent `Matrix`.
- `residuals::Vector{Float64}`: per-eigenvalue true residual `‖H·vᵢ − λᵢ·vᵢ‖`, aligned with `vals`.
- `converged::Int`: total converged eigenpairs from Lanczos (before interval filtering; the
  in-interval count is `length(vals)`).
- `layout::L`: the layout of the returned `vecs` — always [`ColumnLayout`](@ref). The Krylov
  layout the run actually used is in the mapping report.
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

# Destructuring: `vals, vecs, report = result`, mirroring `LanczosResult`.
Base.iterate(r::PolfedResult) = (r.vals, Val(:vecs))
Base.iterate(r::PolfedResult, ::Val{:vecs}) = (r.vecs, Val(:report))
Base.iterate(r::PolfedResult, ::Val{:report}) = (r.report, Val(:done))
Base.iterate(::PolfedResult, ::Val{:done}) = nothing
# No `length` method by design: `length(result)` would be ambiguous (3 items, or the
# eigenpair count?). Declaring the size unknown keeps `collect` working without it.
Base.IteratorSize(::Type{<:PolfedResult}) = Base.SizeUnknown()

"""
    eigenvector(r::PolfedResult, i::Integer) -> AbstractVector

The `i`-th eigenvector as a length-`dim` view. Equivalent to `r.vecs[:, i]` — `vecs` is always
column-oriented — but returns a view rather than a copy.

```julia
result = polfed(H, x0, howmany, target)
v1 = eigenvector(result, 1)
```

!!! warning
    From a CPU matrix entrypoint the view is strided (a row of the Krylov storage), and MKLSparse
    3.0, which POLFED loads, multiplies a strided vector by a sparse matrix wrongly, without an
    error (`H * v`, `mul!(y, H, v)`, `v' * H * v`). `collect` it first, or use `r.vecs[:, i]`,
    which copies.
"""
eigenvector(r::PolfedResult, i::Integer) = _result_eigvec(r.layout, r.vecs, i)

"""
    eigenpair(r::PolfedResult, i::Integer) -> (val, vec)

The `i`-th `(eigenvalue, eigenvector)` pair. Convenience over [`eigenvector`](@ref) for loops:

```julia
for i in eachindex(result.vals)
    λ, v = eigenpair(result, i)
end
```
"""
eigenpair(r::PolfedResult, i::Integer) = (r.vals[i], eigenvector(r, i))

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

Base.show(io::IO, r::PolfedResult) = print(io, "PolfedResult(", length(r.vals), " eigenpairs)")
