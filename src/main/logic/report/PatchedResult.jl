# PatchedResult: the value `patched_polfed` and `full_polfed` return. Deliberately the same
# shape as `PolfedResult` — it destructures as `(vals, vecs, report)`, so every entrypoint in
# the package unpacks identically. It carries the aggregate `PatchedReport` rather than a
# single-run `Report`, because a patched run is N+2 separate filtered Lanczos solves.
#
# `vecs` has two forms, both handled by `eigenvector`:
#   - count / full mode  → one dense matrix, ALWAYS column-oriented (`vecs[:, i]`)
#   - window mode        → a vector of independently owned eigenvectors, so completed patch
#                          matrices can be released as the run proceeds

"""
    PatchedResult

Result of [`patched_polfed`](@ref) and [`full_polfed`](@ref): the eigenpairs collected from
every patch (deduplicated at the seams, sorted ascending by true energy of `H`) plus the
aggregate [`PatchedReport`](@ref).

It destructures as `(vals, vecs, report)` — the same protocol as [`PolfedResult`](@ref) and
[`LanczosResult`](@ref) — so

```julia
vals, vecs, report = patched_polfed(H, x0, howmany, howmany_per_patch, target)
vals, vecs, report = full_polfed(H, x0, howmany_per_patch)
```

The fields are also available directly. Use [`eigenvector`](@ref) for the `i`-th eigenvector as
a plain length-`dim` vector — it resolves both storage forms — and `length(result.vals)` for the
count. Displaying it (`display(result)`, or typing it at the REPL) prints the aggregate report.

# Fields
- `vals::V`: eigenvalues (true energies of `H`), sorted ascending.
- `vecs::M`: eigenvectors. In count/full mode a dense `dim × n` matrix whose columns are the
  eigenvectors (`vecs[:, i]`), independent of the run's Krylov layout; in window mode a vector of
  independently owned vectors.
- `report::PatchedReport`: the per-run decomposition, convergence and timing summary.
- `layout::L`: the Krylov [`Layout`](@ref) a matrix `vecs` is stored in.

# Notes
There is no `residuals` field: a patched run has no single convergence history, so per-eigenpair
residuals live in the per-patch `FactorizationReport`s inside `report`. See
[`PatchedReport`](@ref).
"""
struct PatchedResult{V<:AbstractVector, M, L<:Layout}
    vals::V
    vecs::M
    report::PatchedReport
    layout::L
end

# Destructuring: `vals, vecs, report = result`, mirroring `PolfedResult`/`LanczosResult`.
Base.iterate(r::PatchedResult) = (r.vals, Val(:vecs))
Base.iterate(r::PatchedResult, ::Val{:vecs}) = (r.vecs, Val(:report))
Base.iterate(r::PatchedResult, ::Val{:report}) = (r.report, Val(:done))
Base.iterate(::PatchedResult, ::Val{:done}) = nothing
# No `length` method by design: `length(result)` would be ambiguous (3 items, or the
# eigenpair count?). Declaring the size unknown keeps `collect` working without it.
Base.IteratorSize(::Type{<:PatchedResult}) = Base.SizeUnknown()

"""
    _patched_eigvec(layout, vecs, i) -> AbstractVector

The `i`-th eigenvector, for either storage form: a layout-resolved view into a dense matrix, or
the `i`-th element of a vector-of-vectors (window mode).
"""
_patched_eigvec(layout::Layout, vecs::AbstractMatrix, i::Int) = _result_eigvec(layout, vecs, i)
_patched_eigvec(::Layout, vecs::AbstractVector, i::Int) = vecs[i]

"""
    eigenvector(r::PatchedResult, i::Integer) -> AbstractVector

The `i`-th eigenvector as a length-`dim` vector, for either storage form. In count/full mode
this is `r.vecs[:, i]`; in **window** mode `vecs` is a vector of vectors rather than a matrix, and
this accessor hides that difference.
"""
eigenvector(r::PatchedResult, i::Integer) = _patched_eigvec(r.layout, r.vecs, Int(i))

"""
    eigenpair(r::PatchedResult, i::Integer) -> (val, vec)

The `i`-th `(eigenvalue, eigenvector)` pair.
"""
eigenpair(r::PatchedResult, i::Integer) = (r.vals[i], eigenvector(r, i))

"""
    show(io::IO, ::MIME"text/plain", r::PatchedResult)

Print the `Patched POLFED result` header followed by the aggregate [`PatchedReport`](@ref).
Honors the report's IO properties (`:color`, `:convergence_details`).
"""
function Base.show(io::IO, ::MIME"text/plain", r::PatchedResult)
    f = Formatter(get(io, :color, false))
    println(io, blue(f, "Patched POLFED result"))
    println(io, blue(f, "═"^60))
    println(io, "eigenpairs returned: ", length(r.vals))
    println(io)
    show(io, MIME"text/plain"(), r.report)
    return nothing
end
