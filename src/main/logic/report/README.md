# `main/report/` — POLFED result & report

What `polfed(...)` hands back: the post-processed eigenpairs plus a structured, human-readable
summary of the run.

## Return value: `PolfedResult`

`polfed(...)` returns a [`PolfedResult`](PolfedResult.jl). It holds the eigenpairs **already
filtered to the target interval** and **sorted ascending by true energy of `H`**, and carries the
`Report`.

- **Iterable** → `(val, vec)` pairs: `for (λ, v) in result ... end`.
- **Indexable**: `result[i]` → `(val, vec)`; `length(result)` = number of eigenpairs.
- `vec` is always a length-`dim` view, whichever layout the eigenvectors are stored in
  (columns for `ColumnLayout`, rows for `RowLayout` — the matrix entrypoint's row-native path).
- Fields: `vals`, `vecs`, `residuals`, `converged` (total from Lanczos, before interval
  filtering; the in-interval count is `length(result)`), `layout`, `report`.

Post-processing lives in [build_report.jl](build_report.jl): the in-interval subset is the
contiguous `searchsorted` range of the sorted values (so the eigenvector slice is a **view**, no
copy), and a `@warn` — never an error — fires if fewer than `howmany` eigenpairs land inside the
interval.

## The `Report` and how it displays

[`Report`](report.jl) aggregates four sections, rendered through
`Base.show(::IO, ::MIME"text/plain", …)` — so `display(report)`, `show(report)`, and typing it at
the REPL all work; there is no `display_report()` verb.

1. **Mapping** ([MappingReport.jl](MappingReport.jl)) — operator/kernel, rescaling
   `H̃ = (H − center)/spread` + bounds, layout, block size, `opdim`, parallelization.
2. **Spectral Transformation** ([SpectralTransformReport.jl](SpectralTransformReport.jl)) —
   target + strategy, rescaled window `[left, right]` and its true-energy image, filter + order,
   and the headline cost `H·v matvecs = K·blocksize·iters + howmany`.
3. **Convergence** (from the Lanczos `FactorizationReport`) — requested/converged counts, how many
   eigenpairs failed the residual (`tol`) and eigen (`eigentol`) tolerances, worst residual, Krylov
   usage; with the per-check history table on demand.
4. **Timings** — POLFED-level split (`bounds` / `DoS` / `transform-build` / `Lanczos`) plus the
   Lanczos-internal decomposition (`matvec` / `reorthog.` / `tridiag.` / `conv.check`). All
   percentages are of wall time.

### Display tuning — two IO properties, no keyword args

Both are read straight off the IO sink:

| property               | default            | effect                                            |
|------------------------|--------------------|---------------------------------------------------|
| `:color`               | auto (TTY → `true`)| ANSI colors; automatically **off** when redirected to a file |
| `:convergence_details` | `false`            | append the per-check convergence-history table (`PrettyTables`) |

```julia
display(result)                          # colored on a TTY, plain in a file — automatically
open("run.txt","w") do io                # → plain text, no ANSI
    show(io, MIME"text/plain"(), result)
end
show(IOContext(stdout, :convergence_details => true), MIME"text/plain"(), result)
```

### Notes

- **GPU runs omit CPU time** — the Timings `total` line drops the `/ … cpu` term (CPU time is
  misleading on GPU). Detected from the `FactorizationReport`'s GPU basis columns.
- **Block vs single Lanczos** — the Convergence header reads `(block Lanczos)` / `(Lanczos)` from
  the factorization type; the same six stage timers back the decomposition either way.
- The Mapping/Transform sections are plain-data **snapshots** (no operator closures retained), so a
  `Report` is cheap to hold and store.
