# [Reports, Logging, and Defaults](@id docs_report_defaults)

Reports are part of every solver result; there is no `produce_report` switch.
[`polfed`](@ref UniPolfed.polfed) returns a [`PolfedResult`](@ref
UniPolfed.PolfedResult), whose `report` field contains mapping,
spectral-transformation, convergence, and timing diagnostics.

The patched and full-spectrum entry points return `(vals, vecs, report)` instead.
Their third value is a [`PatchedReport`](@ref UniPolfed.PatchedReport), which
aggregates the factorization report from every patch.

## Result and Report Types

### [`PolfedResult`](@id UniPolfed.PolfedResult)

The standard solver returns a `PolfedResult`. Its `vals`, `vecs`, `residuals`,
`converged`, and `layout` fields describe the selected eigenpairs, while
`report` contains the structured run diagnostics.

### [`Report`](@id UniPolfed.Report)

A standard report combines a [`MappingReport`](@ref UniPolfed.MappingReport),
a [`SpectralTransformReport`](@ref UniPolfed.SpectralTransformReport), the
Lanczos [`FactorizationReport`](@ref
UniPolfed.Numerics.Lanczos.FactorizationReport), and stage timings.

### [`MappingReport`](@id UniPolfed.MappingReport)

Snapshot of the resolved operator, rescaling, layout, processing unit, and
parallelization strategy.

### [`SpectralTransformReport`](@id UniPolfed.SpectralTransformReport)

Snapshot of the target interval, selected filter, polynomial order, and
estimated transformation cost.

### [`PatchReport`](@id UniPolfed.PatchReport)

Diagnostics for one interval solved by a patched or full-spectrum run.

### [`PatchedReport`](@id UniPolfed.PatchedReport)

Aggregate decomposition, convergence, mapping, and timing diagnostics across
all patches.

### [`FactorizationReport`](@id UniPolfed.Numerics.Lanczos.FactorizationReport)

Lanczos convergence history, residuals, Krylov-budget usage, and internal
timings.

Reports use Julia's normal text display:

```julia
result = polfed(mat, x0, howmany, target)
display(result)         # result header followed by the complete report
display(result.report)  # report only
```

The `:color` and `:convergence_details` IO properties control rendering. For
example, this writes a plain-text report and includes the convergence history:

```julia
open("polfed-report.txt", "w") do io
    report_io = IOContext(
        io,
        :color => false,
        :convergence_details => true,
    )
    show(report_io, MIME"text/plain"(), result.report)
end
```

## [Logging Levels and Helpers](@id docs_logging_defaults)

```@docs
UniPolfed.Common.POLFED_SILENT_LEVEL
UniPolfed.Common.POLFED_WARN_LEVEL
UniPolfed.Common.POLFED_INFO_LEVEL
UniPolfed.Common.POLFED_DEBUG_LEVEL
UniPolfed.Common.POLFED_DEEP_DEBUG_LEVEL
UniPolfed.Common.verbosity
UniPolfed.Common.should_log
UniPolfed.Common.polfed_log
```

## Defaults

Defaults now live in the per-entry-point registry `UniPolfed.Defaults`:

- `Defaults.polfed` contains the standard solver's DoS, mapping, and
  factorization defaults. The patched solver also uses these values for its
  public `mc`, `dosc`, `tol`, `eigentol`, and `overestimate_iters` keywords.
- `Defaults.patched_polfed` contains the patched decomposition and its internal
  factorization defaults.
- `Defaults.full_polfed` contains the corresponding full-spectrum defaults.

Users normally override these through [`DoSConfig`](@ref UniPolfed.DoSConfig),
[`MappingConfig`](@ref UniPolfed.MappingConfig), or solver keyword arguments
instead of modifying the constants.

### Standard `polfed` Defaults

- DoS: `N = 250`, `R = 300`, `kernel = :Jackson`.
- Mapping: `optimize_mapping = false`, `maxdos_edge_margin = 0.01`.
- Lanczos: `rot = FullRO()`, `basistype = MatrixBasis`,
  `tol = 1e-14`, `eigentol = 1e-9`, and
  `overestimate_iters = 1.25`.
- Convergence checks start at `check_start_frac = 1.0`; the
  `conv_checking(expected_dim)` callback then adapts the interval between
  checks to the current residual.

### Patched and Full-Spectrum Decomposition Defaults

Both registries use `howmany_pad = 3` and `max_glue_retries = 3`.
`retry_budget_growth` is `2.0` for `patched_polfed` and `1.5` for
`full_polfed`.
