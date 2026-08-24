# Development roadmap

This page records remaining work that is supported by the current code, rather
than the historical file-layout migration.

## Multi-window POLFED

The filter layer can already resolve two intervals and can build a
`DoubleDeltaFilter` or a two-ended `EdgeFilter`. Public `polfed` execution is
still single-window:

1. Add an entrypoint and pipeline that accept two target intervals.
2. Teach `window_eig_target` how to represent both windows in one `BothEnds`
   factorization.
3. Generalize result trimming from one contiguous interval to the union of two
   intervals while preserving the matching eigenvector layout.
4. Generalize `SpectralTransformReport` to render both targets, passbands, and
   physical-energy windows.
5. Add end-to-end exact-diagonalization tests for two interior windows, both
   spectral edges, and mixed edge/interior placement.

Until those steps land, two-ended filters are low-level building blocks and are
not accepted by the public window entrypoint.

## Filter work

  filter-owned defaults module, coefficient normalization, leakage validation,
  resolution-chain registration, and end-to-end tests. The previous prototype
  dispatched construction through `DeltaUpFilter`, mixed up normalization and
  polynomial-order arguments, and was never reachable from the package.
- Add deterministic regression coverage for filter-order selection across
  target position, interval width, and cutoff.
- Keep the distinction between the requested interval and the usually wider
  validated passband explicit in APIs and reports.

## Mapping and performance

- Benchmark `G`, `W`, and block-size combinations and show the generic `mul!`
  baseline in every mapping report.
- Extend specialized mappings for common structured operators, including XXZ,
  J1-J2, U(1)-conserving spin exchange, and dimension-specific Anderson models.
- Investigate fused Clenshaw kernels and reduce row/column boundary transposes.
- Expose a coherent threading policy for Julia tasks, BLAS, and vendor BLAS
  implementations without overriding requirements of a custom mapping.
- Treat historical speedup numbers as machine-specific observations. Use
  `benchmark_mappings` on the target system before selecting a kernel.

## Lanczos investigations

- Continue benchmarking structured projected-matrix diagonalization. Existing
  dense-versus-banded experiments were incomplete and do not justify a default
  change yet.
- Clarify basis/layout combinations in the public API. `MatrixBasis` supports
  row and column layouts; unsupported combinations should fail at validation
  boundaries with actionable errors.
- Preserve exact-spectrum and direct residual checks for every convergence mode,
  including mixed `BothEnds` specifications.
