# `filter/` — per-filter layout

Filters are organized **per filter**, not per concern: everything about one filter (its type,
coefficients, order/build, validation) lives in its own folder under `filter_types/`, so *adding a
filter = drop a folder + implement the interface* — no touching shared concern-folders.

The slice has three parts, and the folder names say which is which: `filter.jl` is the seam,
`resolution/` is how a filter gets **picked**, `filter_types/` is which filters **exist**.

## Layout

```
filter/
  filter.jl              # AbstractFilter + interface stubs + _scalar_filter (shared) + includes
  resolution/            # HOW a filter is chosen — shared, filter-agnostic
    order_bisection.jl   #   bisect_order — bracket-double + bisect, PREDICATE-driven
    edge_side.jl         #   window geometry: edge_side — the family decision
    resolve_filter.jl    #   geometry + cutoff-LADDER orchestrator (single & pair) + the chains
    resolve_double_strategy.jl  # two windows: cost-aware joint(double)-vs-separate decision
  filter_types/          # WHICH filters exist — one self-contained folder each
    filter_types.jl      # includes each filter folder
    delta_up/            # reference filter — fully documented
      delta_up.jl        #   the struct + type-tag methods + includes
      coefficients.jl    #   _coefficients_nonnormalized + coefficients (normalized to +1)
      build.jl           #   analytical_order + resolve_order! + resolve_bounds_from_order + build_filter
      validate.jl        #   validate_filter (leakage toward the exposed end)
    delta_down/          # lowest-end twin — same shape, sign flipped (shows how a filter slots in)
      delta_down.jl  coefficients.jl  build.jl  validate.jl
    double_delta/        # two windows in one polynomial (peaks +1 / −1), :both exposed ends
      double_delta.jl  coefficients.jl  build.jl  validate.jl
      eig_target.jl    #   the :both Lanczos targets — window 1 is the +1 peak ⇒ the HIGH end
    edge/                # spectrum-edge windows — a MONOTONE monomial power ("basically H³"):
                         #   −((1−x)/2)ᵖ / ((1+x)/2)ᵖ / xᵖ for :lowest/:highest/:both (:Taylor);
                         #   fixed order (default 3), no bisection/passband, cannot leak
      edge.jl  coefficients.jl  build.jl  validate.jl
```

The numerical leakage scan itself lives in `common/utils/validate_leakage.jl`
(`validate_no_leakage_in_interval_numerically`) — a reusable primitive both filters call.

## What's shared vs per-filter

**Shared** (genuinely cross-filter, expressed through the interface):
- `AbstractFilter` + the interface function **stubs** (`coefficients`, `exposed_end`,
  `expected_krylov_dim`, `analytical_order`, `polynomial_type`, `polynomial_order`, `build_filter`,
  `resolve_order!`, `resolve_bounds_from_order`, `validate_filter`) — declared once so filters add
  methods without an `import`.
- `_scalar_filter(filter)` — the "evaluate `P(x)`" primitive (from
  `polynomial_type`/`coefficients`/`polynomial_order`).
- `_fitted_krylov_dim(howmany, blocksize, η, A, α, xc)` — the delta family's Krylov-budget fit
  `A·x^α·(1+x/xc)^(1-α)` **iterations** at `x = howmany/blocksize`, times `blocksize·η`. Shared
  because the budget depends on the count and the block, not on the filter's shape; each delta
  filter passes its own constants from `Defaults.filters.<name>.krylov_fit_*`. `EdgeFilter` does
  not use it (its estimate is a separate, untuned linear guess).
- `bisect_order(analytical_K, suppressed)` — the order mechanic. It takes the filter's own
  **`suppressed(order)::Bool` predicate**, so it's agnostic to how many endpoints a filter tests
  (one for a delta, several for a double/edge filter).
- `edge_side` (window geometry: which spectrum edge, if any, the request sits on) and the
  `resolve_filter` orchestrator: geometry picks the **CHAIN** of filter types
  (`INTERIOR_FILTER_CHAIN = (DeltaUpFilter, DeltaDownFilter)` for interior windows,
  `EDGE_FILTER_CHAIN = (EdgeFilter,)` for edge windows), and each type walks its **own parameter
  ladder** through the one generic `_first_valid(F, request)` loop. The **resolution interface**
  a chain member implements (in its own folder): `_cutoff_values(F)` — its ladder, tried in
  order, first valid wins (DeltaUp cutoffs `0.17…0.35`, DeltaDown the negated ladder, Edge its
  fixed monomial order); and `_build_candidate(F, request, value)` — one fresh filter per rung.
  Fitness is judged by `validate_filter(candidate)` alone: a filter folds its full contract into
  its own validation (the deltas check passband ⊇ the requested `filter.interval` and strictly
  interior — the filter carries its request, so no extra argument is needed). An
  exhausted type hands over to the next in the chain; an exhausted chain errors. Near-edge
  interior windows like `[-0.95, -0.9]` deliberately stay delta cases — rung 0.17 leaks and a
  higher rung passes. The pair method `resolve_filter([iv1, iv2])` tries ONE `DoubleDeltaFilter`
  over its ladder (returns a 1-tuple) and falls back to two independent filters (2-tuple; an
  edge member always resolves separately). A returned passband may be WIDER than the request —
  the pipeline trims to the requested interval.

**Per-filter** (`filter_types/<name>/`): the mutable struct and every method that reads the
filter's own parameters. Its *default values* are not here — they live in the central registry
as `Defaults.filters.<name>` (`src/main/defaults/filters/`), alongside `Defaults.filters.edge_tol`
and the two resolution chains.

## Design decisions

- **No per-filter module** — plain includes into `module Polfed`; only the *defaults* get a
  small module, and those now live centrally as `Defaults.filters.<name>`, which keeps
  `cutoff`/`order`/… from colliding between filters and puts every tunable knob in one registry.
- **Two intervals on the filter**: `interval` (from the DoS — the requested window; sets `target`,
  drives order resolution, and is what results are filtered to) and `passband` (from
  `resolve_bounds_from_order` — the *effective* exposed window `[left, right]` where `P` sits at/above
  `cutoff`; used by `validate_filter` and as diagnostics). The passband is resolved **per side**, so
  it is generally **asymmetric** — `P` is symmetric about `0`, not about a non-central target, so a
  symmetric `target ± δ` would leave the slower-dropping edge above cutoff and `validate_filter` would
  spuriously flag it. Resolving each side to its own cutoff crossing puts both edges exactly on
  `cutoff`, so validation needs **no** overshoot tolerance (`leak_rtol` defaults to `0`).
- **Mutable struct**, filled in two phases by `build_filter` (resolve `order`, then `passband`).
- **Per-filter peak sign, not a shared `peak_value`** — `DeltaUpFilter` normalizes to `+1`;
  `DeltaDownFilter` normalizes to `-1` and carries a **negative `cutoff`** (`-0.17`) so the same
  "oriented" tests read directly (`P ≤ cutoff` is a lowest-end leak). `_window_bounds` is likewise
  dropped — delta filters are symmetric (`target ± δ`); reintroduce a placement helper when an
  edge/asymmetric filter needs it.
- **`resolve_order!` is per-filter** (dispatched on the type); only the bracket-and-bisect
  *mechanic* (`bisect_order`) is shared.
- **`validate_filter` needs no fudge factor.** With both passband edges resolved onto `cutoff`
  (above), the scan just backs off `ε` from each edge and flags any point that reaches `cutoff`
  outside. `leak_rtol` (default `0`) remains only as a knob to treat `cutoff` as softer if desired.

`double_delta` is not in `AVAILABLE_FILTERS` (no single-window chain contains it) — it is reached
through `resolve_double_strategy`, which `double_polfed` calls with two windows. That function owns
the cost model: it resolves both windows singly for `K₁`/`K₂` (free — scalar bisection, no
matvecs), takes the joint filter when `min(K₁,K₂)/max(K₁,K₂) ≥ Defaults.double_polfed.double_ratio`,
and falls back to two runs whenever the joint filter cannot be built (edge-touching window, or an
exhausted ladder). See `Defaults.double_polfed` for why the threshold is the Krylov overhead − 1.
