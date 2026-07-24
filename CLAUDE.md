# UniPolfed.jl — agent guide

## What POLFED is
**Polynomial Filtering Exact Diagonalization.** Solves `H|ψ⟩ = E|ψ⟩` for a *selected*
part of the spectrum of large (sparse) quantum many-body Hamiltonians. It builds a
polynomial (Chebyshev) filter that **amplifies** eigenvectors near a target energy `λ`
and **suppresses** the rest, mapping the target eigenvalues to *extremal* values so a
Krylov/Lanczos method can reach them. Refs: Sierant et al., PRL 125, 156601 (2020)
[arXiv:2005.09534] (original method); the Polfed.jl article (SciPost / arXiv:2605.10191).

## Pipeline and the bottleneck
`polfed(...)`:
1. **DoS** (KPM): estimate the density of states `ρ(E)` to place the target window / count states.
2. **Spectral transform**: resolve interval bounds `[left,right]` (from DoS mass, or a fixed
   order) → Chebyshev order `K` (bisection to a cutoff; `K ≈ 2.65499·√(1−λ²)/δ`) → build the
   filtered operator `f!_transformed` (Clenshaw evaluation of the filter polynomial of `H`).
3. **Lanczos / block-Lanczos** on `f!_transformed`: the filtered target eigenvalues are now
   extremal and converge fast.
4. Map back to true eigenvalues (Rayleigh quotients under the original `H`) + residuals.

**The bottleneck is the spectral transformation.** Every Lanczos matvec applies the order-`K`
filter = `K` sparse matvecs of `H` (Clenshaw recurrence). Total cost ≈
(#Lanczos matvecs) × `K` × cost(`H·v`); `K` grows as the window narrows or the DoS rises.
This is why `Numerics/Clenshaw` (fast, stable filter evaluation), the value-grouped
`Optimization/` mappings, GPU kernels, and parallelization all exist — they all attack this
one cost.

## Architecture (tiers — see `RESTRUCTURE.md` for the full tree + rationale)
- **Backend/**, **Common/** — infrastructure (GPU dispatch, CPU/GPU `ProcessingUnit`, logging,
  timing). `Backend/` is included at the package top level, NOT a submodule, so the CUDA
  extension's `UniPolfed._set_cuda_backend!` / `UniPolfed.is_gpu_array` hooks resolve.
  **Timing = `StageTimer`** (`common/stage_timer.jl`), which replaced `@addtime!` everywhere: stages
  are addressed **by name** and **self-accumulate** (first `@stage :matvec …` creates the entry, each
  later one adds wall/CPU/hit-count), and `@stage` **returns the expression's value**. Two forms —
  ambient `@stages` opens a task-local scope so any `@stage` in its dynamic extent records into it
  (this is what let `lanczositer!` drop its `walltimes`/`cputimes` parameters), and explicit
  `@stage timer :name expr` records into a given timer. Gotchas: `@stages` wraps the block in a
  `try`, whose **scope hides variables first assigned inside** — declare them before (hence
  `result = nothing` in `lanczos_algorithm`), which is also why the pipelines use the explicit form;
  and `task_local_storage` is **not inherited by spawned tasks**, so annotate outside a
  `Threads.@spawn` region or pass a timer explicitly. Missing stages read as `0`, so report code
  needs no `haskey` guards. (`@addtime!` / `common/add_time.jl` were deleted once nothing called them.)
- **Numerics/** — reusable, POLFED-agnostic spectral primitives: **Clenshaw** (filter/poly
  evaluation), **KPM** (DoS), **Lanczos** (eigensolver). Litmus test: usable/testable without
  constructing any POLFED struct.
- **UniPolfed/** — the algorithm, **directly in `module UniPolfed`** (the old `PolfedCore` submodule was
  eliminated; `src/UniPolfed/` files include straight into `UniPolfed.jl`, so the CUDA ext rewire targets
  `UniPolfed.*`): entrypoint, configs (+Plans where there's real resolution), Mapping, Transform
  (+Filters), Factorization, DensityOfStates adapter, Parallel, Optimization, Report. `Defaults` is
  a submodule of default constants, grown incrementally as components land.
- **Models/** — Hamiltonians (QSun, spin-half chains).

## Build / test / run
- Julia CPU build: `$HOME/julia-wrappers/julia-x86_64` (v1.12.5). GPU build: `$HOME/julia-wrappers/julia-aarch64`.
- Load check: `$HOME/julia-wrappers/julia-x86_64 --project=. -e 'using UniPolfed'`
- Tests: `$HOME/julia-wrappers/julia-x86_64 --project=. -e 'import Pkg; Pkg.test()'`
- Coverage: `Pkg.test(coverage=true)`, then read the `*.cov` files (left column = per-line hit
  count; `-` = non-executable line).
- **GPU code paths cannot run on the x86 host.** CPU paths are covered now; GPU is gated for the
  aarch64/CUDA build (the extension stays dormant until CUDA is loaded).

## Restructure status (branch `restructure/file-layout`)
New `src/` is rebuilt component-by-component from `src-OBSOLET/` (read-only reference — copy
from it, do not move). Old tests in `test-OBSOLET/`. Migration order is dependency-first:
infra → Numerics (Clenshaw → KPM → Lanczos) → Models → POLFED core.
- ✅ Common, Backend, **Numerics/Clenshaw**, **Numerics/KPM** — migrated, tested, 100% line coverage.
- ✅ **Numerics/Lanczos** — migrated, tested (CPU ~full coverage; GPU gated). New API:
  composable `EigTarget` (`NumEigs`/`ToValue`/`Window` × `Lowest`/`Highest`/`BothEnds`),
  `LanczosResult` + `has_converged` (gluing), `ConvergenceCriterion`/`State`, breakdown guard.
  **`convergence/` was rewritten** (see its `README.md`) around ONE invariant: *everything
  returned is a contiguous run inward from the targeted transformed extreme, in which every pair
  passed BOTH the Ritz and the eigen tolerance* — on `done`, on `maxiter`, and on breakdown
  alike, so **gaps are not expressible**. The key distinction is two coordinate spaces: `p` (the
  filtered operator's Ritz value — free, and the ONLY space where Lanczos guarantees contiguity)
  vs `E` (whatever `mapvals` produces — needs eigenvector reconstruction, may only *select*).
  The old `Window` path argued completeness in `E`, with a distance-blind `_nearest_converged`
  bracket, no Ritz gate, and a holed set returned verbatim at the budget ceiling — that combination
  produced the reported gapped spectra. Now: `EigTarget.jl` is **types only**; `end_spec.jl` holds a
  **five-method interface** (`pool_indices` / `cheap_gate` / `needs_mapped` / `is_satisfied` /
  `select`) that is the entire "what does convergence mean" surface; `converged_run.jl` holds the
  single walker (`converged_prefix` — the only producer of indices); `check_convergence.jl` is
  **one orchestrator** (the two diverging ones collapsed). `ends(target)` reduces
  `Lowest`/`Highest`/`BothEnds` to `(spec, islow)` pairs, so they are not separate code paths.
  Adding a spec = one section in `end_spec.jl`. **`select`** decides what leaves the module: it drops
  the `ToValue` guard, and for a `Window` returns the in-window pairs **plus one bracket per side**
  (resolving the old `TEMPORARY — bracket-return policy` comment). **What proves completeness is the
  cheap gate, NOT the `E` straddle**: every in-window pair has `p ≥ edge_value` (the window is inside
  the passband, where `|P|` IS monotone in `|E − centre|`), and the run is a `p`-prefix, so once it
  passes `edge_value` it contains the whole window. **The brackets are NOT certified spectral
  neighbours** — they sit *outside* the passband, where the Chebyshev tail oscillates and `p`-order
  inverts vs `E`-order. Measured on the L=8 QREM fixture: `E=−0.9065` has `P=0.127` while the true
  neighbour `E=−0.5516` has `P=0.112` (passband edge 0.185), so the run reaches the distant pair
  first. Monotone non-leaking filters (`EdgeFilter`) DO return the true neighbour and the edge tests
  assert it. Certifying brackets generally needs a DoS-derived margin (widen the window by a few mean
  level spacings, take the outermost of the widened complete set) — **not implemented**.
  `state.complete` is the criterion's
  own completeness verdict (`_window_interval_complete` in `build_report.jl` was deleted rather
  than kept duplicating it), and `state.residual` is now always the WORST residual in the pool (the
  window path used to store the *best*, inverting the signal the check scheduler reads).
  `decide_end(...; force=true)` skips the cheap gate on finalization — without it the partial run
  is short-circuited away and an empty result is returned.
- ✅ **Models** (+ QSun) — migrated, tested (ground-truth: su(2) algebra/Casimir, kron
  cross-check, U(1) sector tiling). `QSun/` moved under `Models/`; the nested `SpinOperators`
  module was flattened into plain `include`s.
- 🚧 **POLFED core** (in progress). Done: **DensityOfStates** (adapter over Numerics/KPM; no Plan),
  **Factorization** (config only, no Plan; uses `EigTarget`, `eig_target=nothing`→`HighestEigs(howmany)`),
  **Parallel** (4 strategies incl. new **dormant** `MulRowsParallel`; real worker spawn gated behind
  `POLFED_TEST_DISTRIBUTED`), **Mapping** (first slice with a real parametric Plan: estimates
  `Emin`/`Emax` via the enhanced `lanczos_extrema` — `BothEnds(1,1)` + power-iteration warm-up +
  `maxdim=100` + `return_result` — then builds the `f!_rescaled=(H-b)/a` closure),
  **Common/Utils** (`bisection` + `chebyshev_integration` — the weighted ∫[ΣcₖTₖ]/√(1-x²) that
  replaces QuadGK for the DoS state count; exact via the `x=cosθ` trig identity),
  **Transform (+Filters)** (`AbstractFilter` seam + `DeltaFilter`; `TransformConfig` is **internal**,
  no Plan; `resolve_target`/`resolve_bounds`/`resolve_order` → `validate_filter` → `define_transformation`
  → `build_transform` returns a NamedTuple). **`expected_krylov_dim` + coefficients + analytical order
  are METHODS on the filter** (delta constants 2.65499, 20.427/1.696 live there, not Defaults). The
  side-lobe check is a **new diagnostic** (warns when the filter leaks above cutoff outside the window).
  **DoS must run on the RESCALED operator** — moments are in rescaled `[-1,1]` coords (KPM blows up
  outside `[-1,1]`); `build_transform` and `resolve_bounds`/`resolve_target` assume this. **CUDA ext
  rewired** (dropped `PolfedCore.*`; Optimization-dependent kernels stubbed behind comments).
- ✅ **Optimization** (value-grouped row kernels) + **entrypoints** — migrated, tested e2e vs exact
  diag (QREM). Two `polfed(...)` methods: the **operator** entrypoint `polfed(f!, x0, …)` (column-native,
  a user's `f!`/`x0` left untouched) and a **matrix** entrypoint `polfed(H, …)` that owns `H`, so it
  probes `Emin`/`Emax`, builds the fast **row-layout** kernels, copies `x0` to row form
  (`permutedims`, never a lazy transpose), and delegates with `layout=RowLayout()` + `NoParallel`.
  `optimize_mapping` (in `MappingConfig`) = `false` | `:auto`/`true` | kernel `Symbol` | `(kernel,G,W)`
  NamedTuple; `false`→generic `mul!` row op. **Only the transform `f!_rescaled` uses the fast Val(bs)
  kernel** (the bottleneck); the **map-back `mapvals` is always a generic `mul!` row op** (applied to a
  strided `1×D` eigenvector view — the SIMD `:shuf` kernel can't even load there). The **DoS runs on a
  generic COLUMN operator** in the matrix path (`build_optimized_mapping`'s row *block* op can't take
  KPM's per-column probes; DoS is one-time, not the bottleneck). Function entrypoint gained internal
  `layout`/`dos_operator` kwargs (column defaults preserve the public API). Row-native Lanczos is
  Stage A (β-transpose fix, `[[lanczos-row-layout-beta-transpose]]`).
- ✅ **Report** — migrated, tested. `polfed(...)` now returns a **`PolfedResult`** (`src/main/report/`):
  post-processed `vals`/`vecs`/`residuals` **filtered to the target interval** (contiguous
  `searchsorted` slice of the sorted vals ⇒ eigenvector subset is a *view*, no copy) and **sorted wrt
  H** (not the transformed `P(H)`); **iterable as `(val, vec)` pairs** (a length-`dim` view per
  `Layout` — column for `ColumnLayout`, row for `RowLayout`), indexable, `length`; `@warn` (never
  throw) if fewer than `howmany` land in-interval; carries a **`Report`**. The `Report` renders via
  **`Base.show(::IO, ::MIME"text/plain", …)`** (so `display(report)` / typing it works — **no
  `display_report()` verb**), tuned by two **IO properties** read off the sink: `:color` (auto →
  colored TTY, plain when redirected to a file) and `:convergence_details` (default off; appends the
  `PrettyTables` per-check history table). Four sections: **Mapping** (`MappingReport`:
  kernel/`optimize_mapping`, rescaling `center`=`b` / `spread`=`a` + `Emin`/`Emax`, layout, block size,
  opdim, parallel), **Spectral Transformation** (`SpectralTransformReport`: target+strategy, rescaled
  **and** true-energy window, filter/order+safety, headline cost `H·v = K·bs·iters + howmany`),
  **Convergence** (from the Lanczos `FactorizationReport`: requested/converged, `failed residual tol`
  & `failed eigen tol` counts, worst residual, krylov usage), **Timings** (POLFED-level
  `bounds`/`DoS`/`transform-build`/`Lanczos` split, plus the Lanczos-internal
  `matvec`/`reorthog.`/`tridiag.`/`conv.check`). **Both breakdowns are `StageTimer`s read by name** —
  `Report.timer` (pipeline stages `:bounds`/`:dos`/`:transform_build`/`:lanczos`) and
  `FactorizationReport.timer` (Lanczos stages) — and the renderer walks whatever stages were recorded,
  in execution order, so a pipeline can add one without editing `render_timings` (`_stage_label` maps
  only the few symbols whose display differs). Migrating off `@addtime!` also **fixed a long-standing
  hole**: the old code allocated 4 slots but only ever filled `Lanczos`, so every report read
  `bounds 0% · DoS 0% · transform-build 0%`; DoS is really ~13% on an L=8 QREM window run.
  **CPU time is suppressed on GPU runs** (misleading); Convergence header reads `(block
  Lanczos)`/`(Lanczos)`. All 5 `note to myself` items in `main.jl` are now resolved.
- ⬜ Stage-2 fused Clenshaw; delete `src-OBSOLET/`+`test-OBSOLET/`+`src/UniPolfed--OBSOLET/` (final).

## Conventions (details: `RESTRUCTURE.md` §4, §6)
- Folders lowercase, no exceptions for acronyms (`cpu/`, `gpu/`, `kpm/`). Files: PascalCase for a
  type/module, snake_case for behavior/actions.
- External packages `using`-ed inside each module that needs them; internal names via relative
  import. **Import dot gotcha:** sibling submodule = `import ..Mod: name`; a name at the package
  top level (e.g. `CPU`/`GPU`) from a deeper submodule = `import ...CPU` (dots + name).
  `import ...: name` (bare dots + colon) is a **parse error**.
- **Config→Plan**: user-facing `XConfig` (may hold `nothing`/auto fields) → `build_X_plan(...)`
  → fully concrete `XPlan`. **Not every config needs a Plan** — apply it only where there is real
  resolution to do (Mapping estimates `Emin`/`Emax`; Transform resolves bounds/order). `DoSConfig`
  and `FactorizationConfig` resolve nothing, so they are config-only (no Plan).

## Code standards
- Thorough docstrings on every function/type/method (Arguments / Returns / Notes; cross-link
  with `[`name`](@ref)`).
- Concrete struct field types; **parametrize callbacks** (`struct S{F<:Function}`). Avoid abstract
  `Function`/`Integer`/`Real` fields — they box and force dynamic dispatch. Verify with
  `@code_warntype`, lock with `@inferred` regression tests.
- Each migrated component gets a `README.md`.

## Testing philosophy
- Assert against **mathematical ground truth, not snapshots**: Clenshaw → `Tₙ(cosθ)=cos nθ`;
  KPM → known DoS / `Tr Tₖ(H)` (use `H = c·I` for a zero-variance deterministic check);
  Lanczos → known eigenvalues of a constructed matrix.
- `test/` mirrors `src/`; one `*_tests.jl` per source file; `runtests.jl` includes them bottom-up.
- Cover element types `Float32`/`Float64`/`ComplexF32`/`ComplexF64` with type-appropriate
  tolerances; add `@inferred` guards for type stability.
- **Don't test config/constant defaults** (`DoSConfig().N == Defaults.N`) — it only duplicates the
  source and gives no behavioral guarantee. Test behavior, keyword overrides, coercion, and logic.

## Gotchas
- **Julia ≥ 1.12 `threadid()` trap:** never index per-thread buffers by `Threads.threadid()`
  (tasks migrate; `threadid()` is not bounded by the thread count → `BoundsError`, hit by a real
  user). Use **chunk-based** parallelism: split the work, each task owns its own accumulator,
  reduce the partials. `fetch(::Task)` returns `Any` — annotate the fetched type to stay
  type-stable.
- `ProcessingUnit` GPU fields are abstractly typed by design (called O(1) at setup, never in a
  hot loop).
- **Distributed worker bootstrap:** `@everywhere w using UniPolfed` inside a function is a precompile
  error (`using` is toplevel-only). Load POLFED on workers with `remotecall_eval(Main, w, :(using
  UniPolfed))` (takes the `using` as quoted data) — this replaced the dropped `main_module_file` include.

## Planned functionality (intent — not all built yet)
**Lanczos redesign (✅ built):** composable `EigTarget` — end-specs `NumEigs(n)` / `ToValue(v)`
composed into `Lowest` / `Highest` / `BothEnds(low, high)` (aliases `LowestEigs(n)` etc.) —
replacing `which::Symbol`; `ConvergenceInfo` god-object split into immutable `ConvergenceCriterion`
+ mutable `ConvergenceState`; `LanczosResult` carries per-eigenvalue residuals;
`has_converged(res, λ; atol, restol)`; `SymTridiagonal`/`Hermitian` eigensolver in the checks.
- **Both ends of the spectrum** (`BothEnds(n_low, n_high)`): a single Krylov space converges to
  *both* extremes simultaneously (e.g. 200 lowest + 200 highest in one factorization). A Lanczos
  *selection* feature — distinct from double-window filtering.
- **Spectrum gluing:** `ToValue(v)` converges *all* eigenpairs out to a window boundary, and
  `has_converged(λ)` queries whether a boundary eigenvalue already converged in a window. Works
  because the *transformed* problem makes boundary eigenvalues extremal; plain interior Lanczos
  converges slowly, so these queries belong on the transformed operator.

**Filters (`src/main/logic/filter/`, `AbstractFilter` seam, per-filter folders):**
- ✅ **Delta filters** — `DeltaUpFilter`/`DeltaDownFilter` (single Chebyshev peak → ±1;
  `2.65499` order constant is delta-specific) + `DoubleDeltaFilter` (two peaks, `:both`).
- ✅ **EdgeFilter** (`filters/edge/`, delta-style split: edge.jl + coefficients/build/validate.jl)
  — spectrum-edge windows: a MONOTONE
  monomial power (`:Taylor`/Horner), `−((1−x)/2)ᵖ` (`:lowest`) / `((1+x)/2)ᵖ` (`:highest`) /
  `xᵖ, p odd` (`:both` — both edges in ONE BothEnds Krylov space). Fixed order (default 3), no
  bisection/passband, cannot leak. Its `expected_krylov_dim` constants (25/4.0/×1.30) are INITIAL
  estimates, not a fit. Edge windows exposed three boundary bugs, all fixed: the `Window`
  criterion waives its straddle on a `±Inf` bound (`window_eig_target` encodes a clamped
  ±1 bound as ±Inf), the result trim keeps everything past a ±1 bound
  (`_interval_energy_window` → ±Inf — Emin/Emax sit ON the extremal eigenvalues, no margin),
  and `_check_rescaled` SNAPS a target within 1e-8 outside `[-1,1]` (Ritz estimates converge
  from inside, so targeting the exact ground-state energy lands at `−(1+ε)`).
  `order_safety_factor` is now an interface accessor (generic fallback `1.0`; the report uses it).
  `:both` via `polfed(...)` still needs the multi-window pipeline (`MULTI_WINDOW_TODO.md`).
  **`window_eig_target(filter, interval)` — units.** It now takes NO `MappingPlan` and states the
  window in **rescaled** coordinates (where `interval` and the filter already live), so callers
  MUST pass a rescaled `mapvals` — a `Window` is tested against whatever `mapvals` produces. Build
  it with `_make_rescaled(mp.f!, mp.a, mp.b)`, **not** `mp.f!_rescaled`: in the matrix entrypoint
  the latter is the fast value-grouped block kernel, which cannot be applied to the strided `1×D`
  eigenvector views the map-back uses. The pipeline passes **`eigentol / mp.a`** to Lanczos, because
  the rescaled residual is `‖H·v − λ·v‖ / a` — so the acceptance test still means
  `‖H·v − λ·v‖ < eigentol`, exactly as for a count target. `build_polfed_result(...;
  vals_rescaled=true)` un-rescales BOTH values and residuals once, at the result boundary; the
  `FactorizationReport` keeps Lanczos' own rescaled numbers (self-consistent, a factor `mp.a` smaller).
  The third `Window` field is the **edge value** = the filter at the finite window bounds, less
  extremal one (`min` at a `Highest` end, `max` at a `Lowest`) — the criterion's cheap gate. It is
  the filter at the *window* edge, NOT at the passband cutoff: the passband is `osf`-widened and
  only guaranteed to contain the window, so gating on it costs many needless iterations.
- ✅ **Smart resolution** (`edge_side.jl` + `resolve_filter.jl`, replaced `pick_filter`): geometry
  picks the **CHAIN** (`edge_side`: window touching ±1 → `EDGE_FILTER_CHAIN = (EdgeFilter,)`;
  else `INTERIOR_FILTER_CHAIN = (DeltaUpFilter, DeltaDownFilter)`); each type walks its OWN
  parameter ladder through the generic `_first_valid(F, request)` loop, first valid wins, an
  exhausted type hands to the next, an exhausted chain errors. **Resolution interface** (per
  filter, in its folder, stubs in filter.jl): `_cutoff_values(F)` (the ladder — DeltaUp
  `0.17…0.35` in `DeltaUpFilterDefaults.cutoff_ladder`, DeltaDown negated, Edge its fixed
  order) and `_build_candidate(F, request, value)`. Fitness = `validate_filter(candidate)`
  alone — each filter folds its full contract into its own validation (deltas: passband ⊇ the
  requested `filter.interval` + strictly interior; no separate `_useful`/`_covers` layer).
  Adding a filter = implement these + append the type to a chain. Near-edge interior windows (`[-0.95, -0.9]`)
  deliberately stay delta — rung 0.17 leaks, a higher rung passes. Pair method
  `resolve_filter([iv1, iv2])`: DoubleDelta over its ladder → `(double,)`, else independent
  `(f1, f2)`; an edge member always resolves separately. Passband may exceed the request — the
  pipeline trims to the REQUESTED interval. Playground notebook:
  `notebooks/test/Filters/resolve_filter.ipynb`. Patched-pipeline helpers (`intervalize_interval`
  → per-patch `filterize_intervals`) consume the single-interval method.
Planned:
- **Custom user filters** — user supplies the filter shape/coefficients via `AbstractFilter`.
Better filters also lower the required order `K`, directly reducing the bottleneck cost.
