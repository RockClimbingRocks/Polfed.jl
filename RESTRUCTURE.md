# UniPolfed.jl — Source Restructure

This document describes the target organization of `src/`, the conventions it
follows, the rationale behind the key decisions, and the migration process we
are using to get there. It is a working document for the `restructure/file-layout`
branch and can be folded into permanent docs (or deleted) once the migration is
complete.

---

## 1. Why restructure

The pre-restructure layout had grown several inconsistencies:

- **Cross-tree includes.** `PolfedCore` reached *up and over* into sibling
  top-level folders with `include(joinpath(@__DIR__, "..", "Optimization", …))`
  and `…("..", "DensetiesOfStates", …)`.
- **Mixed naming.** Folders and files used PascalCase, snake_case, and lowercase
  interchangeably (`PolfedCore.jl`, `polfed_algorithm.jl`, `common.jl`, `cpu/`).
- **A typo'd folder** (`DensetiesOfStates`) whose directory name disagreed with
  the module/file inside it.
- **Duplicated definitions** — `ProcessingUnit` (the `CPU`/`GPU` types) existed
  in two places with *different* field sets.
- **A grab-bag `Structs/` folder** and a vague `Plan/` folder that mixed resolved
  config structs with unrelated Distributed-worker lifecycle code.
- **~90 lines of GPU-dispatch plumbing** sitting at the top of `UniPolfed.jl`,
  burying the actual package facade.

The goal: a layout where the **folder tree mirrors the dependency tree**, every
`include` points *downward* into its own subtree, names are uniform, and each
concept lives in exactly one place.

---

## 2. Design philosophy — four tiers

Code is organized top-to-bottom by *what kind of thing it is*:

1. **Infrastructure** (`Backend/`, `Common/`) — GPU dispatch, the `CPU`/`GPU`
   processing-unit types, logging, formatting, timing. No algorithmic content.
2. **Numerics** (`Numerics/`) — reusable, POLFED-agnostic spectral primitives:
   `Lanczos`, `Clenshaw`, `KPM`. The litmus test for this tier: *a component
   belongs here iff it can be unit-tested without constructing a single POLFED
   struct.*
3. **The POLFED algorithm** (`UniPolfed/`) — everything specific to POLFED: the
   public entry points, configuration/plan structs, the spectral transform,
   density-of-states adapter, optimization kernels, and the engine.
4. **Physics** (`Models/`) — Hamiltonian builders (`QSun`, spin-half chains).

The single underlying rule: **a file may only `include` things in or below its
own directory.** After the move, the folder tree literally mirrors the
dependency tree.

---

## 3. Target directory structure

`«mod»` marks a file that defines a Julia `module`; everything else is plain
`include`d text.

```
src/
├── UniPolfed.jl                              «mod» THE package module — includes every component
│                                          below + public API. The POLFED algorithm files in
│                                          UniPolfed/ are included straight in here (no PolfedCore).
│
├── Backend/                               ── infrastructure ──
│   ├── Backend.jl                         aggregator (NOT a module — see §6)
│   ├── gpu_dispatch.jl                    GPU Ref-dispatch plumbing (lifted from old UniPolfed.jl)
│   └── ProcessingUnit.jl                  unified CPU()/GPU() types (was duplicated)
│
├── Common/
│   ├── Common.jl                          «mod»   (was common.jl)
│   ├── Formatter.jl                       (was formater.jl)
│   ├── add_time.jl                        (was addtime.jl)
│   └── logging.jl
│
├── Numerics/                              ── reusable spectral primitives ──
│   ├── Numerics.jl                        «mod» (or plain include-hub)
│   │
│   ├── Clenshaw/
│   │   ├── Clenshaw.jl                    «mod» (was ClenshawMapping.jl) — module root:
│   │   │                                  polynomial_properties + ClenshawExpansion
│   │   │                                  type + constructors + callables + build_clenshaw_kernels
│   │   └── Algorithm/
│   │       ├── clenshaw_algorithm.jl      (was ClenshawAlgorithm.jl)
│   │       ├── final_sum.jl               (was FinalSum.jl)
│   │       └── recurrence_relations.jl    (was RecurrenceRelations.jl)
│   │
│   ├── KPM/                               density-of-states PRIMITIVE
│   │   ├── KPM.jl                         «mod»
│   │   ├── moments.jl                     (was Moments.jl; moment-doubling + chunk threading)
│   │   ├── kernels.jl                     (was Kernels.jl)
│   │   └── density.jl                     (NEW: kpm_density → ρ(x); lifted out of the old getdos!)
│   │
│   └── Lanczos/
│       ├── Lanczos.jl                     «mod»
│       ├── lanczos.jl                     (entry: lanczos + lanczos_extrema [BothEnds(1,1) + power-
│       │                                   iteration warm-up + maxdim=100 + return_result kwarg])
│       ├── lanczos_algorithm.jl           (was LanczosAlgorithm.jl)
│       ├── lanczos_method.jl              (was LanczosMethod.jl)
│       ├── LanczosIterator.jl             (type; parametrized on the callback)
│       ├── LanczosDefaults.jl             (defaults registry)
│       ├── LanczosResult.jl               (NEW: result struct + has_converged)
│       ├── Layout.jl                       (NEW: Column/Row block-storage trait — static dispatch)
│       ├── FactorizationReport.jl         (type; reads Criterion+State)
│       ├── Convergence/
│       │   ├── Convergence.jl
│       │   ├── EigTarget.jl               (was EigSorter.jl: NumEigs/ToValue · Lowest/Highest/BothEnds)
│       │   ├── ConvergenceCriterion.jl    (immutable config; was half of ConvergenceInfo)
│       │   ├── ConvergenceState.jl        (mutable state; was half of ConvergenceInfo)
│       │   ├── calc_ritz_residuals.jl     (cheap Ritz residuals; SymTridiagonal/Hermitian)
│       │   ├── calc_eigen_residuals.jl    (‖A·v−λ·v‖ eigenpair residuals)
│       │   └── check_convergence.jl       (orchestrator: scheduling + two-gate convergence +
│       │                                   contiguous-run extraction; absorbs old getnumberofconvergedvecs!/
│       │                                   next_checking!/getvals/getvecs)
│       ├── Factorization/
│       │   ├── Factorization.jl           (+ breakdown tol; lanczositer! returns ok)
│       │   ├── diagnostics.jl
│       │   ├── FactorizationType/
│       │   │   ├── Lanczos.jl             (scalar factorization type; breakdown guard; was LanczosFactorization.jl)
│       │   │   └── BlockLanczos.jl        (block factorization type; breakdown guard; was BlockLanczosFactorization.jl)
│       │   └── FactorizationSteps/        (tridiagonalization.jl dropped — was dead)
│       │       ├── calculate_overlap.jl
│       │       ├── mapping.jl
│       │       ├── orthogonalization.jl
│       │       └── reorthogonalization.jl
│       ├── OrthonormalBasis/
│       │   ├── OrthonormalBasis.jl        (abstract type + aggregator)
│       │   ├── MatrixBasis.jl             (type; `all`→`allvecs` to not shadow Base.all)
│       │   └── VectorBasis.jl             (type; fixed size(x0,2) typo)
│       │   NOTE: HybridMatrixBasis dropped — niche GPU-memory hack; restart is the
│       │   principled large-basis answer (future).
│       └── ReOrthTechniques/              (was ReOrthTechnics/)
│           └── ReOrthTechniques.jl        (ReOrthTechnique, FullRO, PartialRO, SelectiveRO)
│       NOTE: Lanczos/ProcessingUnit.jl deleted — uses the unified Backend type.
│
├── UniPolfed/                                ── the POLFED algorithm (NO submodule: files
│   │                                          included straight into `module UniPolfed`) ──
│   ├── entrypoint.jl                      public polfed() methods + input validation
│   ├── Defaults.jl                        «mod» constants registry (was PolfedDefaults.jl);
│   │                                      MINIMAL during migration — grows as components land
│   ├── polfed_algorithm.jl                engine: dos → transform → factorize
│   │
│   ├── Mapping/                            ✅ MIGRATED — first slice with a REAL Plan
│   │   ├── Mapping.jl                      aggregator
│   │   ├── MappingConfig.jl               (immutable; parallel_strategy default nothing = auto)
│   │   ├── MappingPlan.jl                 (parametric {T,Fpar,Fmap,Frescaled,Fcrr,Fcfs} — type-stable)
│   │   └── build_mapping_plan.jl          estimate Emin/Emax (lanczos_extrema, convergence-guarded)
│   │                                       → a/b → f!_rescaled = (H-b)/a closure
│   │
│   ├── Transform/                          ✅ MIGRATED — NO TransformPlan (config is internal)
│   │   ├── Transform.jl                     aggregator
│   │   ├── Filters/
│   │   │   ├── Filters.jl                   AbstractFilter seam + interface (filter_coefficients,
│   │   │   │                                analytical_order, expected_krylov_dim, exposed_end,
│   │   │   │                                window_placement)
│   │   │   └── DeltaFilter.jl               the one filter; ALL delta-specific constants live here
│   │   │                                    (2.65499 order, 20.427/1.696 krylov); coefficients &
│   │   │                                    order/krylov are METHODS, not stored fields
│   │   ├── TargetSpec.jl                    5 target types + normalize_target_spec
│   │   ├── TransformConfig.jl              (immutable, INTERNAL — not a polfed kwarg; filter +
│   │   │                                    left/right/expose gluing hooks; no Plan)
│   │   ├── resolve_target.jl               TargetSpec → rescaled λ̃ (maxdos scans ρ)
│   │   ├── resolve_bounds.jl               window from DoS mass (analytic, chebyshev_integration)
│   │   │                                    or fixed order; + _scalar_filter/_window_bounds helpers
│   │   ├── resolve_order.jl                order by cutoff bisection + safety factor (widens window)
│   │   ├── validate_filter.jl              expose↔normalization check + side-lobe warning (§7)
│   │   ├── clenshaw_dispatch.jl            clenshaw_apply per strategy + get_b_storage
│   │   ├── define_transformation.jl        builds f!_transformed (Clenshaw closure)
│   │   └── build_transform.jl              orchestrates resolve→validate→define; returns a
│   │                                        NamedTuple (f!_transformed, filter, target, l, r, expose)
│   │
│   ├── Factorization/                      ✅ MIGRATED — NO Plan (only the EigTarget needs resolving)
│   │   ├── Factorization.jl                aggregator
│   │   └── FactorizationConfig.jl          (immutable type + resolve_eig_target; uses EigTarget,
│   │                                        not which::Symbol; eig_target=nothing → HighestEigs(howmany))
│   │   NOTE: expected_krylov_dim is NOT here — it moved to the FILTER (DeltaFilter); its fit
│   │   constants (20.427, 1.696) are delta-specific, like the 2.65499 order constant. maxdim is
│   │   computed at the lanczos call site in polfed_algorithm, where the filter is known.
│   │
│   ├── DensityOfStates/                    ✅ MIGRATED — NO DoSPlan (nothing to resolve)
│   │   ├── DensityOfStates.jl              aggregator
│   │   ├── DoSConfig.jl                    (immutable type; N/R/kernel; was DensitiesOfStatesConfig.jl)
│   │   ├── DoSResult.jl                    (NEW: moments + ρ; replaces the old mutable ρ placeholder)
│   │   └── density_of_states.jl           adapter → kpm_moments + kpm_density (was get_dos!/getdos!)
│   │
│   ├── Parallel/                           ✅ MIGRATED
│   │   ├── Parallel.jl                     aggregator
│   │   ├── Parallelization.jl              4 strategy types (all mutable): NoParallel,
│   │   │                                   MulColsParallel, MulRowsParallel (NEW — dormant type,
│   │   │                                   row-chunked matvec lands with Optimization), TwoLevelParallel
│   │   ├── resolve.jl                      resolve_parallelization (default is pu-COMPUTED, not in
│   │   │                                   Defaults: CPU→MulCols, GPU→NoParallel) + warn_parallelization
│   │   └── workers.jl                      set_workers/remove_workers (Distributed; was Plan/workers*.jl).
│   │                                       Workers load POLFED via remotecall_eval(:(using UniPolfed))
│   │                                       (not the dropped main_module_file include)
│   │
│   ├── Optimization/                       (was top-level)
│   │   ├── Optimization.jl                 (was optimization.jl)
│   │   ├── CPU/
│   │   │   ├── cpu.jl
│   │   │   ├── mapping.jl
│   │   │   ├── make_loop.jl
│   │   │   ├── single_row_mappings.jl
│   │   │   ├── clenshaw_recurrence_relation.jl
│   │   │   └── clenshaw_final_sum.jl
│   │   └── GPU/
│   │       ├── gpu.jl
│   │       ├── vector_dispatch.jl
│   │       └── Kernels/
│   │           ├── kernels.jl
│   │           ├── clenshaw.jl
│   │           ├── mapping.jl
│   │           └── device_functions.jl
│   │
│   └── Report/
│       ├── Report.jl
│       ├── SpectralTransformReport.jl
│       └── BenchmarkReport.jl
│
└── Models/                                 ── physics ──
    ├── Models.jl                          «mod»
    ├── spin_half_chains.jl                 (was SpinHalfChains.jl)
    └── QSun/
        ├── QSun.jl                        «mod»
        ├── hilbert_space.jl                (was HilbertSpace.jl)
        ├── quantum_sun.jl                  (was QuantumSun.jl)
        ├── quantum_sun_u1_conserving.jl    (was QuantumSun_U1_conserving.jl)
        ├── random_utils.jl                 (was RandomUtils.jl)
        └── spin_operators.jl               (was SpinOperators.jl)

ext/UniPolfedCUDAExt.jl                         unchanged — Julia requires extensions at root
```

### What dissolved

| Old | New |
|---|---|
| `Structs/` (grab-bag) | vertical slices: `Mapping/`, `Transform/`, `Factorization/`, `DensityOfStates/`, `Parallel/`, `Report/`; `ProcessingUnit`→`Backend/`; defaults→`Defaults.jl` |
| `Plan/` | `Mapping/` + `Transform/` (plan structs) and `Parallel/workers.jl` (worker lifecycle) |
| `DensetiesOfStates/` | `Numerics/KPM/` (primitive) + `UniPolfed/DensityOfStates/` (adapter); typo fixed |
| top-level `Optimization/` | `UniPolfed/Optimization/` |
| top-level `QSun/` | `Models/QSun/` |
| GPU plumbing in `UniPolfed.jl` | `Backend/gpu_dispatch.jl` |
| `PolfedCore` submodule | collapsed — the algorithm lives directly in `module UniPolfed` (`src/UniPolfed.jl`); `src/UniPolfed/` is an organizational folder whose files are `include`d into it. (CUDA ext references become `UniPolfed.*` instead of `UniPolfed.PolfedCore.*`.) |

---

## 4. Naming conventions

> **Folders →** lowercase, no exceptions for acronyms (`cpu/`, `gpu/`, `kpm/`)
> **Files defining a module or a type →** `PascalCase`, matching the name
> **Files holding functions / behavior →** `snake_case`, matching the main function

The test for a borderline file: **is it named after a *thing* (→ PascalCase) or
an *action* (→ snake_case)?** A type definition (`DeltaFilter.jl`,
`MappingConfig.jl`) is a thing; a step (`side_lobe_check.jl`,
`lanczos_algorithm.jl`) is an action.

The `QSun/` internals use strict snake_case (`quantum_sun.jl`,
`spin_operators.jl`, …) because they are function libraries with no single
defining type.

---

## 5. The `Config → Plan` pattern

POLFED separates *what the user asks for* from *what gets executed*:

```
XConfig    user-facing settings (may contain `nothing` / "auto" fields)
   │  build_X_plan(config, …)        resolve auto fields, compute derived quantities
   ▼
XPlan      fully concrete — what the algorithm actually runs on
```

This applies uniformly: `MappingConfig→MappingPlan`, `TransformConfig→TransformPlan`,
`FactorizationConfig→FactorizationPlan` (renamed from `FactorizationConfigFull`),
`DoSConfig→DoSPlan` (renamed from `DoSConfigFull`). The word "Plan" is used in the
sense of an FFTW `plan_fft`: a precomputed, resolved execution strategy.

In the new layout each *concept* owns its config, plan, and builder in one
folder (vertical slicing), rather than scattering configs into `Structs/` and
plans into `Plan/`.

**Exception — `DoSConfig` has no plan.** Its fields (`N`, `R`, `kernel`) are all
concrete values with defaults; there is nothing to resolve. So density-of-states
skips the `Plan` step: `DoSConfig` is consumed directly by `density_of_states(...)`,
which returns a `DoSResult` (moments + ρ). The pattern is applied where it earns its
keep, not as ceremony.

---

## 6. Module system rules (where `using`/`import` go)

Julia's `using`/`import` is **module-scoped, and submodules do NOT inherit the
parent module's imports.** Consequences:

- **External packages** (`LinearAlgebra`, `SparseArrays`, …) are `using`-ed
  **inside each module that needs them**, right after the `module` line and
  before the `include`s. Example: `Clenshaw` declares
  `using LinearAlgebra, SparseArrays, StaticArrays` itself.
- **`include`d files are not modules** — they are pasted into the enclosing
  module and share its namespace. So `using` goes once at the module top; the
  included files never repeat it.
- **Internal shared names** (Common's helpers, Backend's `gpu_*`/`CPU`/`GPU`)
  are pulled in with `import ..Name` from the parent, e.g. `import ..Common: …`,
  `import ..cuda_available`.
- **`Project.toml [deps]` ≠ `using`.** A dependency is listed once for the whole
  package (that makes it loadable); the per-module `using` is only about bringing
  names into *that module's* scope.

### Why `Backend/` is NOT a submodule

The CUDA extension hard-codes top-level hooks: `UniPolfed._set_cuda_backend!` and
`UniPolfed.is_gpu_array` (see `ext/UniPolfedCUDAExt.jl`). If `Backend` were a submodule
those references would break. So `Backend/` is an *organizational folder* whose
files are `include`d at the `UniPolfed` top level — the GPU-dispatch bindings and
the `CPU`/`GPU` types land directly in `module UniPolfed`, exactly where the
extension expects them. (`Common`, by contrast, is a genuine submodule.)

---

## 7. Spectral-transform filter seam (deferred / algorithmic)

The current transform pipeline hard-codes a single strategy: a delta-peaked
Chebyshev filter with symmetric, interior bounds. This fails for near-edge or
asymmetric target intervals (e.g. `[-1, -0.7]`): a Chebyshev side lobe rises
above the `cutoff` threshold in a region that still has density of states, so
those eigenvalues get amplified too and pollute the Krylov subspace.

The restructure introduces a *seam* for this without yet adding new behavior:

- `Filters/AbstractFilter.jl` — the interface a filter must implement
  (resolve bounds, resolve order, coefficients).
- `Filters/DeltaFilter.jl` — the **only** concrete filter for now; it absorbs the
  former `GetBounds.jl` / `GetOrderOfExpansion.jl` logic *and* the Krylov-budget
  heuristic `expected_krylov_dim` — all delta-specific fit constants (the `2.65499`
  order constant, the `20.427`/`1.696` Krylov-dim fit). A future filter brings its
  own. `expected_krylov_dim(filter, howmany, blocksize, η)` dispatches on the filter;
  `η` (overestimate factor) stays a user knob in `FactorizationConfig`/`Defaults`.
- `select_filter.jl` — given the DoS and interval, choose a filter. Returns
  `DeltaFilter` today; the hook for future strategies.
- `side_lobe_check.jl` — validation: scan the built filter weighted by the DoS
  across `[-1, 1]` and flag secondary lobes above cutoff. This is the immediate
  robustness win and is a pure addition.

**Scope for now:** wrap the existing delta filter behind this seam only. New
filter types (window/edge filters) and the side-lobe *remediation* logic are
future, separate, algorithmic work.

---

## 8. Migration process

- **Branch:** `restructure/file-layout`, off `main`. `main` stays untouched.
- **Old code** is parked in `src-OBSOLET/`, old tests in `test-OBSOLET/`. These
  are kept as the **reference we port from** (copy out of them; do not move, so
  they stay complete). They are deleted only at the very end.
- **New `src/`** is built up from an empty skeleton, **one component at a time**,
  with the package kept loading green at every step.
- **Per-component loop:** copy code into its new home → apply naming convention
  and fix `include`/`import` wiring → wire into `UniPolfed.jl` → write tests →
  run green → review.
- **Migration order is dependency-first:** Clenshaw → KPM → Lanczos → POLFED core.
  (KPM uses Clenshaw; the POLFED core uses all three.)
- **History note:** because we copy rather than `git mv` out of `src-OBSOLET`
  (to keep it a complete reference), `git blame` does not follow into the new
  files. The history remains accessible at the `src-OBSOLET/…` paths.

### Testing approach

These are mathematical primitives, so tests assert against **ground truth**, not
captured snapshots:

- **Clenshaw:** exact polynomial values (`Tₙ(cos θ) = cos nθ`).
- **KPM:** known density of states of a constructed matrix.
- **Lanczos:** known eigenvalues of a constructed matrix.

The new `test/` mirrors the `src/` layout, with `runtests.jl` including
per-component files as each lands.

### Documentation

Thorough docstrings are required on all migrated code — preserve existing
docstrings verbatim and add them where missing, matching the package's
Arguments / Returns / Notes / Example style.

### Build / run notes

- Julia is invoked via `$HOME/julia-wrappers/julia-x86_64` (CPU, v1.12.5).
  `julia-aarch64` is the GPU build.
- **GPU code paths cannot be tested on the x86 machine.** CPU paths are covered
  now; GPU tests are gated for later. The CUDA extension stays dormant until
  the POLFED core / `Optimization` are migrated (it only activates when CUDA is loaded).

---

## 9. Status

| Component | State |
|---|---|
| Skeleton (folders + module stubs) | ✅ done |
| `Common/` | ✅ migrated + loads green |
| `Backend/` (gpu_dispatch + unified ProcessingUnit) | ✅ migrated + loads green |
| `Numerics/Clenshaw/` | ✅ migrated + loads green + tested + 100% cov |
| `Numerics/KPM/` | ✅ migrated + loads green + tested + 100% cov (moment doubling; chunk threading) |
| `Numerics/Lanczos/` | ✅ migrated + loads green + tested (CPU ~full cov; GPU gated) — EigTarget redesign, both-ends + ToValue gluing, breakdown guard |
| `Models/` (+ QSun) | ✅ migrated + loads green + tested + 100% cov (ground-truth: su(2) algebra/Casimir, kron cross-check, sector tiling). QSun moved under `Models/`; `SpinOperators` submodule flattened; dead U(1) coupling branches pruned (§10.1) |
| `UniPolfed/Defaults.jl` (minimal) + `UniPolfed/DensityOfStates/` | ✅ migrated + loads green + tested (ground-truth: `H=c·I` exact moments, ∫ρ≈dim sum rule). No DoSPlan — nothing to resolve |
| `UniPolfed/Factorization/` | ✅ migrated + loads green + tested. `FactorizationConfig` only (no Plan); uses `EigTarget` (`eig_target=nothing`→`HighestEigs(howmany)`) + `resolve_eig_target`; `Defaults` gained rot/basistype/tol/eigentol/η. `expected_krylov_dim` deferred to the filter |
| `UniPolfed/Parallel/` | ✅ migrated + loads green + tested (no-op strategies, resolve, warn; real worker spawn gated behind `POLFED_TEST_DISTRIBUTED`). Added `MulRowsParallel` (dormant) |
| `UniPolfed/Mapping/` | ✅ migrated + loads green + tested (ground-truth: exact a/b + `(H-b)/a` rescaling, bound estimation). First slice with a real (parametric) Plan. Enhanced `lanczos_extrema`: `BothEnds(1,1)` + power-iter warm-up + maxdim `1000→100` + `return_result` |
| `Common/Utils/` (bisection + chebyshev_integration) | ✅ migrated + loads green + tested (ground-truth: known roots; ∫Tₖ/√(1-x²) closed forms + fine numeric integral). Weighted Chebyshev integral (Chebyshev-Gauss measure) — replaces QuadGK; exact for the KPM density |
| `UniPolfed/Transform/` (+ Filters) | ✅ migrated + loads green + tested (ground-truth: diagonal H ⇒ f!_transformed acts as P(λ̃) to machine precision, all CPU strategies; peak=normalization; analytic DoS window count ≈ howmany). NO TransformPlan — `TransformConfig` is internal. AbstractFilter seam + DeltaFilter (delta constants + coefficients/order/krylov as methods). Side-lobe check is a NEW diagnostic (§7). DoS must run on the RESCALED operator (moments in rescaled coords) |
| CUDA extension rewire | ✅ done — `UniPolfed.PolfedCore.*` refs dropped; hooks target `UniPolfed.*`; Optimization-dependent kernels (materialize_host_matrix, vector_dispatch) stubbed behind comments until `Optimization/` lands |
| `UniPolfed/` core remainder (Optimization, Report, entrypoint) | ⬜ pending — Optimization next |
| Delete `src-OBSOLET/` + `test-OBSOLET/` | ⬜ final step |

---

## 10. Open questions — to raise with collaborators

### 10.1 Dead coupling branches in the U(1)-conserving Quantum Sun builder ✅ resolved

**File:** `src/Models/QSun/quantum_sun_u1_conserving.jl` (`build_hamiltonian_cons`).

**What.** The grain–ray coupling loop expanded each of `SˣSˣ` and `SʸSʸ` into four
`(local ±, grain ±)` sub-branches. Four of those — the **same-direction** ones
(local *and* grain both raised, or both lowered) — were **unreachable**: each was
gated by `if new_idx != 0 && mag[new_idx] == S_z`, and raising/lowering both spins
shifts total `Sᶻ` by `±2`, so the flipped state always lands in a different
magnetization sector and the guard is always false. They never pushed for any input
(the 12 lines that kept the file below 100% coverage).

**Resolution (collaborator decision):** confirmed dead code — **pruned**.
`build_hamiltonian_cons` only ever realizes the magnetization-conserving XY hopping
`SˣSˣ + SʸSʸ = ½(S⁺S⁻ + S⁻S⁺)`, so only the *cross* terms (`c1·cg2` = local raise +
grain lower, `c2·cg1` = local lower + grain raise) are built. The removal is
**byte-identical in output** (the deleted branches never contributed), makes the U(1)
structure explicit, and brings the file — and all of `Models/` — to 100% coverage.
