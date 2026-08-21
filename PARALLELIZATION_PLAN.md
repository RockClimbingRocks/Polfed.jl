# Parallelization rework — implementation plan

Status: proposed, not started. Written 2026-08-20.

Rework POLFED's mapping/transform parallelization around **two** strategies with clear,
non-overlapping meanings, fold the fused Clenshaw kernel into `optimization/`, and add the
hybrid block decomposition that the `notebooks/benchmarks/clenshaw/` sweep showed to be the
fastest configuration at every production size.

---

## 0. Where we are (verified 2026-08-20)

- `UniPolfed` precompiles. The earlier `Logging` breakage is **resolved** — `common/logging.jl`
  documents that `@info`/`@warn` come from Base, and `Logging` is a test-only dep.
- Four strategies exist in `src/main/logic/mapping/parallel.jl`:
  `WholeBlock` (38 refs), `MulColsParallel` (33), `MulRowsParallel` (17), `TwoLevelParallel` (22).
- `MulRowsParallel` is documented as behaving **identically to `WholeBlock`** today.
- Layout types are `ColumnLayout` / `RowLayout` (note: `ColumnLayout`, not `ColLayout`),
  exported from `numerics/lanczos/Layout.jl`.
- Strategy types are **not exported** at top level; users reach them as `UniPolfed.WholeBlock()`.

### What the benchmark established

`notebooks/benchmarks/clenshaw/` timed five Clenshaw drivers. The finding that drives this plan:
**V3, V4 and V5 are one kernel with three settings of one triple** `(g, w, tpg)`, subject to
`g·w = bs` and `g·tpg = nt`.

| benchmark variant | `(g, w, tpg)` | already in `src/` as |
|---|---|---|
| V3 `fused_index`  | `(1, bs, nt)` | `WholeBlock` (unfused) |
| V4 `fused_vector` | `(bs, 1, 1)`  | `MulColsParallel` (unfused) |
| V5 `fused_block`  | general       | **missing** |

So only ONE decomposition is genuinely new. The other two changes are *fusion* (orthogonal to
decomposition) and *naming/semantics*.

Measured, on 2x32-core EPYC 9354 (32 MB L3 per 4-core CCX), µs/step/Krylov-vector:

- V5 wins 8 of 12 boxes at L=22, the production-relevant size; V1 (MKL) is last in all 36 boxes.
- Best-vs-current (V2, what POLFED runs today) at L=22: **1.14x - 2.52x**.
- Thread placement is worth up to **2.9x** and is invisible to the type system.

---

## 1. Target design

### 1.1 Two strategies

```julia
abstract type Parallelization end

"""Parallelism INSIDE a single recurrence step: one parallel region per Clenshaw step,
`groups` vector groups of width `bs÷groups`, each served by `nthreads()÷groups` threads
that split its state range. `groups = nothing` resolves automatically."""
mutable struct BlockWise <: Parallelization
    groups::Union{Int,Nothing}
end
BlockWise(; groups = nothing) = BlockWise(groups)

"""Parallelism AROUND the Clenshaw call: the recurrence runs once per Krylov vector with
its own buffer triplet, one parallel region for the whole K-loop. Layout-aware — over
columns for `ColumnLayout`, over rows for `RowLayout`."""
struct VectorWise <: Parallelization end
```

- `WholeBlock` -> `BlockWise`. `MulRowsParallel` is **deleted** (it was a documented duplicate).
- `MulColsParallel` -> `VectorWise`.
- `TwoLevelParallel` is **deleted** entirely (see 2.1).

### 1.2 Why `BlockWise` gets ONE field, not three

`g·w = bs` and `g·tpg = nt` leave exactly **one** degree of freedom. Three fields would let a
user write an arithmetically impossible `BlockWise(g=4, w=4, tpg=8)`. Neither `bs` nor `nt`
exists at construction time anyway (`bs` arrives at plan-build, `nt` is `Threads.nthreads()`).

This is the documented Config->Plan seam (CLAUDE.md "Conventions"), and one of the cases with
real resolution:

```julia
resolve_decomposition(cfg::BlockWise, bs, nt, D, ::Type{T}) -> (g, w, tpg, chunk)

struct BlockPlan{F<:Function}
    f!::F        # the fused range kernel, built at Val(w)
    g::Int
    tpg::Int
    chunk::Int   # aligned: align * cld(cld(D, tpg), align)
end
```

**`w` must NOT be a plan field.** `build_kernel(..., Val(w))` specializes the entire generated
inner loop on the pack width — `w` *is* the `bs` the kernel is built at. It selects which
closure gets built; it is never stored. `g`/`tpg`/`chunk` only drive loop bounds, so plain
`Int` fields are correct (satisfies "concrete struct field types; parametrize callbacks").

### 1.3 The one dispatch

There is no "index-threading vs block-threading" fork — index-threading is `g = 1`, the same
code path. The real fork is *where the parallel region sits*, and that is exactly the
strategy distinction:

| | region | why |
|---|---|---|
| `BlockWise`  | inside each recurrence step | a group's chunks must all finish before its rotation |
| `VectorWise` | outside the whole K-loop | nothing is shared, so no per-step barrier is needed |

`VectorWise` therefore saves `K-1` barriers. That is V4's genuine structural advantage, and
under this naming it falls out of the semantics rather than needing a flag.

---

## 2. Phases

Each phase ends green (`Pkg.test()`), so the tree is never broken between phases.

### Phase 0 — Delete `TwoLevelParallel` (no behavior change to kept paths)

Removes a whole dependency surface before anything is renamed.

1. Delete `src/main/logic/mapping/workers.jl` and its `include` in `mapping/mapping.jl:7`.
2. Delete `set_workers` / `remove_workers` and every call site.
3. Delete the `TwoLevelParallel` struct (`mapping/parallel.jl`), its `get_b_storage` method,
   its `clenshaw_apply` method, and `get_clenshaw_shared_buffer!` (`transform/parallel/b_storage.jl`).
4. Drop `using Distributed` / `using SharedArrays` (`src/UniPolfed.jl:22-23`) and remove
   `Distributed` + `SharedArrays` from `Project.toml [deps]`.
5. Delete `Manifest.toml` and re-resolve.

**Acceptance:** `Pkg.test()` green; `grep -r TwoLevelParallel src/ test/` empty.
**Note:** the CLAUDE.md gotcha about `remotecall_eval` worker bootstrap becomes dead — remove it.

### Phase 1 — Rename, semantics unchanged

Pure rename so the diff is reviewable and behavior is provably identical.

1. `WholeBlock` -> `BlockWise` (add the `groups::Union{Int,Nothing}` field, default `nothing`;
   nothing reads it yet).
2. `MulColsParallel` -> `VectorWise` (drop `nt_per_col`; see below).
3. Delete `MulRowsParallel`; every reference becomes `BlockWise`.
4. Update `resolve_parallelization` (`mapping/parallel.jl`): GPU -> `BlockWise()`,
   CPU auto -> `VectorWise()` (preserves today's `MulColsParallel()` default).
5. Update `entrypoint_helpers.jl:42` (`parallel_strategy = WholeBlock()` -> `BlockWise()`).
6. Update `MappingReport.jl:50`, `build_expansion.jl:45,61,67`, `b_storage.jl`, `clenshaw_apply.jl`.
7. Update tests: `test/main/logic/transform/parallel/b_storage_tests.jl`,
   `transform/build_transform_tests.jl`, `transform/clenshaw/build_expansion_tests.jl`,
   `report/MappingReport_tests.jl`, `report/SpectralTransformReport_tests.jl`.
8. Update `notebooks/test/ui.ipynb`.

**On `nt_per_col`:** the `MulColsParallel(nt_per_col>1)` branch existed to serialize BLAS and
hand threads to the kernel — that is what `BlockWise` now means. Fold it: `nt_per_col > 1`
becomes `BlockWise()`. Do not carry the field over.

**Acceptance:** `Pkg.test()` green; no `WholeBlock`/`MulCols`/`MulRows` left; a POLFED run
produces bit-identical eigenvalues to before the rename.

### Phase 2 — Make `VectorWise` layout-aware

Today's threaded `MulColsParallel` method hardcodes `view(Y, :, i)` — column views only.

**The trap:** Julia is column-major. In `RowLayout` the block is `bs x D`, so vector `i` is
`Y[i, :]` with stride `bs` — **strided, not contiguous**. The packed kernels take raw pointers
and require a dense `Matrix` (see CLAUDE.md: `mp.f!_rescaled` "can't act on strided 1xD
eigenvector views"). Per-vector *views* are therefore not an option under `RowLayout`.

Do what benchmark V4 does — materialize per-vector contiguous `1 x D` buffers:

1. `_per_vector_buffers(layout, pu, x0)`:
   - `ColumnLayout` -> per-column `D`-vectors (contiguous views are fine, keep today's path).
   - `RowLayout` -> `bs` separate dense `1 x D` matrices, plus a scatter/gather at the
     transform boundary.
2. `clenshaw_apply(..., ::VectorWise)` dispatching on layout.

**Cost check:** scatter+gather is `2·D·bs` copies **per Lanczos matvec**, against
`K·4·D·bs` of recurrence traffic with `K ≈ 900`. Under 0.1%. Assert this with a timing test.

**Bonus:** `w = 1` gives the packed kernel its widest packing (`P = W/1` states per register),
which is why V4/V5 beat V2 in the sweep.

**Acceptance:** `VectorWise` produces identical results under both layouts; a `RowLayout` run
is not slower than the `ColumnLayout` run of the same problem.

### Phase 3 — Expose a serial range core in `optimization/`

`_packed_apply!` opens its own `Threads.@threads :static` — which the `BlockWise` driver
**cannot** allow, since nested `:static` regions are illegal in Julia.

1. New public builder alongside `build_kernel`:
   ```julia
   build_range_kernel(kernel, s::MappingStructure, ::Val{w}) -> (Y, X, lo, hi) -> nothing
   ```
   Serial, no threading, `[lo, hi)` over basis states.
2. `spin_flip/packed`: `_packed_range!` already exists (`spin_flip_packed_kernel.jl`) — wrap it.
3. `spin_flip/rows`, `hopping/packed`, `hopping/rows`: same split.
4. `universal/row_read`: **has no range core** — extract one from `_row_read!`.
5. Rewrite `_packed_apply!` and friends as `build_range_kernel` + a threading wrapper, so there
   is one implementation of the work and one of the policy.

**Acceptance:** `build_range_kernel(k, s, Val(w))` over `[0, D)` equals `build_kernel(k, s, Val(w))`
for every registered kernel, all four element types.

### Phase 4 — Fused Clenshaw kernel

The largest win in the sweep, and orthogonal to decomposition — it benefits `VectorWise` too.

1. New builder:
   ```julia
   build_clenshaw_kernel(kernel, s, ::Val{w}) -> (b1,b2,X,b3,c,α,β,lo,hi) -> nothing
   ```
   computing `b1 <- c·X + α·(H̃·b2) + β·b3` in one pass, accumulator never leaving registers.
   Reference implementation: `fused_block!` / `fused_range!` in
   `notebooks/benchmarks/clenshaw/clenshaw_variants.jl` (a `@generated` epilogue).
2. Add `fused::Bool` to `KernelInfo` (`optimization/registry.jl`) — **not** a `:packed_fused`
   variant. Fusion is a different call signature of the same kernel, not a different structure
   match. `mapping_kernels(H)` should show it.
3. Implement for `spin_flip/packed` first, then `hopping/packed`. Patterns without one report
   `fused = false` and the transform keeps matvec+combine.
4. Wire into `BlockWise` and `VectorWise`; keep the unfused path as fallback.

**Acceptance:** fused and unfused agree to `1e-14` relative over `K = 40` steps for all four
element types; `@inferred` guard on the builder; the fused path is not slower at `nt = 1`.

### Phase 5 — `BlockWise` decomposition + driver

1. `cores_per_l3()` — read `/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list`,
   fall back to `4` when unreadable (non-Linux, containers). Call **once** at setup, never in
   a hot loop.
2. `resolve_decomposition(cfg, bs, nt, D, T) -> (g, w, tpg, chunk)`:
   - `g = cfg.groups` when given, else auto.
   - `g` must divide **both** `bs` and `nt`, and `w = bs÷g` must be a power of two `≤ W`.
     Pick the largest such `g`. **`min(bs, nt)` is not sufficient** — `nt = 12, bs = 8` gives
     `g = 8, tpg = 1` and four idle threads.
   - Prefer `w = 1` (widest packing).
   - Cap `tpg ≤ cores_per_l3()`. Beyond that a group cannot be cache-co-resident and extra
     threads *cost* time — measured L=18, bs=4: `nt=16 -> 61.7`, `nt=32 -> 151.6`, **2.5x
     slower on twice the threads**.
   - When `nt > bs · cores_per_l3()`, **use fewer threads** and log at INFO which and why.
   - When the resolution lands on `g = bs, tpg = 1`, log at WARN that `VectorWise` strictly
     dominates (same decomposition without the per-step barriers).
3. `BlockPlan{F}` as in 1.2; build the kernel at `Val(w)`.
4. The driver — ONE flat region, since nested `:static` is illegal:
   ```julia
   Threads.@threads :static for t in 1:(g*tpg)
       gi = (t - 1) ÷ tpg + 1
       ci = (t - 1) % tpg
       lo, hi = ci * chunk, min((ci + 1) * chunk, D)
       lo < hi && f!(B[gi]..., Xs[gi], ..., lo, hi)
   end
   ```
   Rotation is **per group** and must happen after all its chunks complete — so it lives
   outside the parallel region.
5. `get_b_storage(::BlockWise, ...)` -> `g` triplets of width `w`.

**Acceptance:** `BlockWise(groups=1)` reproduces V3, `BlockWise()` at `bs≥nt` reproduces V4's
numbers within noise, and the auto resolution matches the sweep's best box at L=20 and L=22.
`@inferred` on the driver.

### Phase 6 — Placement diagnostics

The 2.9x lives in thread placement, which no type can control from inside a library.
Measured rule — **two opposite behaviors from one variable**:

| | correct placement |
|---|---|
| `tpg == 1` (threads share nothing) | **spread** across L3 domains — packing costs 1.18x |
| `tpg > 1` (group shares buffers)   | **pack** each group into one domain — 0.70-0.80x |

Julia cannot portably pin from library code, so **diagnose, do not fix**:

1. At setup, one `Threads.@threads :static` region reading `sched_getcpu()` per thread
   (O(1), never hot).
2. Map each cpu to its L3 domain via sysfs.
3. Check the placement matches what the resolved decomposition assumed; on mismatch emit
   `polfed_log(POLFED_WARN_LEVEL, ...)` naming the expected grouping.
   **Never a raw `@warn`** — it ignores `verbosity[]`.
4. Document the required pinning in the mapping README (`pick_cpus.py` in the benchmark folder
   is the working reference, including that PIN order is semantic).

**Acceptance:** a deliberately scattered pinning triggers the warning; a correct one is silent
at WARN.

### Phase 7 — Reports, docs, defaults

1. `MappingReport` currently prints only the strategy type name. Print the resolved
   `(g, w, tpg)` and the kernel actually chosen — `(auto)` today hides the decision.
2. `PatchedReport` aggregation unchanged.
3. READMEs: `main/logic/mapping/`, `main/logic/transform/parallel/`, `optimization/`.
4. CLAUDE.md: rewrite the parallelization mentions; drop the `remotecall_eval` gotcha
   (dead after Phase 0); add the strided-view and nested-`:static` traps.
5. `Defaults`: if any decomposition constant becomes user-tunable, it goes in the registry —
   **and remember the include-order invariant**: `defaults/` loads after `logic/`, so every
   `Defaults.*` use in `logic/` must stay a keyword-argument default evaluated at call time.
   Never a top-level `const x = Defaults....` in `logic/`.

---

## 3. Gotchas (each has already bitten this work)

- **Nested `Threads.@threads :static` is illegal.** Hence the flat `g*tpg` region and the
  serial range cores (Phase 3).
- **`threadid()` trap (Julia ≥ 1.12):** never index per-thread buffers by `Threads.threadid()`.
  Index by the loop variable `t` of a `:static` region, or by chunk.
- **Packed kernels need dense `Matrix`.** No views, no lazy transposes, no strided `1 x D`
  rows. This is what forces the Phase 2 scatter/gather.
- **Alignment is correctness, not tuning.** Every chunk start must be a multiple of `P·G`, and
  `w` a power of two `≤ W` — preconditions of the lane permutation.
- **`g` must divide both `bs` and `nt`**, else threads silently idle.
- **The tail loop in `_packed_range!` is required** — a dropped tail leaves stale columns.
- **Rotation must follow the barrier** in `BlockWise`, or a group reads a half-written buffer.
- **`mapvals` for a `Window` must use `_make_rescaled(mp.f!, ...)`**, not `mp.f!_rescaled` —
  unchanged by this work, but the same strided-view reason.

---

## 4. Measure before hard-coding

Every constant below came from a **shared** node (load ~29; only 1 of 16 CCXs fully idle).
The *shape* reproduces across L=18/20/22 and matches the L3-capacity model, but the numbers
should be re-measured on an idle machine before they are frozen:

- the `tpg ≤ cores_per_l3()` cap;
- the "use fewer threads when `nt > bs·cores_per_l3()`" branch;
- the L3-capacity crossover, predicted at `4·D·w·sizeof(T) = L3` (L ≈ 20 at `w=1`, Float64),
  where packing stops paying — measured co-resident vs scattered: **2.84x at L=18, 2.89x at
  L=20, 1.00x at L=22**.

Also re-run the sweep in ONE process per L so thread-scaling columns are comparable —
the current `clenshaw_bench.csv` has each thread count in a separate process over a long
window, so cross-column (scaling) comparisons drift with machine load. Within-box
(variant vs variant) comparisons are sound.

---

## 5. Suggested order of work

Phases 0-1 are mechanical and unblock everything. **Phase 4 before Phase 5** — fusion is the
bigger measured win, it benefits the strategies that already exist, and it needs no new
decomposition policy to tune. `BlockWise`'s hybrid is the last piece and the only one whose
constants still want an idle machine.

```
0 delete TwoLevel  ->  1 rename  ->  2 VectorWise layouts  ->  3 range cores
                                                                   |
                                       7 reports/docs  <-  5 BlockWise  <-  4 fusion
                                                                   |
                                                            6 placement warnings
```
