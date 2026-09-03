# [Entrypoints](@id api_entrypoints)

The public solver functions. Everything here is part of the supported API; package
internals are deliberately excluded from this reference.

Each entrypoint has an **operator** method (`f(f!, x0, …)`) and a **matrix** method
(`f(H, x0, …)`). The operator method is column-native and leaves your `f!` and `x0`
untouched; the matrix method builds optimized row-layout kernels and works in row layout.
See [The operator interface](@ref operator_interface) below.

## [`polfed`](@id UniPolfed.polfed)

```julia
polfed(H,  x0, howmany, target;       mc, dosc, tol, eigentol, overestimate_iters)
polfed(f!, x0, howmany, target;       mc, dosc, tol, eigentol, overestimate_iters)
polfed(H,  x0, target_window;         mc, dosc, tol, eigentol, overestimate_iters)
polfed(f!, x0, target_window;         mc, dosc, tol, eigentol, overestimate_iters)
```

Single-filter POLFED. Count-target calls return up to `howmany` eigenpairs near a point
target; window-target calls omit `howmany` and return the eigenpairs found inside the
requested energy interval.

**Arguments**

- `H` — a Hermitian matrix (sparse, dense, or a device array).
- `f!` — an operator such that `f!(Y, X)` overwrites `Y` with `H * X`.
- `x0` — starting vector or block. A vector selects Lanczos; a matrix selects block Lanczos
  with block size equal to the column count.
- `howmany` — number of eigenpairs to target (count mode only).
- `target` / `target_window` — see [Targets](@ref api_targets).

**Keywords**

- `mc::MappingConfig` — operator, rescaling and kernel configuration.
- `dosc::DoSConfig` — density-of-states estimation.
- `tol` — Lanczos breakdown / Ritz tolerance.
- `eigentol` — acceptance tolerance, ``\lVert H v - \lambda v \rVert < \texttt{eigentol}``.
- `overestimate_iters` — Krylov budget inflation factor.

**Returns** a [`PolfedResult`](@ref UniPolfed.PolfedResult), which destructures as
`vals, vecs, report`.

Tutorial: [POLFED](@ref tutorial_polfed).

## [`patched_polfed`](@id UniPolfed.patched_polfed)

```julia
patched_polfed(H,  x0, howmany, howmany_per_patch, target; kwargs...)
patched_polfed(f!, x0, howmany, howmany_per_patch, target; kwargs...)
patched_polfed(H,  x0, howmany_per_patch, target;          kwargs...)
patched_polfed(f!, x0, howmany_per_patch, target;          kwargs...)
```

Edge-first patched POLFED. The **arity distinguishes the two modes**: with `howmany` it is a
count target near a point; without it, `target` must be an explicit window —
`(:rescaled, (lo, hi))`, `(:unrescaled, (lo, hi))`, or a bare unrescaled `(lo, hi)` tuple.

**Extra keywords**, in addition to those of `polfed`:

- `edge_distance` — how far from ``\pm 1`` the edge peels reach (default `0.2`).
- `howmany_pad` — overlap between neighbouring patches (default `3`).
- `max_retries` — coverage retries per patch (default `3`).
- `retry_growth` — budget multiplier per retry (default `2.0`).

**Returns** a [`PatchedResult`](@ref UniPolfed.PatchedResult), destructuring as
`vals, vecs, report` with a [`PatchedReport`](@ref UniPolfed.PatchedReport). In count mode
`vecs` is one preallocated matrix; in window mode each eigenvector is independently owned (a
vector of vectors), so completed patch matrices can be released as the run proceeds. Use
[`eigenvector`](@ref UniPolfed.eigenvector) to get one regardless of form.

Tutorial: [Patched POLFED](@ref tutorial_patched).

## [`double_polfed`](@id UniPolfed.double_polfed)

```julia
double_polfed(H,  x0, howmany, target1, target2; strategy, kwargs...)
double_polfed(f!, x0, howmany, target1, target2; strategy, kwargs...)
double_polfed(H,  x0, target1, target2;          strategy, kwargs...)
double_polfed(f!, x0, target1, target2;          strategy, kwargs...)
```

Two disjoint spectral windows in one call. The two targets are **separate arguments**, because
a bare `(lo, hi)` tuple is already a window spec. `howmany` may be a single count applied to
both windows, or a `(n₁, n₂)` tuple giving one count per window.

Both windows are served from one bounds probe and one DoS estimate. Whether they also share a
Krylov space is decided per problem: a joint [`DoubleDeltaFilter`](@ref filters) run when the
two single-window orders are comparable, two independent runs otherwise. Force it with
`strategy = :double` or `:separate`.

**Returns** a `DoublePolfedResult` — the two windows as ordinary
[`PolfedResult`](@ref UniPolfed.PolfedResult)s, in the caller's argument order, plus the
decision that produced them:

```julia
result    = double_polfed(H, x0, (-0.8, -0.5), (0.3, 0.6))
low, high = result                # the two windows, in the order you passed them
vals, vecs, report = low          # each window unpacks like any other run

result.strategy      # :double | :separate
result.ratio         # min(K₁,K₂)/max(K₁,K₂) — what the decision was made on
```

Tutorial: [Double POLFED](@ref tutorial_double).

## [`full_polfed`](@id UniPolfed.full_polfed)

```julia
full_polfed(H,  x0, howmany_per_patch; kwargs...)
full_polfed(f!, x0, howmany_per_patch; kwargs...)
```

Resolve the complete spectrum edge-first. There is no target argument. Accepts the same
keywords as `patched_polfed`, but defaults come from `Defaults.full_polfed` (notably
`retry_growth = 1.5` and `optimize_mapping` on).

**Returns** a [`PatchedResult`](@ref UniPolfed.PatchedResult), destructuring as
`vals, vecs, report`, where `vecs` is one dense ``D \times D`` matrix. If completeness cannot be
certified, the converged pairs are returned with a warning.

Tutorial: [Full POLFED](@ref tutorial_full).

## [The operator interface](@id operator_interface)

POLFED never needs your Hamiltonian as a matrix. Every entrypoint above has an **operator
method** taking a function instead — the route to take when your Hamiltonian is generated on the
fly, lives in a custom format, or is simply too large to assemble.

### The `f!` contract

```julia
f!(Y, X)   # must overwrite Y with H * X, and return nothing useful
```

That is the whole interface. `X` and `Y` are a vector (standard Lanczos) or a block of
columns (block Lanczos), matching the shape of the `x0` you pass.

!!! important "`f!` must also accept a single Krylov vector"
    Even when `x0` is a block, `f!` has to work on **one** vector — a length-`dim` vector under
    [`ColumnLayout`](@ref UniPolfed.Numerics.Lanczos.ColumnLayout), a `1 × dim` matrix under
    [`RowLayout`](@ref UniPolfed.Numerics.Lanczos.RowLayout). Two stages need it: the
    spectral-bounds probe, and the DoS, which deliberately probes one vector at a time so the
    estimated density does not depend on your block size.

    A `mul!` wrapper satisfies this for free. An operator hard-wired to a fixed block width does
    not, and must be written shape-generically.

### Reproducibility

The DoS is estimated with **stochastic** KPM probes, so two otherwise identical calls can resolve
slightly different target windows and return slightly different eigenpair sets. Seed it when the
result has to be reproducible:

```julia
result = polfed(H, x0, 100, :maxdos; dosc = DoSConfig(rng = MersenneTwister(2024)))
```

With a seeded `rng` the answer is reproducible and **independent of the Krylov block size** — the
same seed gives the same eigenpairs at `block_size = 1` and `block_size = 8`. On a
[`GPU`](@ref UniPolfed.GPU) the device generator is used and `rng` is ignored, so a GPU DoS is not
reproducible from a host seed.

```julia
using UniPolfed
using LinearAlgebra

f!(Y, X) = mul!(Y, H, X)          # the trivial wrapper, as an example

x0 = normalize!(rand(size(H, 1)))
result = polfed(f!, x0, 100, :maxdos)
```

Every entrypoint has this form:

```julia
polfed(f!, x0, howmany, target)
polfed(f!, x0, target_window)
patched_polfed(f!, x0, howmany, howmany_per_patch, target)
patched_polfed(f!, x0, howmany_per_patch, target)
full_polfed(f!, x0, howmany_per_patch)
```

### Operator path versus matrix path

They are genuinely different code paths, and it is worth knowing which one you are on:

| | operator method `polfed(f!, …)` | matrix method `polfed(H, …)` |
| :--- | :--- | :--- |
| Layout | **column-native** — your `f!` and `x0` are used as given | row layout; `x0` is copied with `permutedims` |
| Kernels | yours | POLFED builds value-grouped row kernels from `H` |
| Bounds | probed, unless you supply `Emin`/`Emax` | probed from `H` |
| Ownership | POLFED touches neither `f!` nor `x0` | POLFED owns the operator |

So the matrix method is a convenience layer that *builds an operator for you* and picks a
faster layout. If you bring your own `f!`, you keep control and POLFED stays out of the way.

### Give it the bounds if you have them

The first thing POLFED does with an unknown operator is a short both-ends Lanczos probe to
estimate `Emin`/`Emax`. If you already know them — analytically, or from a previous run —
skip it:

```julia
mc = MappingConfig(Emin = Emin, Emax = Emax)
result = polfed(f!, x0, 100, :maxdos; mc = mc)
```

You can also run the probe yourself:

```julia
Emin, Emax = UniPolfed.Numerics.Lanczos.lanczos_extrema(f!, x0; maxdim=1000)
```

### Supply the rescaled operator too

This is the one that actually matters for speed. POLFED works with
``\tilde H = (H - b)/a``, and unless you provide that action it has to rescale at runtime —
inside the hottest loop in the program, once per Clenshaw step.

The naive rescaling costs two extra passes over the data:

```julia
f!_rescaled_bad = (Y, X) -> begin
    f!(Y, X)
    @. Y *= 1 / spread
    @. Y -= (center / spread) * X
end
```

Better: fold the rescaling into your kernel, or — the cheap version of the same idea —
pre-build a rescaled matrix once and multiply by that:

```julia
spread = (Emax - Emin) / 2
center = (Emax + Emin) / 2

H_rescaled = copy(H)
H_rescaled .*= 1 / spread
H_rescaled += spdiagm(0 => fill(-center / spread, size(H, 1)))

mc = MappingConfig(
    f!_rescaled = (Y, X) -> mul!(Y, H_rescaled, X),
    Emin = Emin,
    Emax = Emax,
)

result = polfed(f!, x0, 100, :maxdos; mc = mc)
```

Note both `Emin` and `Emax` must be supplied alongside `f!_rescaled` — POLFED needs the same
``a`` and ``b`` you rescaled with in order to map results back to true energies.

A fully worked structured example, with benchmark numbers, is in
[Optimized kernels](@ref optimized_kernels).

### Custom Clenshaw kernels

For model-specific tuning you can replace the two Clenshaw steps themselves:

```julia
crr! = (b1, b2, b3, c, X) -> begin      # recurrence step
    tmp = similar(X)
    mul!(tmp, H_rescaled, b2)
    @. b1 = c * X + 2 * tmp - b3
    return nothing
end

cfs! = (b1, b2, c, Y, X) -> begin       # final sum
    tmp = similar(X)
    mul!(tmp, H_rescaled, b1)
    @. Y = c * X + tmp - b2
    return nothing
end

mc = MappingConfig(clenshaw_recurrence = crr!, clenshaw_finalsum = cfs!)
```

In production you would replace the bodies with fused kernels that avoid the temporary
allocation entirely; the integration point stays
[`MappingConfig`](@ref UniPolfed.MappingConfig). What the recurrence is doing mathematically
is on [On-the-fly mapping](@ref on_the_fly_mapping).

### Benchmark the kernel before you benchmark POLFED

Since the transform dominates the runtime, a faster `f!` is a faster POLFED — and you can
verify the first part in seconds without running the solver at all:

```julia
using BenchmarkTools

X = rand(size(H, 1)); Y = similar(X)
@btime mul!($Y, $H, $X)
@btime $f_custom!($Y, $X)
```

Doing it this way isolates the mapping, avoids run-to-run allocation noise, and tells you
whether a full POLFED comparison is even worth setting up.

### Checking which path you are on

Two lines of the report answer it:

- **Mapping → operator** says whether POLFED used your `f!_rescaled` or fell back to rescaling at
  runtime. If you passed one and it is not reflected here, check that you also passed `Emin` and
  `Emax`.
- **Mapping → layout** reads `ColumnLayout` on the operator path. Seeing `RowLayout` means you
  are on the matrix path after all.

Comparing a plain `mul!` wrapper against a structured kernel, the matrix-multiplication count
should be **identical** and only the timings should move. A changed count means the two runs did
not solve the same problem — most likely the spectral bounds differ.

## Filter sizing and introspection

### [`expected_krylov_dim`](@id UniPolfed.expected_krylov_dim)

```julia
expected_krylov_dim(filter, howmany, block_size, eta)
```

Estimate the Krylov-space dimension for a concrete filter. `eta` is the budget-inflation
factor supplied as `overestimate_iters`.

The delta family shares one empirical fit for the **iteration count**, evaluated at the
eigenpairs-per-block ratio ``x = \texttt{howmany}/b`` and multiplied back by the block size:

```math
m \;=\; \Bigl\lceil A\,x^{\alpha}\Bigl(1 + \frac{x}{x_c}\Bigr)^{1-\alpha} b\,\eta \Bigr\rceil ,
\qquad A = 16.725632,\; \alpha = 0.40598213,\; x_c = 50.265139 ,
```

with the constants in `Defaults.filters.<name>.krylov_fit_A` / `_alpha` / `_xc`.
[`DoubleDeltaFilter`](@ref double_filter) applies the same fit inflated by
`Defaults.double_polfed.krylov_overhead`; [`EdgeFilter`](@ref filters) keeps its own
(untuned) linear estimate.

### [`order_safety_factor`](@id UniPolfed.order_safety_factor)

```julia
order_safety_factor(filter)
```

The safety factor applied when resolving a filter's polynomial order. Fixed-order filters
return `1.0`.

## Mapping-optimization helpers

Used by the matrix path; exposed for callers building their own kernels. They live in
`UniPolfed.Optimization`.

### [`classify`](@id UniPolfed.Optimization.classify)

```julia
classify(H; tol=1e-12, round_digits=12)
```

Analyze a structured matrix and return its value-grouped representation, a
`MappingStructure`.

### [`resolve_mapping_kernel`](@id UniPolfed.Optimization.resolve_mapping_kernel)

```julia
resolve_mapping_kernel(choice, H, Val(block_size); secs=0.5)
```

Resolve `true`/`:auto`, a kernel name, or a `(kernel = …, …)` named tuple into
a concrete kernel plus tuning parameters.

### [`build_optimized_mapping`](@id UniPolfed.Optimization.build_optimized_mapping)

```julia
build_optimized_mapping(H, choice, Val(block_size), a, b; secs=0.5)
```

Build the optimized row-layout action of ``\tilde H = (H - b)/a``. Returns a named tuple
containing `f!_rescaled`, the selected kernel id, and its tuning parameters.

## Low-level Lanczos utilities

### [`lanczos`](@id UniPolfed.Numerics.Lanczos.lanczos)

```julia
lanczos(A_or_f!, x0, target; kwargs...)
lanczos(A_or_f!, x0, n; which=:SR, kwargs...)
```

Run the Hermitian Lanczos or block-Lanczos eigensolver directly, with no filtering. Use
`LowestEigs(n)`, `HighestEigs(n)` or `BothEnds(n_low, n_high)` as explicit targets, or the
integer convenience form with `which=:SR` / `:LR`. Returns a `LanczosResult`.

### [`lanczos_extrema`](@id UniPolfed.Numerics.Lanczos.lanczos_extrema)

```julia
lanczos_extrema(H; x0=nothing, maxdim=100, power_iters=3)
lanczos_extrema(f!, x0; maxdim=100, power_iters=3)
```

Estimate the smallest and largest eigenvalues in one both-ends run — the same probe POLFED
uses for `Emin`/`Emax`. Pass `return_result=true` for the full `LanczosResult` instead of
the `(Emin, Emax)` pair.

## A full call with explicit configuration

```julia
using UniPolfed: polfed, MappingConfig, DoSConfig, BlockWise
using UniPolfed.Models: qsun_hamiltonian
using LinearAlgebra

H  = qsun_hamiltonian(12, 2, 1.0, 0.5; use_sparse=true)
x0 = normalize!(rand(size(H, 1)))
f!(Y, X) = mul!(Y, H, X)

result = polfed(
    f!, x0, 100, 0.0;
    mc   = MappingConfig(parallel_strategy = BlockWise()),
    dosc = DoSConfig(N = 200, R = 300),
    tol  = 1e-15,
    eigentol = 1e-10,
)

vals, vecs, report = result
```
