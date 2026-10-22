# [Patched POLFED](@id tutorial_patched)

## The problem it solves

It is **not** the polynomial order. Recall the order scaling from [Filters](@ref filters):
for a target interval of half-width ``\delta``,

```math
K \approx \frac{2.655\sqrt{1-\tilde\lambda^{2}}}{\delta}.
```

A large `howmany` means a *wide* interval, so ``\delta`` is large and ``K`` is **small**.
Asking for a lot of eigenpairs makes the filter cheap, not expensive.

The bottleneck moves somewhere else entirely: **the projected Lanczos matrix**. Its size
grows with `howmany` — sublinearly at small counts, essentially linearly once a block holds
more than ~50 eigenpairs,

```math
m \;\approx\; A\,x^{\alpha}\Bigl(1 + \frac{x}{x_c}\Bigr)^{1-\alpha} b\,\eta ,
\qquad x = \frac{\texttt{howmany}}{b},
```

with ``A = 16.726``, ``\alpha = 0.406``, ``x_c = 50.265`` — sublinear while a block holds fewer
than ``x_c`` eigenpairs, then straightening out to about ``1.63`` extra dimensions per eigenpair,

and it is diagonalized **densely**, at ``\mathcal{O}(m^3)``. Ask for 10 000 eigenpairs and
``m`` lands around 20 000 — perfectly possible, but the dense diagonalization of that matrix
now dominates the entire run, and the Krylov basis holding ``m`` vectors of length
``\mathcal{D}`` dominates the memory.

So the trade is the other way around from what you might expect:

| | one big run | patched into 3 runs |
| :--- | :--- | :--- |
| polynomial order ``K`` | low | **higher** per patch |
| projected matrix ``m`` | ~20 000 | ~6 900 each |
| dense diagonalization | ``\mathcal{O}(20000^3)`` | ``3\times\mathcal{O}(6900^3)`` — ~9× less work |
| Krylov basis in memory | one huge basis | one small basis at a time |

Patched POLFED **pays a higher polynomial order in exchange for a much smaller projected
problem**, by splitting the request into several independent, slightly overlapping runs —
3 × 3333 eigenpairs instead of 1 × 10 000. Each run's diagonalization is cheap and its basis
fits comfortably, and the extra matvecs from the higher ``K`` cost far less than the cubic
term they replace.

## Edge-first

The decomposition is not a plain left-to-right tiling. Each spectrum edge (``\pm 1``) is
peeled **first**, with an [`EdgeFilter`](@ref filters) and a *count* target, and only then
is the strict interior tiled with delta-filter window patches.

The reason is stage ①: `Emin` and `Emax` are **estimates**. A true extreme eigenvalue can
land exactly on ``\pm 1`` or a hair beyond, and a closed interval bound would silently drop
it. A count target ("give me the lowest ``n``") does not care where ``\pm 1`` actually fell,
and the `EdgeFilter` is monotone, so it cannot leak. Robustness where the geometry is
least trustworthy.

```text
   ├── edge peel ──┤├─ patch ─┤├─ patch ─┤├─ patch ─┤├── edge peel ──┤
  -1                        interior                                +1
   EdgeFilter                delta filters                  EdgeFilter
   count target              window targets                 count target
```

## Calling it

Two modes, distinguished by arity — exactly like [`polfed`](@ref Polfed.polfed):

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

H  = qsun_hamiltonian(12, 2, 1.0, 0.5; use_sparse = true)
x0 = normalize!(rand(size(H, 1)))

# count mode: 10_000 eigenpairs near a point, in patches of ~3333
vals, vecs, report = patched_polfed(H, x0, 10_000, 3333, :maxdos)

# window mode: everything in an interval, in patches of ~3333
vals, vecs, report = patched_polfed(H, x0, 3333, (-0.5, 0.5))
```

The return is a [`PatchedResult`](@ref Polfed.PatchedResult), unpacking as
`(vals, vecs, report)` exactly like [`polfed`](@ref Polfed.polfed) — only the report type
differs, because a patched run is N+2 separate solves with no single convergence history.
Use `eigenvector(result, i)` for individual eigenvectors: in **window mode** `vecs` is not a
matrix at all but a vector of independently owned vectors, so each completed patch matrix can be
released as the run proceeds.

`howmany_per_patch` is the tuning knob, and it sets exactly the trade above. Smaller patches
mean a smaller projected matrix per run — the thing you came here to shrink — but a
**higher** polynomial order per patch, more patches, and more per-patch setup and Krylov
overhead. Larger patches push you back toward the single-run cost you were escaping. A few
thousand is a reasonable starting point; tune it by watching where the *Timings* balance
between matvecs and the dense diagonalization.

## The extra keywords

```julia
vals, vecs, report = patched_polfed(H, x0, 3333, (-0.5, 0.5);
    edge_distance = 0.2,     # how far from ±1 the edge peels reach
    howmany_pad   = 3,       # overlap between neighbouring patches
    max_retries   = 3,       # coverage retries per patch
    retry_growth  = 2.0,     # budget multiplier on each retry
)
```

`mc`, `dosc`, `tol`, `eigentol` and `overestimate_iters` are accepted too, and mean exactly what
they mean in [`polfed`](@ref Polfed.polfed).

## How it avoids returning garbage

Splitting a spectrum into pieces and gluing the results back together is where this kind of
algorithm normally goes wrong. Two layers guard against it:

**1. Nothing uncertified gets in.** Each interior patch uses a one-sided `Window` target
with `return_brackets=false`, so only pairs that are genuinely inside the passband come
back — the out-of-passband bracketing pairs that Lanczos produces at the ends of a window
are discarded rather than reported. Chebyshev tails never leak into your results.

**2. Gaps are detected and retried.** If a patch fails to bracket the interval it owns, that
patch is re-run with both a wider overlap (`howmany_pad`) and a bigger budget
(`overestimate_iters × retry_growth`), up to `max_retries` times. When the tolerance is
achievable this recovers cleanly. When it is not, POLFED **warns and returns the partial
result** — it never crashes and never silently pretends a gap is not there.

Finally, eigenvalues found by both an edge peel and its neighbouring interior patch are
deduplicated at the end, so seam states appear exactly once.

## Reading the output

The third return value is a [`PatchedReport`](@ref Polfed.PatchedReport), which
aggregates one [`PatchReport`](@ref Polfed.PatchReport) per run:

```julia
display_report(report)
```

Sections:

- **Decomposition** — how many runs, a histogram of filter types, polynomial order
  min/max/mean, and the aggregate cost ``\sum_p K_p \cdot bs_p \cdot iters_p``. This is
  where you see whether `howmany_per_patch` was sensible.
- **Mapping** — as in a standard report.
- **Convergence** — aggregated across all runs. The number quoted is the worst **eigen**
  residual, not the pool's worst Ritz residual.
- **Timings** — aggregate.

```julia
display_report(report; convergence_details = true)
```

adds a per-run table: kind, filter, order, interval, budget, coverage and retries for every
patch.

!!! note "Kind is not filter type"
    A run's **kind** (`:edge_lowest`, `:edge_highest`, `:interior`) is not the same as its
    **filter type**. An interior patch that happens to sit near ``\pm 1`` can legitimately
    be assigned an `EdgeFilter`, so in the histogram the `EdgeFilter` count is always
    ``\geq`` the number of edge peels. This is expected, not a bug.

### What to look at first

| Symptom in the report | Likely reading |
| :--- | :--- |
| retries > 0 on several patches | patches are too narrow, or `eigentol` is too tight for the budget |
| order min/max spread very wide | the DoS varies a lot across the window — dense regions get narrow patches and high orders. Expected, not a fault |
| a coverage warning | there is a gap — the returned set is incomplete, and the report says where |
| aggregate cost dominated by a few patches | those sit in the dense part of the spectrum |

## Next

- [Full POLFED](@ref tutorial_full) — the same machinery over the entire spectrum.
- [Guidelines](@ref guidelines) — choosing between `polfed` and `patched_polfed`.
