# [The bottleneck](@id going_faster)

Before tuning anything, know what you are tuning. POLFED's runtime is, to a very good
approximation, one product:

```math
\text{total cost} \;\approx\;
\underbrace{(\#\ \text{Lanczos iterations})}_{\text{Krylov space}}
\times
\underbrace{K}_{\text{filter order}}
\times
\underbrace{\mathrm{cost}(H\cdot v)}_{\text{your kernel}} .
```

Every Lanczos iteration applies the filter once, and each filter application is ``K`` sparse
matvecs through the Clenshaw recurrence (see
[On-the-fly mapping](@ref on_the_fly_mapping)). Nothing else in the pipeline comes close —
the bounds probe, the DoS estimate and the filter resolution are all one-off costs measured
in a handful of matvecs.

So there are exactly three levers.

## Lever 1 — make ``H\cdot v`` cheaper

The single most productive direction, because it multiplies against everything else.
POLFED is **memory-bound**, not FLOP-bound: for a sparse many-body Hamiltonian the limit is
how fast you can stream indices and values, not how fast you can multiply.

- **[Optimized kernels](@ref optimized_kernels)** — if your Hamiltonian has few distinct
  offdiagonal values (most spin models do), `optimize_mapping=true` builds value-grouped row
  kernels that stop re-reading what they already know. Or write your own.
- **[GPU](@ref going_faster_gpu)** — the workload is data-parallel and bandwidth-hungry,
  which is exactly what GPUs are for. The API barely changes.

## Lever 2 — do the products in parallel

**[Parallelization](@ref parallelization)** — spread the work across threads, either inside
each Clenshaw step or around the whole recurrence. Block Lanczos interacts strongly with this:
a block exposes independent Krylov vectors for threads to work on, and the block size is what
selects between the two strategies — which is why factorization choice and parallelization
strategy are decided together.

## Lever 3 — make ``K`` smaller

``K`` is set by the geometry, ``K \approx 2.655\sqrt{1-\tilde\lambda^2}/\delta``, so a
**narrow** window is the expensive one. The lever is therefore:

- **Raise `howmany`.** Counter-intuitive but often correct: asking for *more* eigenpairs
  widens the interval, which **lowers** ``K`` and can make the whole run faster.

It is not free, and the thing it costs you is the subject of the next section.

## The other bottleneck: the projected matrix

Lowering ``K`` by raising `howmany` runs into a second cost that has nothing to do with
matvecs. The Krylov space grows with `howmany` — sublinearly at small counts, essentially
linearly once a block holds more than ~50 eigenpairs,

```math
m \;\approx\; A\,x^{\alpha}\Bigl(1 + \frac{x}{x_c}\Bigr)^{1-\alpha} b\,\eta ,
\qquad x = \frac{\texttt{howmany}}{b},
```

with ``A = 16.726``, ``\alpha = 0.406``, ``x_c = 50.265`` — sublinear while a block holds fewer
than ``x_c`` eigenpairs, then straightening out to about ``1.63`` extra dimensions per eigenpair,

and the projected Lanczos matrix of size ``m`` is diagonalized **densely**, at
``\mathcal{O}(m^3)``. At `howmany = 10 000`, ``m`` is around 20 000 and that cubic term —
not the filter — becomes the dominant cost of the run. The Krylov basis, ``m`` vectors of
length ``\mathcal{D}``, becomes the dominant memory cost at the same time.

**[Patched POLFED](@ref tutorial_patched)** exists for exactly this regime: split the
request into several independent, overlapping runs (3 × 3333 rather than 1 × 10 000). Each
patch has a narrower window and therefore a **higher** ``K``, which is the price — but its
projected matrix is a third of the size, so the cubic term drops by roughly 9×, and only one
small basis is in memory at a time.

So the two levers pull against each other, and which one binds depends on where you are:

| Regime | What dominates | What to do |
| :--- | :--- | :--- |
| few eigenpairs, narrow window | large ``K``, matvecs | raise `howmany`; levers 1 and 2 |
| many eigenpairs, wide window | dense diagonalization of ``m``, memory | split with `patched_polfed` |

## Measure first

Every run tells you where its time went:

```julia
result = polfed(H, x0, howmany, target)
display_report(result)
```

Read in this order:

1. **Spectral Transformation → polynomial order and matvec count.** If ``K`` is large, you
   have a lever-3 problem, and no amount of kernel tuning will rescue it.
2. **Timings → Lanczos versus preparation.** Preparation should be a small fraction. If it
   is not, your bounds probe or DoS settings are doing too much work.
3. **Timings inside Lanczos → matvecs versus reorthogonalization.** If matvecs dominate,
   go to levers 1 and 2. If reorthogonalization dominates, your Krylov space is too big —
   look at `howmany` and `block_size`.

Then change one thing and compare reports. The matrix-multiplication count should be
identical between two runs that solve the same problem; if it moved, you changed the
problem, not just its speed.

## A rough order of operations

1. `optimize_mapping=true` if your matrix is structured — one keyword, often a large win.
2. Check block size and parallel strategy against [Guidelines](@ref guidelines).
3. Move to GPU if you have one.
4. Write a structured kernel, and give POLFED the *rescaled* action too.
5. If `howmany` is large enough that the projected diagonalization or the Krylov basis
   dominates, switch to `patched_polfed`.
