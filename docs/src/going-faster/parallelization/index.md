# [Parallelization](@id parallelization)

Lever 2 from [The bottleneck](@ref going_faster): apply the operator to many things at once.
The strategy is set through [`MappingConfig`](@ref UniPolfed.MappingConfig).

!!! warning "`VectorWise` does not run yet"
    Above one thread, `VectorWise` is currently downgraded to `BlockWise` with a warning —
    an explicit request as much as an automatic one — because the Clenshaw combine opens its
    own `Threads.@threads :static` region inside `VectorWise`'s, and Julia rejects nested
    `:static` regions. On the **matrix** entrypoints it is coerced unconditionally for a
    second, independent reason: the transform operator is built whole-block at `Val(bs)` and
    shared with the DoS stage.

    The strategy itself is implemented and layout-aware; only its *selection* is blocked. Until
    that clears, every run is `BlockWise`, and the rest of this page describes `VectorWise` as
    designed rather than as currently reachable.

## The strategies

Both strategies use every thread. They differ in **where the parallel region opens** relative
to the Clenshaw recurrence that evaluates the filter.

| Strategy | Parallel region | Typical use |
| :--- | :--- | :--- |
| [`BlockWise`](@ref UniPolfed.BlockWise) | **inside** each Clenshaw step — the block goes to the operator in one call and is split across threads there, one region per step | GPU runs; single vectors; anything where the block is not a multiple of the thread count |
| [`VectorWise`](@ref UniPolfed.VectorWise) | **around** the whole recurrence — one Clenshaw run per Krylov vector, each with its own buffers | block Lanczos where the block size equals the thread count |

```julia
using UniPolfed

mc_block  = MappingConfig(parallel_strategy = BlockWise())
mc_vector = MappingConfig(parallel_strategy = VectorWise())

result = polfed(H, x0, howmany, target; mc = mc_block)
display_report(result)
```

The trade is synchronization against sharing. `VectorWise` shares nothing between threads, so
it needs no per-step barrier and opens exactly one parallel region for the whole ``K``-step
loop. `BlockWise` opens one per step and pays a barrier each time, but it is the only option
that parallelizes a **single** Krylov vector, because there is nothing else to split.

!!! note "Neither strategy is serial"
    `BlockWise` means POLFED hands the operator the whole block — not that the work runs on one
    thread. The optimized row kernels chunk their own output rows across `Threads.nthreads()`,
    and on the GPU the device owns the decomposition. With `BlockWise` and `julia -t 4` the
    matvec still uses four threads.

### `BlockWise(; groups)`

`groups` cuts the block into that many Krylov-vector groups, each served by
`nthreads() ÷ groups` threads splitting its basis-state range. `groups = 1` is the pure
basis-state split — the only parallelism available at block size 1.

!!! warning "`groups` is not read yet"
    The field is validated at construction, but the block decomposition that consumes it is not
    implemented. Until it lands, every `BlockWise` behaves as a plain whole-block call
    regardless of `groups`.

## Automatic selection

Leaving `parallel_strategy = nothing` resolves the strategy from the processing unit, the block
size, the thread count and whether the mapping is optimized:

The branches are checked in this order, and the first match wins:

| Condition | Resolved |
| :--- | :--- |
| GPU | `BlockWise` — there is no device `VectorWise` kernel; an explicit request is coerced with a warning |
| an explicit `parallel_strategy` | that strategy — but still filtered by the downgrade above |
| `block_size == 1` | `BlockWise` — `VectorWise` has a single vector and nothing to split |
| `optimize_mapping === false` | `BlockWise` — the generic `mul!` takes its own MKL threads |
| `optimize_mapping === true` **and** `block_size == nthreads()` | `VectorWise` — one whole vector per thread |
| anything else | `BlockWise` |

Mind the two different tests. The `BlockWise` short-circuit is `=== false`, so `:auto`, a
`NamedTuple` and an explicit kernel all count as optimized and fall past it. The `VectorWise`
branch is the stricter `=== true`, so those same values do **not** reach it — they land on
`anything else`.

## Vector parallelism needs vectors

`VectorWise` splits the Krylov block, so with a *vector* `x0` there is exactly one vector and
nothing to split. Vector-level parallelism and block Lanczos are the same decision seen from
two sides:

```julia
x0_vec = normalize!(rand(size(H, 1)))            # → Lanczos, 1 vector
x0_mat = Matrix(qr(rand(size(H, 1), 8)).Q)       # → block Lanczos, 8 vectors
```

If you are running plain Lanczos and want threads to help, use `BlockWise` — it parallelizes
inside the kernel instead.

!!! note "The split follows the layout"
    `VectorWise` splits along whichever axis holds the Krylov vectors: **columns** under
    [`ColumnLayout`](@ref UniPolfed.Numerics.Lanczos.ColumnLayout) (each thread takes a
    length-`D` column) and **rows** under
    [`RowLayout`](@ref UniPolfed.Numerics.Lanczos.RowLayout), which is what the matrix
    entrypoint uses (each thread takes a `1 × D` row). A row-layout block is therefore split
    into `bs` vectors of length `D`, never into `D` fragments of length `bs`.

!!! note "`VectorWise` serializes BLAS"
    Both CPU `VectorWise` paths set `BLAS.set_num_threads(1)` around the call and restore it
    afterwards, so the optimized kernels own the threads rather than competing with BLAS for
    them. `BlockWise` does not.

## Choosing a block size

Bigger is not better. The block widens the mapping, but it also enlarges the projected problem:
the estimated Krylov dimension grows **linearly** in the block size,

```julia
expected_krylov_dim(filter, howmany, blocksize, η) =
    ceil(Int, (krylov_iters_per_block * blocksize + krylov_iters_per_eig * howmany) * η)
```

where `η` is `overestimate_iters` and the two coefficients come from the filter's own defaults
(`Defaults.filters.delta_up.krylov_iters_per_block`, …). An oversized block therefore costs you
more iterations *and* more memory for the Krylov basis, which is usually the dominant memory
consumer.

The working rules:

```math
\frac{\texttt{howmany}}{\texttt{block\_size}} \gtrsim 100,
\qquad \texttt{block\_size} \le 16 .
```

Push the ratio much below 100 and block factorization typically needs many more iterations to
converge — enough to erase the mapping gain and then some.

There is now a second reason to care about the block size: it is what selects `VectorWise`
automatically. A block size equal to your thread count is the configuration that strategy is
built for.

## [Choosing a strategy](@id parallelization_reasoning)

- **GPU** → `BlockWise`. The device provides its own parallelism; host-side decomposition only
  gets in the way. Already the default, but setting it explicitly documents intent.
- **Single vector** → `BlockWise`. It is the only strategy that can parallelize one vector.
- **Block Lanczos with `block_size == nthreads()`** → `VectorWise`. One vector per thread, no
  per-step barrier. This is what auto-resolution *selects* — but the downgrade then takes it
  away again, so today this configuration also runs `BlockWise`.
- **Block Lanczos with any other block size** → `BlockWise`.

Raising `howmany` is also a lever here — it lowers the polynomial order — but it raises memory
use and makes the projected diagonalization more expensive. See [Guidelines](@ref guidelines).

## Reading the output

The **Mapping** section of the report echoes the resolved strategy, the block size and the
layout — check it first, since a strategy that could not be applied falls back.

The **Timings** section is where the answer is. Compare wall time against CPU time: for a
threaded run, CPU time well above wall time means threads are actually working. If they track
each other, your parallelism is not engaging — usually a block with too few vectors.

Between two configurations of the same problem the matrix-multiplication count must be
identical — parallelization changes *when* the work happens, never how much of it there is. A
changed count means something else moved with it, and the runs are not comparable.

!!! note "Comparing the two strategies"
    A loop over `BlockWise()` and `VectorWise()` does not currently measure anything: both
    runs resolve to `BlockWise`, so the difference you see is noise. Check the **Mapping**
    section's `parallel strategy` line to confirm which one actually ran.

!!! note "CPU time on GPU runs"
    CPU time is suppressed in the report for GPU runs, where it would be misleading.

## Next

- [GPU](@ref going_faster_gpu) — where `BlockWise` is the right answer.
- [Guidelines](@ref guidelines) — the tuning rules in one place.
