# [Parallelization](@id parallelization)

Lever 2 from [The bottleneck](@ref going_faster): apply the operator to many things at once.
The strategy is set through [`MappingConfig`](@ref Polfed.MappingConfig).

## The strategies

Both strategies use every thread. They differ in **where the parallel region opens** relative
to the Clenshaw recurrence that evaluates the filter.

| Strategy | Parallel region | Typical use |
| :--- | :--- | :--- |
| [`BlockWise`](@ref Polfed.BlockWise) | **inside** each Clenshaw step — the block goes to the operator in one call and is split across threads there, one region per step | GPU runs; single vectors; anything where the block is not a multiple of the thread count |
| [`VectorWise`](@ref Polfed.VectorWise) | **around** the whole recurrence — one Clenshaw run per Krylov vector, each with its own buffers | block Lanczos where the block size equals the thread count |

```julia
using Polfed

mc_block  = MappingConfig(parallel_strategy = BlockWise())
mc_vector = MappingConfig(parallel_strategy = VectorWise())

result = polfed(H, x0, howmany, target; mc = mc_block)
display_report(result)
```

The trade is synchronization against sharing. `VectorWise` shares nothing between threads, so
it needs no per-step barrier and opens exactly one parallel region for the whole ``K``-step
loop. `BlockWise` opens one per step and pays a barrier each time, but it is the only option
that parallelizes a **single** Krylov vector, because there is nothing else to split.

### What a Clenshaw step runs on the matrix entrypoints

`polfed(H, …)` always runs `BlockWise`, and evaluates the filter so that a step never switches
from one thread pool to another — measured on 16 threads, that switch cost 2–8× the step itself:

- `optimize_mapping = false`: each step is one accumulating MKL product
  (`Y ← 2·X·H̃ − Y`) plus one MKL `axpy`, all on MKL's own threads.
- an optimized kernel with a serial range form (every auto-selected kernel except
  `row_bucketed` at block size 1): the whole ``K``-step recurrence runs in **one** Julia parallel
  region, the tasks meeting at spin barriers.

During the filtered Lanczos runs MKL's idle threads are told to sleep at once (Intel OpenMP
blocktime `0`, restored afterwards), so they do not spin on the cores the next filter application
needs. `polfed(f!, …)` keeps the per-step combine, since POLFED cannot see inside your `f!`.

### `BlockWise(; groups)`

!!! warning "`groups` is not read yet"
    The field is validated at construction, but the block decomposition that consumes it is not
    implemented. Until it lands, every `BlockWise` behaves as a plain whole-block call
    regardless of `groups`.

`groups` cuts the block into that many Krylov-vector groups, each served by
`nthreads() ÷ groups` threads splitting its basis-state range. `groups = 1` is the pure
basis-state split — the only parallelism available at block size 1.

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
    [`ColumnLayout`](@ref Polfed.Numerics.Lanczos.ColumnLayout) (each thread takes a
    length-`D` column) and **rows** under
    [`RowLayout`](@ref Polfed.Numerics.Lanczos.RowLayout), which is what the matrix
    entrypoint uses (each thread takes a `1 × D` row). A row-layout block is therefore split
    into `bs` vectors of length `D`, never into `D` fragments of length `bs`.

## Choosing a block size

Bigger is not better 😉. The block widens the mapping, but it also enlarges the projected problem.
The estimate is a fit for the number of **iterations**, evaluated at the eigenpairs-per-block
ratio `x = howmany / blocksize` and multiplied back by the block size to give a dimension:

```julia
expected_krylov_dim(filter, howmany, blocksize, η) =
    ceil(Int, A * x^α * (1 + x / xc)^(1 - α) * blocksize * η)   # x = howmany / blocksize
```

where `η` is `overestimate_iters` and the three constants come from the filter's own defaults
(`Defaults.filters.delta_up.krylov_fit_A`, `…_alpha`, `…_xc` — `16.726`, `0.406`, `50.265` for
the whole delta family). Because the fit is sublinear in `x`, halving `x` by doubling the block
does **not** halve the iteration count, so the dimension still climbs with the block size — at
`howmany = 1000`, `η = 1`, it is `1681` at `blocksize = 1`, `2080` at `10` and `2318` at `16`.
An oversized block therefore costs you more matvecs *and* more memory for the Krylov basis,
which is usually the dominant memory consumer.

The working rules:

```math
\frac{\texttt{howmany}}{\texttt{block\_size}} \gtrsim 100 .
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

## Next

- [GPU](@ref going_faster_gpu) — where `BlockWise` is the right answer.
- [Guidelines](@ref guidelines) — the tuning rules in one place.
