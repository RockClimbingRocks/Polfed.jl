# [Configs and Targets](@id api_configs_targets)

Configuration structs, execution-strategy types, and the accepted `target` spellings.
Mapping configuration is passed as `mc`; density-of-states configuration as `dosc`.
Factorization tolerances (`tol`, `eigentol`, `overestimate_iters`) are direct keywords on
the entrypoints. The transform and the filter are resolved by the pipeline and are not user
configuration.

## [`MappingConfig`](@id UniPolfed.MappingConfig)

```julia
MappingConfig(;
    optimize_mapping=false,
    f!_rescaled=nothing,
    clenshaw_recurrence=nothing,
    clenshaw_finalsum=nothing,
    parallel_strategy=nothing,
    Emin=nothing,
    Emax=nothing,
)
```

Controls operator rescaling and how the transform is applied.

- `optimize_mapping` — selects the matrix row kernel. Accepts `false`, `true`/`:auto`, a
  kernel name, or a `(kernel = …, …)` named tuple carrying that kernel's own parameters.
  `true` is the supported entry point; the individual kernels are internal.
- `f!_rescaled` — supply the already-rescaled operator action ``\tilde H = (H-b)/a``
  yourself. Requires `Emin` and `Emax` to be given as well.
- `clenshaw_recurrence`, `clenshaw_finalsum` — supply custom Clenshaw kernels.
- `parallel_strategy` — `nothing` selects the backend-dependent default
  (resolved from the processing unit, block size and thread count — see
  [Parallelization](@ref parallelization)).
- `Emin`, `Emax` — supply spectral bounds and skip the preliminary Lanczos probe.

The struct is immutable; the pipeline resolves it into a concrete mapping plan without
mutating your configuration.

Usage: [Bring your own operator](@ref operator_interface),
[Optimized kernels](@ref optimized_kernels).

## [`DoSConfig`](@id UniPolfed.DoSConfig)

```julia
DoSConfig(; N=250, R=300, kernel=:Jackson)
```

Controls the kernel-polynomial density-of-states estimate.

- `N` — number of Chebyshev moments. Higher `N` gives finer spectral resolution.
- `R` — number of stochastic probe vectors. Higher `R` reduces noise.
- `kernel` — the damping kernel used to suppress Gibbs oscillations.

Because `:maxdos` is defined as the peak of this estimate, `N` and `R` can shift that
target. The mathematics is on [The workflow](@ref how_polfed_works).

## [Targets](@id api_targets)

Count-based calls (`polfed(…, howmany, target)`) accept point targets:

| Spelling | Meaning |
| :--- | :--- |
| `:maxdos` | the rescaled energy where the DoS is maximal |
| `:middle` | center of the unrescaled spectrum; identical to `(:rescaled, 0.0)` |
| `(:offset, frac)` | interpolate from the DoS maximum toward an edge, `frac ∈ [-1, 1]` |
| `(:unrescaled, E)` or plain `E::Real` | a true energy |
| `(:rescaled, e)` | an already-rescaled energy, `e ∈ [-1, 1]` |

Window calls (`polfed(…, target_window)`) accept:

| Spelling | Meaning |
| :--- | :--- |
| `(:unrescaled, (Elo, Ehi))` or plain `(Elo, Ehi)` | true-energy interval |
| `(:rescaled, (elo, ehi))` | rescaled interval |

`full_polfed` takes no target at all. Worked comparison of the point targets:
[Target modes](@ref polfed_targets).

## Parallelization strategies

### [`Parallelization`](@id UniPolfed.Parallelization)

Abstract supertype for mapping and transform execution strategies.

### [`BlockWise`](@id UniPolfed.BlockWise)

```julia
BlockWise(; groups = nothing)
```

Parallelize **inside** each Clenshaw step: the whole `bs × D` block goes to the operator in one
call and is split across threads there, opening one parallel region per step. The effective
strategy on GPU arrays, whose kernels provide their own decomposition, and the only strategy
that parallelizes a single Krylov vector.

`groups` cuts the block into that many Krylov-vector groups, each served by
`nthreads() ÷ groups` threads splitting its basis-state range; `groups = 1` is the pure
basis-state split. `nothing` resolves it automatically. Values below `1` throw an
`ArgumentError` at construction.

!!! warning
    `groups` is validated but **not yet read** — the block decomposition that consumes it is
    unimplemented, so every `BlockWise` currently behaves as a plain whole-block call.

### [`VectorWise`](@id UniPolfed.VectorWise)

```julia
VectorWise()
```

Parallelize **around** the recurrence: run Clenshaw once per Krylov vector, each with its own
buffer triplet. Nothing is shared between threads, so there is no per-step barrier and one
parallel region covers the whole `K`-step loop. Gives no parallelism at block size 1 — use
[`BlockWise`](@ref UniPolfed.BlockWise) there.

Currently splits by columns only; row splitting under `RowLayout` is not implemented yet. Not
supported on the GPU, where an explicit request is coerced to `BlockWise` with a warning.

Choosing between them: [Parallelization](@ref parallelization).

## [Processing units](@id docs_processing_units)

### [`ProcessingUnit`](@id UniPolfed.ProcessingUnit)

Abstract backend descriptor used by allocation and storage dispatch.

### [`CPU`](@id UniPolfed.CPU)

```julia
CPU()
```

The CPU allocation and array adapter.

### [`GPU`](@id UniPolfed.GPU)

```julia
GPU()
```

The CUDA adapter. Construction throws an informative error when the CUDA extension is
unavailable or nonfunctional.

## Lanczos reorthogonalization and basis storage

### [`ReOrthTechnique`](@id UniPolfed.Numerics.Lanczos.ReOrthTechnique)

Abstract supertype for reorthogonalization strategies.

### [`FullRO`](@id UniPolfed.Numerics.Lanczos.FullRO)

Reorthogonalize on every iteration. The default.

### [`PartialRO`](@id UniPolfed.Numerics.Lanczos.PartialRO)

`PartialRO(skip)` reorthogonalizes only on selected iterations.

### [`SelectiveRO`](@id UniPolfed.Numerics.Lanczos.SelectiveRO)

Reserved for a selective scheme; **not yet implemented**.

### [`OrthonormalBasis`](@id UniPolfed.Numerics.Lanczos.OrthonormalBasis)

Abstract supertype for Krylov-basis storage.

### [`MatrixBasis`](@id UniPolfed.Numerics.Lanczos.MatrixBasis)

Store all basis vectors in one dense matrix. Supports both block layouts. The default.

### [`VectorBasis`](@id UniPolfed.Numerics.Lanczos.VectorBasis)

Store the basis as a vector of vector or block arrays.

## Block layout

### [`Layout`](@id UniPolfed.Numerics.Lanczos.Layout)

Abstract trait selecting the orientation of a Krylov block.

### [`ColumnLayout`](@id UniPolfed.Numerics.Lanczos.ColumnLayout)

Vectors as columns (``D \times \texttt{block\_size}``); the operator is applied as `H * X`.
This is what the operator entrypoints use.

### [`RowLayout`](@id UniPolfed.Numerics.Lanczos.RowLayout)

Vectors as rows (``\texttt{block\_size} \times D``); the operator is applied as `X * H`.
This is what the matrix entrypoints use.
