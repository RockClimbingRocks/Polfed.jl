# [Configs and Parallelization Types](@id docs_parallel_types)

This page documents the current configuration structs and execution strategy
types used by [`polfed`](@ref UniPolfed.polfed). Pass mapping configuration as
the `mc` keyword and density-of-states configuration as `dosc`. Factorization
tolerances and Krylov-budget inflation are direct `polfed` keywords (`tol`,
`eigentol`, and `overestimate_iters`); the transform and filter are resolved by
the pipeline.

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

`MappingConfig` controls operator rescaling and transform application:

- `optimize_mapping` selects the matrix row kernel. It accepts `false`,
  `true`/`:auto`, a kernel symbol, or a `(kernel=..., G=..., W=...)` named
  tuple.
- `f!_rescaled` optionally supplies the already-rescaled operator action.
- `clenshaw_recurrence` and `clenshaw_finalsum` optionally supply custom
  Clenshaw kernels.
- `parallel_strategy=nothing` selects the backend-dependent default.
- `Emin` and `Emax` optionally provide spectral bounds and avoid estimating
  them with a preliminary Lanczos run.

The struct is immutable. The pipeline resolves it into a concrete mapping plan
without mutating the user's configuration.

## [`DoSConfig`](@id UniPolfed.DoSConfig)

```julia
DoSConfig(; N=250, R=300, kernel=:Jackson)
```

`DoSConfig` controls the kernel-polynomial density-of-states estimate:

- `N` is the number of Chebyshev moments.
- `R` is the number of stochastic probes.
- `kernel` selects the damping kernel.

## Parallelization Strategy Types

### [`Parallelization`](@id UniPolfed.Parallelization)

Abstract supertype for mapping and transform execution strategies.

### [`NoParallel`](@id UniPolfed.NoParallel)

```julia
NoParallel()
```

Run the transform without host-side decomposition. This is also the effective
strategy for GPU arrays, whose kernels provide their own parallelism.

### [`MulColsParallel`](@id UniPolfed.MulColsParallel)

```julia
MulColsParallel(nt_per_col=1)
```

Spread independent Krylov-block columns across Julia threads. This is most
useful when the block size exposes enough independent columns.

### [`MulRowsParallel`](@id UniPolfed.MulRowsParallel)

```julia
MulRowsParallel(nt_per_row=1)
```

Apply the transform to the whole block while allowing the operator kernel to
parallelize over output rows. Unlike column decomposition, this strategy can
also expose parallelism for a single vector.

### [`TwoLevelParallel`](@id UniPolfed.TwoLevelParallel)

```julia
TwoLevelParallel(nt_per_col)
```

Distribute block columns across worker processes and use `nt_per_col` threads
inside each worker.

## [Processing Unit Types](@id docs_processing_units)

### [`ProcessingUnit`](@id UniPolfed.ProcessingUnit)

Abstract backend descriptor used by allocation and storage dispatch.

### [`CPU`](@id UniPolfed.CPU)

```julia
CPU()
```

Construct the CPU allocation and array adapter.

### [`GPU`](@id UniPolfed.GPU)

```julia
GPU()
```

Construct the CUDA adapter. Construction throws an informative error when the
CUDA extension is unavailable or nonfunctional.

## Lanczos Reorthogonalization and Basis Types

### [`ReOrthTechnique`](@id UniPolfed.Numerics.Lanczos.ReOrthTechnique)

Abstract supertype for Lanczos reorthogonalization strategies.

### [`FullRO`](@id UniPolfed.Numerics.Lanczos.FullRO)

Reorthogonalize on every iteration.

### [`PartialRO`](@id UniPolfed.Numerics.Lanczos.PartialRO)

Construct as `PartialRO(skip)` to reorthogonalize only on selected iterations.

### [`SelectiveRO`](@id UniPolfed.Numerics.Lanczos.SelectiveRO)

Reserved for a selective scheme and not yet implemented.

### [`OrthonormalBasis`](@id UniPolfed.Numerics.Lanczos.OrthonormalBasis)

Abstract supertype for Krylov-basis storage.

### [`MatrixBasis`](@id UniPolfed.Numerics.Lanczos.MatrixBasis)

Store all basis vectors in one dense matrix. This backend supports both block
layouts.

### [`VectorBasis`](@id UniPolfed.Numerics.Lanczos.VectorBasis)

Store the basis as a vector of vector or block arrays.

## Lanczos Block Layout Types

### [`Layout`](@id UniPolfed.Numerics.Lanczos.Layout)

Abstract trait selecting the orientation of a Krylov block.

### [`ColumnLayout`](@id UniPolfed.Numerics.Lanczos.ColumnLayout)

Store vectors as columns (`dimension x block_size`) and apply the operator as
`H * X`.

### [`RowLayout`](@id UniPolfed.Numerics.Lanczos.RowLayout)

Store vectors as rows (`block_size x dimension`) and apply the operator as
`X * H`.

## Supported `target` Specifications

Count-based calls (`polfed(..., howmany, target)`) accept point targets:

- `:maxdos`
- `:middle`
- `(:offset, frac)`
- `(:unrescaled, E)` or plain `E::Real`
- `(:rescaled, e)`

Window calls (`polfed(..., target_window)`) accept:

- `(:unrescaled, (Elo, Ehi))` or plain `(Elo, Ehi)`
- `(:rescaled, (elo, ehi))`
