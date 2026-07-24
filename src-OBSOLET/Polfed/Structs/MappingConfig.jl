"""
    MappingConfig(; kwargs...)

Configuration for the mapping stage of [`polfed`](@ref), including:
- mapping parallelization strategy,
- optional matrix-entrypoint auto-optimization,
- optional pre-rescaled mapping callback,
- optional custom Clenshaw kernels, and
- optional explicit spectral bounds (`Emin`, `Emax`).

Pass this struct as `mapping=...` in [`polfed`](@ref).

# Fields
- `parallel_strategy::Union{Parallelization,Nothing}`
  Mapping parallelization strategy.
  - `nothing` means "auto":
    - CPU input -> [`MulColsParallel`](@ref)
    - GPU input -> [`NoParallel`](@ref)
  - You can explicitly pass [`MulColsParallel`](@ref),
    [`TwoLevelParallel`](@ref), or [`NoParallel`](@ref).

- `optimize_mapping::Bool`
  Enables matrix-entrypoint optimization (`polfed(mat, ...)`) for structured
  matrices with repeated offdiagonal values.
  This can reduce memory traffic by building optimized mapping and Clenshaw
  kernels under the hood.

- `f!_rescaled::Union{Nothing,Function}`
  Optional user-provided mapping for the *rescaled* operator
  `H_tilde = (H - b)/a`.
  Expected signature:
  `f!_rescaled(Y, X) -> nothing`, with in-place write to `Y`.
  It should support both vector and matrix `X` (for Lanczos and Block Lanczos).

- `clenshaw_recurrence::Union{Nothing,Function}`
  Optional custom Clenshaw recurrence kernel.
  Accepted signatures:
  - `(b1, b2, b3, c, k, X)`, or
  - `(b1, b2, b3, c, X)`.
  It should update `b1` in place and avoid allocations.

- `clenshaw_finalsum::Union{Nothing,Function}`
  Optional custom Clenshaw final-sum kernel with signature:
  `(b2, b3, c0, Y, X)`.
  It should write the final transformed output into `Y`.

- `Emin::Union{Real,Nothing}`, `Emax::Union{Real,Nothing}`
  Optional unrescaled spectral bounds used for internal rescaling and target
  conversion.
  If either is `nothing`, POLFED estimates missing bounds with short Lanczos
  probes.

# Keyword Defaults
- `parallel_strategy = PolfedDefaults.parallel_strategy`
- `optimize_mapping = PolfedDefaults.optimize_mapping`
- `f!_rescaled = nothing`
- `clenshaw_recurrence = nothing`
- `clenshaw_finalsum = nothing`
- `Emin = nothing`
- `Emax = nothing`

# Notes
- `optimize_mapping=true` is intended for matrix input; for callback input
  (`polfed(f!, ...)`) it is ignored unless you provide custom mapping kernels.
- If you provide `f!_rescaled`, keep it consistent with `Emin`/`Emax` and your
  original operator scaling.
- If all mapping parallelization is handled externally (custom mapping or GPU
  kernel strategy), prefer [`NoParallel`](@ref).

# Example
```julia
mapping = MappingConfig(
    parallel_strategy = MulColsParallel(),
    optimize_mapping = true,
)
vals, vecs, report = polfed(mat, x0, howmany, target;
    produce_report = true,
    mapping = mapping,
)
```
"""
mutable struct MappingConfig
    parallel_strategy::Union{Parallelization,Nothing}
    optimize_mapping::Bool
    f!_rescaled::Union{Nothing, Function}
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    Emin::Union{Real, Nothing}
    Emax::Union{Real, Nothing}

    """Build a `MappingConfig` from keyword arguments."""
    function MappingConfig(;
        parallel_strategy::Union{Parallelization,Nothing} = PolfedDefaults.parallel_strategy,
        optimize_mapping::Bool                          = PolfedDefaults.optimize_mapping,
        f!_rescaled::Union{Nothing, Function}           = nothing,
        clenshaw_recurrence::Union{Nothing, Function}   = nothing,
        clenshaw_finalsum::Union{Nothing, Function}     = nothing,
        Emin::Union{Real, Nothing}                      = nothing,
        Emax::Union{Real, Nothing}                      = nothing,
    )
        new(parallel_strategy, optimize_mapping, f!_rescaled, clenshaw_recurrence, clenshaw_finalsum, Emin, Emax)
    end
end
