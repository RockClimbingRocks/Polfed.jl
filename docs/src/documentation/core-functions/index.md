# [Documentation for UniPolfed.jl](@id docs_main_reference)

This page documents POLFED solver entry points and the public Hamiltonian
constructors used throughout the tutorials.

## [`polfed`](@id UniPolfed.polfed)

The current entrypoint has count-target and window-target forms for both a
matrix and an in-place operator callback:

```julia
polfed(H,  x0, howmany, target; mc, dosc, tol, eigentol, overestimate_iters)
polfed(f!, x0, howmany, target; mc, dosc, tol, eigentol, overestimate_iters)
polfed(H,  x0, target_window; mc, dosc, tol, eigentol, overestimate_iters)
polfed(f!, x0, target_window; mc, dosc, tol, eigentol, overestimate_iters)
```

For callback input, `f!(Y, X)` must overwrite `Y` with `H * X`. A vector `x0`
selects ordinary Lanczos; a matrix selects block Lanczos. Count-target calls
return up to `howmany` eigenpairs near a point target. Window-target calls omit
`howmany` and return the eigenpairs found inside the requested energy interval.

The result is a `PolfedResult`. Its `vals`, `vecs`, `residuals`, `converged`,
and `report` fields contain the filtered eigenpairs and run diagnostics.

## Model Constructors

Public Hamiltonian builders live under the common `UniPolfed.Models` namespace.
Their full signatures, keyword arguments, return types, and conventions are
documented in [Models](@ref docs_models_reference).

## Mapping-Optimization Helpers

The matrix entrypoint selects optimized row kernels through
[`MappingConfig`](@ref UniPolfed.MappingConfig). The underlying optimization
helpers live in `UniPolfed.Optimization`:

### [`value_grouping`](@id UniPolfed.Optimization.value_grouping)

```julia
value_grouping(H; tol=1e-12, rtol=1e-9)
```

Analyze a structured matrix and return its value-grouped representation for
the optimized row kernels.

### [`resolve_mapping_kernel`](@id UniPolfed.Optimization.resolve_mapping_kernel)

```julia
resolve_mapping_kernel(choice, H, Val(block_size); secs=0.5)
```

Resolve `true` or `:auto`, a kernel symbol, or a
`(kernel=..., G=..., W=...)` named tuple to a concrete optimized kernel and
tuning parameters.

### [`build_optimized_mapping`](@id UniPolfed.Optimization.build_optimized_mapping)

```julia
build_optimized_mapping(H, choice, Val(block_size), a, b; secs=0.5)
```

Build the optimized row-layout action of the rescaled operator
`H_tilde = (H - b) / a`. It returns a named tuple containing
`f!_rescaled`, the selected kernel id, and its tuning parameters.

## Filter Sizing and Introspection

### [`expected_krylov_dim`](@id UniPolfed.expected_krylov_dim)

```julia
expected_krylov_dim(filter, howmany, block_size, eta)
```

Estimate the Krylov-space dimension for the concrete filter. The final
argument is the budget-inflation factor used by `overestimate_iters`.

### [`order_safety_factor`](@id UniPolfed.order_safety_factor)

```julia
order_safety_factor(filter)
```

Return the safety factor applied when resolving the filter's polynomial order.
Fixed-order filters return `1.0`.

## Low-Level Lanczos Utilities

### [`lanczos`](@id UniPolfed.Numerics.Lanczos.lanczos)

```julia
lanczos(A_or_f!, x0, target; kwargs...)
lanczos(A_or_f!, x0, n; which=:SR, kwargs...)
```

Run the Hermitian Lanczos or block-Lanczos eigensolver directly. Use
`LowestEigs(n)`, `HighestEigs(n)`, or `BothEnds(n_low, n_high)` for explicit
target objects, or use the integer convenience form with `which=:SR` or
`:LR`. The return value is a `LanczosResult`.

### [`lanczos_extrema`](@id UniPolfed.Numerics.Lanczos.lanczos_extrema)

```julia
lanczos_extrema(H; x0=nothing, maxdim=100, power_iters=3)
lanczos_extrema(f!, x0; maxdim=100, power_iters=3)
```

Estimate the smallest and largest eigenvalues in one both-ends Lanczos run.
Pass `return_result=true` to obtain the full `LanczosResult` instead of the
`(Emin, Emax)` pair.

## Example: Passing Current Configuration into `polfed`

```julia
using UniPolfed: polfed, MappingConfig, DoSConfig, MulColsParallel
using UniPolfed.Models: qsun_hamiltonian
using LinearAlgebra

L_loc = 12
L_grain = 2
g0 = 1.0
α = 0.5
mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
x0 = rand(size(mat, 1)); x0 ./= norm(x0)
howmany = 100
target = 0.0

f!(Y, X) = mul!(Y, mat, X)

mapping_cfg = MappingConfig(
    parallel_strategy = MulColsParallel(),
)

dos_cfg = DoSConfig(
    N = 200,
    R = 300,
)

result = polfed(
    f!,
    x0,
    howmany,
    target;
    mc = mapping_cfg,
    dosc = dos_cfg,
    tol = 1e-15,
    eigentol = 1e-10,
)

vals = result.vals
vecs = result.vecs
report = result.report
```
