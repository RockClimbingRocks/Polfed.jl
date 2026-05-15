# [Automatic Optimization](@id tutorial_xxz_baseline)

## When to use automatic optimization

Automatic optimization should be used when the Hamiltonian has a simple
structure, in particular when all offdiagonal elements have the same value
(or only a few distinct values).

In that regime, we do not need to repeatedly look up matrix entries, read
them, and multiply them in a generic sparse-matrix path. Instead, POLFED can
decompose mapping into diagonal and offdiagonal parts and use a lower-memory
execution path.

This idea was already introduced in the beginner section
[Optimized Mapping](@ref tutorial_optimized_mapping), but in
[Quantum Sun (QSun)](@ref qsun_model) the optimized mapping was beneficial
because of the structure of the previously mentioned Hamiltonian.

## Speedup of the XXZ Hamiltonian

For the XXZ model, you only need to construct the Hamiltonian with
[`xxz_hamiltonian`](@ref Polfed.Models.xxz_hamiltonian) from the
[XXZ](@ref xxz_model) model page, then enable `optimize_mapping=true`; here
one should observe a large speedup compared to the generic `mul!` mapping.

```julia
using Polfed
using Polfed.Models: xxz_hamiltonian
using LinearAlgebra

L = 18
Lup = L ÷ 2
Delta = 0.55

mat = xxz_hamiltonian(L, Lup, 1.0, Delta, 0.0; boundary=:periodic, field=0.0, use_sparse=true)
x0 = rand(size(mat, 1)); x0 ./= norm(x0)
howmany = 120
target = 0.0

mapping_auto = MappingConfig(optimize_mapping=true)
vals_auto, vecs_auto, report_auto = polfed(mat, x0, howmany, target; produce_report=true, mapping=mapping_auto)
display_report(report_auto)
```

Timing behavior should be similar to what you observed in
[Custom Mapping](@ref tutorial_xxz_custom_mapping): for this structured XXZ
case, automatic optimization typically reaches nearly the same speedup while
keeping the interface simpler.

In other words, `optimize_mapping` essentially does what is done in the
[Custom Mapping](@ref tutorial_xxz_custom_mapping) section.
*

