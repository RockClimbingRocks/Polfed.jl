# [XXZ](@id xxz_model)

The spin-1/2 XXZ chain is available from the common `Polfed.Models`
namespace through [`xxz_hamiltonian`](@ref Polfed.Models.xxz_hamiltonian).

```julia
using Polfed.Models: xxz_hamiltonian

H = xxz_hamiltonian(L, Lup, J, Delta, W; kwargs...)
```

For the full Julia constructor signature, argument types, keyword meanings,
and return types, see [Models](@ref docs_models_reference), in particular
[`xxz_hamiltonian`](@ref Polfed.Models.xxz_hamiltonian).

## Model Definition

The Hamiltonian is

```math
H = J \sum_{i=1}^{L}
\left(
    S_i^x S_{i+1}^x
    + S_i^y S_{i+1}^y
    + \Delta S_i^z S_{i+1}^z
\right)
+ \sum_{i=1}^{L} h_i S_i^z.
```

The constructor is fixed to spin-1/2 chains and works directly in a chosen
magnetization sector:
- `L` is the chain length,
- `Lup` is the number of spin-up sites,
- the total magnetization is ``S_z = Lup - L/2``.

For even `L`, the zero-magnetization sector is therefore given by
`Lup = L ÷ 2`.

Disorder and boundary conventions follow the common `Polfed.Models` API:
- `W` is the random-field disorder strength,
- if `fields` is not provided, then `h_i` are sampled uniformly from
  `[field - W, field + W]`,
- `boundary=:periodic` wraps bonds around the chain,
- `boundary=:open` omits boundary-crossing bonds.

## Parameters

- `L`: chain length.
- `Lup`: number of spin-up sites in the fixed spin-1/2 sector.
- `J`: nearest-neighbor exchange scale.
- `Delta`: Ising anisotropy ``\Delta``.
- `W`: disorder width for longitudinal fields.
- `boundary`: either `:periodic` or `:open`.
- `field`: center of the disorder window.
- `fields`: explicit field realization. When passed, it is used directly.
- `rng`: random-number generator used when disorder is sampled internally.
- `use_sparse`: if `true`, return a sparse matrix; otherwise return a dense
  matrix.

## Basic Example

```julia
using Polfed
using Polfed.Models: xxz_hamiltonian
using LinearAlgebra
using Random

rng = MersenneTwister(1234)

# constructing the Hamiltonian
L = 16
Lup = L ÷ 2
J = 1.0
Delta = 1.0
W = 0.5

H = xxz_hamiltonian(
    L,
    Lup,
    J,
    Delta,
    W;
    boundary=:periodic,
    rng=rng,
    use_sparse=true,
)

# generating a random initial state
x0 = rand(rng, size(H, 1))
x0 ./= norm(x0)

# setting POLFED parameters
howmany = 40
target = :middle

vals, vecs = polfed(H, x0, howmany, target)
```

## Relation to Advanced Tutorials

The advanced tutorial pages build on this same model and show how to exploit
its structure for faster mappings and GPU workflows:
- [Optimization of the XXZ Model](@ref optimization_xxz_model)
- [Custom Mapping](@ref tutorial_xxz_custom_mapping)
- [Automatic Optimization](@ref tutorial_xxz_baseline)
- [GPU Implementation](@ref tutorial_xxz_rescaled_clenshaw)

For the detailed Julia constructor reference, see
[Models](@ref docs_models_reference).
*

