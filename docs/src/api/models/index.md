# [Models](@id docs_models_reference)

Public Hamiltonian constructors exposed through `UniPolfed.Models`.

```julia
using UniPolfed.Models: qsun_hamiltonian, xxz_hamiltonian, j1j2_hamiltonian
```

These are **optional**. They exist so the tutorials have something concrete to run and so
you have a known-good test problem — POLFED works with any Hermitian matrix or any operator
`f!`. See [Bring your own operator](@ref operator_interface).

The model pages give the physical background and usage examples:
[Quantum Sun (QSun)](@ref qsun_model), [XXZ](@ref xxz_model), [J1-J2](@ref j1j2_model).
This page is the constructor reference.

## Quantum Sun

```@docs
UniPolfed.Models.qsun_hamiltonian
```

See also: [Quantum Sun (QSun)](@ref qsun_model).

## XXZ

```@docs
UniPolfed.Models.xxz_hamiltonian
```

See also: [XXZ](@ref xxz_model) and the worked optimization example in
[Optimized kernels](@ref optimized_kernels_xxz).

## J1-J2

```@docs
UniPolfed.Models.j1j2_hamiltonian
```

See also: [J1-J2](@ref j1j2_model).
