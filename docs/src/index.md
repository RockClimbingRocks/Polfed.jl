<!-- ```@meta
CurrentModule = Polfed
```

# Polfed

Documentation for [Polfed](https://github.com/RockClimbingRocks/Polfed.jl).

```@index

```

```@autodocs
Modules = [Polfed]
``` -->

# Polfed.jl

**Polfed.jl** is a Julia package for performing **Polynomial Filtering (PolFed)** simulations of quantum systems.  
It provides efficient numerical routines for working with large-spare Hamiltonians, including support for eigenvalue filtering, kernel polynomial methods, and custom iterative algorithms.

Polfed is designed for **high-performance simulations**, integrates smoothly with other Julia scientific computing packages, and includes GPU acceleration support.

## Overview

Polfed.jl accepts general functions or callable objects as for example, any tipe of matrices (sparse, dense, CuMatrix...).

The high level interface of Polfed is provided by the following functions:

- [`polfed`](@ref): solves eigenvalue problem `H*v =E*v` at the targeted part of the spectrum
<!-- - [`dos`](@ref): calulates denseties of states of the hamiltonian `ρ(E)=∑_i δ(E_i-E)` -->
