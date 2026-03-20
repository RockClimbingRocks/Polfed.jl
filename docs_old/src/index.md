# Polfed.jl

```@meta
CurrentModule = Polfed
```

Documentation for [Polfed](https://github.com/RockClimbingRocks/Polfed.jl).
This documentation provides an overview of the Polfed.jl package, including its main features and usage examples.

**Polfed.jl** is a Julia package for performing **POLynomial Filtering Exact Diagonalization (POLFED)** simulations of quantum systems. The idea about POLFED is to combine polynomial filtering with Lanczos or Block Lanczos factorization to efficiently extract eigenvalues and eigenvectors of large sparse matrices in the middle of the spectrum.

## Features

- Efficient implementation of polfed algorithm
- Support for Lanczos and Block Lanczos factorizations
- Easy-to-use API for defining and solving problems
- Automatic optimizations for matrices with only few different values of off-diagonal elements
- Automatic parallelization using Julia's built-in multithreading capabilities
- Support for two level parallelization
- Support for Nvidia GPUs via CUDA.jl (real-valued arrays)
- CPU support for complex Hermitian matrices

## Installation

To install Polfed, you can use the Julia package manager. Run Julia REPL, usually with `julia` command (provided that Julia is in your 'PATH'), once your interactivce Julia session is running, add Polfed package by executing:

```julia
import Pkg
Pkg.add("Polfed")
```

Polfed.jl is a pure Julia package; no dependencies (aside from the Julia standard library) are required.

## Usage Example

Here is a simple example of how to use Polfed.jl to solve an eigenvalue problem a the middle of the spectrum on a case of quantum sun model

```julia
using Polfed
using Polfed.QSun: quantum_sun_hamiltonian
using LinearAlgebra

mat = quantum_sun_hamiltonian(12, 2; sparse=true) # define Hamiltonian matrix
v0 = rand(size(mat, 1)); v0 ./= norm(v0) # initial vector
howmany = 100
target = :maxdos

vals, vecs = polfed(mat, v0, howmany, target)
```

Here all of the default settings of polfed are used. For more advanced usage and customization options, please refer to the documentation pages generated from the Literate tutorials.

## Tutorial and Documentation flow/setup

In tutorial page we will guide u tghrou basic usage examples of polfed,

- In section [My first polfed run](@ref My_first_POLFED_run) we demonstrate how to run polfed with `Lanczos factorization` and `Block lanczos Factorization`
- In section [Knowinge your parallelization](@ref Knowing_your_parallelization) We comment on different types of parallelization, in particular we stress advanteges and disadventages of [NoParallel]())
  My_first_POLFED_run
  aaaaa
