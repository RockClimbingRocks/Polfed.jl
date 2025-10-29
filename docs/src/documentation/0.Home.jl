# # Polfed.jl

# ```@meta
# CurrentModule = Polfed
# ```
# Documentation for [Polfed](https://github.com/RockClimbingRocks/Polfed.jl).
# This documentation provides an overview of the Polfed.jl package, including its main features, usage examples and documentation. PPackage is based on papers REF1(2020 PRL), REF2(MY).

# **Polfed.jl** is a Julia package for performing **POLynomial Filtering Exact Diagonalization (POLFED)** simulations of quantum systems. The idea about POLFED is to combine polynomial filtering with Lanczos or Block Lanczos factorization to efficiently extract eigenvalues and eigenvectors of large sparse matrices in the middle of the spectrum.



# ## Features
# - Efficient implementation of polfed algorithm
# - Support for Lanczos and Block Lanczos factorizations
# - Easy-to-use API for defining and solving problems
# - Automatic optimizations for matrices with only few different values of off-diagonal elements
# - Automatic parallelization using Julia's built-in multithreading capabilities
# - Support for two level parallelization
# - Support for Nvidia GPUs via CUDA.jl
# 
# ## Installation
# To install Polfed, you can use the Julia package manager. Run Julia REPL, usually with `julia` command (provided that Julia is in your 'PATH'), once your interactivce Julia session is running, add Polfed package by executing:

# ```julia
# import Pkg
# Pkg.add("Polfed")
# ```
# Polfed.jl is a pure Julia package; no dependencies (aside from the Julia standard library) are required.

# ## Usage Example

# Here is a simple example of how to use Polfed.jl to solve an eigenvalue problem a the middle of the spectrum on a case of quantum sun model

# ```julia
# using Polfed
# using Polfed.QSun: quantum_sun_hamiltonian
# using LinearAlgebra

# mat = quantum_sun_hamiltonian(12, 2; sparse=true) # define Hamiltonian matrix
# v0 = rand(size(mat, 1)); v0 ./= norm(v0) # initial vector
# howmany = 100
# target = 0.

# vals, vecs = polfed(mat, v0, howmany, target)
# ```

# Here all of the default settings of polfed are used. For more advanced usage and customization options, please refer to the (upcoming) full documentation.



# ## Tutorial and Documentation flow/setup

# In tutorial page we will guide u tghrou basic usage examples of polfed,

# - In section [My first polfed run](@ref My_first_POLFED_run) we demonstrate how to run polfed with `Lanczos factorization` and `Block lanczos Factorization`
# - In section [Knowinge your parallelization](@ref Knowing_your_parallelization) we comment on different types of parallelization, in particular we stress advanteges and disadventages of [`NoParallel`](@ref), [`MulColsParallel`](@ref) and [`TwoLevelParallel`](@ref) parallelization strategyies.
# - In section [Constructing optimized mapping](@ref Constructing_Optimized_Mapping) we demonstrate how to construct optimized mapping for disordered `XXZ` model with constant offdiagonal elements, this way we can reduce memory acces and make mapping faster.
# - In section [Reducing memory access](@ref Reducing_Memory_Access) we further reduced unnecesary memory access to construct allready rescaled mapping and custom clenshaw functions.
# - In section [Pre-Optimized polfed](@ref preoptimized_polfed) we demonstrate how to use all of the previously discussed optimizations in one go, by simply passing in the keyword argument `optimized_mapping=true` into [`polfed`](@ref) function.


# ## Citation
# If you use Polfed.jl in your research, please cite the following papers: