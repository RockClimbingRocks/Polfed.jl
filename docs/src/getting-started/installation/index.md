# [Installation](@id installation)

## Install the package

```julia
import Pkg
Pkg.add("Polfed")
```

Then load it:

```julia
using Polfed
```

That is everything you need for the CPU workflows in these docs. `Polfed.jl` requires
Julia 1.10 or newer.

## Optional: GPU support

GPU execution is provided by a package extension that stays dormant until CUDA is loaded.
Nothing extra needs to be installed at package level — add CUDA to your own environment:

```julia
import Pkg
Pkg.add("CUDA")

using Polfed
using CUDA          # activates the GPU code paths
```

From then on, passing `CuArray`/`CuSparseMatrixCSR` inputs to [`polfed`](@ref Polfed.polfed)
routes the work to the GPU. See [GPU](@ref going_faster_gpu).

## Optional: models

The Hamiltonian builders used throughout the tutorials live in a submodule:

```julia
using Polfed.Models: qsun_hamiltonian, xxz_hamiltonian, j1j2_hamiltonian
```

These are convenience test Hamiltonians only. POLFED never requires them — you can always
pass your own matrix or your own operator (see
[Bring your own operator](@ref operator_interface)).

## Check the install

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

H  = qsun_hamiltonian(8, 2, 1.0, 0.5; use_sparse=true)
x0 = normalize!(rand(size(H, 1)))

vals, vecs, report = polfed(H, x0, 10, :maxdos)
length(vals)       # number of eigenpairs returned
```

If that runs and returns eigenpairs, you are set. Continue with the
[Quickstart](@ref quickstart).
