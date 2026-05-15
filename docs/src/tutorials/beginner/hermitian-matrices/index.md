# Hermitian matrices

POLFED supports complex Hermitian workflows while keeping the same
[`polfed`](@ref Polfed.polfed) interface.

This is important for many physics workflows where Hamiltonians are Hermitian
but not purely real, for example when working in symmetry-reduced sectors
(such as translational-symmetry momentum sectors) that naturally produce
complex matrix elements.

## Simple Example

```julia
using Polfed
using SparseArrays
using LinearAlgebra

n = 5000
H = sprandn(ComplexF64, n, n, 0.01)
H = H + H'
H = H + spdiagm(0 => randn(Float64, n))

x0 = randn(ComplexF64, n)
x0 ./= norm(x0)

howmany = 40
target = 0.0

vals, vecs = polfed(H, x0, howmany, target)
```
*

