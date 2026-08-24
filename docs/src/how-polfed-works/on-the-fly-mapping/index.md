# [On-the-fly mapping](@id on_the_fly_mapping)

The filter from the [previous page](@ref filters) is a polynomial of degree ``K`` in
``\tilde H``. The obvious thing to do with it — form ``P(\tilde H)`` and hand the matrix to
Lanczos — is exactly what POLFED refuses to do.

## Why nothing is stored

Suppose ``H`` is sparse with ``\mathcal{O}(\mathcal{D})`` nonzeros. Then:

- ``\tilde H^2`` is denser than ``\tilde H``, ``\tilde H^3`` denser still. A degree-``K``
  polynomial in a sparse matrix is, for any interesting ``K``, **dense**. Storing it costs
  ``\mathcal{O}(\mathcal{D}^2)`` — the very thing sparsity was supposed to avoid, and for a
  many-body Hilbert space it is hopeless.
- The alternative classical route to interior eigenvalues, **shift-invert**, needs a
  factorization of ``H - \sigma I``. Sparse factorizations fill in catastrophically for
  many-body Hamiltonians, and the memory blows up long before the compute does.

POLFED does neither. There is **no stored ``P(H)``, no shift-invert, no factorization** —
nothing is allocated whose size grows faster than a handful of Krylov vectors.

## What happens instead

Lanczos never needs the transformed operator as an object. It only ever needs to *apply* it
to a vector. So POLFED builds a closure:

```julia
f!_transformed(Y, X)   #  Y ← P(H̃) · X
```

and each call evaluates the polynomial directly by the **Clenshaw recurrence**, using only
sparse products ``\tilde H \cdot v``. For coefficients ``c_0,\dots,c_K``, Clenshaw runs the
backward recurrence

```math
b_{k} = 2\,\tilde H\, b_{k+1} - b_{k+2} + c_k\, x ,
\qquad k = K, K-1, \dots, 1,
```

and finishes with

```math
P(\tilde H)\,x = \tilde H\, b_1 - b_2 + c_0\, x .
```

The working set is three vectors (or three blocks), independent of ``K``. Cost per
application: **``K`` sparse matvecs**. Memory: flat.

Clenshaw is used rather than summing ``\sum_k c_k T_k(\tilde H) x`` term by term because it
is both cheaper and numerically stabler — the explicit Chebyshev sum loses accuracy badly at
the orders POLFED reaches.

## The consequence: this is the bottleneck

Every single Lanczos matvec costs ``K`` sparse matvecs. So

```math
\text{total cost} \;\approx\;
(\#\ \text{Lanczos matvecs}) \times K \times \mathrm{cost}(H\cdot v),
```

and ``K`` grows as the target window narrows or the local density of states rises.

This one line explains the shape of the whole package: the fast value-grouped kernels, the
GPU backend, the parallelization strategies, and the patched decomposition are all attacks
on this same product. See [Going faster](@ref going_faster).

## Where you can intervene

The transform is built from a rescaled operator action, and every part of it is
replaceable through [`MappingConfig`](@ref UniPolfed.MappingConfig):

| Keyword | What it replaces |
| :--- | :--- |
| `optimize_mapping` | lets POLFED build value-grouped row kernels from your matrix |
| `f!_rescaled` | your own action of ``\tilde H = (H-b)/a``, skipping runtime rescaling |
| `clenshaw_recurrence`, `clenshaw_finalsum` | your own fused Clenshaw steps |
| `Emin`, `Emax` | skip the bounds probe when you already know the spectrum |

Supplying `f!_rescaled` is usually the highest-value change for a structured model: the
naive route applies `f!` and then rescales in two extra vector passes,

```julia
f!_rescaled_bad = (Y, X) -> begin
    f!(Y, X)
    @. Y *= 1 / spread
    @. Y -= (center / spread) * X
end
```

which adds memory traffic to the hottest loop in the program. Folding the rescaling into
the kernel — or even just pre-building a rescaled matrix once and using `mul!` — removes it.
Worked examples are in [Optimized kernels](@ref optimized_kernels).

!!! note "A subtlety worth knowing"
    The fast kernel is used for the transform only. The *map-back* step (`mapvals`, which
    turns filtered Ritz values into true energies) always uses a generic `mul!` row
    operation, because it acts on strided ``1 \times D`` eigenvector views that the blocked
    `Val(bs)` kernel cannot handle. Likewise the DoS estimate runs on a generic column
    operator. Only the bottleneck gets the specialized path.

## Read next

- [Going faster](@ref going_faster) — the three levers on that cost formula.
- [Bring your own operator](@ref operator_interface) — supplying `f!` and `f!_rescaled`.
- [Patched POLFED](@ref tutorial_patched) — attacking ``K`` instead of ``\mathrm{cost}(H\cdot v)``.
