# [Double POLFED](@id tutorial_double)

Two energy windows in one call. `double_polfed` serves both windows from one bounds probe and
one DoS estimate, and picks — per problem — whether one Krylov space or two is cheaper.

## The two ways to serve two windows

A `DoubleDeltaFilter` places **two peaks** in a single polynomial: `+1` on the lower window,
`−1` on the upper. Both windows then sit at opposite *extremes* of the transformed spectrum, so
ONE filtered Lanczos run (a `BothEnds` target) converges them together.

That is not always the better deal. The combined polynomial has a single degree,
`K_d = max(K₁, K₂)`, so **both** windows pay the sharper peak's order on every matvec. If one
window needs a far higher order than the other — a narrow window next to a wide one, or a window
near a spectral edge next to one in the bulk — the cheap window is dragged along at the expensive
window's price, and two independent runs cost less.

So `double_polfed` measures before it chooses. Both windows are resolved through the ordinary
single-window filter chain first (free: order resolution is scalar bisection, not matvecs),
giving `K₁` and `K₂`. With `r = min(K₁,K₂)/max(K₁,K₂)`:

| decision | condition | what runs |
|:---|:---|:---|
| joint | `r ≥ 0.4` | one `DoubleDeltaFilter` run, `BothEnds` target |
| separate | `r < 0.4` | two single-filter runs, sequentially |

The threshold is not arbitrary. The joint run's Krylov space is inflated by
`Defaults.double_polfed.krylov_overhead = 1.4` over a single-window run for `max(n₁, n₂)` states
— one space, both ends — so the joint run wins exactly when `1.4 < 1 + r`, i.e. when `r ≥ 0.4`.
The two constants are derived from each other and must move together.

Whatever the ratio says, geometry wins: if either window touches a spectrum edge (`±1`, where a
monotone `EdgeFilter` is mandatory) or the double filter's cutoff ladder finds no valid
candidate, the run falls back to two separate runs.

## Calling it

Everything below runs against this setup:

```julia
using UniPolfed
using UniPolfed.Models: qsun_hamiltonian
using LinearAlgebra

H  = qsun_hamiltonian(8, 2, 1.0, 0.5; use_sparse = true)
x0 = normalize!(rand(size(H, 1)))
```

The two modes mirror [`polfed`](@ref UniPolfed.polfed) and are distinguished by **arity** — the
two targets are separate arguments, because a bare `(lo, hi)` is already a window spec:

```julia
# window mode: every eigenpair inside each window
result = double_polfed(H, x0, (-0.8, -0.5), (0.3, 0.6))

# count mode: `howmany` eigenpairs near each target …
result = double_polfed(H, x0, 100, -0.65, 0.45)

# … or one count per window
result = double_polfed(H, x0, (50, 150), -0.65, 0.45)
```

The two windows must be **disjoint with a gap between them** — a single connected region is a job
for [`polfed`](@ref UniPolfed.polfed), or `patched_polfed` if it is wide. The targets may be given
in either order; results come back in the order you passed them.

Force a path with `strategy` when you want to measure rather than trust:

```julia
w1, w2 = (-0.8, -0.5), (0.3, 0.6)

double_polfed(H, x0, w1, w2; strategy = :double)     # one joint run
double_polfed(H, x0, w1, w2; strategy = :separate)   # two runs
```

## The result

`double_polfed` returns a `DoublePolfedResult`: two ordinary
[`PolfedResult`](@ref UniPolfed.PolfedResult)s plus the decision that produced them. It
destructures, so the two windows are separated at the call site:

```julia
result = double_polfed(H, x0, (-0.8, -0.5), (0.3, 0.6))
lower, upper = result

for i in eachindex(lower.vals)
    λ, v = eigenpair(lower, i)
    # …
end

upper.vals           # eigenvalues in the second window
result.strategy      # :double | :separate
result.reason        # :ratio | :forced | :edge | :double_failed
result.ratio         # min(K₁,K₂)/max(K₁,K₂) — what the decision was made on
result.order_double  # K_d, or 0 on the separate path
```

Walk the pairs with [`eigenpair`](@ref UniPolfed.eigenpair), not by iterating the result:
a `PolfedResult` iterates as the three-element `(vals, vecs, report)` destructuring, so
`for (λ, v) in lower` does **not** loop over eigenpairs.

Each sub-result carries its own full `Report`. On the joint path both carry the **same** report —
one Lanczos run produced both windows, so its convergence counts, `H·v` cost and timings describe
that one run, not one window's share of it. Displaying the pair says so explicitly:

```julia
display_report(result)
```

## Notes

- The two runs on the separate path are **sequential** by design: parallelism already lives inside
  the matvec, and two Lanczos runs competing for the same cores only slow each other down.
- In window mode each window is returned *bracketed* (one eigenpair just outside each edge). On
  the joint path a window can never adopt the other window's eigenpair as its bracket — the
  bracket search is cut off at the midpoint of the valley between the two windows.
- Exactly two windows are supported. For many windows, use `patched_polfed`.
