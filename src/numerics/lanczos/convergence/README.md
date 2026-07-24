# Lanczos convergence

This directory implements target selection, convergence state, Ritz/eigen residual
checks, and extraction of converged eigenpairs. It is loaded by `Lanczos.jl`.

## Window targets

`Window` selects eigenpairs by the mapped value produced by the `mapvals` callback.
Two forms serve different completeness contracts.

```julia
Window(low, high)
```

The two-argument form walks inward from the selected transformed-spectrum end until
the converged run straddles both finite mapped bounds. It returns in-window values
and one available bracket candidate on each finite side. A bracket is the closest
value in the converged transformed run, not necessarily the immediate spectral
neighbour.

```julia
Window(low, high, edge_value; return_brackets=false)
```

The three-argument form uses `edge_value` as a transformed-space completeness
boundary, normally the validated filter cutoff. It converges one guard pair past
that boundary, then selects mapped values in `[low, high]`. This is the safe form
for assembling overlapping spectral patches.

The distinction matters because Lanczos orders the filtered eigenvalues
`p = P(E)`, while patch assembly orders mapped energies `E`. Outside a validated
passband, a high-order Chebyshev tail oscillates, so transformed order does not
certify mapped-energy adjacency.

## Patched-spectrum contract

Patch assembly uses:

1. overlapping mapped solve intervals;
2. each filter's transformed passband boundary as `edge_value`; and
3. `return_brackets=false`.

The overlapping intervals must contain at least one common eigenpair. The result
buffer can then reconcile repeated pairs and trim the assembled spectrum to the
requested ownership interval. Merge tolerances should be consistent with mapped
eigenvalue accuracy (`eigentol`), rather than the transformed Ritz tolerance alone.

## Residual handling

Ritz residuals use the current scalar residual norm or current block residual Gram
matrix for both column and row layouts. A coupling below the factorization's
breakdown tolerance is treated as zero, including full-dimensional breakdown.
Window progress reports the first required blocker while incomplete and a
required-run maximum when complete.

Run the package test suite with:

```julia
using Pkg
Pkg.test()
```
