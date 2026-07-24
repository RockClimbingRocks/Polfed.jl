# `convergence3`

This folder is a source-compatible replacement candidate for `lanczos/convergence`.
It is intentionally not wired into `Lanczos.jl`, because this task permits changes only
inside `convergence3`.

## What was wrong

The failed patches do have common eigenvalues, but not a common contiguous suffix/prefix.
The old Window rule stops when a transformed-order prefix contains any mapped value below
the window and any mapped value above it. It then calls the nearest member of that prefix a
"bracket".

That is not a spectral-neighbour certificate. Lanczos orders the filtered eigenvalues
`p = P(E)`, while patching orders the mapped energies `E`. A high-order Chebyshev tail
oscillates, so a distant energy may be more extremal in `p` than the actual seam neighbour.
In the supplied failure the second patch therefore begins with a distant value, skips most
of the first patch's suffix, and the strict merger correctly rejects it.

There is also an independent residual bug in both old implementations: the current Ritz
residual was evaluated with the stored coupling `β_(m-1)`. The `m`-dimensional Ritz residual
requires the current coupling `β_m`, which is still represented by the unnormalised
`factorization.r`. At a full-dimensional breakdown the old formula can report a large
residual even though the true current residual is zero.

## Compatibility

The live call shape remains valid:

```julia
Window(low, high)
```

It retains the legacy mapped-straddle behavior and returns the in-window values plus one
available bracket candidate per finite side. The candidates are deliberately not described
as immediate spectral neighbours.

The old three-argument shape is also accepted, with a precise transformed-boundary meaning:

```julia
Window(low, high, edge_value; return_brackets=false)
```

Here `edge_value` is the validated transformed-space passband boundary (normally the filter
cutoff). The internal prefix must converge one pair past that boundary. Selection then keeps
mapped values in `[low, high]`; brackets default off because oscillatory-tail brackets are not
safe seam evidence.

All existing target types, aliases, criterion/state fields, `decide` signatures, row/column
layouts, and CPU/GPU-aware extraction paths are retained.

## Required contract for patched spectra

A strict suffix/prefix merger needs a genuinely shared energy interval. Convergence cannot
manufacture that from adjacent intervals which only touch at one real-valued endpoint.
For patched runs, each call must therefore use:

1. overlapping mapped intervals, normally the validated `filter.passband` intervals;
2. the corresponding transformed cutoff as `edge_value`; and
3. `return_brackets=false`.

Conceptually, a highest-end delta filter should construct:

```julia
Highest(Window(filter.passband.left, filter.passband.right, filter.cutoff))
```

and a lowest-end filter analogously uses `Lowest`. The overlapping passbands must contain at
least one eigenvalue; otherwise no exact eigenpair overlap exists for a value-based merger.
The final assembled result can then be trimmed to the originally requested full interval.

The current `window_eig_target` passes only the adjacent requested interval and discards the
passband/cutoff. No implementation confined to this folder can recover that missing metadata
or guarantee immediate neighbours in a non-monotone tail. Thus swapping only the loader to
`convergence3` improves residual correctness and keeps the legacy API, but the patched
pipeline also needs to adopt the three-argument overlap contract before its strict merge is
mathematically valid.

There is a separate caller-side tolerance hazard: the supplied run passes `tol = 1e-14` to
the patch merger even though independently reconstructed mapped eigenvalues are accepted at
roughly `eigentol = 1.17e-10`. One shared value in the supplied output already differs between
patches by about `1.2e-14`. Once the seam topology is corrected, the merge tolerance should be
chosen consistently with the mapped eigenvalue accuracy rather than the transformed Ritz
tolerance.

## Other corrections

- Ritz residuals use the current scalar residual norm or the current block residual Gram
  matrix, for both column and row layouts.
- A numerical breakdown below the factorization's own breakdown tolerance is treated as a
  zero current coupling.
- The accidental `println("n = …")` / `println("vals = …")` diagnostics are removed.
- `Window(-Inf, Inf)` no longer certifies an empty run.
- Window progress reports the first required blocker while incomplete and a required-run
  maximum when complete, rather than an unrelated maximum over the whole projected spectrum.
- Documentation distinguishes an internal contiguous transformed prefix from mapped
  selection; it no longer promises energy contiguity after an oscillatory-tail selection.

## Activation and verification

The repository currently loads `convergence/Convergence.jl`. To evaluate this candidate in
an isolated copy, replace that one include path with `convergence3/Convergence.jl` (do not
include both, because the types have the same names), then run the existing Lanczos
convergence suite and `convergence3/test/runtests.jl`.
