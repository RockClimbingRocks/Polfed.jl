# [Unitary POLFED](@id tutorial_unitary)

!!! warning "Preview — not implemented yet"
    There is no unitary entrypoint in the package today. This page is a placeholder so the
    section exists in the navigation; fill it in as the work lands.

## The idea

For a Floquet system the object of interest is not a Hermitian Hamiltonian but a **unitary**
evolution operator ``U``, whose eigenvalues sit on the unit circle:

```math
U|\psi\rangle = e^{-i\theta}|\psi\rangle .
```

The eigenphases ``\theta`` play the role energies play in the Hermitian case, and the same
question arises — how do you reach the ones in the middle of the quasi-energy spectrum,
rather than the ones a Krylov method happens to converge to first?

The POLFED answer transfers: build a filter that is large near a target eigenphase and small
elsewhere, apply it matrix-free, and let the Krylov method see the targets as extremal. What
changes is the geometry. There is no ``[-1,1]`` interval to rescale into and no spectral
edge to peel — the spectrum is a circle, so "edge" and "interior" stop meaning what they
mean in [Filters](@ref filters).

## To write

- [ ] What replaces the ``[-1,1]`` rescaling for a spectrum on the unit circle.
- [ ] The filter family, and what "passband" and "leakage" mean without spectral edges.
- [ ] Which Krylov method — the operator is unitary, not Hermitian, so plain Lanczos does
      not apply.
- [ ] Entrypoint signature and how a target eigenphase is specified.
- [ ] Reading the output — residuals against ``U``, and what convergence means here.

## Current status

Not started in `src/`. Track it on the
[development roadmap](../../development/roadmap.md).
