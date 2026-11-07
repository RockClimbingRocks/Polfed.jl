# Lanczos

Lanczos and **block** Lanczos eigensolvers for Hermitian/symmetric operators. Given
a matrix or a matrix-free action `f!(Y, X) = A·X`, it converges the **extremal**
parts of the spectrum (smallest, largest, both ends at once, or out to a chosen
energy) via a Krylov factorization, and reports per-eigenvalue residuals.

Like `Clenshaw` and `KPM`, this is a **POLFED-agnostic primitive** — it needs only
an operator action and a starting vector, constructs no POLFED structs, and is
tested against the known eigenvalues of constructed matrices. In POLFED it runs on
the *filtered* operator (where the target window has become extremal), with the
*unfiltered* `H` supplied as `mapvals` to recover true eigenvalues.

---

## Targeting — what to converge

A run is described by an [`EigTarget`](convergence/EigTarget.jl), built from per-end
**specs**:

```julia
NumEigs(n)              # a fixed count of eigenpairs at the end
ToValue(v)              # ALL eigenpairs out to transformed value v
Window(lo, hi, edge)    # ALL eigenpairs whose MAPPED value lands in [lo, hi]

Lowest(spec)            #  low end
Highest(spec)           #  high end
BothEnds(low, high)     #  both extremes in ONE factorization
```

Whatever the spec, the returned set is a **contiguous run** inward from the targeted
extreme in which every pair passed **both** the Ritz and the eigen tolerance — it can be
short, but it never has a gap. See [`convergence/README.md`](convergence/README.md).

with convenience aliases for the common count case:

```julia
LowestEigs(n)  ≡ Lowest(NumEigs(n))
HighestEigs(n) ≡ Highest(NumEigs(n))
BothEnds(a, b) ≡ BothEnds(NumEigs(a), NumEigs(b))
```

Examples:

```julia
lanczos(A, x0, LowestEigs(200))                       # 200 smallest
lanczos(A, x0, BothEnds(200, 200))                    # 200 low + 200 high, one Krylov space
lanczos(A, x0, BothEnds(ToValue(-2.0), ToValue(2.0))) # everything out to ±2.0 (both windows)
lanczos(A, x0, Lowest(ToValue(-2.0)))                 # all eigenpairs ≤ -2.0
```

`ToValue` converges *all* extremal eigenpairs out to `v` and confirms completeness
via a guard (the first eigenvalue *past* `v` must also converge), so the count is
**dynamic**. This is the spectrum-gluing primitive: target the window boundary and
let the solver capture every state up to it.

> Both-ends targeting is a *Lanczos selection* feature (one Krylov space → both
> extremes). It is distinct from POLFED's double/multi-window *filter*.

---

## Result and the gluing query

`lanczos` returns a [`LanczosResult`](LanczosResult.jl):

```julia
res = lanczos(A, x0, BothEnds(200, 200))
res.vals        # converged eigenvalues, ascending
res.vecs        # eigenvectors (columns; rows for RowLayout) — a view into the MatrixBasis storage
res.residuals   # per-eigenvalue true residual ‖A·v − λ·v‖
res.converged   # count meeting the eigen tolerance
res.report      # FactorizationReport (timings, convergence history)

vals, vecs, report = res     # also iterates as the legacy 3-tuple
```

For **spectrum gluing**, ask whether a specific eigenvalue already converged in a
window:

```julia
has_converged(res, λ; atol=1e-6, restol=res.report.eigentol)
```

(Reliable when `λ` is one the run actually targeted — e.g. a window boundary on the
transformed operator.)

---

## Entry points

```julia
lanczos(mat::AbstractMatrix, x0, target; …)    # matrix interface
lanczos(f!::Function,        x0, target; …)    # operator interface, f!(Y,X)=A·X
lanczos(A_or_f!, x0, n::Int; which=:SR, …)     # convenience: :SR→LowestEigs, :LR→HighestEigs
lanczos_extrema(A; x0=nothing, …)              # (Emin, Emax) via two short probes
```

Key keywords: `basistype` (`MatrixBasis` default, or `VectorBasis`), `rot`
(`FullRO`/`PartialRO`), `maxdim` (Krylov budget; defaults to `500·count` capped at
the operator dimension), `tol`/`eigentol`, `mapvals` (operator for eigenvalues), and
`collect_garbage`. CPU vs GPU is detected from `x0`. The seed is normalized (vector)
or orthonormalized (block) automatically.

**On the GPU** the Krylov vectors, blocks and basis are device arrays, while everything
projected stays on the host: the block-tridiagonal matrix, its `eigen`, the Ritz vectors,
the coupling blocks (copied to device scratch before each projection) and the returned
eigenvalues. Every factorization, layout and basis runs there, real and complex; the row
layout orthonormalizes by a device QR of `Xᴴ` (CUSOLVER has no LQ).

---

## Internals

- **Config→State split:** an immutable `ConvergenceCriterion{T<:EigTarget,M}` (target,
  tolerances, budget, `mapvals`) + a mutable `ConvergenceState` (counters, per-check
  history) — replacing the old abstract-typed `ConvergenceInfo` god-object, with
  concrete field types throughout.
- **Fast convergence checks:** the projected matrix is `SymTridiagonal` (scalar) or
  `Hermitian` (block), so `eigen` uses the symmetric solver; checks are scheduled
  adaptively from `ConvergenceCriterion.next_check`.
- **Two-stage convergence:** the cheap Ritz residual gates the expensive eigenvector
  reconstruction, and acceptance requires both. One orchestrator serves every target
  type; per-spec rules live behind a five-method interface. See
  [`convergence/README.md`](convergence/README.md).
- **Breakdown handling:** if the Krylov residual norm vanishes (an invariant subspace
  / eigenvector seed), the iteration stops gracefully and force-extracts the exact
  Ritz pairs instead of dividing by zero.
- **Memory — the basis is the peak:** intermediate eigen checks rebuild eigenvectors 64
  at a time and drop them (`calc_eigen_residuals_chunked`); the final extraction writes
  them into the leading slots of the `MatrixBasis` storage block by block
  (`extract_eigenpairs!`) and the ascending sort permutes them in place
  (`permute_eigenvectors!`). No full eigenvector matrix is ever allocated next to the
  basis, so `res.vecs` is a view that keeps the basis storage alive.
- **Thread configuration is the caller's responsibility** — the primitive does not
  mutate global BLAS/MKL/Julia thread settings (that would fight `f!`'s own
  parallelization, e.g. POLFED's per-mapping `ParallelizationStrategy`).
- **Linear-algebra backend:** block QR/LQ and the projected symmetric eigendecomposition
  use LAPACK; dense overlap, projection, MatrixBasis reorthogonalization, and Ritz-vector
  reconstruction use BLAS `mul!`; scalar recurrence/reorthogonalization uses BLAS-style
  `dot`/`axpy!`/`norm`. Loading `MKL.jl` in the application switches the CPU BLAS *and*
  LAPACK calls through Julia's libblastrampoline. Lanczos never imports MKL itself, so it
  cannot unexpectedly switch the process-wide backend used by the mapping callback.
- **Factorization layout:** the abstract base (`KrylovFactorization`, `_breakdown_tol`,
  `update!`, `lanczositer`, `lanczositer!`) lives in `Factorization/Factorization.jl`;
  the two concrete types are in `Factorization/FactorizationType/Lanczos.jl` (scalar)
  and `Factorization/FactorizationType/BlockLanczos.jl` (block).

`ProcessingUnit`/`CPU`/`GPU` and the `gpu_*` helpers come from the package-top
`Backend/` (imported via `...Name`); `Common` provides logging/formatting/timing.

---

## Testing

Tests in `test/Numerics/Lanczos/` assert against **known eigenvalues** of constructed
symmetric matrices (cross-checked with dense `eigen`): `LowestEigs`/`HighestEigs`,
`BothEnds` (count and `ToValue`), block, element types, `has_converged`, the
invariant-subspace breakdown path, and the report.

> **Use deterministic seeds in tests.** A `randn` seed can, for a given RNG state,
> align with an eigenvector of a random test matrix (a scaled eigenvector of a
> random-`Q` matrix is statistically Gaussian-looking), causing a spurious
> iteration-1 breakdown. The tests use fixed generic seed vectors instead.

Run with `Pkg.test()` (`coverage=true` for line coverage). CPU paths are ~fully
covered; the device paths are covered by `gpu_lanczos_tests.jl`, which is gated on a
CUDA device and does not run under `Pkg.test()` (see the top-level CLAUDE.md).
