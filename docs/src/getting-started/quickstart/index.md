# [Quickstart](@id quickstart)

Your first eigenpairs, in about five minutes. Every step below is a normal Julia session —
no configuration files, no tuning.

## 1. Build a problem

We use the Quantum Sun model as a stand-in for your Hamiltonian. Any sparse Hermitian
matrix works just as well.

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

L_loc   = 12
L_grain = 2
g0      = 1.0
α       = 0.5

H  = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
x0 = normalize!(rand(size(H, 1)))
```

`x0` is the starting vector for the Krylov iteration. It only has to be normalized —
its direction does not matter.

## 2. Ask for eigenpairs

Two questions POLFED can answer, distinguished by how you call it:

```julia
# "Give me 100 eigenpairs near the maximum of the density of states"
result = polfed(H, x0, 100, :maxdos)

# "Give me every eigenpair between these two energies"
result_window = polfed(H, x0, (-0.5, 0.5))
```

The first form is a **count target**: `howmany` eigenpairs closest to a point. The second is
a **window target**: everything inside an interval, however many that turns out to be. See
[Target modes](@ref polfed_targets) for all the ways to name a target.

## 3. Use the answer

`polfed` returns a result that unpacks as `vals, vecs, report`:

```julia
vals, vecs, report = polfed(H, x0, 100, :maxdos)
```

Or keep the object and take fields off it:

```julia
result = polfed(H, x0, 100, :maxdos)

result.vals          # eigenvalues, sorted, true energies of H
result.vecs          # matching eigenvectors
result.residuals     # ‖H·v − λ·v‖ per pair
result.report        # the run report

length(result.vals)  # how many came back
```

`vecs` is a `D × n` matrix whose **columns** are the eigenvectors, so:

```julia
v1 = vecs[:, 1]                   # the 1st eigenvector
```

This holds for every entrypoint and both call paths — the matrix entrypoints run in row layout
internally and hand back a lazy `transpose` of those rows (no copy), so you never have to think
about it. `vecs` is a view into the solver's storage; `copy(vecs)` gives an independent `Matrix`.
If you want a view of a single eigenvector, [`eigenvector`](@ref Polfed.eigenvector) does the
same thing without allocating:

```julia
v1   = eigenvector(result, 1)     # a view of column 1
λ, v = eigenpair(result, 1)       # value and vector together
```

To multiply such a view by a sparse `H` on the CPU, `collect` it first: the view is strided,
and MKLSparse (loaded by POLFED) gets strided vectors wrong — see
[`eigenvector`](@ref Polfed.eigenvector).

## 4. Read the output

```julia
display_report(report)
```

```@raw html
<style>
.pf-report {
  --pfr-bg: #f6f8fa; --pfr-fg: #24292f; --pfr-border: #d0d7de;
  --pfr-hd: #0550ae; --pfr-v: #0969a5; --pfr-ok: #1a7f37; --pfr-wn: #9a6700; --pfr-er: #cf222e;
  margin: 1.2rem 0; padding: 0.9rem 1rem; overflow-x: auto;
  border: 1px solid var(--pfr-border); border-radius: 8px; background: var(--pfr-bg);
}
html.theme--documenter-dark .pf-report,
html.theme--catppuccin-frappe .pf-report,
html.theme--catppuccin-macchiato .pf-report,
html.theme--catppuccin-mocha .pf-report {
  --pfr-bg: #15191a; --pfr-fg: #d7dde3; --pfr-border: #58636a;
  --pfr-hd: #79b8ff; --pfr-v: #56d4dd; --pfr-ok: #5ddba0; --pfr-wn: #e0a336; --pfr-er: #ff7b72;
}
/* Documenter styles every .content pre with its own border and background at a
   higher specificity — without !important the report renders as a box in a box. */
.pf-report pre {
  margin: 0 !important; padding: 0 !important;
  background: none !important; border: 0 !important; border-radius: 0 !important;
  font-size: 0.83rem; line-height: 1.5; white-space: pre; color: var(--pfr-fg);
}
.pf-report .pf-hd { color: var(--pfr-hd); font-weight: 700; }
.pf-report .pf-v  { color: var(--pfr-v);  font-weight: 700; }
.pf-report .pf-ok { color: var(--pfr-ok); font-weight: 700; }
.pf-report .pf-wn { color: var(--pfr-wn); font-weight: 700; }
.pf-report .pf-er { color: var(--pfr-er); font-weight: 700; }
.pf-report .pf-b  { font-weight: 700; }
</style>
<div class="pf-report"><pre><span class="pf-hd">Spectral Transformation ────────────────────────────────────────────────</span>
- target: <span class="pf-v">:maxdos</span>
- interval: resc. [<span class="pf-v">+1.56e-01</span>,<span class="pf-v">+2.23e-01</span>], unresc. [<span class="pf-v">+4.66e-01</span>,<span class="pf-v">+7.32e-01</span>]
- filter: <span class="pf-v">DeltaUpFilter</span> · order K = <span class="pf-v">78</span> · cutoff <span class="pf-v">0.14</span>
- mat-vec muls: <span class="pf-v">4_700</span>
<span class="pf-hd">Mapping ────────────────────────────────────────────────────────────────</span>
- operator: <span class="pf-v">generic operator (mul! / user f!)</span>
- layout: <span class="pf-v">RowLayout</span>    block size <span class="pf-v">1</span>    opdim <span class="pf-v">256</span>
- parallel strategy: <span class="pf-v">BlockWise</span>
- Energy: Emin = <span class="pf-v">-4.170</span>   Emax = <span class="pf-v">3.847</span>
<span class="pf-hd">Convergence (block lanczos) ────────────────────────────────────────────</span>
- Number of converged eigenpairs: <span class="pf-ok">20</span> (<span class="pf-v">20</span> requested)
- Ritz  conv tol=<span class="pf-v">1.0e-14</span> passed:  <span class="pf-ok">20</span> (max res. <span class="pf-wn">1.72e-16</span>)
- Eigen conv tol=<span class="pf-v">1.0e-09</span> passed:  <span class="pf-ok">20</span> (max res. <span class="pf-wn">1.12e-13</span>)
- Iterations needed: <span class="pf-b">60</span> (out of <span class="pf-v">68</span> reserved, overestimated by <span class="pf-ok">11.76%</span>)
- Returned set: <span class="pf-ok">certified complete</span>
<span class="pf-hd">Timings ────────────────────────────────────────────────────────────────</span>
- Total polfed run took: <span class="pf-v">5.56 seconds</span> (walltime), <span class="pf-v">5.53 seconds</span> (CPU time)
- Walltime distribution: Lanczos <span class="pf-v">81.29%</span> · preparation <span class="pf-v">18.71%</span>
- Lan walltime: matvec <span class="pf-v">1.59%</span> · reorthog. <span class="pf-v">20.10%</span> · tridiag. <span class="pf-v">11.53%</span> · conv.check <span class="pf-v">66.79%</span>
- Lan CPU time: matvec <span class="pf-v">1.59%</span> · reorthog. <span class="pf-v">20.13%</span> · tridiag. <span class="pf-v">11.45%</span> · conv.check <span class="pf-v">66.82%</span></pre></div>
```

Four sections, and on a first run each answers one question:

| Section | Answers |
| :--- | :--- |
| **Spectral Transformation** | what filter was built — `order K` is what every matvec costs |
| **Mapping** | how the operator was set up, and the bounds it was rescaled with |
| **Convergence** | did it work — how many pairs converged, and to what residual |
| **Timings** | where the time went |

The number to look at first is **`order K`**. Every Lanczos iteration applies the filter once,
and each application costs `K` sparse products — `mat-vec muls` is the running total.

A line-by-line walkthrough of every field is in
[Reading the report](@ref polfed_reading_report). The patched and full variants return a
different report type, covered in their own tutorials.

## Next steps

- [**How POLFED works**](@ref how_polfed_works) — the six stages, with an interactive
  explorer for the auto-piloted part.
- [**Get to know `polfed` in more detail**](@ref tutorial_polfed) — every target mode, block
  Lanczos, complex operators, and a line-by-line read of the report.
- [**How to run code faster**](@ref going_faster) — the three levers, in the order worth
  trying them.
- [**Using other variants of POLFED**](../../index.md) — patched, full, double and unitary,
  laid out as cards on the home page so you can pick the one that fits your problem.
