# Polfed.jl

```@meta
CurrentModule = Polfed
```

**Polynomial Filtering Exact Diagonalization** — eigenpairs from *anywhere* in the spectrum
of large sparse matrices.

```@raw html
<style>
/* ---------- shared tokens ---------- */
.polfed-home {
  --polfed-card-bg: #ffffff;
  --polfed-card-border: #d0d7de;
  --polfed-card-shadow: 0 1px 2px rgba(27,31,36,.08);
  --polfed-card-hover: #4e79a7;
  --polfed-muted: #57606a;
  --polfed-accent: #4e79a7;
  --polfed-accent-soft: #eaf0f6;
  --polfed-segment-border: #dbdbdb;
}
html.theme--documenter-dark .polfed-home,
html.theme--catppuccin-frappe .polfed-home,
html.theme--catppuccin-macchiato .polfed-home,
html.theme--catppuccin-mocha .polfed-home {
  --polfed-card-bg: #1f2428;
  --polfed-card-border: #58636a;
  --polfed-card-shadow: none;
  --polfed-card-hover: #7ea6cc;
  --polfed-muted: #a7b0b8;
  --polfed-accent: #7ea6cc;
  --polfed-accent-soft: #232c33;
  --polfed-segment-border: #5e6d6f;
}

/* ---------- figures (JS-rendered, true Chebyshev filters) ---------- */
.polfed-fig {
  --pf-fg: #24292f;
  --pf-muted: #9aa4ad;
  --pf-grid: #d0d7de;
  --pf-blue: #4e79a7;
  --pf-green: #2ca05a;
  --pf-bg: #ffffff;
  --pf-border: #d0d7de;
}
html.theme--documenter-dark .polfed-fig,
html.theme--catppuccin-frappe .polfed-fig,
html.theme--catppuccin-macchiato .polfed-fig,
html.theme--catppuccin-mocha .polfed-fig {
  --pf-fg: #f3f5f7;
  --pf-muted: #6f7a83;
  --pf-grid: #58636a;
  --pf-blue: #7ea6cc;
  --pf-green: #3fd07f;
  --pf-bg: #15191a;
  --pf-border: #58636a;
}
.polfed-hero-figure {
  margin: 1.75rem 0 !important;   /* same override as above */
  padding: 0.5rem;
  border: 1px solid var(--pf-border);
  border-radius: 10px;
  background: var(--pf-bg);
}
.polfed-hero-figure svg { display: block; width: 100%; max-width: 900px; margin: 0 auto; height: auto; }
.polfed-hero-figure figcaption {
  margin: 0.4rem 0.4rem 0.2rem;
  font-size: 0.88rem;
  color: #57606a;
}
html.theme--documenter-dark .polfed-hero-figure figcaption,
html.theme--catppuccin-frappe .polfed-hero-figure figcaption,
html.theme--catppuccin-macchiato .polfed-hero-figure figcaption,
html.theme--catppuccin-mocha .polfed-hero-figure figcaption { color: #a7b0b8; }

/* ---------- action buttons ---------- */
.polfed-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin: 1.25rem 0;
}
.polfed-split-button {
  display: inline-flex;
  align-items: stretch;
  border: 1px solid var(--polfed-segment-border);
  border-radius: 8px;
  overflow: hidden;
}
.polfed-split-button .button {
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
}
.polfed-split-button .button:first-child { pointer-events: none; }
.polfed-split-button .button:not(:first-child) {
  border-left: 1px solid var(--polfed-segment-border) !important;
}

/* ---------- cards ---------- */
.polfed-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: 1rem;
  margin: 1.25rem 0 2rem;
}
.polfed-card {
  position: relative;
  display: flex;
  flex-direction: column;
  /* The figure fills the card's content box, so THIS is the knob that controls
     how close the illustration gets to the card edge. Lower it to widen the
     figure; the figure never overflows because it simply stretches to fit. */
  padding: 0.9rem 0.9rem 1.2rem;
  border: 1px solid var(--polfed-card-border);
  border-radius: 10px;
  background: var(--polfed-card-bg);
  box-shadow: var(--polfed-card-shadow);
  text-decoration: none !important;
  color: inherit !important;
  transition: border-color .15s ease, transform .15s ease;
}
.polfed-card:hover {
  border-color: var(--polfed-card-hover);
  transform: translateY(-2px);
}
/* Fixed-height figure box: every card's illustration occupies the same space,
   so every title starts at exactly the same height. */
.polfed-card figure {
  position: relative;
  /* No explicit width and no negative margin: the figure stretches to the card's
     content box on its own. Widen it with the card's side padding above, which
     cannot overflow. */
  margin: 0 0 1.1rem !important;   /* beats Documenter's .content figure { margin: 2em } */
  padding: 0;
  border-radius: 8px;
  background: var(--polfed-accent-soft);
  display: flex;
  align-items: center;
  justify-content: center;
  aspect-ratio: 200 / 112;
}
/* The variant illustrations use a taller canvas than the feature icons, so the
   box has to match — an SVG scales to *fit* its box, it does not stretch. */
.polfed-cards--variants figure { aspect-ratio: 200 / 130; }
.polfed-card svg { display: block; width: 100%; height: 100%; }
.polfed-card h3 {
  margin: 0 0 0.35rem !important;   /* beats .content h3 { margin-bottom: 0.6666em } */
  padding: 0;
  font-size: 1.28rem !important;
  font-weight: 700;
  color: var(--polfed-accent);
}
.polfed-card p {
  margin: 0 !important;
  padding: 0;
  font-size: 0.9rem;
  color: var(--polfed-muted);
}
.polfed-card code { font-size: 0.85em; }
/* Written directly over the artwork, in its top-right corner. Absolutely
   positioned inside the figure and with no box of its own, so it takes up no
   layout space and never moves the illustration. */
.polfed-card figure .polfed-tag {
  position: absolute;
  top: 0.3rem;
  right: 0.5rem;
  z-index: 2;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--polfed-muted);
  pointer-events: none;
}
</style>

<div class="polfed-home">
<p class="polfed-actions">
  <a class="button is-primary is-medium" href="getting-started/quickstart/">▶ Run it in 5 minutes</a>
</p>
<p class="polfed-actions">
  <span class="polfed-split-button">
    <span class="button is-static">Citation</span>
    <a class="button is-primary" href="citation/#article">article</a>
    <a class="button is-primary" href="citation/#code">code</a>
  </span>
  <span class="polfed-split-button">
    <span class="button is-static">Code</span>
    <a class="button is-primary" href="https://github.com/RockClimbingRocks/Polfed.jl" target="_blank" rel="noopener noreferrer">GitHub</a>
  </span>
  <span class="polfed-split-button">
    <span class="button is-static">Article</span>
    <a class="button is-primary" href="https://scipost.org/submissions/2605.10191v2/" target="_blank" rel="noopener noreferrer">SciPost</a>
    <a class="button is-primary" href="https://arxiv.org/abs/2605.10191" target="_blank" rel="noopener noreferrer">arXiv</a>
  </span>
</p>
</div>
```

## What POLFED does

Krylov methods converge fast at the **edges** of a spectrum and slowly in the middle. POLFED
builds a **polynomial filter** that maps the part of the spectrum you actually want onto the
extreme values of a transformed problem — so the eigenpairs you asked for become the easy
ones — and then runs **Lanczos** (or block Lanczos) on that transformed operator.

The transformation is applied **on the fly**: the filter polynomial is never built as a
matrix. Each application is evaluated directly from sparse products ``H\cdot v`` through the
Clenshaw recurrence. That single design choice is what makes the method practical, and it
buys two things at once:

- **A much faster matrix–vector mapping.** The hot loop is a sparse product you control, and
  it can be specialized, threaded, or moved to a GPU.
- **A low memory footprint.** There is no dense ``P(H)`` to store and no shift-invert
  factorization to hold — memory stays flat in the polynomial order.

```@raw html
<figure class="polfed-fig polfed-hero-figure">
  <svg id="pf-hero" viewBox="0 0 880 430" role="img" aria-label="A Chebyshev delta filter centred at lambda equals zero. Eigenvalues whose filtered value clears the cutoff are highlighted in green."></svg>
  <figcaption>
    The lower axis is the original spectrum. The curve is the Chebyshev delta filter centred
    on the target λ. Every eigenvalue is mapped to a point on that curve; the ones that clear
    the cutoff (dashed) become <span style="color:var(--pf-green);font-weight:600;">extremal</span>
    in the transformed problem, which is exactly where Lanczos is fast.
  </figcaption>
</figure>
```

## Pick a POLFED variant suitable for your problem

```@raw html
<div class="polfed-home">
<div class="polfed-cards polfed-cards--variants">

  <a class="polfed-card polfed-fig" href="tutorials/polfed/">
    <figure><svg id="pf-card-single" viewBox="0 0 200 130" role="img" aria-label="A single delta filter peak over a spectrum"></svg></figure>
    <h3>POLFED</h3>
    <p>One polynomial filter run to obtain eigenpairs at the desired part of the spectrum.</p>
  </a>

  <a class="polfed-card polfed-fig" href="tutorials/double-polfed/">
    <figure><svg id="pf-card-double" viewBox="0 0 200 130" role="img" aria-label="A double delta filter with one peak pointing up and one pointing down"></svg></figure>
    <h3>Double POLFED</h3>
    <p>For problems where two distinct, disconnected parts of the spectrum are wanted at
    once — for example ETH studies needing 1000 eigenpairs around <code>E = 0</code> and
    another window at <code>ω = 10</code>. One Krylov space, two peaks.</p>
  </a>

  <a class="polfed-card polfed-fig" href="tutorials/patched-polfed/">
    <figure><svg id="pf-card-patched" viewBox="0 0 200 130" role="img" aria-label="One wide target window equals three narrower delta filters covering it in sequence"></svg></figure>
    <h3>Patched POLFED</h3>
    <p>For when you need a huge number of eigenpairs. The polynomial order is low, but the
    Lanczos matrix becomes enormous — at some point too large to diagonalize. This version
    of POLFED splits the job into several independent, overlapping runs and patches them
    together.</p>
  </a>

  <a class="polfed-card polfed-fig" href="tutorials/full-polfed/">
    <figure><svg id="pf-card-full" viewBox="0 0 200 130" role="img" aria-label="The whole spectrum tiled by patches, each with its own filter peak"></svg></figure>
    <h3>Full POLFED</h3>
    <p>The entire spectrum, patch by patch — this is essentially exact diagonalization, done
    matrix-free. Uses close to <strong>3× less memory</strong> than dense ED.
    <em>TODO: add the timing comparison.</em></p>
  </a>

  <a class="polfed-card polfed-fig" href="tutorials/unitary-polfed/">
    <figure>
      <span class="polfed-tag">Preview</span>
      <svg viewBox="0 0 200 130" role="img" aria-label="Eigenphases spread around the unit circle, with a targeted cluster highlighted">
        <circle cx="100" cy="65" r="56" fill="none" stroke="var(--pf-blue)" stroke-width="3.6"/>
        <!-- eigenphases sitting exactly on the circle, at multiples of 22.5° -->
        <g fill="var(--pf-muted)">
          <circle cx="139.60" cy="25.40"  r="4.5"/><circle cx="121.43" cy="13.26"  r="4.5"/>
          <circle cx="100.00" cy="9.00"   r="4.5"/><circle cx="78.57"  cy="13.26"  r="4.5"/>
          <circle cx="60.40"  cy="25.40"  r="4.5"/><circle cx="48.26"  cy="43.57"  r="4.5"/>
          <circle cx="44.00"  cy="65.00"  r="4.5"/><circle cx="48.26"  cy="86.43"  r="4.5"/>
          <circle cx="60.40"  cy="104.60" r="4.5"/><circle cx="78.57"  cy="116.74" r="4.5"/>
          <circle cx="100.00" cy="121.00" r="4.5"/><circle cx="121.43" cy="116.74" r="4.5"/>
          <circle cx="139.60" cy="104.60" r="4.5"/>
        </g>
        <!-- the targeted eigenphases -->
        <g fill="var(--pf-green)">
          <circle cx="151.74" cy="43.57" r="5.5"/><circle cx="156.00" cy="65.00" r="5.5"/>
          <circle cx="151.74" cy="86.43" r="5.5"/>
        </g>
      </svg>
    </figure>
    <h3>Unitary POLFED</h3>
    <p>Filtering eigenphases of unitary (Floquet) operators rather than eigenvalues of a
    Hermitian one.</p>
  </a>

</div>
</div>
```

## Further functionalities

```@raw html
<div class="polfed-home">
<div class="polfed-cards">

  <a class="polfed-card polfed-fig" href="going-faster/optimized-kernels/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="Repeated matrix values compressed into a few groups">
        <g fill="var(--pf-blue)" opacity=".85">
          <rect x="22" y="32" width="16" height="16" rx="2"/><rect x="42" y="32" width="16" height="16" rx="2"/><rect x="62" y="32" width="16" height="16" rx="2"/>
          <rect x="22" y="52" width="16" height="16" rx="2"/><rect x="42" y="52" width="16" height="16" rx="2"/><rect x="62" y="52" width="16" height="16" rx="2"/>
        </g>
        <path d="M92 56 L 118 56 M110 50 L118 56 L110 62" fill="none" stroke="var(--pf-muted)" stroke-width="2.5"/>
        <g fill="var(--pf-green)" opacity=".9"><rect x="130" y="48" width="16" height="16" rx="2"/><rect x="150" y="48" width="16" height="16" rx="2"/></g>
      </svg>
    </figure>
    <h3>Optimized kernels</h3>
    <p>Value-grouped row kernels for structured Hamiltonians. <code>optimize_mapping=true</code>
    and the transform gets a lot cheaper.</p>
  </a>

  <a class="polfed-card polfed-fig" href="going-faster/gpu/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="A grid of GPU threads, each row trailing off into further threads">
        <rect x="36" y="26" width="128" height="60" rx="6" fill="none" stroke="var(--pf-blue)" stroke-width="3"/>
        <g fill="var(--pf-green)">
          <!-- row 1 -->
          <rect x="48" y="38" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="65" y="38" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="82" y="38" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="99" y="38" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="116" y="38" width="12" height="12" rx="2" opacity=".45"/>
          <rect x="133" y="38" width="12" height="12" rx="2" opacity=".18"/>
          <!-- row 2 -->
          <rect x="48" y="60" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="65" y="60" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="82" y="60" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="99" y="60" width="12" height="12" rx="2" opacity=".85"/>
          <rect x="116" y="60" width="12" height="12" rx="2" opacity=".45"/>
          <rect x="133" y="60" width="12" height="12" rx="2" opacity=".18"/>
        </g>
      </svg>
    </figure>
    <h3>GPU</h3>
    <p>Move the arrays to <code>CuArray</code>; the call stays the same. The memory-bound
    transform is what GPUs are good at.</p>
  </a>

  <a class="polfed-card polfed-fig" href="going-faster/parallelization/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="Work split across several threads">
        <g stroke="var(--pf-blue)" stroke-width="3" fill="none">
          <line x1="24" y1="38" x2="176" y2="38"/><line x1="24" y1="56" x2="176" y2="56"/><line x1="24" y1="74" x2="176" y2="74"/>
        </g>
        <g fill="var(--pf-green)"><circle cx="62" cy="38" r="5"/><circle cx="110" cy="56" r="5"/><circle cx="148" cy="74" r="5"/></g>
      </svg>
    </figure>
    <h3>Parallelization</h3>
    <p>Two strategies: one thread per Krylov vector, or every thread on the whole block
    at once. Threads split the matvec either way.</p>
  </a>

  <a class="polfed-card polfed-fig" href="api/results-reports-logging/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="A report panel with a converging residual curve">
        <rect x="28" y="26" width="144" height="60" rx="5" fill="none" stroke="var(--pf-blue)" stroke-width="2.5"/>
        <g stroke="var(--pf-muted)" stroke-width="2"><line x1="42" y1="40" x2="124" y2="40"/><line x1="42" y1="51" x2="102" y2="51"/></g>
        <path d="M42 76 C 72 76, 86 69, 102 62 C 124 52, 144 66, 160 66" fill="none" stroke="var(--pf-green)" stroke-width="2.5"/>
      </svg>
    </figure>
    <h3>Reports &amp; logging</h3>
    <p>Every run returns filter order, convergence, residuals and a timing split. Five
    verbosity levels when you need more.</p>
  </a>

  <a class="polfed-card polfed-fig" href="how-polfed-works/krylov/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="A single Krylov vector beside a block of vectors">
        <rect x="30" y="32" width="14" height="48" rx="2" fill="var(--pf-blue)" opacity=".85"/>
        <text x="37" y="94" text-anchor="middle" font-size="10" fill="var(--pf-muted)">vector</text>
        <text x="80" y="60" text-anchor="middle" font-size="14" fill="var(--pf-muted)">or</text>
        <g fill="var(--pf-green)" opacity=".85">
          <rect x="108" y="32" width="14" height="48" rx="2"/><rect x="126" y="32" width="14" height="48" rx="2"/>
          <rect x="144" y="32" width="14" height="48" rx="2"/><rect x="162" y="32" width="14" height="48" rx="2"/>
        </g>
        <text x="142" y="94" text-anchor="middle" font-size="10" fill="var(--pf-muted)">block</text>
      </svg>
    </figure>
    <h3>Lanczos &amp; block Lanczos</h3>
    <p>Both factorizations are supported and chosen automatically by the shape of your
    starting vector — a vector runs Lanczos, a matrix runs block Lanczos with that many
    columns.</p>
  </a>

  <a class="polfed-card polfed-fig" href="tutorials/polfed/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="A real number line and a complex plane">
        <line x1="18" y1="56" x2="86" y2="56" stroke="var(--pf-blue)" stroke-width="3" stroke-linecap="round"/>
        <g fill="var(--pf-blue)"><circle cx="34" cy="56" r="4"/><circle cx="52" cy="56" r="4"/><circle cx="70" cy="56" r="4"/></g>
        <text x="52" y="80" text-anchor="middle" font-size="11" fill="var(--pf-muted)">real</text>
        <g stroke="var(--pf-green)" stroke-width="3" stroke-linecap="round">
          <line x1="114" y1="56" x2="182" y2="56"/><line x1="148" y1="26" x2="148" y2="86"/>
        </g>
        <g fill="var(--pf-green)"><circle cx="166" cy="40" r="4"/><circle cx="130" cy="70" r="4"/><circle cx="160" cy="72" r="4"/></g>
        <text x="148" y="102" text-anchor="middle" font-size="11" fill="var(--pf-muted)">complex</text>
      </svg>
    </figure>
    <h3>Real &amp; complex</h3>
    <p>Real symmetric and complex Hermitian operators are equally supported — momentum
    sectors and other symmetry-reduced bases work out of the box, in single or double
    precision.</p>
  </a>

  <a class="polfed-card polfed-fig" href="models/qsun/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="A spin chain coupled to an ergodic grain">
        <circle cx="36" cy="56" r="18" fill="var(--pf-blue)" opacity=".8"/>
        <g stroke="var(--pf-muted)" stroke-width="2"><line x1="54" y1="56" x2="82" y2="56"/><line x1="96" y1="56" x2="124" y2="56"/><line x1="138" y1="56" x2="166" y2="56"/></g>
        <g fill="var(--pf-green)"><circle cx="89" cy="56" r="7"/><circle cx="131" cy="56" r="7"/><circle cx="173" cy="56" r="7"/></g>
      </svg>
    </figure>
    <h3>Models</h3>
    <p>Ready-to-use Quantum Sun, XXZ and J1–J2 Hamiltonian builders, with symmetry sectors.
    Or pass any <code>H</code> or operator <code>f!</code> of your own.</p>
  </a>

  <a class="polfed-card polfed-fig" href="benchmarks/">
    <figure>
      <svg viewBox="0 0 200 112" role="img" aria-label="Bar chart comparing configurations">
        <g fill="var(--pf-blue)" opacity=".85"><rect x="34" y="52" width="18" height="34" rx="2"/><rect x="76" y="40" width="18" height="46" rx="2"/><rect x="118" y="28" width="18" height="58" rx="2"/></g>
        <g fill="var(--pf-green)" opacity=".85"><rect x="56" y="68" width="18" height="18" rx="2"/><rect x="98" y="62" width="18" height="24" rx="2"/><rect x="140" y="58" width="18" height="28" rx="2"/></g>
      </svg>
    </figure>
    <h3>Benchmarks</h3>
    <p>POLFED benchmarks across variants, parameters and hardware — accuracy, timing and
    scaling, so you can see what a given setting actually costs.</p>
  </a>

</div>
</div>
```

## Quick run

Already installed? This is the whole thing. If not, the setup steps are on the
[Installation](@ref installation) page, and a longer walkthrough is in the
[Quickstart](@ref quickstart).

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

H  = qsun_hamiltonian(12, 2, 1.0, 0.5; use_sparse=true)
x0 = normalize!(rand(size(H, 1)))

result = polfed(H, x0, 100, :maxdos)   # 100 eigenpairs at the DoS maximum
display_report(result)                 # order, convergence, timings
```

Everything between "target" and "answer" — the density of states, the energy interval, the
filter, and its polynomial order — is resolved automatically. See
[The workflow](@ref how_polfed_works).

## Citing

If Polfed.jl supports your research, please cite **both** the overview article and the
code — it keeps the method and the software visible, reusable, and maintainable. Details
are on the [citation page](@ref citation).

The original POLFED method was introduced by Piotr Sierant and collaborators,
[PRL 125, 156601 (2020)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.156601)
/ [arXiv:2005.09534](https://arxiv.org/pdf/2005.09534). The package itself is described in
[SciPost](https://scipost.org/submissions/2605.10191v2/) /
[arXiv:2605.10191](https://arxiv.org/abs/2605.10191).

Version of the code: `v0.1.0`

```@raw html
<p class="polfed-actions">
  <a class="button is-primary" href="https://juliapkgstats.com/pkg/Polfed?timeframe=30d&trendingPeriod=14d&userData=true&ciData=true&missingData=true" target="_blank" rel="noopener noreferrer">Download Statistics</a>
</p>

<script>
(function () {
  const NS = "http://www.w3.org/2000/svg";

  /* Rescaled eigenvalues of a small QSun sample, spanning [-1, 1]. */
  const EIG = [
    -1.0,-0.8862397328904057,-0.8138365850501954,-0.7643225411127533,-0.7187609110983968,
    -0.6997605808304965,-0.6781724053153465,-0.6628769734462476,-0.6555008533116723,
    -0.6267389263172705,-0.6069325920558576,-0.6002697427080279,-0.5784162092016458,
    -0.5638818226007932,-0.5309181092225888,-0.5197656009827994,-0.5104954691723352,
    -0.5088660884158803,-0.502195481672162,-0.4881248013948633,-0.4724939941512107,
    -0.4697075462260785,-0.4449499266605002,-0.43677100051172824,-0.4243705078324369,
    -0.42009175481101313,-0.41627304357147416,-0.41119006162761274,-0.4052875659614221,
    -0.39225647335893027,-0.3851810983151796,-0.3799463497733936,-0.3737745003460363,
    -0.36508961483402547,-0.3608739904704648,-0.35163453692905794,-0.3469545307678019,
    -0.33991063817989614,-0.32747721682499314,-0.3246955654604279,-0.3172804068667309,
    -0.3122363925302267,-0.30871338178175345,-0.30163616603643506,-0.2994281274946959,
    -0.28956444712674695,-0.28418013090233296,-0.2746346375848551,-0.2683452800939827,
    -0.26329369973122363,-0.25504123255503164,-0.25251725438319267,-0.24679102758060065,
    -0.23658361409521564,-0.2314840584554421,-0.2301222669904662,-0.22855784733457538,
    -0.22108833084033364,-0.21540027655705468,-0.21099987743893953,-0.2033370267187006,
    -0.19712602535127383,-0.19161715382325875,-0.18239060161332082,-0.1809874217975776,
    -0.173864611043623,-0.17031650017325128,-0.16712914105240806,-0.1621283772103342,
    -0.15894974145075402,-0.15269510391876634,-0.1471918818891129,-0.1420416180619292,
    -0.13871738368798317,-0.13588341801987786,-0.13246457153472915,-0.130127852071681,
    -0.12589553076887303,-0.11891165408789882,-0.11539229178489978,-0.11142936222869644,
    -0.10325531881385082,-0.09404115488468896,-0.0892641044448668,-0.08474082513987984,
    -0.08227428188208782,-0.08094542463359694,-0.07845417700480453,-0.07494941563761186,
    -0.07329354656510167,-0.06863519250851163,-0.06378224595255806,-0.062494584553020704,
    -0.05930252817499551,-0.047478133864845444,-0.040243455193717644,-0.03167090854535132,
    -0.029108668031192397,-0.024759870329775606,-0.022104449767448974,-0.018782930979589794,
    -0.017863726135231402,-0.011915376269962204,-0.005997862725306875,0.0002514069329118227,
    0.0022093438372056315,0.008689271448713986,0.009911066055664134,0.014545581192528868,
    0.01990513989628108,0.021471310068868057,0.024686044604475103,0.025945802650147728,
    0.03216032682617953,0.03800461006548309,0.03856014226704779,0.04543405119118332,
    0.0507051142705264,0.05627169592490444,0.06126334264160843,0.0672931436680215,
    0.07009003878114714,0.0718836232280733,0.07384239806812845,0.08066382079865139,
    0.08582789159684678,0.08939682187285115,0.0904054271055243,0.09555923962018127,
    0.09855140803690669,0.102745908921033,0.10523728726186328,0.10674882290861616,
    0.11240430521941186,0.11767418642222564,0.12620295760540703,0.13138479876666811,
    0.13374914052446277,0.13627721010803942,0.1402721263504261,0.14484272006761953,
    0.14876277273055652,0.15330679455611046,0.15624948685304535,0.16025501036628853,
    0.16637517083852663,0.16878921059903265,0.17759945757882037,0.1818197297604848,
    0.18349275480126576,0.18759046681048175,0.19005884234454704,0.19446935150261835,
    0.1983362124713565,0.20335193393592105,0.20478260096569167,0.21250091281234645,
    0.21510537329081536,0.22065106083208885,0.2208682299892854,0.22234280695720804,
    0.22547089362266895,0.23211050308410885,0.23511749180305097,0.24002410188427092,
    0.24192233935294913,0.249677508727342,0.2552365091017531,0.25918271372364593,
    0.2624301731343962,0.26733025423682827,0.2702836011953202,0.27611102783337643,
    0.2833798512663728,0.2904226500474597,0.29387179884183245,0.30007247287366534,
    0.3030913231354488,0.3042555865594446,0.3085684814358684,0.31950839763431915,
    0.3228722143612751,0.32770917659383053,0.33339455506645915,0.3364569764669032,
    0.3395355415658398,0.3439433061515341,0.3479507575160735,0.3497871844905475,
    0.3582338352854033,0.3597197025114584,0.36384328214574874,0.3659444742324435,
    0.36968317545749074,0.37486362446369,0.3802472107635389,0.3947838803172028,
    0.3996449036408523,0.4018512356356278,0.4072236101666943,0.41256330309081407,
    0.4177027464059101,0.4184771551820011,0.4207775757144963,0.43488242307180874,
    0.43810123163733394,0.45347462680502587,0.4585575428189736,0.461345684663321,
    0.4680305090451441,0.46914154817501386,0.4829435708957753,0.4839584942763934,
    0.4896822273997405,0.4977434139759097,0.5031898471113406,0.5143302885141922,
    0.5168415582005367,0.5236323603522955,0.5298001679353167,0.5368080502372451,
    0.5394626880103116,0.5439955900432788,0.5553613306040573,0.5626420839599869,
    0.5656874587152707,0.5666289494857834,0.6017987878033213,0.610024487089707,
    0.614724212529222,0.629574281120805,0.6312720462429319,0.6394729042740595,
    0.6458717827126227,0.6554744155358416,0.6658900672735714,0.6819210952705744,
    0.6894070080198288,0.6954113090841277,0.7049098194273632,0.7123945893014302,
    0.7370828347513815,0.7430083240104194,0.7751256570031169,0.7975483758070174,
    0.8237938362888106,0.8489466052013971,0.8677184367062846,0.8761113539314241,
    0.9023070888293631,0.9066938567486342,1.0
  ];

  function mk(name, attrs) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    return n;
  }

  function cssVar(node, name) {
    return getComputedStyle(node).getPropertyValue(name).trim();
  }

  function chebT(n, x) {
    const c = Math.max(-1, Math.min(1, x));
    return Math.cos(n * Math.acos(c));
  }

  /* The Chebyshev expansion of a Dirac delta at `lam`, normalized to P(lam) = 1.
     This is the same polynomial POLFED builds — not a sketch of one. */
  function deltaFilter(x, lam, K) {
    let num = 0, norm = 0;
    for (let n = 0; n <= K; n += 1) {
      const c = (2 - (n === 0 ? 1 : 0)) * Math.cos(n * Math.acos(lam));
      num  += c * chebT(n, x);
      norm += c * chebT(n, lam);
    }
    return num / Math.max(Math.abs(norm), 1e-12);
  }

  /* Sum of delta filters, renormalized so the peaks sit at 1. */
  function multiFilter(x, lams, K) {
    let v = 0;
    for (const lam of lams) v += deltaFilter(x, lam, K);
    return v;
  }

  function curvePath(fn, xs, ys, x0, x1, samples) {
    const pts = [];
    for (let i = 0; i <= samples; i += 1) {
      const x = x0 + (i / samples) * (x1 - x0);
      pts.push(xs(x).toFixed(2) + "," + ys(fn(x)).toFixed(2));
    }
    return "M" + pts.join(" L");
  }

  /* ---------- the hero figure ---------- */
  function renderHero() {
    const svg = document.getElementById("pf-hero");
    if (!svg) return;
    svg.replaceChildren();

    const fg    = cssVar(svg, "--pf-fg");
    const muted = cssVar(svg, "--pf-muted");
    const grid  = cssVar(svg, "--pf-grid");
    const blue  = cssVar(svg, "--pf-blue");
    const green = cssVar(svg, "--pf-green");

    const W = 880, L = 70, R = 40, TOP = 26, AXIS = 320, SPEC = 380;
    const PW = W - L - R;
    const XMIN = -1.12, XMAX = 1.12, YMIN = -0.25, YMAX = 1.0;

    const K = 50, lam = 0.0, cutoff = 0.17;

    const xs = x => L + ((x - XMIN) / (XMAX - XMIN)) * PW;
    const ys = y => AXIS - ((Math.max(YMIN, Math.min(YMAX, y)) - YMIN) / (YMAX - YMIN)) * (AXIS - TOP);

    /* cutoff line */
    svg.appendChild(mk("line", {
      x1: xs(-1), y1: ys(cutoff), x2: xs(1), y2: ys(cutoff),
      stroke: grid, "stroke-width": 2, "stroke-dasharray": "7 7", "stroke-linecap": "round"
    }));
    const clab = mk("text", { x: xs(-1) - 8, y: ys(cutoff) + 4, "text-anchor": "end", "font-size": 13, fill: muted });
    clab.textContent = "cutoff";
    svg.appendChild(clab);

    /* spectrum axis with arrow */
    svg.appendChild(mk("line", {
      x1: L, y1: SPEC, x2: W - R - 18, y2: SPEC,
      stroke: fg, "stroke-width": 2.5, "stroke-linecap": "round"
    }));
    svg.appendChild(mk("path", {
      d: "M " + (W - R - 18) + " " + (SPEC - 8) + " L " + (W - R) + " " + SPEC +
         " L " + (W - R - 18) + " " + (SPEC + 8) + " Z",
      fill: fg
    }));

    for (const t of [{ l: "-1", x: -1 }, { l: "+1", x: 1 }]) {
      svg.appendChild(mk("line", {
        x1: xs(t.x), y1: SPEC - 10, x2: xs(t.x), y2: SPEC + 10,
        stroke: fg, "stroke-width": 3, "stroke-linecap": "round"
      }));
      const n = mk("text", { x: xs(t.x), y: SPEC + 32, "text-anchor": "middle", "font-size": 17, fill: fg });
      n.textContent = t.l;
      svg.appendChild(n);
    }

    /* eigenvalue ticks: green only above the cutoff */
    for (const x of EIG) {
      const above = deltaFilter(x, lam, K) > cutoff;
      svg.appendChild(mk("line", {
        x1: xs(x), y1: SPEC - 7, x2: xs(x), y2: SPEC - (above ? 46 : 32),
        stroke: above ? green : muted, "stroke-width": 2,
        opacity: above ? 0.95 : 0.6, "stroke-linecap": "round"
      }));
    }

    /* transformed eigenvalues on the curve */
    for (const x of EIG) {
      const v = deltaFilter(x, lam, K);
      const above = v > cutoff;
      svg.appendChild(mk("circle", {
        cx: xs(x), cy: ys(v), r: above ? 6 : 5,
        fill: above ? green : muted, "fill-opacity": above ? 0.55 : 0.35,
        stroke: above ? green : muted, "stroke-opacity": above ? 0.95 : 0.55,
        "stroke-width": 2
      }));
    }

    /* target marker */
    svg.appendChild(mk("line", {
      x1: xs(lam), y1: SPEC + 14, x2: xs(lam), y2: SPEC - 14,
      stroke: fg, "stroke-width": 5, "stroke-linecap": "round"
    }));
    const lt = mk("text", { x: xs(lam), y: SPEC + 34, "text-anchor": "middle", "font-size": 19, fill: fg });
    lt.textContent = "λ";
    svg.appendChild(lt);

    /* the filter itself */
    svg.appendChild(mk("path", {
      d: curvePath(x => deltaFilter(x, lam, K), xs, ys, -1, 1, 900),
      fill: "none", stroke: blue, "stroke-width": 4,
      "stroke-linejoin": "round", "stroke-linecap": "round"
    }));

    const kl = mk("text", { x: W - R, y: TOP + 4, "text-anchor": "end", "font-size": 13, fill: muted });
    kl.textContent = "K = " + K;
    svg.appendChild(kl);
  }

  /* ---------- small card figures (all on a 200 x 112 canvas) ---------- */
  const CW = 200, CH = 130, CL = 5, CR = 5, CPW = CW - CL - CR;
  const cx = x => CL + ((x + 1.1) / 2.2) * CPW;
  const TICK_STEP = 0.04;   /* peak centres are placed on this grid so the
                               highlighted ticks sit symmetrically on the peak */

  function renderMiniFilter(id, lams, K, cutoff) {
    const svg = document.getElementById(id);
    if (!svg) return;
    svg.replaceChildren();

    const fg    = cssVar(svg, "--pf-fg");
    const muted = cssVar(svg, "--pf-muted");
    const grid  = cssVar(svg, "--pf-grid");
    const blue  = cssVar(svg, "--pf-blue");
    const green = cssVar(svg, "--pf-green");

    const TOP = 6, AXIS = 100, SPEC = 122;
    const ys = y => AXIS - ((Math.max(-0.25, Math.min(1, y)) + 0.25) / 1.25) * (AXIS - TOP);
    const P = x => multiFilter(x, lams, K);

    svg.appendChild(mk("line", {
      x1: cx(-1), y1: ys(cutoff), x2: cx(1), y2: ys(cutoff),
      stroke: grid, "stroke-width": 1.5, "stroke-dasharray": "5 4"
    }));

    svg.appendChild(mk("line", {
      x1: cx(-1.05), y1: SPEC, x2: cx(1.05), y2: SPEC,
      stroke: fg, "stroke-width": 1.5, opacity: .6, "stroke-linecap": "round"
    }));

    for (let i = 0; i <= Math.round(2 / TICK_STEP); i += 1) {
      const x = -1 + i * TICK_STEP;
      const sel = P(x) > cutoff;
      svg.appendChild(mk("line", {
        x1: cx(x), y1: SPEC - 3, x2: cx(x), y2: SPEC - (sel ? 12 : 7),
        stroke: sel ? green : muted, "stroke-width": 2,
        opacity: sel ? 0.95 : 0.5, "stroke-linecap": "round"
      }));
    }

    svg.appendChild(mk("path", {
      d: curvePath(P, cx, ys, -1, 1, 520),
      fill: "none", stroke: blue, "stroke-width": 2.8,
      "stroke-linejoin": "round", "stroke-linecap": "round"
    }));
  }

  /* Double: one peak up (mapped to +1), one peak down (mapped to -1).
     A state is selected when |P(x)| clears the cutoff at either end. */
  function renderDouble(id, lamUp, lamDown, K, cutoff) {
    const svg = document.getElementById(id);
    if (!svg) return;
    svg.replaceChildren();

    const fg    = cssVar(svg, "--pf-fg");
    const muted = cssVar(svg, "--pf-muted");
    const grid  = cssVar(svg, "--pf-grid");
    const blue  = cssVar(svg, "--pf-blue");
    const green = cssVar(svg, "--pf-green");

    const TOP = 4, BOT = 104, SPEC = 124;
    const ys = y => TOP + ((1 - Math.max(-1, Math.min(1, y))) / 2) * (BOT - TOP);
    const P = x => deltaFilter(x, lamUp, K) - deltaFilter(x, lamDown, K);

    svg.appendChild(mk("line", {
      x1: cx(-1.05), y1: ys(0), x2: cx(1.05), y2: ys(0),
      stroke: grid, "stroke-width": 1, opacity: .8
    }));

    for (const c of [cutoff, -cutoff]) {
      svg.appendChild(mk("line", {
        x1: cx(-1), y1: ys(c), x2: cx(1), y2: ys(c),
        stroke: grid, "stroke-width": 1.4, "stroke-dasharray": "5 4"
      }));
    }

    svg.appendChild(mk("line", {
      x1: cx(-1.05), y1: SPEC, x2: cx(1.05), y2: SPEC,
      stroke: fg, "stroke-width": 1.5, opacity: .6, "stroke-linecap": "round"
    }));

    for (let i = 0; i <= Math.round(2 / TICK_STEP); i += 1) {
      const x = -1 + i * TICK_STEP;
      const sel = Math.abs(P(x)) > cutoff;
      svg.appendChild(mk("line", {
        x1: cx(x), y1: SPEC - 3, x2: cx(x), y2: SPEC - (sel ? 11 : 6),
        stroke: sel ? green : muted, "stroke-width": 2,
        opacity: sel ? 0.95 : 0.5, "stroke-linecap": "round"
      }));
    }

    svg.appendChild(mk("path", {
      d: curvePath(P, cx, ys, -1, 1, 620),
      fill: "none", stroke: blue, "stroke-width": 2.8,
      "stroke-linejoin": "round", "stroke-linecap": "round"
    }));
  }

  /* Patched: one wide green window = a sum of three narrower filters.
     Laid out as an equation down the left edge:  window = p1 + p2 + p3. */
  function renderPatched() {
    const svg = document.getElementById("pf-card-patched");
    if (!svg) return;
    svg.replaceChildren();

    const fg    = cssVar(svg, "--pf-fg");
    const muted = cssVar(svg, "--pf-muted");
    const blue  = cssVar(svg, "--pf-blue");
    const green = cssVar(svg, "--pf-green");

    /* leave a gutter on the left for the = and + operators */
    const PL = 20, PR = 5;
    const px = x => PL + ((x + 1.1) / 2.2) * (CW - PL - PR);
    const OPX = 9;

    function op(sym, y, size) {
      const t = mk("text", { x: OPX, y: y, "text-anchor": "middle", "font-size": size, fill: muted });
      t.textContent = sym;
      svg.appendChild(t);
    }

    /* the wanted window, -0.5 .. 0.25 */
    svg.appendChild(mk("line", {
      x1: px(-1.05), y1: 16, x2: px(1.05), y2: 16,
      stroke: fg, "stroke-width": 1.8, opacity: .6, "stroke-linecap": "round"
    }));
    for (const t of [-1, 1]) {
      svg.appendChild(mk("line", { x1: px(t), y1: 10, x2: px(t), y2: 22, stroke: fg, "stroke-width": 2, opacity: .7 }));
    }
    svg.appendChild(mk("line", {
      x1: px(-0.5), y1: 16, x2: px(0.25), y2: 16,
      stroke: green, "stroke-width": 6, "stroke-linecap": "round"
    }));
    const wl = mk("text", { x: px(-0.125), y: 8, "text-anchor": "middle", "font-size": 9, fill: green });
    wl.textContent = "wanted";
    svg.appendChild(wl);

    /* Each patch is the real delta filter centred on its own slice, at the order
       that actually resolves that slice: K = 2.655*sqrt(1-lam^2)/delta. */
    const patches = [[-0.5, -0.25], [-0.25, 0.0], [0.0, 0.25]];
    const bases = [62, 92, 122];
    const ROWH = 24;

    op("=", (22 + (bases[0] - ROWH)) / 2 + 4, 17);

    patches.forEach(function (p, i) {
      const base = bases[i];
      const top  = base - ROWH;
      const lam  = (p[0] + p[1]) / 2;
      const dlt  = (p[1] - p[0]) / 2;
      const K    = Math.round(2.655 * Math.sqrt(1 - lam * lam) / dlt);

      const YLO = -0.4;
      const ys = y => base - ((Math.max(YLO, Math.min(1, y)) - YLO) / (1 - YLO)) * (base - top);

      if (i > 0) op("+", (bases[i - 1] + top) / 2 + 4, 16);

      svg.appendChild(mk("line", {
        x1: px(-1.05), y1: ys(0), x2: px(1.05), y2: ys(0),
        stroke: fg, "stroke-width": 1.1, opacity: .4, "stroke-linecap": "round"
      }));
      svg.appendChild(mk("line", {
        x1: px(p[0]), y1: ys(0), x2: px(p[1]), y2: ys(0),
        stroke: green, "stroke-width": 3.6, "stroke-linecap": "round"
      }));
      svg.appendChild(mk("path", {
        d: curvePath(x => deltaFilter(x, lam, K), px, ys, -1, 1, 700),
        fill: "none", stroke: blue, "stroke-width": 1.9,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
    });
  }

  /* Full: centred boxes, each holding its own filter peak. The peak itself is
     green; the tails outside the passband are grey. */
  function renderFull() {
    const svg = document.getElementById("pf-card-full");
    if (!svg) return;
    svg.replaceChildren();

    const blue  = cssVar(svg, "--pf-blue");
    const green = cssVar(svg, "--pf-green");
    const muted = cssVar(svg, "--pf-muted");
    const fg    = cssVar(svg, "--pf-fg");

    const n = 6, gap = 4, boxW = 25, boxH = 38;
    const total = n * boxW + (n - 1) * gap;
    const x0 = (CW - total) / 2;
    const yTop = 34;
    const K = 14, cutoff = 0.17;

    /* x-range over which the peak clears the cutoff */
    let xa = 0, xb = 0;
    for (let x = -1; x <= 1; x += 0.002) {
      if (deltaFilter(x, 0, K) > cutoff) { if (xa === 0) xa = x; xb = x; }
    }

    for (let i = 0; i < n; i += 1) {
      const bx = x0 + i * (boxW + gap);
      svg.appendChild(mk("rect", {
        x: bx, y: yTop, width: boxW, height: boxH, rx: 3,
        fill: "none", stroke: blue, "stroke-width": 1.6, opacity: .9
      }));

      /* inset so the curve always terminates inside the box */
      const bxs = t => bx + 2 + ((t + 1) / 2) * (boxW - 4);
      const bys = v => (yTop + boxH - 5) - Math.max(0, Math.min(1, v)) * (boxH - 11);
      const f = t => deltaFilter(t, 0.0, K);

      svg.appendChild(mk("path", {
        d: curvePath(f, bxs, bys, -1, 1, 160),
        fill: "none", stroke: muted, "stroke-width": 1.5, opacity: .75,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
      svg.appendChild(mk("path", {
        d: curvePath(f, bxs, bys, xa, xb, 80),
        fill: "none", stroke: green, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
    }

    /* spectrum axis under the tiles, ticked at the rescaled extremes */
    const baseY = yTop + boxH + 12;
    const bxL = x0, bxR = x0 + total;

    svg.appendChild(mk("line", {
      x1: bxL, y1: baseY, x2: bxR, y2: baseY,
      stroke: fg, "stroke-width": 1.5, opacity: .55, "stroke-linecap": "round"
    }));

    for (const t of [{ x: bxL, l: "-1" }, { x: bxR, l: "+1" }]) {
      svg.appendChild(mk("line", {
        x1: t.x, y1: baseY - 6, x2: t.x, y2: baseY + 6,
        stroke: fg, "stroke-width": 2, opacity: .75, "stroke-linecap": "round"
      }));
      const lab = mk("text", {
        x: t.x, y: baseY + 20, "text-anchor": "middle", "font-size": 12, fill: fg
      });
      lab.textContent = t.l;
      svg.appendChild(lab);
    }
  }

  function renderAll() {
    renderHero();
    renderMiniFilter("pf-card-single", [0.0], 50, 0.17);
    renderDouble("pf-card-double", -0.44, 0.44, 52, 0.30);
    renderPatched();
    renderFull();
  }

  renderAll();

  /* re-render on theme switch so the colours follow */
  new MutationObserver(renderAll).observe(document.documentElement, {
    attributes: true, attributeFilter: ["class"]
  });
})();
</script>
```
