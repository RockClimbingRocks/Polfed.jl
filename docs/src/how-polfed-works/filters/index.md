# [Filters](@id filters)

A POLFED filter is a polynomial ``P`` that is large where you want eigenvalues and small
everywhere else. Getting one is stage ⑤ of [the workflow](@ref how_polfed_works), and you
never have to choose it by hand — but understanding the vocabulary makes the report
readable and the failure modes obvious.

## The delta filter

The workhorse. Since Chebyshev polynomials live on ``[-1,1]``, the Hamiltonian is rescaled
first (stage ①). For a rescaled target ``\tilde\lambda`` and order ``K``,

```math
P_{\tilde{\lambda}}^{K}(\tilde{H})
= \frac{1}{\chi}
  \sum_{n=0}^{K} c_n^{\tilde{\lambda}} T_n(\tilde{H}),
\qquad
c_n^{\tilde{\lambda}}
=
(2 - \delta_{n,0})\cos\!\left(n\arccos(\tilde{\lambda})\right).
```

These coefficients are the Chebyshev expansion of a Dirac delta at ``\tilde\lambda``, and
``\chi`` normalizes the filter so that ``P_{\tilde{\lambda}}^{K}(\tilde{\lambda}) = 1``.
The result is a sharp peak at the target with small oscillating side lobes elsewhere.

Move the slider: with ``\tilde\lambda`` fixed at 0, increasing ``K`` sharpens the peak, and
fewer eigenvalues clear the threshold.

```@raw html
<div id="polfed-filter-demo" class="polfed-filter-demo">
  <style>
    .polfed-filter-demo {
      --polfed-filter-bg: #ffffff;
      --polfed-filter-fg: #24292f;
      --polfed-filter-muted: #6e7781;
      --polfed-filter-grid: #d0d7de;
      --polfed-filter-blue: #4e79a7;
      --polfed-filter-orange: #ffa62b;
      --polfed-filter-red: #2ca05a;
      --polfed-filter-panel: #f6f8fa;
      --polfed-filter-border: #d0d7de;
      margin: 1.5rem 0;
    }

    html.theme--documenter-dark .polfed-filter-demo,
    html.theme--catppuccin-frappe .polfed-filter-demo,
    html.theme--catppuccin-macchiato .polfed-filter-demo,
    html.theme--catppuccin-mocha .polfed-filter-demo {
      --polfed-filter-bg: #15191a;
      --polfed-filter-fg: #f3f5f7;
      --polfed-filter-muted: #a7b0b8;
      --polfed-filter-grid: #58636a;
      --polfed-filter-panel: #1f2428;
      --polfed-filter-border: #58636a;
      --polfed-filter-red: #3fd07f;
    }

    .polfed-filter-controls {
      display: grid;
      gap: 0.75rem;
      margin: 0 0 1rem;
      max-width: 880px;
      padding: 1rem;
      border: 1px solid var(--polfed-filter-border);
      border-radius: 8px;
      background: var(--polfed-filter-panel);
    }

    .polfed-filter-control-row {
      display: grid;
      grid-template-columns: minmax(8rem, auto) 1fr minmax(5rem, auto);
      gap: 0.75rem;
      align-items: center;
    }

    .polfed-filter-control-row input { width: 100%; }

    .polfed-filter-readout {
      color: var(--polfed-filter-muted);
      font-size: 0.95rem;
    }

    .polfed-filter-canvas {
      display: block;
      width: 100%;
      max-width: 880px;
      height: auto;
      border: 1px solid var(--polfed-filter-border);
      border-radius: 8px;
      background: var(--polfed-filter-bg);
    }

    .polfed-filter-demo text { font-family: inherit; }

    @media (max-width: 620px) {
      .polfed-filter-control-row { grid-template-columns: 1fr; }
    }
  </style>

  <div class="polfed-filter-controls">
    <div class="polfed-filter-control-row">
      <label for="polfed-filter-order">Polynomial order K</label>
      <input id="polfed-filter-order" type="range" min="10" max="100" step="1" value="50">
      <output id="polfed-filter-order-output" for="polfed-filter-order">K = 50</output>
    </div>
    <div id="polfed-filter-readout" class="polfed-filter-readout"></div>
  </div>

  <svg id="polfed-filter-svg" class="polfed-filter-canvas" viewBox="0 0 880 420" role="img" aria-labelledby="polfed-filter-title polfed-filter-desc">
    <title id="polfed-filter-title">Interactive polynomial filtering transformation</title>
    <desc id="polfed-filter-desc">Chebyshev polynomial filter with target lambda fixed at zero and adjustable polynomial order K.</desc>
  </svg>

  <script>
    (function () {
      const root = document.getElementById("polfed-filter-demo");
      if (!root) return;

      const svg = document.getElementById("polfed-filter-svg");
      const slider = document.getElementById("polfed-filter-order");
      const orderOutput = document.getElementById("polfed-filter-order-output");
      const readout = document.getElementById("polfed-filter-readout");
      const ns = "http://www.w3.org/2000/svg";

      const width = 880;

      const plot = { left: 72, right: 36, top: 24, axis: 315, spectrum: 375 };
      const plotWidth = width - plot.left - plot.right;

      const lambda = 0;
      const threshold = 0.17;

      const xMin = -1.1;
      const xMax = 1.1;
      const yMin = -0.25;
      const yMax = 1.00;

      const eigenvalues = [
        -1.0, -0.8862397328904057, -0.8138365850501954, -0.7643225411127533,-0.7187609110983968, -0.6997605808304965, -0.6781724053153465,-0.6628769734462476, -0.6555008533116723, -0.6267389263172705,-0.6069325920558576, -0.6002697427080279, -0.5784162092016458,-0.5638818226007932, -0.5309181092225888, -0.5197656009827994,-0.5104954691723352, -0.5088660884158803, -0.502195481672162,-0.4881248013948633, -0.4724939941512107, -0.4697075462260785,-0.4449499266605002, -0.43677100051172824, -0.4243705078324369,-0.42009175481101313, -0.41627304357147416, -0.41119006162761274,-0.4052875659614221, -0.39225647335893027, -0.3851810983151796,-0.3799463497733936, -0.3737745003460363, -0.36508961483402547,-0.3608739904704648, -0.35163453692905794, -0.3469545307678019,-0.33991063817989614, -0.32747721682499314, -0.3246955654604279,-0.3172804068667309, -0.3122363925302267, -0.30871338178175345,-0.30163616603643506, -0.2994281274946959, -0.28956444712674695,-0.28418013090233296, -0.2746346375848551, -0.2683452800939827,-0.26329369973122363, -0.25504123255503164, -0.25251725438319267,-0.24679102758060065, -0.23658361409521564, -0.2314840584554421,-0.2301222669904662, -0.22855784733457538, -0.22108833084033364,-0.21540027655705468, -0.21099987743893953, -0.2033370267187006,-0.19712602535127383, -0.19161715382325875, -0.18239060161332082,-0.1809874217975776, -0.173864611043623, -0.17031650017325128,-0.16712914105240806, -0.1621283772103342, -0.15894974145075402,-0.15269510391876634, -0.1471918818891129, -0.1420416180619292,-0.13871738368798317, -0.13588341801987786, -0.13246457153472915,-0.130127852071681, -0.12589553076887303, -0.11891165408789882,-0.11539229178489978, -0.11142936222869644, -0.10325531881385082,-0.09404115488468896, -0.0892641044448668, -0.08474082513987984,-0.08227428188208782, -0.08094542463359694, -0.07845417700480453,-0.07494941563761186, -0.07329354656510167, -0.06863519250851163,-0.06378224595255806, -0.062494584553020704, -0.05930252817499551,-0.047478133864845444, -0.040243455193717644, -0.03167090854535132,-0.029108668031192397, -0.024759870329775606, -0.022104449767448974,-0.018782930979589794, -0.017863726135231402, -0.011915376269962204,-0.005997862725306875, 0.0002514069329118227, 0.0022093438372056315,0.008689271448713986, 0.009911066055664134, 0.014545581192528868,0.01990513989628108, 0.021471310068868057, 0.024686044604475103,0.025945802650147728, 0.03216032682617953, 0.03800461006548309,0.03856014226704779, 0.04543405119118332, 0.0507051142705264,0.05627169592490444, 0.06126334264160843, 0.0672931436680215,0.07009003878114714, 0.0718836232280733, 0.07384239806812845,0.08066382079865139, 0.08582789159684678, 0.08939682187285115,0.0904054271055243, 0.09555923962018127, 0.09855140803690669,0.102745908921033, 0.10523728726186328, 0.10674882290861616,0.11240430521941186, 0.11767418642222564, 0.12620295760540703,0.13138479876666811, 0.13374914052446277, 0.13627721010803942,0.1402721263504261, 0.14484272006761953, 0.14876277273055652,0.15330679455611046, 0.15624948685304535, 0.16025501036628853,0.16637517083852663, 0.16878921059903265, 0.17759945757882037,0.1818197297604848, 0.18349275480126576, 0.18759046681048175,0.19005884234454704, 0.19446935150261835, 0.1983362124713565,0.20335193393592105, 0.20478260096569167, 0.21250091281234645,0.21510537329081536, 0.22065106083208885, 0.2208682299892854,0.22234280695720804, 0.22547089362266895, 0.23211050308410885,0.23511749180305097, 0.24002410188427092, 0.24192233935294913,0.249677508727342, 0.2552365091017531, 0.25918271372364593,0.2624301731343962, 0.26733025423682827, 0.2702836011953202,0.27611102783337643, 0.2833798512663728, 0.2904226500474597,0.29387179884183245, 0.30007247287366534, 0.3030913231354488,0.3042555865594446, 0.3085684814358684, 0.31950839763431915,0.3228722143612751, 0.32770917659383053, 0.33339455506645915,0.3364569764669032, 0.3395355415658398, 0.3439433061515341,0.3479507575160735, 0.3497871844905475, 0.3582338352854033,0.3597197025114584, 0.36384328214574874, 0.3659444742324435,0.36968317545749074, 0.37486362446369, 0.3802472107635389,0.3947838803172028, 0.3996449036408523, 0.4018512356356278,0.4072236101666943, 0.41256330309081407, 0.4177027464059101,0.4184771551820011, 0.4207775757144963, 0.43488242307180874,0.43810123163733394, 0.45347462680502587, 0.4585575428189736,0.461345684663321, 0.4680305090451441, 0.46914154817501386,0.4829435708957753, 0.4839584942763934, 0.4896822273997405,0.4977434139759097, 0.5031898471113406, 0.5143302885141922,0.5168415582005367, 0.5236323603522955, 0.5298001679353167,0.5368080502372451, 0.5394626880103116, 0.5439955900432788,0.5553613306040573, 0.5626420839599869, 0.5656874587152707,0.5666289494857834, 0.6017987878033213, 0.610024487089707,0.614724212529222, 0.629574281120805, 0.6312720462429319,0.6394729042740595, 0.6458717827126227, 0.6554744155358416,0.6658900672735714, 0.6819210952705744, 0.6894070080198288,0.6954113090841277, 0.7049098194273632, 0.7123945893014302,0.7370828347513815, 0.7430083240104194, 0.7751256570031169,0.7975483758070174, 0.8237938362888106, 0.8489466052013971,0.8677184367062846, 0.8761113539314241, 0.9023070888293631,0.9066938567486342, 1.0
      ];

      function cssVar(name) {
        return getComputedStyle(root).getPropertyValue(name).trim();
      }

      function el(name, attrs, children) {
        const node = document.createElementNS(ns, name);
        for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
        for (const child of children || []) node.appendChild(child);
        return node;
      }

      function text(value, x, y, attrs) {
        const node = el("text", Object.assign({ x, y, fill: cssVar("--polfed-filter-fg") }, attrs || {}));
        node.textContent = value;
        return node;
      }

      function xScale(x) {
        return plot.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
      }

      function yScale(y) {
        const clipped = Math.max(yMin, Math.min(yMax, y));
        return plot.axis - ((clipped - yMin) / (yMax - yMin)) * (plot.axis - plot.top);
      }

      function chebyshevT(n, x) {
        const clipped = Math.max(-1, Math.min(1, x));
        return Math.cos(n * Math.acos(clipped));
      }

      function coefficient(n) {
        return (2 - (n === 0 ? 1 : 0)) * Math.cos(n * Math.acos(lambda));
      }

      function filterValue(x, order) {
        let numerator = 0;
        let normalization = 0;
        for (let n = 0; n <= order; n += 1) {
          const c = coefficient(n);
          numerator += c * chebyshevT(n, x);
          normalization += c * chebyshevT(n, lambda);
        }
        return numerator / Math.max(Math.abs(normalization), 1e-12);
      }

      function pathFromSamples(order) {
        const samples = [];
        const fMin = -1.0;
        const fMax = 1.0;
        for (let i = 0; i <= 720; i += 1) {
          const x = fMin + (i / 720) * (fMax - fMin);
          samples.push([xScale(x), yScale(filterValue(x, order))]);
        }
        return samples.map(function (p, i) {
          return (i === 0 ? "M" : "L") + p[0].toFixed(2) + "," + p[1].toFixed(2);
        }).join(" ");
      }

      function render() {
        const order = Number(slider.value);
        orderOutput.textContent = "K = " + order;
        svg.replaceChildren();

        const title = el("title", { id: "polfed-filter-title" });
        title.textContent = "Interactive polynomial filtering transformation";
        svg.appendChild(title);

        const desc = el("desc", { id: "polfed-filter-desc" });
        desc.textContent =
          "Chebyshev polynomial filter with target lambda fixed at zero and adjustable polynomial order K.";
        svg.appendChild(desc);

        const fg = cssVar("--polfed-filter-fg");
        const grid = cssVar("--polfed-filter-grid");
        const blue = cssVar("--polfed-filter-blue");
        const orange = cssVar("--polfed-filter-orange");
        const red = cssVar("--polfed-filter-red");

        /* Cutoff line at transformation value P = 0.17. */
        svg.appendChild(el("line", {
          x1: xScale(-1), y1: yScale(threshold),
          x2: xScale(1),  y2: yScale(threshold),
          stroke: grid, "stroke-width": 2,
          "stroke-dasharray": "7 7", "stroke-linecap": "round"
        }));

        /* Untransformed spectrum axis. */
        svg.appendChild(el("line", {
          x1: plot.left, y1: plot.spectrum,
          x2: width - plot.right - 18, y2: plot.spectrum,
          stroke: fg, "stroke-width": 2.5, "stroke-linecap": "round"
        }));

        svg.appendChild(el("path", {
          d: "M " + (width - plot.right - 18) + " " + (plot.spectrum - 8) +
             " L " + (width - plot.right) + " " + plot.spectrum +
             " L " + (width - plot.right - 18) + " " + (plot.spectrum + 8) + " Z",
          fill: fg
        }));

        const tickPositions = [{ label: "-1", x: -1.0 }, { label: "+1", x: 1.0 }];

        for (const tick of tickPositions) {
          const tickX = xScale(tick.x);
          svg.appendChild(el("line", {
            x1: tickX, y1: plot.spectrum - 10,
            x2: tickX, y2: plot.spectrum + 10,
            stroke: fg, "stroke-width": 3, "stroke-linecap": "round"
          }));
          svg.appendChild(text(tick.label, tickX - 12, plot.spectrum + 34, { "font-size": 18, fill: fg }));
        }

        /* Eigenvalue ticks; those above the cutoff are highlighted. */
        let highlighted = 0;

        for (const x of eigenvalues) {
          const value = filterValue(x, order);
          const isTarget = value > threshold;
          const color = isTarget ? red : orange;
          if (isTarget) highlighted += 1;

          svg.appendChild(el("line", {
            x1: xScale(x), y1: plot.spectrum - 7,
            x2: xScale(x), y2: plot.spectrum - (isTarget ? 50 : 35),
            stroke: color, "stroke-width": 2.0,
            opacity: isTarget ? 0.92 : 0.72, "stroke-linecap": "round"
          }));
        }

        /* Transformed eigenvalue markers. */
        for (const x of eigenvalues) {
          const value = filterValue(x, order);
          const circColor = value < threshold ? orange : red;

          svg.appendChild(el("circle", {
            cx: xScale(x), cy: yScale(value), r: 6.0,
            fill: circColor, "fill-opacity": 0.5,
            stroke: circColor, "stroke-opacity": 0.95, "stroke-width": 2
          }));
        }

        /* Target marker at lambda = 0. */
        const lambdaX = xScale(lambda);

        svg.appendChild(el("line", {
          x1: lambdaX, y1: plot.spectrum + 14,
          x2: lambdaX, y2: plot.spectrum - 14,
          stroke: fg, "stroke-width": 5, "stroke-linecap": "round"
        }));

        svg.appendChild(text("λ", lambdaX - 5, plot.spectrum + 40, { "font-size": 20, fill: fg }));

        /* The filter curve. */
        svg.appendChild(el("path", {
          d: pathFromSamples(order),
          fill: "none", stroke: blue, "stroke-width": 4,
          "stroke-linejoin": "round", "stroke-linecap": "round"
        }));

        readout.textContent = highlighted + " transformed eigenvalues are above the cutoff.";
      }

      slider.addEventListener("input", render);

      const observer = new MutationObserver(render);
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

      render();
    })();
  </script>
</div>
```

## Three words you will see in the report

**Cutoff.** The threshold value a transformed eigenvalue must clear to be considered part of
the target set — the dashed line in the figure. It is a design parameter, not a physical
quantity.

**Passband.** The set of rescaled energies whose filter value clears the cutoff: the region
under the main peak. This is what the filter actually admits. It may come out **wider** than
the interval you requested — the pipeline trims the result down to what you asked for.

**Leakage.** A side lobe that also climbs above the cutoff. That would let eigenpairs from a
completely disconnected part of the spectrum into your window, mixed in with the ones you
want and indistinguishable from them. Filter validation exists to prevent exactly this, and
it is why the order cannot simply be set to "whatever is cheap".

## Choosing the order

The peak width scales as ``\delta \propto 1/K``, and for the delta filters the order that
puts the interval half-width ``\delta`` at the cutoff is close to

```math
K \approx \frac{2.655\sqrt{1-\tilde\lambda^{2}}}{\delta}.
```

POLFED resolves ``K`` by bisection against the actual cutoff rather than trusting that
formula, but the scaling is the thing to remember:

- **Sharper peak** ⇒ better separation of the target eigenvalues ⇒ fewer Lanczos iterations.
- **Sharper peak** ⇒ larger ``K`` ⇒ more sparse matvecs *per* Lanczos iteration.
- **Narrower window** ⇒ larger ``K``. Equivalently, a larger `howmany` widens the interval
  and **lowers** the order — asking for more eigenpairs makes the filter cheaper, not more
  expensive. What grows with `howmany` instead is the projected Lanczos matrix, and that is
  the cost [Patched POLFED](@ref tutorial_patched) addresses.

## The filter types

| Filter | Shape | Order | Can leak? | Used when |
| :--- | :--- | :--- | :--- | :--- |
| `DeltaUpFilter` | one peak mapped to ``+1`` | resolved by bisection from the cutoff | yes — validated | interior windows |
| `DeltaDownFilter` | one peak mapped to ``-1`` | resolved by bisection from the cutoff | yes — validated | interior windows |
| `DeltaUpSmoothFilter` | one peak mapped to ``+1``, Jackson-damped | bisection, ≈ 2× the sharp order | side lobes ~1000× smaller | fallback when a sharp peak leaks |
| `EdgeFilter` | monotone monomial power | fixed (default 3) | **no** | window touches ``\pm 1`` |
| [`DoubleDeltaFilter`](@ref double_filter) | two peaks, one up one down | one order **per peak**, from the cutoff | yes — including *between* the peaks | two windows at once |

The `EdgeFilter` is the odd one out, and usefully so. It is monotone with
``P(\pm 1) = \pm 1``, so it has no side lobes and **cannot leak** — there is no passband to
validate, only an exposed unbounded half. That robustness is why the patched solver peels
spectrum edges with it first.

## [See the shapes](@id filter_shapes)

Each family looks quite different, and the differences are the whole story. Switch between them:

```@raw html
<style>
.pf-shapes {
  --pfs-bg: #ffffff; --pfs-fg: #24292f; --pfs-muted: #6e7781; --pfs-grid: #d0d7de;
  --pfs-blue: #4e79a7; --pfs-green: #2ca05a; --pfs-amber: #d97706; --pfs-red: #cf222e;
  --pfs-purple: #8250df;
  --pfs-panel: #f6f8fa; --pfs-border: #d0d7de;
  margin: 1.5rem 0;
}
html.theme--documenter-dark .pf-shapes,
html.theme--catppuccin-frappe .pf-shapes,
html.theme--catppuccin-macchiato .pf-shapes,
html.theme--catppuccin-mocha .pf-shapes {
  --pfs-bg: #15191a; --pfs-fg: #f3f5f7; --pfs-muted: #a7b0b8; --pfs-grid: #58636a;
  --pfs-blue: #7ea6cc; --pfs-green: #3fd07f; --pfs-amber: #e0a336; --pfs-red: #ff7b72;
  --pfs-purple: #c49bff;
  --pfs-panel: #1f2428; --pfs-border: #58636a;
}
.pf-shapes-controls {
  display: grid; gap: 0.7rem; margin: 0 0 1rem; padding: 1rem;
  border: 1px solid var(--pfs-border); border-radius: 8px; background: var(--pfs-panel);
}
.pf-shapes-row {
  display: grid; grid-template-columns: minmax(9rem, auto) 1fr minmax(6rem, auto);
  gap: 0.75rem; align-items: center;
}
.pf-shapes-row input, .pf-shapes-row select { width: 100%; }
.pf-shapes-row output { color: var(--pfs-muted); font-size: 0.92rem; font-variant-numeric: tabular-nums; }
.pf-shapes svg {
  display: block; width: 100%; height: auto;
  border: 1px solid var(--pfs-border); border-radius: 8px; background: var(--pfs-bg);
}
.pf-shapes text { font-family: inherit; }
.pf-shapes-note {
  margin-top: 0.7rem; font-size: 0.88rem; color: var(--pfs-muted);
}
@media (max-width: 640px) { .pf-shapes-row { grid-template-columns: 1fr; } }
</style>

<div class="pf-shapes" id="pf-shapes">
  <div class="pf-shapes-controls">
    <div class="pf-shapes-row">
      <label for="pfs-kind">filter</label>
      <select id="pfs-kind">
        <option value="up" selected>DeltaUpFilter</option>
        <option value="down">DeltaDownFilter</option>
        <option value="smooth">DeltaUpSmoothFilter</option>
        <option value="edge_hi">EdgeFilter · :highest</option>
        <option value="edge_lo">EdgeFilter · :lowest</option>
        <option value="edge_both">EdgeFilter · :both</option>
        <option value="double">DoubleDeltaFilter</option>
      </select>
      <output id="pfs-kind-out"></output>
    </div>
    <div class="pf-shapes-row" id="pfs-cmp-row">
      <label for="pfs-cmp">compare with</label>
      <select id="pfs-cmp">
        <option value="none" selected>— (single filter)</option>
        <option value="up">DeltaUpFilter</option>
        <option value="smooth">DeltaUpSmoothFilter</option>
        <option value="down">DeltaDownFilter</option>
        <option value="double">DoubleDeltaFilter</option>
      </select>
      <output id="pfs-cmp-out"></output>
    </div>
    <div class="pf-shapes-row" id="pfs-order-row">
      <label for="pfs-order" id="pfs-order-label">order K</label>
      <input id="pfs-order" type="range" min="4" max="160" step="1" value="40">
      <output id="pfs-order-out">40</output>
    </div>
    <div class="pf-shapes-row" id="pfs-lam-row">
      <label for="pfs-lam" id="pfs-lam-label">target λ</label>
      <input id="pfs-lam" type="range" min="-90" max="90" step="1" value="0">
      <output id="pfs-lam-out">0.00</output>
    </div>
    <div class="pf-shapes-row" id="pfs-cut-row">
      <label for="pfs-cut">cutoff</label>
      <input id="pfs-cut" type="range" min="1" max="40" step="1" value="17">
      <output id="pfs-cut-out">0.17</output>
    </div>
  </div>

  <svg id="pfs-svg" viewBox="0 0 880 428" role="img" aria-label="The shapes of the POLFED filter families"></svg>
  <div class="pf-shapes-note" id="pfs-note"></div>
</div>

<script>
(function () {
  const root = document.getElementById("pf-shapes");
  if (!root) return;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("pfs-svg");
  const elKind = document.getElementById("pfs-kind");
  const elOrd = document.getElementById("pfs-order");
  const elLam = document.getElementById("pfs-lam");
  const elCut = document.getElementById("pfs-cut");
  const elCmp = document.getElementById("pfs-cmp");
  const rowCmp = document.getElementById("pfs-cmp-row");
  const outCmp = document.getElementById("pfs-cmp-out");
  const outKind = document.getElementById("pfs-kind-out");
  const outOrd = document.getElementById("pfs-order-out");
  const outLam = document.getElementById("pfs-lam-out");
  const outCut = document.getElementById("pfs-cut-out");
  const note = document.getElementById("pfs-note");
  const rowOrd = document.getElementById("pfs-order-row");
  const rowLam = document.getElementById("pfs-lam-row");
  const rowCut = document.getElementById("pfs-cut-row");
  const lblOrd = document.getElementById("pfs-order-label");
  const lblLam = document.getElementById("pfs-lam-label");

  function mk(n, a) {
    const e = document.createElementNS(NS, n);
    for (const k in a || {}) e.setAttribute(k, a[k]);
    return e;
  }
  function txt(v, x, y, a) {
    const e = mk("text", Object.assign({ x: x, y: y }, a || {}));
    e.textContent = v;
    return e;
  }
  function cssVar(n) { return getComputedStyle(root).getPropertyValue(n).trim(); }

  const chebT = (n, x) => Math.cos(n * Math.acos(Math.max(-1, Math.min(1, x))));
  const cRaw = (n, lam) => (2 - (n === 0 ? 1 : 0)) * Math.cos(n * Math.acos(lam));

  /* A Gaussian-distributed spectrum — the generic shape of a many-body density of
     states — seeded so the picture is stable between reloads. */
  const EIG = (function () {
    let seed = 20260728;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const out = [];
    while (out.length < 698) {
      const u = Math.max(1e-12, rnd()), v = rnd();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 0.30;
      if (g > -0.99 && g < 0.99) out.push(g);
    }
    /* The extreme eigenvalues sit exactly ON ±1 after rescaling — that is what the
       rescaling is defined to do — so pin one to each end rather than leaving a gap. */
    out.push(-1, 1);
    return out.sort((x, y) => x - y);
  })();

  /* Jackson damping — the same kernel the DoS uses, applied to the delta expansion. */
  function jackson(n, N) {
    const q = Math.PI / (N + 1);
    return ((N - n + 1) * Math.cos(n * q) + Math.sin(n * q) / Math.tan(q)) / (N + 1);
  }

  function deltaAt(x, lam, K, damp) {
    let num = 0, nrm = 0;
    for (let n = 0; n <= K; n += 1) {
      const g = damp ? jackson(n, K) : 1;
      const c = cRaw(n, lam) * g;
      num += c * chebT(n, x);
      nrm += c * chebT(n, lam);
    }
    return num / Math.max(Math.abs(nrm), 1e-12);
  }

  /* P = a·U + b·D, with a, b from the 2×2 solve that pins BOTH peaks. */
  function doubleAt(x, l1, l2, K) {
    const U = t => { let s = 0; for (let n = 0; n <= K; n += 1) s += cRaw(n, l1) * chebT(n, t); return s; };
    const D = t => { let s = 0; for (let n = 0; n <= K; n += 1) s += cRaw(n, l2) * chebT(n, t); return s; };
    const U1 = U(l1), U2 = U(l2), D1 = D(l1), D2 = D(l2);
    const det = U1 * D2 - D1 * U2;
    const a = (D1 + D2) / det, b = -(U1 + U2) / det;
    return a * U(x) + b * D(x);
  }

  const EDGE_ORDER = 3;   /* Defaults.filters.edge.order — what POLFED actually uses */

  const SPEC = {
    up:        { order: [4,160,40], lam: true,  cut: "upper",  group: "delta",  name: "DeltaUpFilter" },
    down:      { order: [4,160,40], lam: true,  cut: "lower",  group: "delta",  name: "DeltaDownFilter" },
    smooth:    { order: [4,160,60], lam: true,  cut: "upper",  group: "delta",  name: "DeltaUpSmoothFilter" },
    edge_hi:   { order: [1,12,3],   step: 1, lam: false, cut: "none", group: "edge", name: "EdgeFilter :highest" },
    edge_lo:   { order: [1,12,3],   step: 1, lam: false, cut: "none", group: "edge", name: "EdgeFilter :lowest" },
    edge_both: { order: [1,13,3],   step: 2, lam: false, cut: "none", group: "edge", name: "EdgeFilter :both" },
    double:    { order: [4,120,40], lam: true,  cut: "both",   group: "double", name: "DoubleDeltaFilter" }
  };

  /* Only like-for-like overlays are offered. A delta filter and an edge filter are drawn on
     different footings — different cutoff regime, different order semantics — so putting them
     on one pair of axes would invite a comparison that means nothing. The double filter is
     alone in its group: it carries two peaks and two cutoffs, and nothing else does. */
  function refreshCompareOptions() {
    const k = elKind.value, g = SPEC[k].group;
    const peers = Object.keys(SPEC).filter(key => key !== k && SPEC[key].group === g);

    if (peers.length === 0) {
      elCmp.disabled = true;
      elCmp.replaceChildren(new Option("— not comparable", "none"));
      elCmp.value = "none";
      outCmp.textContent = g === "double" ? "two peaks — nothing to compare against" : "";
      rowCmp.style.opacity = "0.55";
      return;
    }

    const keep = elCmp.value;
    elCmp.disabled = false;
    rowCmp.style.opacity = "";
    elCmp.replaceChildren(new Option("— (single filter)", "none"));
    for (const key of peers) elCmp.appendChild(new Option(SPEC[key].name, key));
    elCmp.value = peers.includes(keep) ? keep : "none";
  }

  function syncControls() {
    const k = elKind.value, sp = SPEC[k];
    refreshCompareOptions();
    rowOrd.style.display = sp.order ? "" : "none";
    if (sp.order) {
      const step = sp.step || 1;
      if (Number(elOrd.min) !== sp.order[0] || Number(elOrd.max) !== sp.order[1] ||
          Number(elOrd.step) !== step) {
        elOrd.min = sp.order[0]; elOrd.max = sp.order[1];
        elOrd.step = step; elOrd.value = sp.order[2];
      }
      /* :both is monotone only for odd powers, so the slider steps by 2 rather than
         silently rounding an even choice up — the number shown is the number plotted. */
      lblOrd.textContent = k.startsWith("edge") ? "power p" : "order K";
    }
    rowLam.style.display = sp.lam ? "" : "none";
    rowCut.style.display = sp.cut === "none" ? "none" : "";
    lblLam.textContent = k === "double" ? "separation ±λ" : "target λ";
    if (k === "double" && Number(elLam.value) < 20) elLam.value = 45;
  }

  /* One factory for both the selected filter and the comparison overlay, so the two are
     always built at the SAME order and target — otherwise the comparison says nothing. */
  function makeP(kind, K, lam) {
    if (kind === "up")        return { P: x => deltaAt(x, lam, K, false), targets: [lam] };
    if (kind === "down")      return { P: x => -deltaAt(x, lam, K, false), targets: [lam] };
    if (kind === "smooth")    return { P: x => deltaAt(x, lam, K, true), targets: [lam] };
    if (kind === "edge_hi")   return { P: x => Math.pow((1 + x) / 2, K), targets: [] };
    if (kind === "edge_lo")   return { P: x => -Math.pow((1 - x) / 2, K), targets: [] };
    if (kind === "edge_both") return { P: x => Math.pow(x, K), targets: [] };
    const l1 = -Math.abs(lam), l2 = Math.abs(lam);
    return { P: x => doubleAt(x, l1, l2, K), targets: [l1, l2] };
  }

  function render() {
    const kind = elKind.value, sp = SPEC[kind];
    const K = sp.order ? Number(elOrd.value) : EDGE_ORDER;
    const lam = Number(elLam.value) / 100;
    const cut = Number(elCut.value) / 100;
    if (sp.order) outOrd.textContent = K;
    outLam.textContent = lam.toFixed(2);
    outCut.textContent = cut.toFixed(2);
    outKind.textContent = sp.name;

    const built = makeP(kind, K, lam);
    const P = built.P, targets = built.targets;

    const cmpKind = elCmp.value;
    const cmp = cmpKind === "none" || cmpKind === kind ? null : makeP(cmpKind, K, lam);
    outCmp.textContent = cmp ? "same K, same λ" : "";

    const fg = cssVar("--pfs-fg"), muted = cssVar("--pfs-muted"), grid = cssVar("--pfs-grid");
    const blue = cssVar("--pfs-blue"), green = cssVar("--pfs-green");
    const amber = cssVar("--pfs-amber"), red = cssVar("--pfs-red");
    const purple = cssVar("--pfs-purple");

    const W = 880, L = 62, R = 28, PW = W - L - R, TOP = 26, BOT = 330;
    const X = x => L + ((x + 1.04) / 2.08) * PW;
    const Y = v => TOP + ((1.12 - Math.max(-1.12, Math.min(1.12, v))) / 2.24) * (BOT - TOP);

    const cuts = sp.cut === "upper" ? [cut] : sp.cut === "lower" ? [-cut] :
                 sp.cut === "both" ? [cut, -cut] : [];
    const admits = v => cuts.some(c => (c > 0 ? v >= c : v <= c));

    /* Find every contiguous admitted run. A run holding a target is the intended
       passband; any OTHER admitted run is leakage — states from a disconnected part
       of the spectrum that would be mixed in with the ones you asked for. */
    const runs = [];
    if (cuts.length) {
      const step = 0.0008;
      let start = null;
      for (let x = -1; x <= 1.0001; x += step) {
        const ok = admits(P(x));
        if (ok && start === null) start = x;
        if ((!ok || x + step > 1.0001) && start !== null) {
          const lo = start, hi = Math.min(1, x);
          runs.push({ lo: lo, hi: hi, main: targets.some(t => t >= lo - 1e-3 && t <= hi + 1e-3) });
          start = null;
        }
      }
    }
    const leaks = runs.filter(r => !r.main);
    const inRun = x => runs.find(r => x >= r.lo && x <= r.hi);

    svg.replaceChildren();

    /* frame */
    svg.appendChild(mk("line", { x1: X(-1.04), y1: Y(0), x2: X(1.04), y2: Y(0),
      stroke: grid, "stroke-width": 1.2 }));
    for (const v of [1, -1]) {
      svg.appendChild(mk("line", { x1: X(-1.04), y1: Y(v), x2: X(1.04), y2: Y(v),
        stroke: grid, "stroke-width": 1, opacity: .55 }));
      svg.appendChild(txt(v > 0 ? "+1" : "−1", L - 10, Y(v) + 4,
        { "text-anchor": "end", "font-size": 12, fill: muted }));
    }

    /* admitted regions: green where intended, red where it leaks */
    for (const r of runs) {
      svg.appendChild(mk("rect", { x: X(r.lo), y: TOP, width: Math.max(1.5, X(r.hi) - X(r.lo)),
        height: BOT - TOP, fill: r.main ? green : red, opacity: r.main ? .13 : .2 }));
    }

    for (const c of cuts) {
      svg.appendChild(mk("line", { x1: X(-1), y1: Y(c), x2: X(1), y2: Y(c),
        stroke: grid, "stroke-width": 2, "stroke-dasharray": "7 6" }));
      svg.appendChild(txt("cutoff " + c.toFixed(2), X(1) + 4, Y(c) + 4,
        { "font-size": 10.5, fill: muted }));
    }

    /* the polynomial */
    let d = "";
    for (let i = 0; i <= 1400; i += 1) {
      const x = -1 + (2 * i) / 1400;
      d += (i === 0 ? "M" : " L") + X(x).toFixed(1) + "," + Y(P(x)).toFixed(1);
    }
    svg.appendChild(mk("path", { d: d, fill: "none", stroke: blue, "stroke-width": 2.6,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));

    /* the comparison overlay, dashed so the two never read as one curve */
    if (cmp) {
      let dc = "";
      for (let i = 0; i <= 1400; i += 1) {
        const x = -1 + (2 * i) / 1400;
        dc += (i === 0 ? "M" : " L") + X(x).toFixed(1) + "," + Y(cmp.P(x)).toFixed(1);
      }
      svg.appendChild(mk("path", { d: dc, fill: "none", stroke: purple, "stroke-width": 2.4,
        "stroke-dasharray": "7 5", "stroke-linejoin": "round", opacity: .95 }));
      svg.appendChild(mk("line", { x1: W - 250, y1: 18, x2: W - 222, y2: 18,
        stroke: blue, "stroke-width": 2.6 }));
      svg.appendChild(txt(SPEC[kind].name, W - 216, 22, { "font-size": 11, fill: muted }));
      svg.appendChild(mk("line", { x1: W - 250, y1: 34, x2: W - 222, y2: 34,
        stroke: purple, "stroke-width": 2.4, "stroke-dasharray": "7 5" }));
      svg.appendChild(txt(SPEC[cmpKind].name, W - 216, 38, { "font-size": 11, fill: muted }));
    }

    /* the spectrum, mapped through the filter */
    let nMain = 0, nLeak = 0;
    for (const e of EIG) {
      const v = P(e);
      const r = cuts.length ? inRun(e) : null;
      const kindOf = !r ? "none" : (r.main ? "main" : "leak");
      if (kindOf === "main") nMain += 1;
      if (kindOf === "leak") nLeak += 1;
      const col = kindOf === "main" ? green : kindOf === "leak" ? red : muted;
      svg.appendChild(mk("circle", {
        cx: X(e), cy: Y(v), r: kindOf === "none" ? 3.4 : 4.6,
        fill: col, "fill-opacity": kindOf === "none" ? 0.35 : 0.55,
        stroke: col, "stroke-opacity": kindOf === "none" ? 0.5 : 0.95, "stroke-width": 1.8
      }));
      svg.appendChild(mk("line", { x1: X(e), y1: BOT + 18, x2: X(e), y2: BOT + (kindOf === "none" ? 25 : 30),
        stroke: col, "stroke-width": kindOf === "none" ? 1.1 : 2,
        opacity: kindOf === "none" ? 0.45 : 0.95 }));
    }

    /* target markers */
    for (const m of targets) {
      svg.appendChild(mk("line", { x1: X(m), y1: TOP, x2: X(m), y2: BOT,
        stroke: amber, "stroke-width": 1.6, "stroke-dasharray": "5 4", opacity: .8 }));
      svg.appendChild(txt("λ", X(m) + 5, TOP + 12, { "font-size": 12, fill: amber }));
    }

    /* ε axis */
    svg.appendChild(mk("line", { x1: X(-1.04), y1: BOT + 14, x2: X(1.04), y2: BOT + 14,
      stroke: fg, "stroke-width": 1.5, opacity: .6 }));
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      svg.appendChild(txt(t.toFixed(1).replace("0.0", "0"), X(t), BOT + 52,
        { "text-anchor": "middle", "font-size": 12, fill: fg }));
    }
    svg.appendChild(txt("rescaled energy ε", X(0), BOT + 76,
      { "text-anchor": "middle", "font-size": 16, "font-weight": 600, fill: fg }));

    const NOTES = {
      up: "One peak pinned to +1 at λ. The circles are a real spectrum mapped through the filter; green ones clear the cutoff and are kept.",
      down: "The mirror image: the peak is pinned to −1, so the window lands at the transformed <em>lowest</em> end. Same polynomial, opposite sign.",
      smooth: "The same expansion with a Jackson kernel applied to the coefficients before normalization. The side lobes drop by roughly three orders of magnitude — but the main peak is about twice as wide, so the same window costs about twice the order. That trade is why it sits <em>last</em> in the interior chain: a fallback for windows the sharp filters cannot serve cleanly, not the default.",
      edge_hi: "((1+x)/2)^p — monotone for any p, so it has no side lobes and <em>cannot</em> leak: there is no cutoff and no passband to validate. Raising p steepens the exposed edge and squashes the rest harder toward 0. <strong>POLFED fixes p = 3</strong> and never resolves it — the slider is here to show the family, not a knob you set.",
      edge_lo: "−((1−x)/2)^p, the mirror: a left-edge window maps to the transformed lowest end. Same story — monotone at any p, and fixed at 3 in practice.",
      edge_both: "x^p, with <strong>p odd</strong> so the map stays monotone — the slider steps by 2 for exactly that reason. Both spectrum edges map to both transformed extremes at once. Even powers would fold the spectrum onto itself and are not a valid edge filter.",
      double: "Two peaks, +1 at the lower window and −1 at the upper one, with the mixing solved from a 2×2 system so both land exactly. Watch the crossing between them — that swing is the third region validation has to check."
    };
    let msg = NOTES[kind];
    if (cmp) {
      msg = '<strong style="color:' + purple + '">Dashed:</strong> ' + SPEC[cmpKind].name +
            ", at the same order and target. The circles and the shading always follow the " +
            "<em>solid</em> filter. " + msg;
    }
    if (leaks.length) {
      msg = '<strong style="color:' + red + '">Leakage: ' + leaks.length +
            ' side region' + (leaks.length > 1 ? 's' : '') + ' clears the cutoff' +
            (nLeak ? ', admitting ' + nLeak + ' unwanted eigenvalue' + (nLeak > 1 ? 's' : '') : '') +
            '.</strong> A candidate like this fails <code>validate_filter</code> and POLFED moves ' +
            'to the next cutoff on the ladder. ' + msg;
    } else if (cuts.length) {
      msg = '<strong style="color:' + green + '">No leakage</strong> — ' + nMain +
            ' eigenvalue' + (nMain === 1 ? '' : 's') + ' admitted, all from the intended window. ' + msg;
    }
    note.innerHTML = msg;
  }

  elKind.addEventListener("change", function () { syncControls(); render(); });
  elCmp.addEventListener("change", render);
  elOrd.addEventListener("input", render);
  elLam.addEventListener("input", render);
  elCut.addEventListener("input", render);
  new MutationObserver(render).observe(document.documentElement,
    { attributes: true, attributeFilter: ["class"] });

  syncControls();
  render();
})();
</script>
```


Three things worth doing with it:

- **Raise the order on a delta filter.** The peak narrows and the side lobes crowd in. The
  shaded band is what the filter actually admits.
- **Switch to the Jackson-damped one.** The ringing nearly disappears — at the cost of a much
  wider peak for the same order. This is the classic smoothing trade, and it is why POLFED
  validates raw delta filters rather than damping them: a damped filter would need a far higher
  order to isolate the same window, and order is what every matvec pays for.
- **Switch to an edge filter.** Monotone, no lobes, no passband — nothing to leak. That
  robustness is why the patched solver peels spectrum edges with it first.

## [The double filter](@id double_filter)

The one filter that is not just "a peak somewhere". [`double_polfed`](@ref tutorial_double)
needs **two** disjoint windows served by a single polynomial, and the trick is to send them to
*opposite* ends of the transformed spectrum:

```math
P(\tilde\lambda_1) = +1, \qquad P(\tilde\lambda_2) = -1 ,
```

with ``\tilde\lambda_1`` the midpoint of the **lower-energy** window and ``\tilde\lambda_2`` the
midpoint of the upper one. Both windows then sit at extremes of the filtered operator, so one
Lanczos run with a `BothEnds` target converges both at once.

### Why it is not simply "add two peaks"

The obvious construction — take the delta expansion ``U`` peaked at ``\tilde\lambda_1``, the
expansion ``D`` peaked at ``\tilde\lambda_2``, and add them — does not work. Each peak has tails
that reach the *other* target, so the two **cross-talk**: normalizing ``U`` to 1 at
``\tilde\lambda_1`` is spoiled by whatever ``D`` contributes there. A single overall scale can
pin one peak or the other, never both.

So the two are mixed with coefficients solved from a 2×2 system,

```math
P = a\,U + b\,D,
\qquad
\begin{pmatrix} U(\tilde\lambda_1) & D(\tilde\lambda_1) \\
                U(\tilde\lambda_2) & D(\tilde\lambda_2) \end{pmatrix}
\begin{pmatrix} a \\ b \end{pmatrix}
=
\begin{pmatrix} +1 \\ -1 \end{pmatrix},
```

which puts **both** peaks exactly where they belong — the off-diagonal entries are precisely the
cross-talk, and the solve absorbs it.

### Two orders, not one

``U`` and ``D`` carry their **own** orders, ``K_1`` and ``K_2``, each resolved from its own
window's width. A narrow window still gets a sharp peak while a wide one stays cheap.

What this does *not* do is make the two windows independent in cost: they share one polynomial,
so every matvec pays the combined degree. That is exactly the trade
[`double_polfed`](@ref tutorial_double) measures before choosing between one joint run and two
separate ones.

### Cutoffs, and a safety factor that runs backwards

The double filter has **two** cutoffs, taken symmetrically from a shared ladder of magnitudes
``0.17, 0.18, \dots, 0.35``, consumed as ``\texttt{cutoff\_upper} = +c`` and
``\texttt{cutoff\_lower} = -c``. A state is admitted when its filtered value clears ``+c`` (upper
window) or falls below ``-c`` (lower window).

Its `order_safety_factor` is **`0.97`** — less than one, which is the opposite of what "safety
factor" usually suggests. Shrinking the order *widens* the passband, so the margin here buys
coverage rather than sharpness: better to admit slightly too much (the pipeline trims afterwards)
than to cut a requested state off.

### A third place to leak

Single-peak filters are validated inside the passband and outside it. A double filter has one
more region to worry about: **between the two peaks**. The polynomial has to come all the way
down from ``+1`` to ``-1``, and the swing can overshoot either cutoff on the way. Validation
therefore samples three regions — within each passband, on the outer sides, and across the gap,
the last with a point count that scales as ``10\max(K_1, K_2)`` because that is where the
oscillation is fastest.

If no candidate on the ladder survives all three checks, the double filter is abandoned and the
two windows are solved separately. That fallback is why an unresolvable double filter degrades
into two ordinary runs rather than an error.

### Constraints

- The two intervals must be **disjoint**, with ``\text{interval}_1`` strictly left of
  ``\text{interval}_2``. Overlapping or touching windows are a single connected region — a job
  for [`polfed`](@ref Polfed.polfed), or `patched_polfed` if it is wide.
- Each target is the **midpoint** of its interval; you do not set the peaks directly.
- Exactly two windows are supported. More than that is what `patched_polfed` is for.

## How one gets picked

```text
window touches ±1 ?
├─ yes → EDGE_FILTER_CHAIN     = (EdgeFilter,)
└─ no  → INTERIOR_FILTER_CHAIN = (DeltaUpFilter, DeltaDownFilter, DeltaUpSmoothFilter)
```

The order of the interior chain is the cost order. The two sharp filters are tried first because
they resolve to the lowest order; `DeltaUpSmoothFilter` is last because its damped peak needs
roughly twice the order for the same window. It earns its place by removing the Gibbs side lobes
at the source rather than retreating to a laxer cutoff — so it can serve a window where every
sharp candidate leaks.

Each type in the chain walks its own **cutoff ladder** — a sequence of candidate cutoffs,
sharpest first. For each value a fresh candidate is built and checked by `validate_filter`;
the first valid one wins. If a type exhausts its ladder, the next type in the chain is
tried; if the chain is exhausted, POLFED raises an error rather than returning a filter that
might leak.

Adding a new filter means implementing `_cutoff_values` / `_build_candidate` and
`validate_filter` in its own folder and appending the type to a chain — the seam is
deliberately narrow.

## Take-home

- The transformation maps the targeted part of the spectrum to the **edge** of the
  transformed problem, which is where Krylov methods are efficient.
- Eigenpairs far from the target are suppressed, so Lanczos concentrates on your window
  instead of on the original spectral edges.
- Increasing ``K`` narrows the window and reduces how many eigenvalues the filter captures.
- Only transformed eigenpairs above the cutoff are kept; the order is chosen so the main
  peak clears the cutoff while the side lobes stay below it.
- Keeping side lobes below the cutoff is the whole game: otherwise eigenpairs from
  disconnected energy regions contaminate your results.
