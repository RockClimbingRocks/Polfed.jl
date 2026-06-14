# Polynomial Filtering

As mentioned in the introduction, conventional Krylov/Lanczos methods are most efficient for extracting eigenvalues and eigenvectors, which we call eigenpairs, close to the spectral edges. If we are interested in mid-spectrum eigenpairs, we first need to apply a spectral transformation that maps the desired part of the spectrum to an extremal part of the transformed spectrum.

The POLFED filter is built from Chebyshev polynomials. Since Chebyshev polynomials are defined on the interval ``[-1,1]``, the Hamiltonian must first be rescaled so that its spectrum lies in this interval. We denote rescaled quantities by placing a tilde above them. For a rescaled target energy ``\tilde{\lambda}`` and polynomial order ``K``, the transformation has the form

```math
P_{\tilde{\lambda}}^{K}(\tilde{H})
= \frac{1}{\chi}
  \sum_{n=0}^{K} c_n^{\tilde{\lambda}} T_n(\tilde{H}),
```

with coefficients

```math
c_n^{\tilde{\lambda}}
=
(2 - \delta_{n,0})\cos(n\arccos(\tilde{\lambda})).
```

Here, ``\chi`` is a normalization factor chosen so that the filter is normalized at the target energy, i.e. ``P_{\tilde{\lambda}}^{K}(\tilde{\lambda}) = 1``. The coefficients above are obtained by expanding the Dirac delta function in Chebyshev polynomials.

The interactive figure below illustrates how the spectral transformation exposes the desired part of the Hamiltonian spectrum. The target energy is fixed at ``\tilde{\lambda}=0``, while the polynomial order ``K`` can be changed with the slider. Increasing ``K`` sharpens the filter around the target energy.


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
      --polfed-filter-red: #e63946;
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

    .polfed-filter-control-row input {
      width: 100%;
    }

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

    .polfed-filter-demo text {
      font-family: inherit;
    }

    @media (max-width: 620px) {
      .polfed-filter-control-row {
        grid-template-columns: 1fr;
      }
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
      const height = 120;

      const plot = {
        left: 72,
        right: 36,
        top: 24,
        axis: 315,
        spectrum: 375
      };

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
        const node = el("text", Object.assign({
          x,
          y,
          fill: cssVar("--polfed-filter-fg")
        }, attrs || {}));
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

        /*
          Threshold line at transformation value P = 0.17.
        */
        svg.appendChild(el("line", {
          x1: xScale(-1),
          y1: yScale(threshold),
          x2: xScale(1),
          y2: yScale(threshold),
          stroke: grid,
          "stroke-width": 2,
          "stroke-dasharray": "7 7",
          "stroke-linecap": "round"
        }));

        /*
          Untransformed spectrum axis.
          The arrow remains as long as in the original version.
        */
        svg.appendChild(el("line", {
          x1: plot.left,
          y1: plot.spectrum,
          x2: width - plot.right - 18,
          y2: plot.spectrum,
          stroke: fg,
          "stroke-width": 2.5,
          "stroke-linecap": "round"
        }));

        svg.appendChild(el("path", {
          d: "M " + (width - plot.right - 18) + " " + (plot.spectrum - 8) +
             " L " + (width - plot.right) + " " + plot.spectrum +
             " L " + (width - plot.right - 18) + " " + (plot.spectrum + 8) + " Z",
          fill: fg
        }));

        /*
          Ticks for -1 and +1.
          They are placed slightly inside the axis, not exactly at the endpoints.
        */
        const tickPositions = [
          { label: "-1", x: -1. },
          { label: "+1", x: 1. }
        ];

        for (const tick of tickPositions) {
          const tickX = xScale(tick.x);

          svg.appendChild(el("line", {
            x1: tickX,
            y1: plot.spectrum - 10,
            x2: tickX,
            y2: plot.spectrum + 10,
            stroke: fg,
            "stroke-width": 3,
            "stroke-linecap": "round"
          }));

          svg.appendChild(text(tick.label, tickX - 12, plot.spectrum + 34, {
            "font-size": 18,
            fill: fg
          }));
        }

        /*
          Eigenvalue ticks on the untransformed spectrum.
          All eigenvalues whose transformed value is above the threshold are red.
        */
        let highlighted = 0;

        for (const x of eigenvalues) {
          const value = filterValue(x, order);
          const isTarget = value > threshold;
          const color = isTarget ? red : orange;

          if (isTarget) highlighted += 1;

          svg.appendChild(el("line", {
            x1: xScale(x),
            y1: plot.spectrum - 7,
            x2: xScale(x),
            y2: plot.spectrum - (isTarget ? 50 : 35),
            stroke: color,
            "stroke-width": isTarget ? 2. : 2.,
            opacity: isTarget ? 0.92 : 0.72,
            "stroke-linecap": "round"
          }));
        }


        /*
          Transformed eigenvalue markers.
          All markers above the threshold are red.
        */
        for (const x of eigenvalues) {
          const value = filterValue(x, order);

          circ_color = value < threshold ? orange : red

          svg.appendChild(el("circle", {
            cx: xScale(x),
            cy: yScale(value),
            r: 6.0,

            fill: circ_color,
            "fill-opacity": 0.5,

            stroke: circ_color,
            "stroke-opacity": 0.95,
            "stroke-width": 2
          }));
        }

        /*
          Target marker at lambda = 0.
          SVG text cannot directly render LaTeX, so we use the Greek symbol λ.
        */
        const lambdaX = xScale(lambda);

        svg.appendChild(el("line", {
          x1: lambdaX,
          y1: plot.spectrum + 14,
          x2: lambdaX,
          y2: plot.spectrum - 14,
          stroke: fg,
          "stroke-width": 5,
          "stroke-linecap": "round"
        }));

        svg.appendChild(text("λ", lambdaX-5, plot.spectrum + 40, {
          "font-size": 20,
          fill: fg
        }));

        /*
          Transformed polynomial filter curve.
          The vertical scale is normalized so that the peak reaches the top.
        */
        svg.appendChild(el("path", {
          d: pathFromSamples(order),
          fill: "none",
          stroke: blue,
          "stroke-width": 4,
          "stroke-linejoin": "round",
          "stroke-linecap": "round"
        }));

        readout.textContent =
          highlighted + " transformed eigenvalus are above the threshold.";
      }

      slider.addEventListener("input", render);

      const observer = new MutationObserver(render);
      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class"]
      });

      render();
    })();
  </script>
</div>
```

## Take-home message

- The polynomial transformation maps the targeted part of the spectrum toward the spectral edge of the transformed problem. This makes eigenpairs close to the target energy appear extremal to the Krylov/Lanczos iteration.

- Eigenpairs far from the target are suppressed by the filter, so the Krylov/Lanczos method focuses on the desired energy window rather than on the original spectral edges.

- Increasing ``K`` sharpens the filter around the fixed target. More precisely, the width of the main peak scales inversely with the polynomial order, ``\delta \propto 1/K``. Therefore, larger ``K`` selects a narrower energy window and reduces the number of eigenvalues captured by the filter.

- In practice, POLFED keeps only the transformed eigenpairs whose transformed eigenvalues lie above a chosen threshold. The polynomial order is chosen so that the main peak around the target lies above this threshold, while the secondary peaks remain below it.

- Keeping the secondary peaks below the threshold is important: otherwise, eigenpairs from disconnected energy regions could also pass the filter and be mixed with the eigenpairs from the desired target window.

- A sharper peak separates the transformed target eigenvalues more clearly, which usually reduces the number of Lanczos iterations needed for convergence. However, a larger ``K`` also means that each filtering step requires more matrix multiplications.