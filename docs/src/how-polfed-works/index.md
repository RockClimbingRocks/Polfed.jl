# [The workflow](@id how_polfed_works)

POLFED is auto-piloted. You name a target and how many eigenpairs you need; the density of
states decides the rest.

Rather than describe that, the explorer below *runs* it. Play with it first, then read the
stage-by-stage walkthrough underneath. Two sub-pages go deeper on the parts that matter most:
the [filters](@ref filters) themselves, and the
[on-the-fly mapping](@ref on_the_fly_mapping) that applies them without ever building a matrix.

## [Try it](@id workflow_explorer)

Everything here runs on a real spectrum: the 256 rescaled eigenvalues of a small Quantum Sun
Hamiltonian (`L_loc = 6`, `L_grain = 2`), with ``E_{\mathrm{min}} = -4.170`` and
``E_{\mathrm{max}} = 3.847``.

Pick a **target kind** from the dropdown; the kinds that take a value (`:rescaled`,
`:unrescaled`, `:offset`) reveal a second slider in their own units. Then move `howmany` and
watch the interval widen and the order **fall**; move the cutoff and watch the order change
with the interval fixed.

```@raw html
<style>
.pf-explorer {
  --pfx-bg: #ffffff; --pfx-fg: #24292f; --pfx-muted: #6e7781; --pfx-grid: #d0d7de;
  --pfx-blue: #4e79a7; --pfx-green: #2ca05a; --pfx-amber: #d97706;
  --pfx-panel: #f6f8fa; --pfx-border: #d0d7de;
  margin: 1.5rem 0;
}
html.theme--documenter-dark .pf-explorer,
html.theme--catppuccin-frappe .pf-explorer,
html.theme--catppuccin-macchiato .pf-explorer,
html.theme--catppuccin-mocha .pf-explorer {
  --pfx-bg: #15191a; --pfx-fg: #f3f5f7; --pfx-muted: #a7b0b8; --pfx-grid: #58636a;
  --pfx-blue: #7ea6cc; --pfx-green: #3fd07f; --pfx-amber: #e0a336;
  --pfx-panel: #1f2428; --pfx-border: #58636a;
}
.pf-explorer-controls {
  display: grid; gap: 0.7rem; margin: 0 0 1rem; padding: 1rem;
  border: 1px solid var(--pfx-border); border-radius: 8px; background: var(--pfx-panel);
}
.pf-explorer-row {
  display: grid; grid-template-columns: minmax(9rem, auto) 1fr minmax(6.5rem, auto);
  gap: 0.75rem; align-items: center;
}
.pf-explorer-row input, .pf-explorer-row select { width: 100%; }
.pf-explorer-row output { color: var(--pfx-muted); font-size: 0.92rem; font-variant-numeric: tabular-nums; }
.pf-explorer svg {
  display: block; width: 100%; height: auto;
  border: 1px solid var(--pfx-border); border-radius: 8px; background: var(--pfx-bg);
}
.pf-explorer text { font-family: inherit; }
.pf-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
  gap: 0.6rem; margin-top: 1rem;
}
.pf-stat {
  padding: 0.55rem 0.7rem; border: 1px solid var(--pfx-border);
  border-radius: 8px; background: var(--pfx-panel);
}
.pf-stat .k {
  display: block; font-size: 0.72rem; letter-spacing: .02em;
  color: var(--pfx-muted);
}
.pf-stat .v {
  display: block; font-size: 1.05rem; font-weight: 700;
  font-variant-numeric: tabular-nums; color: var(--pfx-fg);
}
.pf-stat.hot .v { color: var(--pfx-amber); }
@media (max-width: 640px) { .pf-explorer-row { grid-template-columns: 1fr; } }
</style>

<div class="pf-explorer" id="pf-explorer">
  <div class="pf-explorer-controls">
    <div class="pf-explorer-row">
      <label for="pfx-howmany">howmany</label>
      <input id="pfx-howmany" type="range" min="4" max="100" step="1" value="40">
      <output id="pfx-howmany-out">40</output>
    </div>
    <div class="pf-explorer-row">
      <label for="pfx-target">target</label>
      <select id="pfx-target">
        <option value="maxdos" selected>:maxdos</option>
        <option value="middle">:middle</option>
        <option value="resc">(:rescaled, ε)</option>
        <option value="unresc">(:unrescaled, E)</option>
        <option value="offset">(:offset, η)</option>
      </select>
      <output id="pfx-target-out"></output>
    </div>
    <div class="pf-explorer-row" id="pfx-value-row">
      <label for="pfx-value" id="pfx-value-label">value</label>
      <input id="pfx-value" type="range" min="-95" max="95" step="1" value="20">
      <output id="pfx-value-out">0.20</output>
    </div>
    <div class="pf-explorer-row">
      <label for="pfx-cutoff">cutoff</label>
      <input id="pfx-cutoff" type="range" min="14" max="36" step="1" value="17">
      <output id="pfx-cutoff-out">0.17</output>
    </div>
  </div>

  <svg id="pfx-svg" viewBox="0 0 880 548" role="img" aria-labelledby="pfx-t pfx-d">
    <title id="pfx-t">Interactive POLFED workflow explorer</title>
    <desc id="pfx-d">Top panel: the density of states with the resolved target and the interval holding the requested number of states. Bottom panel: the Chebyshev filter built for that interval.</desc>
  </svg>

  <div class="pf-stats" id="pfx-stats"></div>
</div>

<script>
(function () {
  const root = document.getElementById("pf-explorer");
  if (!root) return;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("pfx-svg");
  const stats = document.getElementById("pfx-stats");
  const elHow = document.getElementById("pfx-howmany");
  const elTgt = document.getElementById("pfx-target");
  const elVal = document.getElementById("pfx-value");
  const elValRow = document.getElementById("pfx-value-row");
  const elValLbl = document.getElementById("pfx-value-label");
  const outVal = document.getElementById("pfx-value-out");
  const elCut = document.getElementById("pfx-cutoff");
  const outHow = document.getElementById("pfx-howmany-out");
  const outTgt = document.getElementById("pfx-target-out");
  const outCut = document.getElementById("pfx-cutoff-out");

  /* Rescaled spectrum of the sample QSun Hamiltonian, and the bounds it came from. */
  const EMIN = -4.170, EMAX = 3.847;
  const A = (EMAX - EMIN) / 2, B = (EMAX + EMIN) / 2;

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
  ].slice().sort(function (a, b) { return a - b; });

  /* shared with the KPM widget further down the page */
  window.PFDATA = { EIG: EIG, EMIN: EMIN, EMAX: EMAX, A: A, B: B, MAXDOS: null };

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

  /* --- the density of states: a smoothed estimate over the sample spectrum --- */
  const H_KDE = 0.045;
  const GRID = [];
  for (let i = 0; i <= 600; i += 1) GRID.push(-1 + (2 * i) / 600);
  const DOS = GRID.map(function (x) {
    let s = 0;
    for (const e of EIG) { const u = (x - e) / H_KDE; s += Math.exp(-0.5 * u * u); }
    return s;
  });
  const DOS_MAX = Math.max.apply(null, DOS);
  const MAXDOS_X = GRID[DOS.indexOf(DOS_MAX)];
  window.PFDATA.MAXDOS = MAXDOS_X;
  function dosAt(x) {
    let lo = 0, hi = GRID.length - 1;
    if (x <= -1) return DOS[0];
    if (x >= 1) return DOS[hi];
    const i = Math.round(((x + 1) / 2) * 600);
    return DOS[Math.max(0, Math.min(hi, i))];
  }

  /* --- the Chebyshev delta filter, exactly as POLFED builds it --- */
  function chebT(n, x) { const c = Math.max(-1, Math.min(1, x)); return Math.cos(n * Math.acos(c)); }
  function deltaFilter(x, lam, K) {
    let num = 0, nrm = 0;
    for (let n = 0; n <= K; n += 1) {
      const c = (2 - (n === 0 ? 1 : 0)) * Math.cos(n * Math.acos(lam));
      num += c * chebT(n, x);
      nrm += c * chebT(n, lam);
    }
    return num / Math.max(Math.abs(nrm), 1e-12);
  }

  /* How far the contiguous passband reaches from λ in one direction (dir = ±1).
     The delta filter is NOT symmetric about λ unless λ = 0, so the two sides differ. */
  function halfWidth(lam, K, cutoff, dir) {
    /* Walk outward from λ and stop at the FIRST crossing below the cutoff — the edge of
       the CONTIGUOUS main lobe. A bisection is wrong here: as soon as a side lobe rises
       above the cutoff the predicate stops being monotone, and the search can jump the
       valley and report a "passband" spanning lobes that are not part of it at all. */
    const STEP = 0.0015;
    let prev = 0;
    for (let w = STEP; w <= 2; w += STEP) {
      const x = lam + dir * w;
      if (x < -1 || x > 1) return dir > 0 ? Math.max(0, 1 - lam) : Math.max(0, lam + 1);
      if (deltaFilter(x, lam, K) <= cutoff) {
        let a = prev, b = w;                       /* refine the crossing */
        for (let i = 0; i < 26; i += 1) {
          const m = (a + b) / 2;
          if (deltaFilter(lam + dir * m, lam, K) > cutoff) a = m; else b = m;
        }
        return a;
      }
      prev = w;
    }
    return 2;
  }

  /* Everything else that clears the cutoff is leakage: states from a disconnected part of
     the spectrum that the filter would admit alongside the ones actually requested. */
  function leakRegions(lam, K, cutoff, passLo, passHi) {
    const out = [];
    const step = 0.0015;
    let start = null;
    for (let x = -1; x <= 1.0001; x += step) {
      const inMain = x >= passLo - 1e-9 && x <= passHi + 1e-9;
      const hot = !inMain && deltaFilter(x, lam, K) > cutoff;
      if (hot && start === null) start = x;
      if ((!hot || x + step > 1.0001) && start !== null) {
        out.push([start, Math.min(1, x)]);
        start = null;
      }
    }
    return out;
  }

  /* stage ⑤: the order must be small enough that the passband still COVERS the requested
     interval — a higher order narrows the peak, so we want the LARGEST K that still
     reaches both endpoints. (A passband wider than the request is fine: the pipeline
     trims the results afterwards. A passband narrower than it would silently drop states.) */
  function covers(lam, K, cutoff, reqL, reqR) {
    return halfWidth(lam, K, cutoff, -1) >= reqL - 1e-9 &&
           halfWidth(lam, K, cutoff, +1) >= reqR - 1e-9;
  }

  function resolveOrder(lam, lo, hi, cutoff) {
    const reqL = Math.max(0, lam - lo), reqR = Math.max(0, hi - lam);
    if (!covers(lam, 1, cutoff, reqL, reqR)) return 1;
    let good = 1, probe = 2;
    while (probe < 4000 && covers(lam, probe, cutoff, reqL, reqR)) { good = probe; probe *= 2; }
    let a = good, b = Math.min(probe, 4000);
    while (a < b) {
      const mid = Math.ceil((a + b) / 2);
      if (covers(lam, mid, cutoff, reqL, reqR)) a = mid; else b = mid - 1;
    }
    return a;
  }

  /* Each target kind carries its own slider range and its own units. */
  const KINDS = {
    maxdos: { takesValue: false, label: "-", spec: function () { return ":maxdos"; } },
    middle: { takesValue: false, label: "-", spec: function () { return ":middle"; } },
    resc:   { takesValue: true,  label: "\u03b5", min: -95,  max: 95,  scale: 0.01,
              spec: function (v) { return "(:rescaled, " + v.toFixed(2) + ")"; } },
    unresc: { takesValue: true,  label: "E", min: -410, max: 380, scale: 0.01,
              spec: function (v) { return "(:unrescaled, " + v.toFixed(2) + ")"; } },
    offset: { takesValue: true,  label: "\u03b7", min: -100, max: 100, scale: 0.01,
              spec: function (v) { return "(:offset, " + v.toFixed(2) + ")"; } }
  };

  /* stage ③: resolve the target kind + value to a rescaled centre */
  function resolveTarget(kind, v) {
    if (kind === "maxdos") return MAXDOS_X;
    if (kind === "middle") return 0;
    if (kind === "resc") return v;
    if (kind === "unresc") return (v - B) / A;
    /* offset: interpolate from the DoS maximum toward an edge */
    return v >= 0 ? MAXDOS_X + v * (1 - MAXDOS_X) : MAXDOS_X + v * (1 + MAXDOS_X);
  }

  /* re-range the value slider whenever the kind changes */
  function syncValueControl() {
    const k = KINDS[elTgt.value];
    elValRow.style.display = k.takesValue ? "" : "none";
    if (!k.takesValue) return;
    if (Number(elVal.min) !== k.min || Number(elVal.max) !== k.max) {
      elVal.min = k.min; elVal.max = k.max;
      elVal.value = Math.round((k.min + k.max) / 4);
    }
    elValLbl.textContent = k.label;
  }

  /* stage ④: grow a SYMMETRIC window [λ-δ, λ+δ], clamped to [-1,1], until it holds at
     least `howmany` states — exactly what `_size_window_from_dos` does by bisecting δ. */
  function countIn(lo, hi) {
    let n = 0;
    for (const e of EIG) if (e >= lo && e <= hi) n += 1;
    return n;
  }
  function windowAt(lam, d) {
    return [Math.max(-1, lam - d), Math.min(1, lam + d)];
  }
  function resolveInterval(lam, howmany) {
    let [wl, wh] = windowAt(lam, 1.0);
    if (countIn(wl, wh) < howmany) return [wl, wh, countIn(wl, wh), 1.0];
    let a = 0, b = 1;
    for (let i = 0; i < 50; i += 1) {
      const mid = (a + b) / 2;
      const [l, h] = windowAt(lam, mid);
      if (countIn(l, h) >= howmany) b = mid; else a = mid;
    }
    const [lo, hi] = windowAt(lam, b);
    return [lo, hi, countIn(lo, hi), b];
  }

  function stat(k, v, hot) {
    return '<div class="pf-stat' + (hot ? ' hot' : '') + '"><span class="k">' + k +
           '</span><span class="v">' + v + '</span></div>';
  }

  function render() {
    const howmany = Number(elHow.value);
    const cutoff = Number(elCut.value) / 100;
    const kindKey = elTgt.value;
    const kind = KINDS[kindKey];
    const rawVal = Number(elVal.value) * (kind.scale || 1);

    outHow.textContent = howmany;
    outCut.textContent = cutoff.toFixed(2);
    if (kind.takesValue) outVal.textContent = rawVal.toFixed(2);

    const lam = Math.max(-0.995, Math.min(0.995, resolveTarget(kindKey, rawVal)));
    const [lo, hi, nStates, delta] = resolveInterval(lam, howmany);
    const K = resolveOrder(lam, lo, hi, cutoff);
    const wL = halfWidth(lam, K, cutoff, -1);
    const wR = halfWidth(lam, K, cutoff, +1);
    const passLo = Math.max(-1, lam - wL), passHi = Math.min(1, lam + wR);
    const leaks = leakRegions(lam, K, cutoff, passLo, passHi);

    outTgt.textContent = kind.spec(rawVal) + "  \u2192  \u03b5 = " + lam.toFixed(3);

    const fg = cssVar("--pfx-fg"), muted = cssVar("--pfx-muted"), grid = cssVar("--pfx-grid");
    const blue = cssVar("--pfx-blue"), green = cssVar("--pfx-green"), amber = cssVar("--pfx-amber");

    const W = 880, L = 58, R = 28, PW = W - L - R;
    const X = x => L + ((x + 1.06) / 2.12) * PW;

    /* panel A: density of states */
    const A_TOP = 42, A_BASE = 214;
    const yA = v => A_BASE - (v / DOS_MAX) * (A_BASE - A_TOP);
    /* panel B: the filter */
    const B_TOP = 290, B_BASE = 470, B_LO = -0.35;
    const yB = v => B_TOP + ((1 - Math.max(B_LO, Math.min(1, v))) / (1 - B_LO)) * (B_BASE - B_TOP);

    svg.replaceChildren();

    /* ---- shaded target interval, spanning both panels ---- */
    svg.appendChild(mk("rect", {
      x: X(lo), y: A_TOP - 8, width: Math.max(1.5, X(hi) - X(lo)), height: B_BASE - A_TOP + 8,
      fill: green, opacity: 0.13
    }));

    /* ---- panel A ---- */
    svg.appendChild(txt("② density of states", L, 26, { "font-size": 13, "font-weight": 700, fill: muted }));

    let d = "M" + X(-1).toFixed(1) + "," + yA(DOS[0]).toFixed(1);
    for (let i = 1; i < GRID.length; i += 1) d += " L" + X(GRID[i]).toFixed(1) + "," + yA(DOS[i]).toFixed(1);
    svg.appendChild(mk("path", { d: d + " L" + X(1).toFixed(1) + "," + A_BASE + " L" + X(-1).toFixed(1) + "," + A_BASE + " Z",
      fill: blue, opacity: 0.18 }));
    svg.appendChild(mk("path", { d: d, fill: "none", stroke: blue, "stroke-width": 2.6, "stroke-linejoin": "round" }));

    svg.appendChild(mk("line", { x1: X(-1.03), y1: A_BASE, x2: X(1.03), y2: A_BASE,
      stroke: fg, "stroke-width": 1.6, opacity: .6 }));

    /* eigenvalues inside the interval are highlighted */
    for (const e of EIG) {
      const inside = e >= lo && e <= hi;
      svg.appendChild(mk("line", {
        x1: X(e), y1: A_BASE + 3, x2: X(e), y2: A_BASE + (inside ? 14 : 9),
        stroke: inside ? green : muted, "stroke-width": inside ? 1.8 : 1.2,
        opacity: inside ? 0.95 : 0.45
      }));
    }

    /* target marker */
    svg.appendChild(mk("line", { x1: X(lam), y1: A_TOP - 8, x2: X(lam), y2: A_BASE + 16,
      stroke: amber, "stroke-width": 2.2, "stroke-dasharray": "5 4" }));
    svg.appendChild(txt("③ λ", X(lam) + 6, A_TOP + 4, { "font-size": 12, "font-weight": 700, fill: amber }));

    /* interval callout */
    svg.appendChild(mk("line", { x1: X(lo), y1: A_BASE + 26, x2: X(hi), y2: A_BASE + 26,
      stroke: green, "stroke-width": 3, "stroke-linecap": "round" }));
    for (const b of [lo, hi]) {
      svg.appendChild(mk("line", { x1: X(b), y1: A_BASE + 21, x2: X(b), y2: A_BASE + 31,
        stroke: green, "stroke-width": 2.4 }));
    }
    svg.appendChild(txt("④ interval holding " + nStates + " states",
      (X(lo) + X(hi)) / 2, A_BASE + 46, { "text-anchor": "middle", "font-size": 12, fill: green }));

    /* ---- panel B ---- */
    svg.appendChild(txt("⑤ the filter this produces", L, 274, { "font-size": 13, "font-weight": 700, fill: muted }));

    svg.appendChild(mk("line", { x1: X(-1.03), y1: yB(0), x2: X(1.03), y2: yB(0),
      stroke: grid, "stroke-width": 1.2 }));
    svg.appendChild(mk("line", { x1: X(-1), y1: yB(cutoff), x2: X(1), y2: yB(cutoff),
      stroke: grid, "stroke-width": 2, "stroke-dasharray": "7 6" }));
    svg.appendChild(txt("cutoff " + cutoff.toFixed(2), X(-1) - 6, yB(cutoff) + 4,
      { "text-anchor": "end", "font-size": 11, fill: muted }));

    let f = "";
    for (let i = 0; i <= 900; i += 1) {
      const x = -1 + (2 * i) / 900;
      f += (i === 0 ? "M" : " L") + X(x).toFixed(1) + "," + yB(deltaFilter(x, lam, K)).toFixed(1);
    }
    svg.appendChild(mk("path", { d: f, fill: "none", stroke: blue, "stroke-width": 2.8,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));

    /* leakage first, so the green passband draws on top of it */
    const RED = "#e5534b";
    for (const [a, b] of leaks) {
      svg.appendChild(mk("rect", { x: X(a), y: B_TOP, width: Math.max(1.5, X(b) - X(a)),
        height: B_BASE - B_TOP, fill: RED, opacity: .22 }));
      svg.appendChild(mk("line", { x1: X(a), y1: yB(cutoff), x2: X(b), y2: yB(cutoff),
        stroke: RED, "stroke-width": 5, "stroke-linecap": "round" }));
    }
    if (leaks.length) {
      const mid = (X(leaks[0][0]) + X(leaks[0][1])) / 2;
      svg.appendChild(txt("leak", mid, yB(cutoff) - 9,
        { "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: RED }));
    }

    /* the passband: the CONTIGUOUS lobe around λ, nothing beyond it */
    svg.appendChild(mk("line", {
      x1: X(passLo), y1: yB(cutoff), x2: X(passHi), y2: yB(cutoff),
      stroke: green, "stroke-width": 5, "stroke-linecap": "round", opacity: .95
    }));
    svg.appendChild(txt("passband", (X(passLo) + X(passHi)) / 2, yB(cutoff) - 9,
      { "text-anchor": "middle", "font-size": 11, fill: green }));

    svg.appendChild(mk("line", { x1: X(-1.03), y1: B_BASE + 16, x2: X(1.03), y2: B_BASE + 16,
      stroke: fg, "stroke-width": 1.6, opacity: .6 }));
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      svg.appendChild(mk("line", { x1: X(t), y1: B_BASE + 11, x2: X(t), y2: B_BASE + 21,
        stroke: fg, "stroke-width": 1.6, opacity: .7 }));
      svg.appendChild(txt(t.toFixed(1).replace("0.0", "0"), X(t), B_BASE + 36,
        { "text-anchor": "middle", "font-size": 12, fill: fg }));
    }
    svg.appendChild(txt("rescaled energy ε", X(0), B_BASE + 52,
      { "text-anchor": "middle", "font-size": 11, fill: muted }));

    /* ---- readouts ---- */
    const krylov = Math.ceil((20.427 * 1 + 1.696 * howmany) * 1.25);
    stats.innerHTML =
      stat("target λ", lam.toFixed(3) + "   ·   E = " + (lam * A + B).toFixed(3)) +
      stat("interval (resc.)", "[" + lo.toFixed(3) + ", " + hi.toFixed(3) + "]") +
      stat("half-width δ", delta.toFixed(4)) +
      stat("passband", "[" + passLo.toFixed(3) + ", " + passHi.toFixed(3) + "]") +
      stat("leakage", leaks.length === 0 ? "none" : leaks.length + " region" + (leaks.length > 1 ? "s" : ""), leaks.length > 0) +
      stat("states inside", nStates) +
      stat("order K", K, true) +
      stat("cutoff", cutoff.toFixed(2)) +
      stat("Krylov dim ≈", krylov) +
      stat("estimated total H·v", (K * krylov).toLocaleString(), true);
  }

  elHow.addEventListener("input", render);
  elTgt.addEventListener("change", function () { syncValueControl(); render(); });
  elVal.addEventListener("input", render);
  elCut.addEventListener("input", render);
  syncValueControl();
  new MutationObserver(render).observe(document.documentElement,
    { attributes: true, attributeFilter: ["class"] });

  render();
})();
</script>
```

### What to notice

**More eigenpairs make the filter cheaper.** Drag `howmany` from 10 up to 100: the interval
widens, ``\delta`` grows, and the order ``K`` *falls* — often by a large factor. This is the
single most counter-intuitive fact about POLFED, and it is why "ask for fewer states to go
faster" is bad advice — usually the more the better.

**Both numbers move, but not at the same rate.** Raising `howmany` pushes the Krylov dimension
— the number of Lanczos iterations — **up**, linearly. At the same time it pushes the order
``K`` **down**. The total work is the product of the two, and ``K`` falls *faster* than the
Krylov dimension rises, so the `estimated total H·v` tile **drops** as you drag the slider
right. More eigenpairs, less work.

**So why not just ask for the whole spectrum?** Because the matvecs are not the only cost. At
the end of the run the projected Lanczos matrix has to be diagonalized to recover the eigenpairs
of the factorized problem, and that matrix grows with `howmany`. Past some point the
matrix–vector products stop being the bottleneck and the diagonalization takes over. So
`howmany` sets an interplay between the two: the **transformation** on one side, the
**diagonalization** (and the convergence checking that goes with it) on the other. Pushing it
too far is exactly the regime [Patched POLFED](@ref tutorial_patched) is built for.

**`:maxdos` is not `:middle`.** Switch between them. For this spectrum the DoS peak sits well
off centre, so the two targets produce visibly different intervals — and `:maxdos` gives a
*narrower* interval for the same `howmany`, because states are packed more densely there.

**Targets near the edges get cheap fast.** Try `(:rescaled, 0.6)` or `(:offset, 0.5)`. The DoS
thins out, so the interval holding the same number of states must be much wider, and ``K``
drops accordingly.

## The six stages

```@raw html
<style>
/* Palette for the two static flow diagrams below. Without these the SVG
   `fill="var(--pf-box)"` attributes are undefined and every box renders black. */
.polfed-flow {
  --pf-fg: #24292f;
  --pf-muted: #57606a;
  --pf-line: #8c959f;
  --pf-box: #ffffff;
  --pf-border: #d0d7de;
  --pf-auto: #eaf0f6;
  --pf-auto-border: #4e79a7;
  --pf-core: #fdf0e3;
  --pf-core-border: #d97706;
  margin: 1.5rem 0;
  overflow-x: auto;
}
html.theme--documenter-dark .polfed-flow,
html.theme--catppuccin-frappe .polfed-flow,
html.theme--catppuccin-macchiato .polfed-flow,
html.theme--catppuccin-mocha .polfed-flow {
  --pf-fg: #f3f5f7;
  --pf-muted: #a7b0b8;
  --pf-line: #7d8792;
  --pf-box: #1f2428;
  --pf-border: #58636a;
  --pf-auto: #232c33;
  --pf-auto-border: #7ea6cc;
  --pf-core: #2e2519;
  --pf-core-border: #e0a336;
}
.polfed-flow svg { display: block; min-width: 720px; width: 100%; height: auto; }
.polfed-flow .pf-t { fill: var(--pf-fg); font-size: 13px; font-weight: 600; }
.polfed-flow .pf-s { fill: var(--pf-muted); font-size: 11px; }
.polfed-flow .pf-n { fill: var(--pf-muted); font-size: 11px; font-weight: 700; }
.polfed-flow .pf-band { fill: var(--pf-muted); font-size: 11.5px; font-style: italic; }
</style>
```


```@raw html
<div class="polfed-flow">
<svg viewBox="0 0 900 210" role="img" aria-labelledby="pf-title pf-desc">
  <title id="pf-title">The POLFED pipeline, six stages</title>
  <desc id="pf-desc">Rescaling, density of states, interval, filter order, the matrix-free spectral transformation, and Lanczos. Stages two to four are resolved automatically from the density of states; stage five is the matrix-free core.</desc>

  <rect x="150" y="14" width="392" height="112" rx="10" fill="var(--pf-auto)" stroke="var(--pf-auto-border)" stroke-width="1.5" stroke-dasharray="6 5"/>
  <text class="pf-band" x="346" y="122" text-anchor="middle">auto from the DoS — you do not set these</text>

  <rect x="556" y="14" width="150" height="112" rx="10" fill="var(--pf-core)" stroke="var(--pf-core-border)" stroke-width="1.5" stroke-dasharray="6 5"/>
  <text class="pf-band" x="631" y="122" text-anchor="middle">matrix-free core</text>

  <g fill="var(--pf-box)" stroke="var(--pf-border)" stroke-width="1.5">
    <rect x="14"  y="36" width="118" height="60" rx="8"/>
    <rect x="164" y="36" width="104" height="60" rx="8"/>
    <rect x="288" y="36" width="104" height="60" rx="8"/>
    <rect x="412" y="36" width="116" height="60" rx="8"/>
    <rect x="570" y="36" width="122" height="60" rx="8"/>
    <rect x="726" y="36" width="160" height="60" rx="8"/>
  </g>

  <g text-anchor="middle">
    <text class="pf-n" x="73"  y="54">①</text><text class="pf-t" x="73"  y="72">Rescaling</text><text class="pf-s" x="73"  y="88">H → H̃ on [-1,1]</text>
    <text class="pf-n" x="216" y="54">②</text><text class="pf-t" x="216" y="72">DoS</text><text class="pf-s" x="216" y="88">KPM: N, R</text>
    <text class="pf-n" x="340" y="54">③</text><text class="pf-t" x="340" y="72">Interval</text><text class="pf-s" x="340" y="88">holds howmany</text>
    <text class="pf-n" x="470" y="54">④</text><text class="pf-t" x="470" y="72">Filter + order</text><text class="pf-s" x="470" y="88">chain + bisection</text>
    <text class="pf-n" x="631" y="54">⑤</text><text class="pf-t" x="631" y="72">Transform</text><text class="pf-s" x="631" y="88">Clenshaw, nothing stored</text>
    <text class="pf-n" x="806" y="54">⑥</text><text class="pf-t" x="806" y="72">Lanczos</text>
  </g>

  <g stroke="var(--pf-line)" stroke-width="2" fill="none">
    <path d="M132 66 H 164"/><path d="M268 66 H 288"/><path d="M392 66 H 412"/>
    <path d="M528 66 H 570"/><path d="M692 66 H 726"/>
  </g>
  <g fill="var(--pf-line)">
    <path d="M164 66 l -8 -4 v 8 z"/><path d="M288 66 l -8 -4 v 8 z"/><path d="M412 66 l -8 -4 v 8 z"/>
    <path d="M570 66 l -8 -4 v 8 z"/><path d="M726 66 l -8 -4 v 8 z"/>
  </g>
</svg>
</div>
```

### ① Rescaling — get the spectrum into ``[-1,1]``

Chebyshev polynomials are defined on ``[-1,1]``, so the operator is rescaled first:

```math
\tilde H = \frac{H - E_c}{\Delta E},\qquad
\Delta E = \frac{E_{\max}-E_{\min}}{2},\qquad
E_c = \frac{E_{\max}+E_{\min}}{2}.
```

``E_c`` is the **centre** of the spectrum and ``\Delta E`` its **half-width**, so
``\varepsilon = (E - E_c)/\Delta E`` maps ``E_{\min} \mapsto -1`` and ``E_{\max} \mapsto +1``.

``E_{\min}`` and ``E_{\max}`` come from a short both-ends Lanczos probe unless you supply them
through [`MappingConfig`](@ref UniPolfed.MappingConfig). They are **estimates**: a true extreme
eigenvalue can land exactly on ``\pm 1``, or a hair beyond. POLFED tolerates that rather than
clamping it.

Everything from here on happens in ``\varepsilon``. Results are converted back at the very end.

### ② Density of states — the map POLFED navigates by

The kernel polynomial method estimates ``\rho(\varepsilon)`` from Chebyshev moments and
stochastic traces:

```math
\rho_{\mathrm{KPM}}(\varepsilon)=
\frac{1}{\pi\sqrt{1-\varepsilon^2}}
\left[\mu_0 g_0 + 2\sum_{n=1}^{N-1}\mu_n g_n T_n(\varepsilon)\right],
\qquad
\mu_n = \mathrm{Tr}\!\left[T_n(\tilde H)\right]
\approx \frac{1}{R}\sum_{r=1}^{R}\langle r|T_n(\tilde H)|r\rangle .
```

Two knobs, both in [`DoSConfig`](@ref UniPolfed.DoSConfig) — and they fail in *different* ways.
Move them:

```@raw html
<div class="pf-explorer" id="pf-kpm">
  <div class="pf-explorer-controls">
    <div class="pf-explorer-row">
      <label for="pfk-n">N (moments)</label>
      <input id="pfk-n" type="range" min="8" max="500" step="2" value="250">
      <output id="pfk-n-out">250</output>
    </div>
    <div class="pf-explorer-row">
      <label for="pfk-r">R (probe vectors)</label>
      <input id="pfk-r" type="range" min="1" max="1000" step="1" value="300">
      <output id="pfk-r-out">300</output>
    </div>
    <div class="pf-explorer-row">
      <label for="pfk-kernel">kernel</label>
      <select id="pfk-kernel">
        <option value="jackson" selected>:Jackson</option>
        <option value="none">none (raw truncation)</option>
      </select>
      <output id="pfk-out"></output>
    </div>
  </div>
  <svg id="pfk-svg" viewBox="0 0 880 280" role="img" aria-label="Kernel polynomial density of states as the number of moments and probe vectors change"></svg>
</div>

<script>
(function () {
  const root = document.getElementById("pf-kpm");
  if (!root || !window.PFDATA) return;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("pfk-svg");
  const elN = document.getElementById("pfk-n");
  const elR = document.getElementById("pfk-r");
  const elK = document.getElementById("pfk-kernel");
  const outN = document.getElementById("pfk-n-out");
  const outR = document.getElementById("pfk-r-out");
  const outMsg = document.getElementById("pfk-out");

  const EIG = window.PFDATA.EIG;
  const D = EIG.length;
  const NMAX = 500;

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

  /* Exact Chebyshev moments of the sample spectrum, μ_n = (1/D) Σ_i T_n(ε_i).
     This is the trace the stochastic estimator is trying to approximate. */
  const MU = new Float64Array(NMAX + 1);
  const M2 = new Float64Array(NMAX + 1);   /* ⟨T_n²⟩, sets the estimator's variance */
  {
    for (const e of EIG) {
      let tkm2 = 1, tkm1 = e;
      MU[0] += 1; M2[0] += 1;
      if (NMAX >= 1) { MU[1] += e; M2[1] += e * e; }
      for (let n = 2; n <= NMAX; n += 1) {
        const t = 2 * e * tkm1 - tkm2;
        MU[n] += t; M2[n] += t * t;
        tkm2 = tkm1; tkm1 = t;
      }
    }
    for (let n = 0; n <= NMAX; n += 1) { MU[n] /= D; M2[n] /= D; }
  }

  /* One fixed noise realization, so moving R rescales the SAME draw and the
     1/√R trend is visible instead of the picture reshuffling each frame. */
  const NOISE = new Float64Array(NMAX + 1);
  {
    let seed = 20260727;
    const rnd = function () {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    for (let n = 0; n <= NMAX; n += 1) {
      /* Box–Muller */
      const u = Math.max(1e-12, rnd()), v = rnd();
      NOISE[n] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }

  /* Jackson kernel — what damps the Gibbs oscillations of a truncated series. */
  function jackson(n, N) {
    const q = Math.PI / (N + 1);
    return ((N - n + 1) * Math.cos(n * q) + Math.sin(n * q) / Math.tan(q)) / (N + 1);
  }

  const GRID = [];
  for (let i = 0; i <= 500; i += 1) GRID.push(-0.999 + (1.998 * i) / 500);

  function density(N, R, useJackson) {
    /* stochastic trace estimate: std ≈ sqrt(2⟨T_n²⟩ / (D·R)) */
    const mu = new Float64Array(N);
    for (let n = 0; n < N; n += 1) {
      const sd = Math.sqrt(Math.max(0, 2 * M2[n]) / (D * R));
      mu[n] = MU[n] + NOISE[n] * sd;
    }
    const g = new Float64Array(N);
    for (let n = 0; n < N; n += 1) g[n] = useJackson ? jackson(n, N) : 1;

    return GRID.map(function (x) {
      let tkm2 = 1, tkm1 = x, s = mu[0] * g[0];
      if (N > 1) s += 2 * mu[1] * g[1] * x;
      for (let n = 2; n < N; n += 1) {
        const t = 2 * x * tkm1 - tkm2;
        s += 2 * mu[n] * g[n] * t;
        tkm2 = tkm1; tkm1 = t;
      }
      return s / (Math.PI * Math.sqrt(Math.max(1e-9, 1 - x * x)));
    });
  }

  function render() {
    const N = Number(elN.value), R = Number(elR.value);
    const useJackson = elK.value === "jackson";
    outN.textContent = N;
    outR.textContent = R;

    const rho = density(N, R, useJackson);
    const ref = density(NMAX, 100000, true);   /* many moments, negligible noise */

    const fg = cssVar("--pfx-fg"), muted = cssVar("--pfx-muted"), grid = cssVar("--pfx-grid");
    const blue = cssVar("--pfx-blue"), green = cssVar("--pfx-green");

    const W = 880, L = 58, RR = 28, PW = W - L - RR, TOP = 24, BASE = 224;
    const X = x => L + ((x + 1.04) / 2.08) * PW;
    const top = Math.max.apply(null, ref) * 1.35;
    const Y = v => BASE - (Math.max(-top * 0.25, Math.min(top, v)) / top) * (BASE - TOP);

    svg.replaceChildren();

    svg.appendChild(mk("line", { x1: X(-1.02), y1: Y(0), x2: X(1.02), y2: Y(0),
      stroke: grid, "stroke-width": 1.2 }));

    /* reference: what the estimate converges to */
    let dref = "";
    for (let i = 0; i < GRID.length; i += 1)
      dref += (i === 0 ? "M" : " L") + X(GRID[i]).toFixed(1) + "," + Y(ref[i]).toFixed(1);
    svg.appendChild(mk("path", { d: dref, fill: "none", stroke: green,
      "stroke-width": 2, "stroke-dasharray": "6 5", opacity: .85 }));

    /* the estimate at the chosen N and R */
    let d = "";
    for (let i = 0; i < GRID.length; i += 1)
      d += (i === 0 ? "M" : " L") + X(GRID[i]).toFixed(1) + "," + Y(rho[i]).toFixed(1);
    svg.appendChild(mk("path", { d: d, fill: "none", stroke: blue,
      "stroke-width": 2.8, "stroke-linejoin": "round" }));

    /* the actual eigenvalues */
    for (const e of EIG) {
      svg.appendChild(mk("line", { x1: X(e), y1: BASE + 6, x2: X(e), y2: BASE + 14,
        stroke: muted, "stroke-width": 1.1, opacity: .5 }));
    }

    svg.appendChild(mk("line", { x1: X(-1.02), y1: BASE + 6, x2: X(1.02), y2: BASE + 6,
      stroke: fg, "stroke-width": 1.5, opacity: .6 }));
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      svg.appendChild(txt(t.toFixed(1).replace("0.0", "0"), X(t), BASE + 34,
        { "text-anchor": "middle", "font-size": 12, fill: fg }));
    }
    svg.appendChild(txt("rescaled energy ε", X(0), BASE + 52,
      { "text-anchor": "middle", "font-size": 11, fill: muted }));

    svg.appendChild(mk("line", { x1: W - 210, y1: 18, x2: W - 186, y2: 18,
      stroke: blue, "stroke-width": 2.8 }));
    svg.appendChild(txt("estimate", W - 180, 22, { "font-size": 11, fill: muted }));
    svg.appendChild(mk("line", { x1: W - 120, y1: 18, x2: W - 96, y2: 18,
      stroke: green, "stroke-width": 2, "stroke-dasharray": "6 5" }));
    svg.appendChild(txt("converged", W - 90, 22, { "font-size": 11, fill: muted }));

    const noiseScale = Math.sqrt(2 / (D * R));
    outMsg.textContent = "noise ~ " + noiseScale.toExponential(1);
  }

  elN.addEventListener("input", render);
  elR.addEventListener("input", render);
  elK.addEventListener("change", render);
  new MutationObserver(render).observe(document.documentElement,
    { attributes: true, attributeFilter: ["class"] });
  render();
})();
</script>
```


- **`N`** is how many moments the series keeps. Too few and the estimate is over-smoothed: real
  structure disappears. It is a **resolution** limit, and it is smooth and reproducible.
- **`R`** is how many random probe vectors estimate each trace. Too few and the curve gets
  **noisy** — wiggles that are not features of the spectrum at all. The error scales as
  ``1/\sqrt{D R}``, so it improves slowly.
- **`g_n`** is the kernel. Switch it off and watch the Gibbs oscillations appear — a truncated
  Chebyshev series rings, and the Jackson kernel is what damps it. The price is extra smoothing.

Drop `R` to ~10 and the noise is obvious; drop `N` to ~20 and the shape flattens instead. Since
`:maxdos` is *defined* as the peak of this estimate, either failure can move your target — which
is why a window target, which skips this stage entirely, is the reproducible choice.

### ③ From the DoS to an interval

Your `target` first resolves to a single rescaled centre ``\tilde\lambda`` — the spellings are in
[Target modes](@ref polfed_targets). Then that centre is **grown** into an interval.

The growth is a bisection on the half-width ``\delta``: widen the symmetric window
``[\tilde\lambda - \delta,\ \tilde\lambda + \delta]`` (clamped to ``[-1,1]``) until it holds at
least `howmany` states. The count comes from integrating the DoS against the exact Chebyshev
weight,

```math
n(\varepsilon_1,\varepsilon_2) = \int_{\varepsilon_1}^{\varepsilon_2}
\frac{\sum_k c_k T_k(\varepsilon)}{\sqrt{1-\varepsilon^2}}\, d\varepsilon ,
```

integrated in closed form rather than sampled, so no quadrature error enters the state count.

**See it happen:** in the [explorer](@ref workflow_explorer), drag `howmany` and watch the green
interval breathe symmetrically about λ while the *states inside* tile tracks it. Where the DoS is
tall the interval barely has to move; out in the tails it has to open up a long way for the same
number of states.

For a **window target** this whole stage is skipped — you supplied the interval yourself.

### ④ Interval → filter and order

The interval's geometry picks the filter. Touching a spectrum edge (``\pm 1``) sends POLFED down
the edge-filter chain; anything interior goes to the delta filters. Each candidate is validated,
and the first valid one wins.

The order ``K`` is then resolved so the filter's passband **covers** the interval. For the delta
filters it lands near

```math
K \approx \frac{2.655\sqrt{1-\tilde\lambda^{2}}}{\delta},
```

though POLFED bisects against the actual cutoff rather than trusting the formula. Note which way
this runs: a **narrow** window needs a **high** order, so a larger `howmany` — a wider interval —
makes the filter *cheaper*.

**See it happen:** the explorer's cutoff slider changes ``K`` with the interval held fixed, which
isolates the cutoff's role: a higher cutoff is a laxer admission test, so a lower-order polynomial
suffices. Details in [Filters](@ref filters).

### ⑤ The spectral transformation — matrix-free

Now the interesting part. ``P(\tilde H)`` is a degree-``K`` polynomial in a sparse matrix, which
is **dense** — forming it would cost ``\mathcal{O}(\mathcal{D}^2)`` and defeat the whole point.
So it is never formed.

Lanczos never needs the operator as an object; it only needs to *apply* it to a vector. So POLFED
hands it a closure, and each call evaluates the polynomial by the **Clenshaw recurrence** —
walking the coefficients backwards through three rotating buffers, using nothing but sparse
products ``\tilde H \cdot v``:

```@raw html
<div class="polfed-flow">
<svg viewBox="0 0 900 250" role="img" aria-label="The Clenshaw recurrence: three rotating buffers, K sparse matrix-vector products, constant memory">
  <text class="pf-t" x="14" y="22">what is NOT built</text>
  <g fill="none" stroke="var(--pf-line)" stroke-width="1.5" opacity=".55">
    <rect x="14" y="34" width="180" height="72" rx="6" stroke-dasharray="5 4"/>
  </g>
  <text class="pf-s" x="104" y="62" text-anchor="middle">P(H̃) as a matrix</text>
  <text class="pf-s" x="104" y="80" text-anchor="middle">dense, D × D</text>
  <text class="pf-s" x="104" y="98" text-anchor="middle">never allocated</text>
  <g stroke="var(--pf-core-border)" stroke-width="2.5">
    <line x1="30" y1="98" x2="178" y2="42"/><line x1="30" y1="42" x2="178" y2="98"/>
  </g>

  <text class="pf-t" x="248" y="22">what runs instead — Clenshaw, K steps</text>

  <g fill="var(--pf-box)" stroke="var(--pf-border)" stroke-width="1.5">
    <rect x="248" y="34" width="104" height="52" rx="7"/>
    <rect x="392" y="34" width="104" height="52" rx="7"/>
    <rect x="536" y="34" width="104" height="52" rx="7"/>
  </g>
  <g text-anchor="middle">
    <text class="pf-t" x="300" y="56">b₁</text><text class="pf-s" x="300" y="74">current</text>
    <text class="pf-t" x="444" y="56">b₂</text><text class="pf-s" x="444" y="74">previous</text>
    <text class="pf-t" x="588" y="56">b₃</text><text class="pf-s" x="588" y="74">the one before</text>
  </g>

  <g stroke="var(--pf-line)" stroke-width="2" fill="none">
    <path d="M352 60 H 392"/><path d="M496 60 H 536"/>
  </g>
  <g fill="var(--pf-line)">
    <path d="M392 60 l -8 -4 v 8 z"/><path d="M536 60 l -8 -4 v 8 z"/>
  </g>
  <path d="M588 86 V 108 H 300 V 86" fill="none" stroke="var(--pf-line)" stroke-width="2" stroke-dasharray="5 4"/>
  <path d="M300 86 l -4 8 h 8 z" fill="var(--pf-line)"/>
  <text class="pf-s" x="444" y="124" text-anchor="middle">buffers rotate — three vectors, whatever K is</text>

  <rect x="248" y="146" width="392" height="46" rx="8" fill="var(--pf-core)" stroke="var(--pf-core-border)" stroke-width="1.5"/>
  <text class="pf-t" x="444" y="166" text-anchor="middle">b₁ ← 2·H̃·b₂ − b₃ + cₖ·x</text>
  <text class="pf-s" x="444" y="184" text-anchor="middle">one sparse H·v per step — the only expensive thing here</text>

  <g fill="var(--pf-box)" stroke="var(--pf-border)" stroke-width="1.5">
    <rect x="688" y="146" width="198" height="46" rx="8"/>
  </g>
  <text class="pf-t" x="787" y="166" text-anchor="middle">P(H̃)·x = H̃·b₁ − b₂ + c₀·x</text>
  <text class="pf-s" x="787" y="184" text-anchor="middle">final sum, after K steps</text>

  <text class="pf-s" x="14" y="150">cost</text>
  <text class="pf-t" x="14" y="170">K sparse H·v</text>
  <text class="pf-s" x="14" y="150" opacity="0"> </text>
  <text class="pf-s" x="14" y="192">memory: 3 vectors</text>
</svg>
</div>
```

Two consequences, and they are the reason the method is practical at all:

- **Cost per application: ``K`` sparse matvecs.** Nothing cheaper, nothing hidden.
- **Memory: three vectors, independent of ``K``.** No dense ``P(H)``, no shift-invert
  factorization. Doubling the order costs time, not memory.

Clenshaw is used rather than summing ``\sum_k c_k T_k(\tilde H)`` term by term because it is both
cheaper and numerically stabler — the explicit sum loses accuracy badly at the orders POLFED
reaches. Full detail in [On-the-fly mapping](@ref on_the_fly_mapping).

#### How the recurrence is actually computed

Clenshaw evaluates ``\sum_{n=0}^{K} c_n T_n`` without ever forming a single ``T_n(\tilde H)``.
It walks the coefficients **backwards**, carrying three vectors:

```julia
b₁ = b₂ = b₃ = 0                       # three buffers, that is the whole working set

for k in K:-1:1
    b₃, b₂ = b₂, b₁                    # rotate: everything shifts one slot
    b₁ = 2 * (H̃ * b₂) - b₃ + c[k] * v  # one sparse H̃·v, then two vector updates
end

result = H̃ * b₁ - b₂ + c[0] * v        # the final sum
```

`result` is exactly ``P(\tilde H)\,v``. Three things make this work:

**Why backwards.** The Chebyshev polynomials obey ``T_{n+1} = 2xT_n - T_{n-1}``. Running that
recurrence *forwards* would mean building ``T_0(\tilde H)v, T_1(\tilde H)v, \dots`` and
accumulating ``c_n`` times each — correct, but it needs the running sum *and* two previous
vectors, and it is numerically poor at high order. Clenshaw folds the coefficients into the
recurrence itself, so the partial sums never appear.

**Why three buffers.** Each step reads `b₂` and `b₃` and writes `b₁`, then everything rotates
by one. Nothing from more than two steps ago is ever needed, so the older vector is overwritten.
The working set is three vectors — or three *blocks*, in a block-Lanczos run — regardless of
whether ``K`` is 10 or 10 000.

**Where the cost is.** The only expensive line is `H̃ * b₂`. Everything else is an axpy. The
loop runs ``K`` times but the first pass has `b₂ = 0`, so that product is skipped: with the one
in the final sum, the total is exactly **``K`` sparse matrix–vector products**.

In POLFED this loop is not written out at the call site — it is packaged into the closure that
Lanczos receives, and both the recurrence step and the final sum can be swapped for fused,
model-specific kernels via `clenshaw_recurrence` and `clenshaw_finalsum` in
[`MappingConfig`](@ref UniPolfed.MappingConfig). See
[Bring your own operator](@ref operator_interface).

### ⑥ Lanczos on the filtered operator

Here is the part worth being precise about: **Lanczos is never given ``H``.** It is given the
*mapping* — the closure from stage ⑤ that applies ``P(\tilde H)`` to a vector — and nothing else.
As far as the eigensolver is concerned it is doing an ordinary Lanczos run on an ordinary
operator; it has no idea a filter is involved.

That is what makes the design modular: the filter and the eigensolver never need to know about
each other. It also means the eigensolver's own behaviour — how a Krylov space is built, why
extremal eigenvalues converge first, what block Lanczos changes — is a completely separate topic,
and the one that explains why any of this works.

Two details POLFED does add on top:

- **Ritz values are converted to true energies during the run**, so convergence is tested against
  ``H`` rather than against the filter. A pair is accepted when
  ``\lVert H v - \lambda v \rVert < \texttt{eigentol}``.
- **At the end**, eigenvalues are recovered as Rayleigh quotients under ``H``, residuals computed,
  and the pairs sorted and trimmed to the interval you actually requested.

For how the eigensolver itself works, read on: [Krylov background](@ref krylov_background).

## Where the time goes

Total cost is, to a good approximation,

```math
(\#\ \text{Lanczos iterations}) \times K \times \mathrm{cost}(H\cdot v).
```

Every Lanczos iteration pays for ``K`` sparse products. So there are three levers on it, and
[Going faster](@ref going_faster) is organized around them:

| Lever | How | Where |
| :--- | :--- | :--- |
| Make ``H\cdot v`` cheaper | value-grouped kernels, GPU | [Optimized kernels](@ref optimized_kernels), [GPU](@ref going_faster_gpu) |
| Make ``K`` smaller | raise `howmany` — a wider interval means a lower order | [Going faster](@ref going_faster) |

There is a competing cost, though. The Krylov space grows linearly with `howmany`, and the
projected Lanczos matrix of size ``m`` is diagonalized **densely** at ``\mathcal{O}(m^3)``.
Past a few thousand eigenpairs that cubic term, not the filter, dominates the run — which is
what [Patched POLFED](@ref tutorial_patched) addresses by splitting the request into several
smaller runs.

## Read next

- [Krylov background](@ref krylov_background) — why edges are easy and the middle is hard.
- [Filters](@ref filters) — delta, edge and double filters; passband, leakage, cutoff.
- [On-the-fly mapping](@ref on_the_fly_mapping) — Clenshaw, and why nothing is stored.
