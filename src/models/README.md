# Models

The **physics tier**: Hamiltonian builders for the quantum many-body systems POLFED
is exercised on. Each builder returns a (by default sparse) matrix `H` suitable for
feeding straight into the spectral-transform pipeline.

This tier is **standalone** — it depends only on the Julia standard library
(`LinearAlgebra`, `SparseArrays`, `Random`) and never reaches into the `Numerics/`
or `UniPolfed/` tiers. That is the litmus test for living here: a model is just an
operator, with no knowledge of how POLFED will diagonalize it.

Two families are provided:

1. **Spin-1/2 chains** — XXZ and J1–J2 chains, built directly in a fixed
   magnetization sector (`spin_half_chains.jl`).
2. **Quantum Sun** — a chaotic GOE "grain" coupled to localized spins through
   long-range couplings, in full and U(1)-symmetric variants (the `QSun/`
   submodule).

---

## File layout

```text
src/models/
├── Models.jl              «module root» — using/includes/exports + the qsun_hamiltonian wrapper
├── spin_half_chains.jl     xxz_hamiltonian, j1j2_hamiltonian (+ private bond/field/assembly helpers)
└── qsun/                   «submodule» the Quantum Sun model
    ├── QSun.jl             «module root» — includes, exports, qsun_hamiltonian dispatch
    ├── spin_operators.jl   SpinBasis + bit-packed on-site Sˣ/Sʸ/Sᶻ, get_spin/set_spin
    ├── random_utils.jl     uniform_array, goe_matrix, random_neighbors, long_range_couplings
    ├── hilbert_space.jl    SpinHilbert, build_hilbert, state_index
    ├── quantum_sun.jl      build_hamiltonian          (full Hilbert space)
    └── quantum_sun_u1_conserving.jl  build_hamiltonian_cons  (fixed-Sᶻ sector)
```

`Models` is a submodule of `UniPolfed`, included directly at the package top level
(sibling to `Numerics`). `QSun` is a submodule of `Models`;
`spin_operators.jl` and the other files under `qsun/` share that namespace. Only
`Models.jl` and `QSun.jl` declare modules.

Public entry points:

```julia
using UniPolfed.Models: xxz_hamiltonian, j1j2_hamiltonian, qsun_hamiltonian
```

---

## Spin-1/2 chains — `spin_half_chains.jl`

Both constructors work in a **fixed magnetization sector**: `Lup` of the `L` sites
are up, fixing `Sᶻ = Lup − L/2`, so the matrix dimension is `binomial(L, Lup)`.

```julia
# XXZ:  H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) + Σ hᵢ Sᶻᵢ
H = xxz_hamiltonian(L, Lup, J, Δ, W; boundary=:periodic, field=0.0, fields=nothing, rng, use_sparse=true)

# J1–J2: adds a next-nearest-neighbor XXZ term with its own (J2, Δ2)
H = j1j2_hamiltonian(L, Lup, J1, J2, Δ1, Δ2, W; boundary=:periodic, …)
```

- **Encoding.** A basis state is an integer `state ∈ 0:(2^L − 1)`; bit `i-1` set ⇔
  site `i` is up (`Sᶻ = +½`). Diagonal entries collect the field term `Σ hᵢ Sᶻᵢ` and
  the Ising term `Σ J Δ Sᶻᵢ Sᶻⱼ`; off-diagonal entries are the `J/2` spin-flip hops
  between anti-aligned bonded sites (which stay in the sector).
- **Disorder.** Longitudinal fields are either supplied explicitly via `fields`
  (length `L`) or sampled from `[field − W, field + W]` with the given `rng`. `W = 0`
  gives a clean uniform field with no RNG draw.
- **Boundaries.** `:periodic` wraps via `mod1`; `:open` drops boundary-crossing
  bonds. Note that very small periodic rings list a physical bond from both
  endpoints (e.g. `L=2` periodic doubles the bond) — a documented enumeration quirk,
  not a bug.
- `J2 = 0` makes `j1j2_hamiltonian` identical to `xxz_hamiltonian` (regardless of
  `Δ2`).

---

## Quantum Sun — `qsun/`

The Quantum Sun model couples a small **chaotic grain** of `L_grain` spins (a GOE
block) to `L_loc` **localized** spins; localized spin `j` attaches to a random grain
site with a long-range coupling `g0 · α^{u_j}`, where the effective distance exponent
`u_j = (j−1) + η_j` carries uniform randomness `η_j ∈ [−ζ, ζ]`.

```julia
H = qsun_hamiltonian(L_loc, L_grain, g0, α;
                     γ=1, w=0.5, hz=1, ζ=0.2, S=0.5,
                     rng, use_U1=false, S_z=0.0, use_sparse=true)
```

Two builders sit behind the `use_U1` flag:

| variant | function | dimension | grain–ray coupling | conserves Sᶻ |
|---|---|---|---|---|
| full Hilbert space | `build_hamiltonian` | `d^{L_loc+L_grain}` | `SˣSˣ` | no |
| U(1)-symmetric sector | `build_hamiltonian_cons` | size of the `S_z` sector | `SˣSˣ + SʸSʸ` | yes |

> **Two intentional differences between the variants** (preserved verbatim from the
> original implementation):
> 1. The full-space model couples via `SˣSˣ` only and therefore does **not**
>    conserve total magnetization — correct for a full-space model. The U(1) variant
>    uses the magnetization-conserving XY form `SˣSˣ + SʸSʸ` and is restricted to a
>    fixed `Sᶻ = S_z` sector.
> 2. The grain block is normalized differently: `γ · GOE / √(d_grain + 1)` in the
>    full case versus a trace normalization `γ · R / √(Tr R² / d_sec)` in the sector
>    case.

`S` may be any integer or half-integer spin (`d = 2S + 1`); the wrapper rejects other
values. `S_z` must be a valid sector (`|S_z| ≤ L·S` with `S_z + L·S ∈ ℤ`).

### Internals

- **`spin_operators.jl`** — `SpinBasis(S)` packs one site into `⌈log₂ d⌉` bits of a
  `UInt64`. `get_spin`/`set_spin` read/write a site's projection; `sigma_x`,
  `sigma_y`, `sigma_z`, `sigma_0` apply the on-site operators, returning the (at most
  two) nonzero matrix elements of the ladder operators as flat tuples.
- **`hilbert_space.jl`** — `build_hilbert(L, B)` enumerates all `d^L` packed states
  (sorted); `state_index` binary-searches them, returning `0` for states outside the
  basis (how the assembly loops skip off-sector hops).
- **`random_utils.jl`** — the random ingredients (disordered fields, GOE grain,
  random attachment sites, long-range coupling profile), all taking an explicit
  `rng` so a whole realization is reproducible from one seed.

---

## Testing

Tests in `test/models/` mirror this layout and assert against **mathematical ground
truth**, not captured snapshots:

- **spin operators** — the on-site `Sˣ/Sʸ/Sᶻ`, materialized as dense matrices, must
  close the su(2) algebra (`[Sˣ,Sʸ] = i Sᶻ` and cyclic) and satisfy the Casimir
  `Sˣ² + Sʸ² + Sᶻ² = S(S+1)·I` for `S ∈ {½, 1, 3⁄2}`; `S⁺ = Sˣ + iSʸ` raises only.
- **hilbert space** — `build_hilbert` enumerates exactly `d^L` sorted, distinct
  states (`0:2^L−1` for spin-1/2); `state_index` round-trips and returns `0`
  off-basis.
- **random utils** — `goe_matrix` is symmetric; `long_range_couplings` equals the
  exact profile `α^{j-1}` at `ζ = 0` and stays inside its randomness band otherwise;
  all are reproducible from a seed.
- **spin chains** — the Heisenberg dimer has the exact spectrum `{−¾, ¼}`; a general
  XXZ matrix matches an **independent kron-of-Pauli-matrices** construction
  (restricted to the sector); `J2 = 0 ⇒ j1j2 ≡ xxz`; dimensions are `binomial(L,Lup)`;
  all matrices are Hermitian; invalid arguments throw `ArgumentError`.
- **Quantum Sun** — Hermiticity; full vs. sector dimensions (the U(1) sectors tile
  the full `2^L` space); the `γ=0, α=0, w=0` limit reduces to the exact local-field
  diagonal; reproducibility and sparse/dense agreement; invalid spin / sector throw.

Run with `Pkg.test()` (add `coverage=true` for line coverage). GPU paths are not
involved — these builders are CPU-only; the module is at 100% line coverage.
