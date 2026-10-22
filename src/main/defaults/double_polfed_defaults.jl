"""
    Polfed.Defaults.double_polfed

Default registry for the `double_polfed` variant, following the per-function pattern of
[`Defaults.polfed`](@ref). Only the knobs that are genuinely *double*-specific are new here —
the strategy decision and the joint-run Krylov overhead. The algorithm-wide knobs (tolerances,
reorthogonalization, convergence scheduling) are aliased from [`Defaults.polfed`](@ref) so the
two entrypoints cannot silently drift apart.
"""
module double_polfed

import ..polfed   # sibling registry: the double variant reuses polfed's algorithm-wide knobs

# ── strategy decision ─────────────────────────────────────────────────────────

"""
    Defaults.double_polfed.strategy::Symbol = :auto

How `double_polfed` decides between ONE joint [`DoubleDeltaFilter`](@ref) run and TWO independent
single-filter runs:

- `:auto`     — decide from the polynomial orders (see [`double_ratio`](@ref)).
- `:double`   — force the joint run (falls back to two runs if the double filter cannot be built).
- `:separate` — force two runs.

The forced values exist for benchmarking: they are the only way to measure the two paths on the
same problem.
"""
const strategy = :auto

"""
    Defaults.double_polfed.krylov_overhead::Float64 = 1.4

Krylov-budget inflation for the JOINT (`DoubleDeltaFilter`) run, relative to a regular `polfed`
run for `max(howmany₁, howmany₂)` states.

One `BothEnds` Krylov space serves both windows at once, so its dimension is **not** the sum of
two single-window budgets: both transformed extremes converge in the same factorization. The
budget is therefore `krylov_overhead × expected_krylov_dim(single-delta fit, max(n₁, n₂))` — see
[`expected_krylov_dim`](@ref)`(::DoubleDeltaFilter, …)`, which reads this constant.

Because `1.4 · fit(max(n₁,n₂))` always exceeds `n₁ + n₂` for the fit in use, the budget also
satisfies the hard floor "the space must be able to hold every requested vector"; no separate
guard is needed.
"""
const krylov_overhead = 1.4

"""
    Defaults.double_polfed.double_ratio::Float64 = 0.4

Order-ratio threshold above which the JOINT double-delta run is preferred (`strategy = :auto`):
with `r = min(K₁, K₂) / max(K₁, K₂)`, take one joint run when `r ≥ double_ratio`, two runs
otherwise.

**Why this number.** Cost ≈ (Krylov dimension) × (matvecs per step). Two runs cost
`K₁·m₁ + K₂·m₂`; the joint run costs `K_d·m_d` with `K_d = max(K₁, K₂)` — the combined
polynomial's degree, so *both* windows pay the sharper peak's price. With `m₁ ≈ m₂ ≈ m` and
`m_d = krylov_overhead · m`, the joint run wins exactly when `krylov_overhead < 1 + r`. The
threshold is therefore `krylov_overhead - 1`, and the two constants must move together.
"""
const double_ratio = krylov_overhead - 1.0

# ── aliased algorithm-wide knobs (see Defaults.polfed for the documentation) ───

"""    Defaults.double_polfed.rot — reorthogonalization, aliased from [`Defaults.polfed`](@ref)."""
const rot = polfed.rot

"""    Defaults.double_polfed.basistype — Krylov basis storage, aliased from [`Defaults.polfed`](@ref)."""
const basistype = polfed.basistype

"""    Defaults.double_polfed.tol — Ritz-residual tolerance, aliased from [`Defaults.polfed`](@ref)."""
const tol = polfed.tol

"""    Defaults.double_polfed.eigentol — eigen-residual tolerance, aliased from [`Defaults.polfed`](@ref)."""
const eigentol = polfed.eigentol

"""    Defaults.double_polfed.overestimate_iters — Krylov safety factor, aliased from [`Defaults.polfed`](@ref)."""
const overestimate_iters = polfed.overestimate_iters

"""    Defaults.double_polfed.conv_checking — check scheduling, aliased from [`Defaults.polfed`](@ref)."""
const conv_checking = polfed.conv_checking

end # module double_polfed
