"""
    UniPolfed.Defaults.patched_polfed

Default registry for the `patched_polfed` variant, following the per-function pattern of
[`Defaults.polfed`](@ref).
"""
module patched_polfed

import ...Numerics.Lanczos: FullRO, MatrixBasis   # for the `rot`/`basistype` defaults below

"""
    Defaults.patched_polfed.edge_distance::Float64 = 0.1

Default inward distance from the rescaled spectral edges `±1` used when constructing
patched POLFED intervals.
"""
const edge_distance = 0.1

"""
    Defaults.patched_polfed.howmany_pad::Int = 10

Per-side overlap, in states, applied to each patch by
[`intervalize_interval_overlapping`](@ref). Each patch is grown outward on both sides by a
band holding about this many states (per the DoS), so adjacent patches share a genuine
overlap region around every boundary — giving the gluing step a redundant band on each side
of a patch boundary instead of a single shared endpoint. `0` disables the overlap.
"""
const howmany_pad = 3

"""
    Defaults.patched_polfed.howmany_edge::Int = 50

Number of extremal eigenpairs peeled from each spectrum edge (`±1`) up front, by a dedicated
[`EdgeFilter`](@ref) run with a COUNT target ([`LowestEigs`](@ref)/[`HighestEigs`](@ref)),
before the interior is tiled with delta filters. The edges are handled this way because the
extreme eigenvalues sit *on* `±1` (where `Emin`/`Emax` are only estimates) and delta filters
cannot be placed there; a count-from-the-extreme target converges them robustly regardless of
where `±1` actually falls. Capped at the number of states in the requested window.
"""
const howmany_edge = 50

"""
    Defaults.patched_polfed.max_glue_retries::Int = 3

Maximum number of times an interior patch is re-run when its converged spectrum fails to bracket
its owned interval (a possible glue gap). Each retry widens **both** the overlap (padding) and the
Krylov budget — they fix different failure modes (an owned edge too close to the bracket edge vs.
too few iterations), so both are grown to be safe. `0` disables retries (fall straight to a warning).
"""
const max_glue_retries = 3

"""
    Defaults.patched_polfed.retry_budget_growth::Float64 = 2.0

Factor the Krylov-budget safety (`overestimate_iters`) is multiplied by on each coverage retry
(see [`max_glue_retries`](@ref)). The padding grows additively (`+howmany_pad` per retry) alongside.
"""
const retry_budget_growth = 2.0








"""
    Defaults.patched_polfed.N::Int = 250

    Default number of Chebyshev moments used in KPM density-of-states estimation.
    Higher `N` sharpens spectral resolution at higher cost.
"""
const N = 1000

"""
    Defaults.patched_polfed.R::Int = 300

    Default number of stochastic random probes for KPM density-of-states estimation.
    Statistical error scales as `1/√R`.
"""
const R = 1000

"""
    Defaults.patched_polfed.kernel::Symbol = :Jackson

    Default KPM damping kernel (see `Numerics.KPM.get_kernel`).
"""
const kernel = :Jackson

# ── Mapping ───────────────────────────────────────────────────────────────────

"""
    Defaults.patched_polfed.optimize_mapping::Bool = false

    Whether to build optimized (value-grouped) mapping/Clenshaw kernels from a matrix
    input. Off by default; effective for structured Hamiltonians with few unique
    off-diagonal values.
"""
const optimize_mapping = false

# ── Target resolution ─────────────────────────────────────────────────────────

"""
    Defaults.patched_polfed.maxdos_edge_margin::Float64 = 0.01

    Margin kept away from the rescaled spectral edges `±1` when searching for the
    max-DoS energy (see [`_resolve_center`](@ref)`(::TargetMaxDoS, …)`). The KPM
    density `ρ(x) = expansion(x)/(π√(1-x²))` carries a Jacobian that diverges as
    `x → ±1`; at finite moment count that divergence is an imperfectly-cancelled
    numerical artifact, so `argmax` snaps to the endpoint unless the search stays a
    physical margin inside `[-1, 1]`. Scanning `[-1 + margin, 1 - margin]` (matching
    the legacy `[-0.99, 0.99]`) lets the bulk mode win.
"""
const maxdos_edge_margin = 0.01

# Transform / filter shape defaults are **per-filter** and live next to their filter
# type (module `<Filter>Defaults`), because they vary from one filter to the next (a delta
# filter's cutoff is not an edge filter's). They are intentionally *not* here — this module
# holds only the filter-agnostic, algorithm-wide defaults.

# ── Factorization (Lanczos) ───────────────────────────────────────────────────

"""
    Defaults.patched_polfed.rot::FullRO = FullRO()

    Default reorthogonalization technique for the Lanczos factorization. Full
    reorthogonalization is the safe default; since the spectral transform dominates
    runtime, its cost is negligible.
"""
const rot = FullRO()

"""
    Defaults.patched_polfed.basistype::Type = MatrixBasis

    Default Krylov basis storage for the Lanczos factorization.
"""
const basistype = MatrixBasis

"""
    Defaults.patched_polfed.tol::Float64 = 1e-14

    Default Ritz-residual tolerance for the Lanczos factorization.
"""
const tol = 1e-14

"""
    Defaults.patched_polfed.eigentol::Float64 = 1e-9

    Default eigen-residual (`‖A·v − λ·v‖`) tolerance for accepting converged eigenpairs.
"""
const eigentol = 1e-9

"""
    Defaults.patched_polfed.overestimate_iters::Float64 = 1.25

    Safety factor multiplying the predicted Krylov dimension, to allow enough
    iterations for all requested eigenpairs to converge. Increase it if a run exhausts
    its iterations without converging everything.
"""
const overestimate_iters = 1.25

# ── Convergence-check scheduling ──────────────────────────────────────────────

"""
    Defaults.patched_polfed.check_start_frac::Float64 = 1.0

    Fraction of the **expected** Krylov dimension at which POLFED starts running convergence checks
    (see [`conv_checking`](@ref)). `1.0` = start exactly at the expected dim; lower values start
    earlier (more, cheaper checks), higher values later.
"""
const check_start_frac = 1.075


"""
    conv_checking(expected_dim::Real) -> next_check

    Build the Lanczos `next_check` scheduling callback for POLFED. The returned
    `(state, criterion, k) -> Int` callback defers all convergence checks until the Krylov dimension
    reaches `check_start_frac × expected_dim` — before that the filtered target eigenpairs cannot have
    converged, so a check is wasted work — then adapts the interval to the current Ritz residual:

    | current residual `ε` | next check after |
    |----------------------|------------------|
    | `ε > 1e-2`           | `k + 20`         |
    | `1e-3 < ε ≤ 1e-2`    | `k + 10`         |
    | `ε ≤ 1e-3`           | `k + 10`          |

    `expected_dim` is the **un-inflated** predicted Krylov dimension (`expected_krylov_dim(filter, …, η=1)`),
    computed by POLFED and captured in the closure — it is not on the Lanczos criterion (which only
    carries the inflated `maxdim`). Passed to `lanczos(…; next_check = conv_checking(expected_dim))`.
"""
conv_checking(expected_dim::Real) =
    (state, criterion, k) -> begin
        start_dim = check_start_frac * expected_dim
        k * criterion.blocksize < start_dim && return ceil(Int, start_dim / criterion.blocksize)
        ε = state.residual

        return ε > 1e-3 ? k + 20 : ε > 1e-3 ? k + 10 : k + 10
    end

    




end # module patched_polfed
