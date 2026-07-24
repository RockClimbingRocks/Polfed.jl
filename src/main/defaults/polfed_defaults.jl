"""
    UniPolfed.Defaults.polfed

Default parameter values for the `polfed` entrypoint and its pipeline, referenced qualified
(e.g. `Defaults.polfed.N`) so the names never leak into `module UniPolfed`'s namespace. Defaults are
**per function**: a variant like `patched_polfed` carries its own [`Defaults.patched_polfed`](@ref).

Filter-specific defaults live beside their filter implementations; this module contains
algorithm-wide defaults shared by the `polfed` pipeline.
"""
module polfed

import ...Numerics.Lanczos: FullRO, MatrixBasis

# ── Density of states (KPM) ───────────────────────────────────────────────────

"""
    Defaults.polfed.N::Int = 250

Default number of Chebyshev moments used in KPM density-of-states estimation.
Higher `N` sharpens spectral resolution at higher cost.
"""
const N = 250

"""
    Defaults.polfed.R::Int = 300

Default number of stochastic random probes for KPM density-of-states estimation.
Statistical error scales as `1/√R`.
"""
const R = 300

"""
    Defaults.polfed.kernel::Symbol = :Jackson

Default KPM damping kernel (see `Numerics.KPM.get_kernel`).
"""
const kernel = :Jackson

# ── Mapping ───────────────────────────────────────────────────────────────────

"""
    Defaults.polfed.optimize_mapping::Bool = false

Whether to build optimized (value-grouped) mapping/Clenshaw kernels from a matrix
input. Off by default; effective for structured Hamiltonians with few unique
off-diagonal values.
"""
const optimize_mapping = false

# ── Target resolution ─────────────────────────────────────────────────────────

"""
    Defaults.polfed.maxdos_edge_margin::Float64 = 0.01

Margin kept away from the rescaled spectral edges `±1` when searching for the
max-DoS energy (see [`_resolve_center`](@ref)`(::TargetMaxDoS, …)`). The KPM
density `ρ(x) = expansion(x)/(π√(1-x²))` carries a Jacobian that diverges as
`x → ±1`; at finite moment count that divergence is an imperfectly-cancelled
numerical artifact, so `argmax` snaps to the endpoint unless the search stays a
physical margin inside `[-1, 1]`. With the default margin, the scan covers
`[-0.99, 0.99]`, allowing the bulk mode to win.
"""
const maxdos_edge_margin = 0.01

# Transform / filter shape defaults are **per-filter** and live next to their filter
# type (module `<Filter>Defaults`), because they vary from one filter to the next (a delta
# filter's cutoff is not an edge filter's). They are intentionally *not* here — this module
# holds only the filter-agnostic, algorithm-wide defaults.

# ── Factorization (Lanczos) ───────────────────────────────────────────────────

"""
    Defaults.polfed.rot::FullRO = FullRO()

Default reorthogonalization technique for the Lanczos factorization. Full
reorthogonalization is the safe default; since the spectral transform dominates
runtime, its cost is negligible.
"""
const rot = FullRO()

"""
    Defaults.polfed.basistype::Type = MatrixBasis

Default Krylov basis storage for the Lanczos factorization.
"""
const basistype = MatrixBasis

"""
    Defaults.polfed.tol::Float64 = 1e-14

Default Ritz-residual tolerance for the Lanczos factorization.
"""
const tol = 1e-14

"""
    Defaults.polfed.eigentol::Float64 = 1e-9

Default eigen-residual (`‖A·v − λ·v‖`) tolerance for accepting converged eigenpairs.
"""
const eigentol = 1e-9

"""
    Defaults.polfed.overestimate_iters::Float64 = 1.25

Safety factor multiplying the predicted Krylov dimension, to allow enough
iterations for all requested eigenpairs to converge. Increase it if a run exhausts
its iterations without converging everything.
"""
const overestimate_iters = 1.25

# ── Convergence-check scheduling ──────────────────────────────────────────────

"""
    Defaults.polfed.check_start_frac::Float64 = 1.0

Fraction of the **expected** Krylov dimension at which POLFED starts running convergence checks
(see [`conv_checking`](@ref)). `1.0` = start exactly at the expected dim; lower values start
earlier (more, cheaper checks), higher values later.
"""
const check_start_frac = 1.0


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
| `ε ≤ 1e-3`           | `k + 5`          |

`expected_dim` is the **un-inflated** predicted Krylov dimension (`expected_krylov_dim(filter, …, η=1)`),
computed by POLFED and captured in the closure — it is not on the Lanczos criterion (which only
carries the inflated `maxdim`). Passed to `lanczos(…; next_check = conv_checking(expected_dim))`.
"""
conv_checking(expected_dim::Real) =
    (state, criterion, k) -> begin
        start_dim = check_start_frac * expected_dim
        k * criterion.blocksize < start_dim && return ceil(Int, start_dim / criterion.blocksize)
        ε = state.residual

        return ε > 1e-3 ? k + 20 : ε > 1e-3 ? k + 10 : k + 5
    end

    

end # module polfed
