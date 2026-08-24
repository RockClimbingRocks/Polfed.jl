# Density-of-states adapter: the POLFED-facing wrapper around the Numerics.KPM
# primitive. DoSConfig (user surface) → density_of_states → DoSResult (moments + ρ).
# Included directly into `module UniPolfed` (no submodule).


"""
    DoSConfig(; N=Defaults.polfed.N, R=Defaults.polfed.R, kernel=Defaults.polfed.kernel)

User-facing configuration for POLFED's density-of-states (DoS) estimation step.

POLFED estimates the DoS once, up front, with the kernel polynomial method (KPM):
`R` stochastic probes and `N` Chebyshev moments, damped by `kernel`. The resulting
density (in rescaled coordinates) is used to place the target window and size the
filter. This struct is the public surface; the estimation itself is
[`density_of_states`](@ref).

# Fields
- `N::Int`: number of Chebyshev moments. Higher ⇒ sharper resolution, more cost.
  Default [`Defaults.polfed.N`](@ref) (`250`).
- `R::Int`: number of random probes. Higher ⇒ less statistical noise (`∝ 1/√R`).
  Default [`Defaults.polfed.R`](@ref) (`300`).
- `kernel::Symbol`: KPM damping kernel (e.g. `:Jackson`, `:Dirichlet`).
  Default [`Defaults.polfed.kernel`](@ref) (`:Jackson`).

# Example
```julia
DoSConfig()                    # defaults
DoSConfig(N = 500, R = 400)    # finer resolution, more probes
```

# Notes
There is intentionally no `DoSConfig` → plan resolution step: every field is a
concrete value with a default, so there is nothing to resolve (unlike the mapping
or transform configs, which estimate spectral bounds). [`density_of_states`](@ref)
consumes a `DoSConfig` directly.
"""
struct DoSConfig
    N::Int
    R::Int
    kernel::Symbol

    function DoSConfig(;
        N::Int       = Defaults.polfed.N,
        R::Int       = Defaults.polfed.R,
        kernel::Symbol = Defaults.polfed.kernel,
    )
        new(N, R, kernel)
    end
end



"""
    DoSResult(moments, ρ)

Result of a density-of-states estimation (see [`density_of_states`](@ref)).

# Fields
- `moments::AbstractVector{<:Real}`: the raw KPM Chebyshev moments `μₖ`. Kept
  because they are cheap and useful downstream — plotting, re-damping with a
  different kernel, or reporting — without re-running the estimation.
  - `coefficients::AbstractVector{<:Real}`: the KPM coefficients used for density reconstruction. Moments + kernel damping.
  - `ρ`: the reconstructed spectral density, a callable `ρ(x)` valid for
  `x ∈ (-1, 1)` in **rescaled** coordinates.

`ρ` is stored as a concrete type parameter (not a boxed `Function` field) so calls
through a `DoSResult` stay type-stable.
"""
struct DoSResult{M<:AbstractVector,N<:AbstractVector,F}
    moments::M
    coefficients::N
    ρ::F
end



"""
    density_of_states(cfg::DoSConfig, f!, dim, E, pu) -> DoSResult

Estimate the spectral density of a Hermitian operator with the kernel polynomial
method (KPM) — POLFED's up-front DoS step.

`f!` must apply the **already-rescaled** operator (`H̃ = (H - b)/a`, spectrum in
`[-1, 1]`) in place: `f!(y, x)` writes `H̃·x` into `y`. The returned density is in
those rescaled coordinates.

This is a thin adapter over the `Numerics.KPM` primitive: it runs
[`kpm_moments`](@ref) with the configured `N`/`R` and reconstructs the density with
[`kpm_density`](@ref) using the configured `kernel`.

# Arguments
- `cfg::DoSConfig`: number of moments `N`, probes `R`, and damping `kernel`.
- `f!`: in-place action of the rescaled Hermitian operator, `f!(y, x) → y = H̃·x`.
- `dim::Int`: Hilbert-space dimension (the operator is `dim × dim`).
- `E::Type{<:Number}`: element type of the probe vectors (e.g. `Float64`).
- `pu::ProcessingUnit`: [`CPU`](@ref) or [`GPU`](@ref) backend.

# Returns
- [`DoSResult`](@ref) holding the raw `moments` and the density `ρ(x)`.

# Notes
Correctness requires `f!` to be Hermitian (real spectrum) — KPM's Chebyshev
expansion and moment-doubling assume it. The density obeys KPM's sum rule
`∫₋₁¹ ρ = g₀·μ₀`, i.e. the total number of states when the moments are
unnormalized traces.
"""
function calculate_dos(
    cfg::DoSConfig,
    f!::Function,
    sys::SysInfo,
)
    polfed_log(POLFED_INFO_LEVEL, "Computing the density of states (KPM) on the rescaled operator.",
               N = cfg.N, R = cfg.R, kernel = cfg.kernel)

    moments = kpm_moments(f!, cfg.N, cfg.R, sys.x0, sys.pu)
    coefficients = kpm_coeffs(moments, cfg.kernel)
    ρ = kpm_density(moments; kernel = cfg.kernel)
    dos = DoSResult(moments, coefficients, ρ)

    _log_dos(dos, sys.dim, cfg)
    return dos
end

"""
    _log_dos(dos::DoSResult, dim::Int, cfg::DoSConfig) -> nothing

Diagnose the estimated density before anything consumes it. Two things are checked, because every
window sizing downstream (`resolve_interval`, `howmany_in_interval`, the patch decomposition)
integrates this density and inherits its error:

- **sum rule** (WARN) — `∫ρ` over `[-1, 1]` must recover the Hilbert-space dimension. A large
  miss means `f!_rescaled` is not the Hermitian, `[-1, 1]`-rescaled operator KPM assumes.
- **profile** (DEBUG) — `ρ` sampled across the spectrum, its peak, the moment decay, and the
  `1/√R` statistical error scale. A negative sample is a Gibbs artifact and is called out.

Deep-debug adds the head of the raw moment sequence. Everything here is scalar work on the
already-computed coefficients — no operator applications — but the profile is still gated so a
level-2 run does not pay for samples it will not print.
"""
function _log_dos(dos::DoSResult, dim::Int, cfg::DoSConfig)
    total = integrate_chebyshev_weighted(dos.coefficients, -1.0, 1.0) / π
    relerr = abs(total - dim) / max(dim, 1)
    relerr > Defaults.polfed.dos_sum_rule_rtol && polfed_log(POLFED_WARN_LEVEL,
        "The KPM sum rule ∫ρ = dim is violated: the estimated density is unreliable, and every " *
        "window/state count derived from it will be too. Check that the operator is Hermitian " *
        "and that its spectrum really is rescaled into [-1, 1].",
        integrated_states = total, dim = dim, relative_error = relerr)

    should_log(POLFED_DEBUG_LEVEL) || return nothing
    xs = range(-0.98, 0.98; length = 9)
    ys = [dos.ρ(x) for x in xs]
    imax = argmax(ys)
    minimum(ys) < 0 && polfed_log(POLFED_WARN_LEVEL,
        "The estimated density is negative somewhere (Gibbs oscillation); increase N or use a " *
        "stronger kernel.", min_ρ = minimum(ys))
    polfed_log(POLFED_DEBUG_LEVEL, "Density of states estimated.",
        total_states = total, dim = dim, sum_rule_error = relerr, μ₀ = first(dos.moments),
        peak = (xs[imax], ys[imax]), moment_decay = abs(last(dos.moments) / first(dos.moments)),
        statistical_error = 1 / sqrt(cfg.R),
        profile = join((@sprintf("%.2g:%.3g", x, y) for (x, y) in zip(xs, ys)), "  "))
    polfed_log(POLFED_DEEP_DEBUG_LEVEL, "KPM moments (head).",
               moments = first(dos.moments, min(10, length(dos.moments))))
    return nothing
end
