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

    # μ₀ is the total state count (unnormalized traces), so it is the one number that says
    # whether the moment estimate is sane before anything downstream consumes ρ.
    polfed_log(POLFED_DEBUG_LEVEL, "Density of states computed.",
               num_moments = length(moments), total_states = first(moments))

    return DoSResult(moments, coefficients, ρ)
end
