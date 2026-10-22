# Density-of-states adapter: the POLFED-facing wrapper around the Numerics.KPM
# primitive. DoSConfig (user surface) → calculate_dos → DoSResult (moments + ρ).
# Included directly into `module Polfed` (no submodule).


"""
    DoSConfig(; N=Defaults.polfed.N, R=Defaults.polfed.R, kernel=Defaults.polfed.kernel,
                rng=Random.default_rng(), block_size=nothing, tasks=nothing)

User-facing configuration for POLFED's density-of-states (DoS) estimation step.

POLFED estimates the DoS once, up front, with the kernel polynomial method (KPM):
`R` stochastic probes and `N` Chebyshev moments, damped by `kernel`. The resulting
density (in rescaled coordinates) is used to place the target window and size the
filter. This struct is the public surface; the estimation itself is
[`calculate_dos`](@ref).

# Fields
- `N::Int`: number of Chebyshev moments. Higher ⇒ sharper resolution, more cost.
  Default [`Defaults.polfed.N`](@ref) (`500`).
- `R::Int`: number of random probes. Higher ⇒ less statistical noise (`∝ 1/√R`).
  Default [`Defaults.polfed.R`](@ref) (`500`).
- `kernel::Symbol`: KPM damping kernel (e.g. `:Jackson`, `:Dirichlet`).
  Default [`Defaults.polfed.kernel`](@ref) (`:Jackson`).
- `rng::AbstractRNG`: generator for the stochastic probes. Default `Random.default_rng()`,
  which makes every call draw fresh probes — so two POLFED runs on the SAME operator resolve
  slightly DIFFERENT windows, and a caller who seeded their own generator for the Hamiltonian
  and starting block still gets an irreproducible target. Pass a seeded generator to pin it.
  Ignored on a [`GPU`](@ref), where probes come from the device generator.
- `block_size::Union{Int,Nothing}`: probes per operator call — the DoS's own width, never the
  Krylov block size. `nothing` (default) is [`Defaults.polfed.dos_block_size`](@ref) (`8`) on the
  matrix entrypoints and `1` on the operator ones. Clamped to `R` and below `dim`; `1` on a GPU.
- `tasks::Union{Int,Nothing}`: probe blocks run concurrently. `nothing` (default) is
  `Threads.nthreads()` on the matrix entrypoints and `1` on the operator ones; `1` on a GPU.

!!! warning "Operator entrypoints"
    On `polfed(f!, …)`, `block_size > 1` hands `f!` a `dim × block_size` block, and `tasks > 1`
    calls it concurrently (it must be safe on distinct arrays). Opt in only if `f!` supports both.

# Example
```julia
DoSConfig()                        # defaults
DoSConfig(N = 500, R = 400)        # finer resolution, more probes
DoSConfig(rng = MersenneTwister(1))# reproducible window
DoSConfig(block_size = 8, tasks = Threads.nthreads()) # polfed(f!, …) with a generic, thread-safe f!
```

# Notes
There is intentionally no `DoSConfig` → plan resolution step: the only automatic fields,
`block_size` and `tasks`, resolve to plain integers per entrypoint ([`_owned_operator_dos`](@ref),
[`_dos_parallelism`](@ref)). [`calculate_dos`](@ref) consumes a `DoSConfig` directly.

For a fixed `rng` seed and `block_size` the moments are bit-identical for every `tasks`, thread
count, Krylov block size and `optimize_mapping`.

`rng` is a type parameter rather than an `AbstractRNG` field: it is read once per probe inside
the moment loop, and an abstract field would box it and force a dynamic dispatch there.
"""
struct DoSConfig{RNG<:AbstractRNG}
    N::Int
    R::Int
    kernel::Symbol
    rng::RNG
    block_size::Union{Int,Nothing}
    tasks::Union{Int,Nothing}

    function DoSConfig(;
        N::Int       = Defaults.polfed.N,
        R::Int       = Defaults.polfed.R,
        kernel::Symbol = Defaults.polfed.kernel,
        rng::AbstractRNG = Random.default_rng(),
        block_size::Union{Int,Nothing} = nothing,
        tasks::Union{Int,Nothing} = nothing,
    )
        isnothing(block_size) || block_size >= 1 || throw(ArgumentError(
            "DoSConfig: block_size must be >= 1 (or nothing), got $block_size."))
        isnothing(tasks) || tasks >= 1 ||
            throw(ArgumentError("DoSConfig: tasks must be >= 1 (or nothing), got $tasks."))
        new{typeof(rng)}(N, R, kernel, rng, block_size, tasks)
    end
end

"""
    _owned_operator_dos(cfg::DoSConfig) -> DoSConfig

`cfg` with the automatic (`nothing`) `block_size`/`tasks` resolved for an operator POLFED owns:
[`Defaults.polfed.dos_block_size`](@ref) and `Threads.nthreads()`. Explicit values are kept.
Used by the matrix entrypoints, whose DoS operator is a generic, reentrant `mul!` on `H`.
"""
_owned_operator_dos(cfg::DoSConfig) =
    DoSConfig(N = cfg.N, R = cfg.R, kernel = cfg.kernel, rng = cfg.rng,
              block_size = something(cfg.block_size, Defaults.polfed.dos_block_size),
              tasks = something(cfg.tasks, Threads.nthreads(:default)))



"""
    DoSResult(moments, ρ)

Result of a density-of-states estimation (see [`calculate_dos`](@ref)).

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
    _dos_probe_template(sys::SysInfo, block_size::Int) -> AbstractVecOrMat

The KPM probe block for the DoS: `dim × block_size` under [`ColumnLayout`](@ref) (a plain
length-`dim` vector for one probe) and `block_size × dim` under [`RowLayout`](@ref).

Deliberately **not** shaped like `sys.x0`, so the density does not depend on the Krylov block
size. `similar` carries the element type and device across.
"""
function _dos_probe_template(sys::SysInfo, block_size::Int)
    sys.layout isa RowLayout && return similar(sys.x0, block_size, sys.dim)
    block_size == 1 && return similar(sys.x0, sys.dim)
    return similar(sys.x0, sys.dim, block_size)
end

"""
    _dos_parallelism(cfg::DoSConfig, sys::SysInfo) -> (block_size, tasks)

The probe-block width and task count the DoS runs with: `(1, 1)` on a GPU; on the CPU
`cfg.block_size` clamped to `R` and below `dim` (KPM reads the larger axis as the Hilbert space)
and `cfg.tasks`, each with `nothing` meaning `1` (the operator-entrypoint default).
"""
function _dos_parallelism(cfg::DoSConfig, sys::SysInfo)
    sys.pu isa GPU && return (1, 1)
    block_size = clamp(something(cfg.block_size, 1), 1, max(1, min(cfg.R, sys.dim - 1)))
    return (block_size, something(cfg.tasks, 1))
end


"""
    calculate_dos(cfg::DoSConfig, f!, sys::SysInfo) -> DoSResult

Estimate the spectral density of a Hermitian operator with the kernel polynomial
method (KPM) — POLFED's up-front DoS step.

`f!` must apply the **already-rescaled** operator (`H̃ = (H - b)/a`, spectrum in
`[-1, 1]`) in place: `f!(Y, X)` writes `H̃·X` into `Y` for a probe block `X` (see
[`_dos_probe_template`](@ref)). The returned density is in those rescaled coordinates.

This is a thin adapter over the `Numerics.KPM` primitive: it runs
[`kpm_moments`](@ref) with the configured `N`/`R`, block width and task count, and reconstructs
the density with [`kpm_density`](@ref) using the configured `kernel`.

# Arguments
- `cfg::DoSConfig`: moments `N`, probes `R`, `kernel`, `rng`, `block_size`, `tasks`.
- `f!`: in-place action of the rescaled Hermitian operator, `f!(Y, X) → Y = H̃·X`.
- `sys::SysInfo`: dimension, element type, device and layout.

# Returns
- [`DoSResult`](@ref) holding the raw `moments` and the density `ρ(x)`.

# Notes
On the CPU this runs inside [`with_serial_libraries`](@ref): BLAS and MKL at one thread, the
parallelism coming from `tasks` concurrent probe blocks.

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
    block_size, tasks = _dos_parallelism(cfg, sys)
    polfed_log(POLFED_INFO_LEVEL, "Computing the density of states (KPM) on the rescaled operator.",
               N = cfg.N, R = cfg.R, kernel = cfg.kernel, rng = typeof(cfg.rng),
               probe_block_size = block_size, tasks = tasks)

    template = _dos_probe_template(sys, block_size)
    run() = kpm_moments(f!, cfg.N, cfg.R, template, sys.pu; rng = cfg.rng, ntasks = tasks)
    moments = sys.pu isa GPU ? run() : with_serial_libraries(run)
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
