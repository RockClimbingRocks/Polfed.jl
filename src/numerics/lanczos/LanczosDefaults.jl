# Default Lanczos settings, referenced by the `lanczos` entry points. Centralized
# here so the public defaults live in one place.

"""Default reorthogonalization strategy ([`FullRO`](@ref))."""
const DEFAULT_ROT = FullRO()

"""Default orthogonality-loss tolerance for the Ritz-residual convergence test."""
const DEFAULT_TOL = 1e-14

"""Default eigen-residual tolerance (`‖A·v − λ·v‖`) for accepting an eigenpair."""
const DEFAULT_EIGENTOL = 1e-10

"""
Default Krylov-dimension budget as a multiple of the number of requested
eigenpairs: `maxdim ≈ DEFAULT_MAXDIM_FACTOR · ntargets`, then **capped at the
operator dimension**. Set to the larger of the old code's two inconsistent values
so the default errs toward "run long enough to converge".

Memory cost is `dim × maxdim` for the basis, so the cap means small/medium
problems effectively run to convergence cheaply, while for large problems you
should pass `maxdim` explicitly to bound memory (POLFED always does).
"""
const DEFAULT_MAXDIM_FACTOR = 500

"""
    default_next_check(state::ConvergenceState, criterion::ConvergenceCriterion, k::Int) -> Int

Default convergence-check scheduling function. Returns the next iteration at which a
full projected-matrix eigendecomposition + residual scan should run, adapting the
interval to the current Ritz residual `state.residual`:

| current residual `ε` | next check after |
|----------------------|------------------|
| `ε > 1e-2`           | `k + 20`         |
| `1e-3 < ε ≤ 1e-2`    | `k + 10`         |
| `ε ≤ 1e-3`           | `k + 5`          |

Checks are cheap to schedule rarely when far from convergence and more often as the
residual shrinks. Pass a custom function as `next_check` to [`ConvergenceCriterion`](@ref)
to override this policy — the callback receives `state` (current counters and history)
and `criterion` (budget: `maxdim`, `maxiter`, `blocksize`) so any adaptive strategy
is expressible.
"""
function default_next_check(state::ConvergenceState, ::ConvergenceCriterion, k::Int)
    ε = state.residual
    if     ε > 1e-2;  return k + 20
    elseif ε > 1e-3;  return k + 10
    else              return k + 5
    end
end

"""Default convergence-check scheduling callback ([`default_next_check`](@ref))."""
const DEFAULT_NEXT_CHECK = default_next_check

# --- dense BLAS threads ------------------------------------------------------------------
#
# Lanczos leans on dense BLAS in three places — reorthogonalization (every iteration), the
# projected-matrix `eigen`, and the eigenvector reconstruction inside a convergence check —
# and none of them is threaded by Julia. They inherit whatever thread count the BLAS library
# started with, which on an HPC login shell is routinely 1: a site profile exports
# `OMP_NUM_THREADS=1` as a "don't accidentally go parallel" default, and OpenBLAS/MKL read it
# when no library-specific variable overrides it. The sparse matvec is unaffected (POLFED
# sets MKL's thread count per call, and the value-grouped kernels use Julia threads), so the
# symptom is narrow and easy to misread: the convergence checks and reorthogonalization run
# ~8x slower than the rest of the same process.

"""
Default `blas_threads` request for [`lanczos`](@ref) — `:auto`, resolved by
[`resolve_blas_threads`](@ref).
"""
const DEFAULT_BLAS_THREADS = :auto

"""One-shot latch so the `:auto` INFO line is emitted once per session, not once per run."""
const _blas_threads_notified = Ref(false)

"""
    resolve_blas_threads(request) -> Union{Int,Nothing}

How many threads dense BLAS should use for one Lanczos run. `nothing` means "leave the
library alone" — no set, no restore.

- `:auto` (the default): raise BLAS to `Threads.nthreads(:default)` **only** when BLAS is
  pinned to a single thread while Julia has several. That combination is the signature of an
  inherited `OMP_NUM_THREADS=1`, not of a deliberate choice — a caller who wants one BLAS
  thread on a one-core allocation also has one Julia thread, and a caller who set
  `OPENBLAS_NUM_THREADS`/`BLAS.set_num_threads` to anything but 1 has expressed an intent
  that is left untouched. Setting the environment variable is therefore also the opt-out.
- an `Int`: use exactly that many.
- `nothing`: leave the library alone.
"""
function resolve_blas_threads(request)
    request === nothing && return nothing
    request isa Integer && return Int(request)
    request === :auto || throw(ArgumentError(
        "blas_threads must be an Integer, `nothing`, or `:auto` (got $(repr(request)))."))

    n_julia = Threads.nthreads(:default)
    n_blas  = BLAS.get_num_threads()
    (n_blas == 1 && n_julia > 1) || return nothing

    # INFO, not WARN: nothing is wrong, and it fires once per session — a run at the default
    # verbosity should say nothing about it. The count is visible on demand for anyone
    # comparing timings, which is the only time it matters.
    if !_blas_threads_notified[] && should_log(POLFED_INFO_LEVEL)
        _blas_threads_notified[] = true
        polfed_log(POLFED_INFO_LEVEL, "Raising dense BLAS threads for the Lanczos runs.",
                   julia_threads=n_julia, blas_threads_before=n_blas, blas_threads=n_julia)
    end
    return n_julia
end

"""
    with_blas_threads(f, nthreads::Union{Int,Nothing})

Run `f()` with dense BLAS set to `nthreads`, restoring the previous count afterwards (also on
error). `nothing` runs `f()` untouched — no library call at all.

Scoped rather than global on purpose: the count is a property of a Lanczos run, and a package
that permanently rewrites a process-wide setting would leak into every other library in the
session. Nests correctly with the inner `BLAS.set_num_threads(1)` that
[`clenshaw_apply`](@ref) uses to keep its own threaded region from oversubscribing.
"""
function with_blas_threads(f::F, nthreads::Union{Int,Nothing}) where {F}
    nthreads === nothing && return f()
    previous = BLAS.get_num_threads()
    BLAS.set_num_threads(nthreads)
    try
        return f()
    finally
        BLAS.set_num_threads(previous)
    end
end
