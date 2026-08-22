# Shared preparation for all matrix entrypoints. The operator-building primitives live with the
# mapping slice; this helper only orchestrates them for the public row-native paths.

"""
    @with_verbosity level expr

Evaluate `expr` with the POLFED [`verbosity`](@ref) temporarily raised to `level`, restoring the
previous level afterwards (also on error). `level === nothing` — the default of every entrypoint's
`verbosity` keyword — evaluates `expr` unchanged, leaving the global level alone.

This is what lets `polfed(H, x0, 50, :maxdos; verbosity = 3)` trace one run without the caller
having to set and reset `UniPolfed.Common.verbosity[]` around it.
"""
macro with_verbosity(level, expr)
    quote
        local lvl = $(esc(level))
        lvl === nothing ? $(esc(expr)) : with_verbosity(() -> $(esc(expr)), Int(lvl))
    end
end

"""
    _prepare_matrix_row_pipeline(H, x0, mc) -> (f!, x0_row, mc_row)

Prepare a concrete matrix for any row-native POLFED entrypoint. The helper resolves missing
spectral bounds, builds the selected row operator, reorients `x0`, and returns a mapping config
that carries the row-specific rescaled operator. Count, window, patched, and full entrypoints
then differ only in the pipeline they call.
"""
function _prepare_matrix_row_pipeline(H::AbstractMatrix, x0::AbstractVecOrMat, mc::MappingConfig)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("Matrix POLFED entrypoints require a square matrix."))

    # x0 orientation: an AbstractMatrix with block_size ≤ dim is already row form (block_size × dim);
    # a column block (dim × block_size) or a vector is not. Used for the probe seed and the reorient.
    is_row = x0 isa AbstractMatrix && size(x0, 1) <= size(x0, 2)

    # Spectral bounds: a single column probe on H (unless the config supplies both). The seed is one
    # length-dim vector — the first column of a column block, the first row of a row block.
    if isnothing(mc.Emin) || isnothing(mc.Emax)
        v0 = x0 isa AbstractVector ? x0 : is_row ? view(x0, 1, :) : view(x0, :, 1)
        polfed_log(POLFED_DEBUG_LEVEL, "Estimating spectral bounds with a Lanczos probe on H.",
                   estimating = isnothing(mc.Emin) ? (isnothing(mc.Emax) ? "Emin+Emax" : "Emin") : "Emax",
                   dim = size(H, 1))
        E1, E2 = lanczos_extrema(H; x0 = collect(v0))
        Emin = isnothing(mc.Emin) ? E1 : mc.Emin
        Emax = isnothing(mc.Emax) ? E2 : mc.Emax
    else
        Emin, Emax = mc.Emin, mc.Emax
    end
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    # Block size comes from the same orientation decision as `is_row`, expressed once so the two
    # cannot drift: under either layout `blocksize` is the smaller axis here by construction.
    bs = x0 isa AbstractVector ? 1 : blocksize(is_row ? RowLayout() : ColumnLayout(), x0)

    # The strategy is resolved BEFORE the operator is built, because it decides how the operator
    # must be built — single-vector or whole-block, self-threading or serial, and how many MKL
    # threads each apply may take. Building first and choosing after is what let a VectorWise run
    # reach a `Val(bs)` kernel it could not call.
    pu = is_gpu_array(x0) ? GPU() : CPU()
    strategy = _row_path_strategy(resolve_parallelization(mc.parallel_strategy, pu, bs))
    rowops = _build_matrix_rowops(H, mc.optimize_mapping, bs, a, b, strategy)

    # x0 → row orientation (block_size × dim); reorient only a column block or a vector (→ 1×D).
    x0_row = is_row ? x0 : permutedims(x0)

    mc_row = MappingConfig(optimize_mapping   = mc.optimize_mapping,
                           f!_rescaled         = rowops.f!_rescaled,
                           parallel_strategy   = strategy,   # already resolved; passed through
                           clenshaw_recurrence = mc.clenshaw_recurrence,
                           clenshaw_finalsum   = mc.clenshaw_finalsum,
                           Emin = Emin, Emax = Emax)

    return rowops.f!, x0_row, mc_row
end

"""
    _row_path_strategy(strategy::Parallelization) -> Parallelization

Coerce [`VectorWise`](@ref) to [`BlockWise`](@ref) on the row-native matrix path, warning when
it does.

`VectorWise` applies the transform once per Krylov vector, over a `1 x D` slice. Two things stop
that working here, and they are independent — neither is a thread-count question alone:

1. **The operator is whole-block.** `f!_rescaled` is built once at `Val(bs)` and shared with the
   DoS stage, which runs KPM over the full `bs x D` block. A `Val(1)` kernel handed that block
   writes row 1 and leaves the rest zero (93% error, measured), so it cannot simply be rebuilt
   per-vector; it would need to be a SECOND operator.
2. **The recurrence threads.** `_threaded_combine!` opens a `Threads.@threads :static` region
   whenever `nthreads() > 1`, and Julia rejects `:static` regions running concurrently — so above
   one thread a `VectorWise` run raises "`@threads :static` cannot be used concurrently or
   nested".

(2) is not specific to this path: it applies to `polfed(f!, …)` equally, and predates the
BlockWise/VectorWise rename — `MulColsParallel` had the same structure and was the CPU default.
It goes unseen because the test suite runs single-threaded, where the combine takes its serial
branch.

Coercing unconditionally rather than gating on `nthreads()`: (1) holds at every thread count, so
a gate would let the single-threaded case through into a `MethodError` instead.
"""
function _row_path_strategy(strategy::Parallelization)
    strategy isa VectorWise || return strategy
    polfed_log(POLFED_WARN_LEVEL,
               "VectorWise is not supported on the matrix entrypoints: the transform operator " *
               "is built whole-block and shared with the DoS, and the Clenshaw combine opens " *
               "its own `@threads :static` region. Using BlockWise instead.",
               threads = Threads.nthreads())
    return BlockWise()
end
