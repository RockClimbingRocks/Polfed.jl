# Shared preparation for all matrix entrypoints. The operator-building primitives live with the
# mapping slice; this helper only orchestrates them for the public row-native paths.

"""
    @with_verbosity level expr

Evaluate `expr` with the POLFED [`verbosity`](@ref) temporarily raised to `level`, restoring the
previous level afterwards (also on error). `level === nothing` — the default of every entrypoint's
`verbosity` keyword — evaluates `expr` unchanged, leaving the global level alone.

This is what lets `polfed(H, x0, 50, :maxdos; verbosity = 3)` trace one run without the caller
having to set and reset `Polfed.Common.verbosity[]` around it.
"""
macro with_verbosity(level, expr)
    quote
        local lvl = $(esc(level))
        lvl === nothing ? $(esc(expr)) : with_verbosity(() -> $(esc(expr)), Int(lvl))
    end
end

"""
    _prepare_matrix_pipeline(H, x0, mc, dosc) -> (f!, x0_l, mc_l, dosc_l, layout)

Prepare a concrete matrix for any matrix POLFED entrypoint. The helper resolves missing spectral
bounds and the Krylov block `layout` ([`_matrix_layout`](@ref)), builds that layout's operators,
orients `x0` for it, and returns a mapping config carrying the rescaled operator. Count, window,
patched, full and double entrypoints then differ only in the pipeline they call.

`dosc_l` is `dosc` resolved by [`_owned_operator_dos`](@ref): the DoS operator here is POLFED's own
`mul!` on `H`, which takes any probe block and is safe to run concurrently (on the GPU it runs as
one task — the device is already parallel).

# Throws
- `ArgumentError`: `H` and `x0` are not on the same device.
"""
function _prepare_matrix_pipeline(H::AbstractMatrix, x0::AbstractVecOrMat, mc::MappingConfig,
                                  dosc::DoSConfig)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("Matrix POLFED entrypoints require a square matrix."))
    on_gpu = is_gpu_array(x0)
    on_gpu == (is_gpu_sparse_matrix(H) || is_gpu_array(H)) || throw(ArgumentError(
        "H and x0 must be on the same device; got $(on_gpu ? "a GPU x0 with a CPU H" :
        "a CPU x0 with a GPU H"). Move both, e.g. `CUDA.CUSPARSE.CuSparseMatrixCSR(H)` and " *
        "`CuArray(x0)`."))

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
        # `copy`, not `collect`: `collect` brings a device vector to the host.
        probe = () -> lanczos_extrema(H; x0 = copy(v0), blas_threads = nothing)
        E1, E2 = _serial_bounds_probe(probe, on_gpu)
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
    pu = on_gpu ? GPU() : CPU()
    layout = _matrix_layout(mc.layout, pu, bs)

    # The strategy is resolved BEFORE the operator is built, because it decides how the operator
    # must be built — single-vector or whole-block, self-threading or serial, and how many MKL
    # threads each apply may take. Building first and choosing after is what let a VectorWise run
    # reach a `Val(bs)` kernel it could not call.
    strategy = _row_path_strategy(
        resolve_parallelization(mc.parallel_strategy, pu, bs, mc.optimize_mapping))
    ops = _build_matrix_ops(H, mc.optimize_mapping, bs, a, b, strategy, layout)

    mc_l = MappingConfig(optimize_mapping   = mc.optimize_mapping,
                         f!_rescaled         = ops.f!_rescaled,
                         parallel_strategy   = strategy,   # already resolved; passed through
                         clenshaw_recurrence = mc.clenshaw_recurrence,
                         clenshaw_finalsum   = mc.clenshaw_finalsum,
                         Emin = Emin, Emax = Emax, layout = layout)

    return ops.f!, _orient_seed(layout, x0, is_row), mc_l, _owned_operator_dos(dosc; gpu = on_gpu),
           layout
end

"""
    _matrix_layout(request, pu, bs) -> Layout

The matrix entrypoint's Krylov block layout: an explicit `request` (`MappingConfig.layout`) as
given; otherwise [`RowLayout`](@ref) on the CPU (the value-grouped kernels are row kernels), and on
the GPU [`ColumnLayout`](@ref) below [`Defaults.polfed.gpu_row_layout_from`](@ref) — a vector `x0`
stays a vector there (scalar Lanczos); a `1 × D` or `D × 1` matrix runs block Lanczos at `bs = 1` —
and row from it on.
"""
_matrix_layout(request::Layout, ::ProcessingUnit, ::Int) = request
_matrix_layout(::Nothing, ::CPU, ::Int) = RowLayout()
_matrix_layout(::Nothing, ::GPU, bs::Int) =
    bs >= Defaults.polfed.gpu_row_layout_from ? RowLayout() : ColumnLayout()

"""
    _orient_seed(layout, x0, is_row) -> AbstractVecOrMat

`x0` in `layout`'s orientation: a `bs × D` block for [`RowLayout`](@ref) (a vector becomes `1 × D`),
a `D × bs` block or the plain vector for [`ColumnLayout`](@ref). Reorients by `permutedims` (a
copy, never a lazy transpose) and only when the orientation differs.
"""
_orient_seed(::RowLayout, x0::AbstractVecOrMat, is_row::Bool) = is_row ? x0 : permutedims(x0)
_orient_seed(::ColumnLayout, x0::AbstractVector, ::Bool) = x0
_orient_seed(::ColumnLayout, x0::AbstractMatrix, is_row::Bool) = is_row ? permutedims(x0) : x0

"""
    _row_path_strategy(strategy::Parallelization) -> Parallelization

Coerce [`VectorWise`](@ref) to [`BlockWise`](@ref) on the row-native matrix path, warning when
it does.

`VectorWise` applies the transform once per Krylov vector, over a `1 x D` slice. Two things stop
that working here, and they are independent — neither is a thread-count question alone:

1. **The operator is whole-block.** `f!_rescaled` is the transform, built once at `Val(bs)` and
   applied to the full `bs x D` Krylov block. A `Val(1)` kernel handed that block writes row 1
   and leaves the rest zero (93% error, measured), so it cannot simply be rebuilt per-vector; it
   would need to be a SECOND operator. (The DoS no longer shares it — it builds its own generic
   operator and probes one vector at a time, see `dos_operator`.)
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
               "is built whole-block at Val(bs), and the Clenshaw combine opens its own " *
               "`@threads :static` region. Using BlockWise instead.",
               threads = Threads.nthreads())
    return BlockWise()
end
