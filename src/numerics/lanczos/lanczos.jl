"""
    _prepare_seed(layout, x0) -> seed

Return a normalized Lanczos seed: a unit vector for vector/single-vector input, or an
orthonormalized block for a multi-vector block (columns via `qr` for [`ColumnLayout`](@ref),
rows via `lq` for [`RowLayout`](@ref)), on the same device as `x0`. Does not mutate the
caller's array.
"""
_prepare_seed(::Layout, x0::AbstractVector) = x0 ./ norm(x0)
function _prepare_seed(layout::Layout, x0::AbstractMatrix)
    blocksize(layout, x0) == 1 && return x0 ./ norm(x0)
    return first(orthonormalize(layout, x0))
end

"""
    lanczos(f!::Function, x0, target::EigTarget; kwargs...) -> LanczosResult
    lanczos(mat::AbstractMatrix, x0, target::EigTarget; kwargs...) -> LanczosResult
    lanczos(A_or_f!, x0, n::Int; which=:SR, kwargs...) -> LanczosResult

Block Lanczos factorization computing the eigenpairs selected by `target` for a
Hermitian/symmetric matrix or linear operator. CPU vs GPU is detected from `x0`.

# Arguments
- `f!::Function`: in-place action `f!(Y, X) = A·X` (operator interface).
- `mat::AbstractMatrix`: concrete matrix (an `f!` is generated internally).
- `x0::AbstractVecOrMat`: starting vector/block; its column count sets the block size.
- `target::EigTarget`: which eigenpairs to compute — [`LowestEigs`](@ref),
  [`HighestEigs`](@ref), [`BothEnds`](@ref), [`NearestTo`](@ref).
- `n::Int` (+ `which`): convenience form; `which=:SR` → `LowestEigs(n)`, `:LR` → `HighestEigs(n)`.

# Keyword Arguments
- `basistype::Type{<:OrthonormalBasis}=MatrixBasis`, `rot::ReOrthTechnique=FullRO()`.
- `maxdim::Union{Int,Nothing}=nothing`: Krylov budget; defaults to
  `DEFAULT_MAXDIM_FACTOR · count` for count targets (or the full dimension for
  value targets), capped at the operator dimension.
- `tol::Real=1e-14`: Ritz-residual tolerance; `eigentol::Real=1e-8`: eigen-residual tolerance.
- `mapvals=f!`: operator used to evaluate eigenvalues/residuals (POLFED passes the
  *unfiltered* `H` here while iterating on the filtered operator).
- `collect_garbage::Bool=true`: force `GC.gc(true)` on cleanup (set `false` for many small probes).
- `next_check::Function=DEFAULT_NEXT_CHECK`: scheduling callback
  `(state, criterion, k) -> Int` controlling when convergence checks run. Receives the
  full [`ConvergenceState`](@ref) and [`ConvergenceCriterion`](@ref) so any adaptive
  strategy is expressible. Defaults to [`default_next_check`](@ref).
- `warn_incomplete::Bool=true`: emit the WARN "Krylov budget exhausted before the target was
  certified complete" message when finalizing a run that hit `maxdim`. Set `false` when the
  caller deliberately over-targets and keeps only a subset (e.g. patched POLFED converges a
  padded window and discards the brackets), so hitting `maxdim` is expected. The
  numerical-breakdown (non-finite residual) warning is unaffected and always fires.
- `blas_threads=:auto`: how many threads dense BLAS may use for this run — reorthogonalization,
  the projected `eigen`, and the convergence check's eigenvector reconstruction are all dense
  BLAS and none of them is threaded by Julia. `:auto` un-pins BLAS from a single thread when
  Julia has several (see [`resolve_blas_threads`](@ref)); an `Int` sets it exactly; `nothing`
  leaves the library alone. Restored after the run either way.

# Returns
- A [`LanczosResult`](@ref) (`vals`, `vecs`, per-eigenvalue `residuals`, `converged`,
  `report`); also iterates as `(vals, vecs, report)` for backward compatibility.
"""
function lanczos(
    f!::Function, x0::AbstractVecOrMat, target::EigTarget;
    basistype::Type{<:OrthonormalBasis} = MatrixBasis,
    rot::ReOrthTechnique                = DEFAULT_ROT,
    maxdim::Union{Int,Nothing}          = nothing,
    tol::Real                           = DEFAULT_TOL,
    eigentol::Real                      = DEFAULT_EIGENTOL,
    mapvals                             = f!,
    layout::Layout                      = ColumnLayout(),
    collect_garbage::Bool               = false,
    next_check::Function                = DEFAULT_NEXT_CHECK,
    warn_incomplete::Bool               = true,
    blas_threads                        = DEFAULT_BLAS_THREADS,
)
    if layout isa RowLayout && basistype <: VectorBasis
        throw(ArgumentError(
            "RowLayout with VectorBasis is not supported; use MatrixBasis for row-oriented blocks.",
        ))
    end

    pu = is_gpu_array(x0) ? GPU() : CPU()

    # A plain vector is orientation-free — always column semantics for the vector path.
    layout = x0 isa AbstractVector ? ColumnLayout() : layout

    x0 = _prepare_seed(layout, x0)   # normalize a vector seed; orthonormalize a block seed

    hilbertspacedim = opdim(layout, x0)
    bs = blocksize(layout, x0)
    rc = requested_count(target)

    # A count-based target can never ask for more eigenpairs than the operator has.
    # For BothEnds this is the *sum* of both ends, so it also forbids the two windows
    # from overlapping. Value-based targets (rc < 0) are dynamic and exempt.
    if rc > 0 && rc > hilbertspacedim
        throw(ArgumentError(
            "Requested $rc eigenpairs but the operator dimension is only $hilbertspacedim " *
            "(for BothEnds the request is the sum of both ends)."))
    end

    # default budget: DEFAULT_MAXDIM_FACTOR·count for count targets; full dimension
    # for value targets (count unknown). Always capped at the operator dimension.
    requested = !isnothing(maxdim) ? maxdim :
                rc > 0             ? DEFAULT_MAXDIM_FACTOR * rc :
                                     hilbertspacedim
    maxdim_capped = min(requested, hilbertspacedim)
    maxiter = max(1, maxdim_capped ÷ bs)
    maxdim_final = maxiter * bs

    iterator = LanczosIterator(f!, x0, rot, layout)
    criterion = ConvergenceCriterion(target, Float64(tol), Float64(eigentol), maxiter, maxdim_final, bs, mapvals, next_check, warn_incomplete)
    state = ConvergenceState(criterion)

    # The dense-BLAS scope covers the whole factorization: reorthogonalization runs every
    # iteration and the convergence check's reconstruction is the single largest GEMM in the
    # run, so the count is resolved once here rather than per call site.
    return with_blas_threads(resolve_blas_threads(blas_threads)) do
        lanczos_method(iterator, criterion, state, basistype, pu; collect_garbage=collect_garbage)
    end
end

"""Matrix-interface overload of [`lanczos`](@ref) (`mul!`-based action)."""
function lanczos(mat::AbstractMatrix, x0::AbstractVecOrMat, target::EigTarget; mapvals=nothing, kwargs...)
    f!(Y::AbstractVecOrMat, X::AbstractVecOrMat) = mul!(Y, mat, X)
    mapvals_ = isnothing(mapvals) ? f! : mapvals
    return lanczos(f!, x0, target; mapvals=mapvals_, kwargs...)
end

"""Map the `(n, which)` convenience form onto an [`EigTarget`](@ref)."""
function _resolve_target(n::Int, which::Symbol)
    which === :SR && return LowestEigs(n)
    which === :LR && return HighestEigs(n)
    throw(ArgumentError("which=:$which is not supported in the Int convenience form; pass an EigTarget (LowestEigs/HighestEigs/BothEnds/NearestTo) directly."))
end

"""Integer convenience overload of [`lanczos`](@ref): `n` eigenpairs at the `which` end."""
function lanczos(A_or_f!, x0::AbstractVecOrMat, n::Int; which::Symbol=:SR, kwargs...)
    return lanczos(A_or_f!, x0, _resolve_target(n, which); kwargs...)
end

# --- extremal eigenvalue probes ---------------------------------------------

function _normalize_extrema_seed(x0::AbstractVector)
    seed = convert.(float(eltype(x0)), x0)
    nrm = norm(seed)
    iszero(nrm) && throw(ArgumentError("lanczos_extrema requires a nonzero starting vector."))
    seed ./= nrm
    return seed
end

@inline _is_lanczos_gpu_matrix(mat::AbstractMatrix) = is_gpu_array(mat) || is_gpu_sparse_matrix(mat)

function _default_extrema_seed(mat::AbstractMatrix)
    T = float(eltype(mat))
    seed = _is_lanczos_gpu_matrix(mat) ? gpu_randn(T, size(mat, 1)) : randn(T, size(mat, 1))
    return _normalize_extrema_seed(seed)
end

"""
    _power_iterate!(f!, v, k) -> v

Apply `k` normalized power-iteration steps in place: `v ← f(v)/‖f(v)‖`. Amplifies the
dominant-magnitude eigencomponent of the seed. Stops early if a step hits the null
space (`f(v) = 0`). `k ≤ 0` is a no-op.
"""
function _power_iterate!(f!::Function, v::AbstractVector, k::Int)
    k <= 0 && return v
    tmp = similar(v)
    for _ in 1:k
        f!(tmp, v)
        nrm = norm(tmp)
        iszero(nrm) && break           # seed hit the operator's null space; stop early
        v .= tmp ./ nrm
    end
    return v
end

"""Cap on the extrema probe's Krylov budget; the actual default is `min(cap, dim)`."""
const EXTREMA_MAXDIM_CAP = 5_000

"""
    _extrema_maxdim(dim::Integer) -> Int

The extrema probe's default Krylov budget: `min(`[`EXTREMA_MAXDIM_CAP`](@ref)`, dim)`.

The probe stores its basis in a [`VectorBasis`](@ref), which grows on demand, so this is a
generous *ceiling* rather than a reservation — a probe that converges in 108 iterations
allocates 108 vectors regardless of the cap. Sizing it this way removes the failure mode a
small fixed budget had: the number of iterations the extremes need grows with `dim`, so a
fixed `100` silently stopped certifying both ends past a few thousand dimensions.
"""
_extrema_maxdim(dim::Integer) = min(EXTREMA_MAXDIM_CAP, Int(dim))

"""
    lanczos_extrema(mat::AbstractMatrix; x0=nothing, maxdim=min(5000, dim), tol=1e-14, eigentol=1e-8, power_iters=3, kwargs...)
    lanczos_extrema(f!::Function, x0::AbstractVector; maxdim=min(5000, dim), tol=1e-14, eigentol=1e-8, power_iters=3, kwargs...)

Estimate the smallest and largest eigenvalues of a Hermitian/symmetric matrix or
operator. Returns `(Emin, Emax)`.

Both extremes are found in a **single** Krylov factorization via `BothEnds(1, 1)` —
about half the matvecs of two separate probes. Before the probe, the seed is refined
by `power_iters` steps of normalized power iteration (`v ← f(v)/‖f(v)‖`), which
amplifies the dominant-magnitude edge and makes the estimate robust against an unlucky
low-overlap random seed. Keep `power_iters` small (default `3`) so the *other* edge
keeps enough overlap to be found too.

A provided `x0` is copied and normalized; for matrix input without `x0`, a random
normalized seed is used.

# Keyword Arguments
- `maxdim::Int=`[`_extrema_maxdim`](@ref)`(dim)`: Krylov budget *cap*, defaulting to
  `min(`[`EXTREMA_MAXDIM_CAP`](@ref)`, dim)`. Paired with the default `basistype` below the cap
  is not a reservation, so it is deliberately generous: the probe pays only for the iterations
  it actually takes.
- `basistype::Type{<:OrthonormalBasis}=VectorBasis`: basis storage. [`VectorBasis`](@ref) grows
  on demand, which is what makes a large `maxdim` free; [`MatrixBasis`](@ref) would preallocate
  `dim × maxdim` and reserve gigabytes for a two-number probe. Column-oriented, so do not pass
  `layout = RowLayout()` through `kwargs` with this backend (see [`lanczos`](@ref)).
- `power_iters::Int=3`: normalized power-iteration steps applied to the seed before the
  probe (`0` disables).
- `return_result::Bool=false`: return the full [`LanczosResult`](@ref) (convergence info,
  vectors, residuals) instead of `(Emin, Emax)` — useful for checking the probe converged.
- `tol`, `eigentol`: forwarded to [`lanczos`](@ref).

!!! note
    Uses `BothEnds(1, 1)` unconditionally, so it requires `dim ≥ 2` (a 1×1 operator
    would request two eigenpairs from a one-dimensional space).
"""
function lanczos_extrema(
    mat::AbstractMatrix;
    x0::Union{Nothing,AbstractVector}=nothing,
    maxdim::Int=_extrema_maxdim(size(mat, 1)),
    tol::Real=1e-14,
    eigentol::Real=1e-8,
    power_iters::Int=3,
    basistype::Type{<:OrthonormalBasis}=VectorBasis,
    kwargs...
)
    size(mat, 1) == size(mat, 2) || throw(DimensionMismatch("lanczos_extrema requires a square matrix."))
    seed = isnothing(x0) ? _default_extrema_seed(mat) : _normalize_extrema_seed(x0)
    length(seed) == size(mat, 1) || throw(DimensionMismatch("x0 length must match the matrix dimension."))
    f!(Y::AbstractVecOrMat, X::AbstractVecOrMat) = mul!(Y, mat, X)
    return lanczos_extrema(f!, seed; maxdim=maxdim, tol=tol, eigentol=eigentol,
                           power_iters=power_iters, basistype=basistype, kwargs...)
end

function lanczos_extrema(
    f!::Function,
    x0::AbstractVector;
    maxdim::Int=_extrema_maxdim(length(x0)),
    tol::Real=1e-14,
    eigentol::Real=1e-8,
    power_iters::Int=3,
    return_result::Bool=false,
    basistype::Type{<:OrthonormalBasis}=VectorBasis,
    kwargs...
)
    seed = _normalize_extrema_seed(x0)
    _power_iterate!(f!, seed, power_iters)
    res = lanczos(f!, seed, BothEnds(1, 1); maxdim=maxdim, tol=tol, eigentol=eigentol,
                  basistype=basistype, collect_garbage=false, kwargs...)
    return_result && return res
    vals = collect(res.vals)
    if length(vals) < 2
        # `BothEnds(1, 1)` returns fewer than two only if an end never certified. With the
        # default budget that means either a genuinely 1-dimensional operator or a spectrum the
        # probe could not resolve — NOT the "dimension < 2" this used to claim unconditionally,
        # which sent a 3432-dimensional failure looking in entirely the wrong place.
        error("lanczos_extrema converged $(length(vals)) of the 2 spectral extremes in " *
              "$(res.report.itersneeded) iterations (maxdim = $maxdim, dim = $(length(seed))). " *
              "If dim ≥ 2, raise `maxdim`, loosen `eigentol`, or pass `Emin`/`Emax` directly.")
    end
    return first(vals), last(vals)
end
