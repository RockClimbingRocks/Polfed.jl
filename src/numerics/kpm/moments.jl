# Chebyshev moment estimation for KPM.
#
# The only shape rule is: `template` has the same shape that `f!(out, in)` expects.
# A vector template supports `H*x`; a `dim x block` matrix supports `H*X`; a
# `block x dim` matrix supports `X*H`. The recurrence itself is identical for all
# three cases.
#
# Probes are applied a block at a time but never mixed: each is drawn on its own
# (`_draw_probe_block!`), its moments are kept in their own column, and the columns
# are summed in probe order at the end. Block width and task count therefore decide
# how fast the moments come out, not which probes are used or how they are summed.

"""
    ChebyshevWorkspace(template)

Three buffers shaped like `template`, reused by [`probe_moments!`](@ref).
"""
struct ChebyshevWorkspace{A}
    prev::A
    curr::A
    next::A
end

ChebyshevWorkspace(template) =
    ChebyshevWorkspace(similar(template), similar(template), similar(template))

# Dot products accumulate in at least Float64: the row kernel sums each probe sequentially.
_dot_acc_type(::Type{E}) where {E<:Number} = promote_type(real(E), Float64)

"""
    probe_moments!(μ, f!, probe, work, info = probe_info(probe)) -> μ

Chebyshev moments of each probe in the block `probe`, one column of `μ` per probe slot:
`μ[k+1, p] = ⟨v_p|T_k(H)|v_p⟩` for `k = 0, …, size(μ, 1) - 1`.

Moment doubling (`μ₂ₙ = 2⟨Tₙv,Tₙv⟩ − μ₀`, `μ₂ₙ₋₁ = 2⟨Tₙv,Tₙ₋₁v⟩ − μ₁`) gives all `N` moments
from `cld(N - 1, 2)` applications of `f!`. `probe` may be `work.prev` itself (no copy).

`f!(out, in)` must write the mapped block into `out` and preserve the shape of `in`.
Correctness assumes the mapped operator is Hermitian and already scaled to `[-1, 1]`.
"""
function probe_moments!(
    μ::AbstractMatrix{<:Real},
    f!,
    probe::AbstractVecOrMat{E},
    work::ChebyshevWorkspace,
    info::ProbeInfo = probe_info(probe),
) where {E<:Number}
    N, nb = size(μ)
    nb == info.block_size || throw(DimensionMismatch(
        "μ has $nb columns but the probe block holds $(info.block_size) probes."))
    N == 0 && return μ

    A = _dot_acc_type(E)
    m0, m1, aa, ab = zeros(A, nb), zeros(A, nb), zeros(A, nb), zeros(A, nb)

    prev = work.prev
    curr = work.curr
    next = work.next

    probe === prev || copyto!(prev, probe)
    _pair_dots!(m0, ab, prev, prev, info)          # μ₀ = ⟨v,v⟩
    μ[1, :] .= m0
    N == 1 && return μ

    f!(curr, prev)
    _pair_dots!(aa, m1, curr, prev, info)          # ⟨T₁v,T₁v⟩ and μ₁ = ⟨T₁v,v⟩
    μ[2, :] .= m1

    max_step = cld(N - 1, 2)
    for n in 1:max_step   # no @inbounds: it would reach into an inlined `f!`
        # Here aa = ⟨Tₙv,Tₙv⟩ and ab = ⟨Tₙv,Tₙ₋₁v⟩, per probe.
        if 2n <= N - 1
            for p in 1:nb
                μ[2n + 1, p] = 2 * aa[p] - m0[p]
            end
        end
        if n >= 2 && 2n - 1 <= N - 1
            for p in 1:nb
                μ[2n, p] = 2 * ab[p] - m1[p]
            end
        end

        if n < max_step
            f!(next, curr)
            _combine_pair_dots!(aa, ab, next, curr, prev, info)   # next ← 2next − prev, then dots
            prev, curr, next = curr, next, prev
        end
    end

    return μ
end

"""
    trace_doubling!(f!, probe, moments, work) -> moments

Add the Chebyshev moments of every probe in `probe` to `moments` (summed over the block, in
probe order). A convenience over [`probe_moments!`](@ref), which keeps the probes apart.
"""
function trace_doubling!(
    f!,
    probe::AbstractVecOrMat{<:Number},
    moments::AbstractVector{<:Real},
    work::ChebyshevWorkspace,
)
    info = probe_info(probe)
    μ = zeros(eltype(moments), length(moments), info.block_size)
    probe_moments!(μ, f!, probe, work, info)
    for p in axes(μ, 2), k in axes(μ, 1)
        moments[k] += μ[k, p]
    end
    return moments
end

# --- per-probe dot products -------------------------------------------------------------
#
# `_pair_dots!(aa, ab, x, y)`: aa[p] = ⟨xₚ,xₚ⟩, ab[p] = Re⟨xₚ,yₚ⟩.
# `_combine_pair_dots!(aa, ab, next, curr, prev)`: next ← 2·next − prev, then the pair dots of
# (next, curr) — the two inner products the NEXT recurrence step needs, read in the same pass.
#
# CPU arrays (`VecOrMat`) get fused loops: one pass over the block, no BLAS call, and a
# summation order fixed by the code alone (so the result does not depend on any library's
# thread count).
# Anything else (GPU arrays, views) uses broadcasts and `dot` on each probe slot.

# A vector probe runs through the column kernel as `dim × 1`, so it vectorizes (and rounds) like
# any other column.
@inline _as_block(x::Vector) = reshape(x, length(x), 1)
@inline _as_block(x::Matrix) = x

@inline _widen(::Type{A}, x::Real) where {A} = A(x)
@inline _widen(::Type{A}, x::Complex) where {A} = Complex{A}(x)
@inline _re_dot(a::Real, b::Real) = a * b
@inline _re_dot(a::Complex, b::Complex) = real(a) * real(b) + imag(a) * imag(b)

function _pair_dots!(aa, ab, x::AbstractVecOrMat, y::AbstractVecOrMat, info::ProbeInfo)
    for p in 1:info.block_size
        xs, ys = _probe_slot(x, info, p), _probe_slot(y, info, p)
        aa[p] = real(dot(xs, xs))
        ab[p] = real(dot(xs, ys))
    end
    return nothing
end

function _combine_pair_dots!(aa, ab, next::AbstractVecOrMat, curr::AbstractVecOrMat,
                             prev::AbstractVecOrMat, info::ProbeInfo)
    @. next = 2 * next - prev
    _pair_dots!(aa, ab, next, curr, info)
    return nothing
end

_pair_dots!(aa::Vector{A}, ab::Vector{A}, x::VecOrMat{E}, y::VecOrMat{E},
            info::ProbeInfo) where {A,E<:Number} =
    _pair_dots_kernel!(aa, ab, _as_block(x), _as_block(y), info)

function _pair_dots_kernel!(aa::Vector{A}, ab::Vector{A}, x::Matrix{E}, y::Matrix{E},
                            info::ProbeInfo) where {A,E<:Number}
    if info.layout === :rows
        fill!(aa, zero(A)); fill!(ab, zero(A))
        @inbounds for j in 1:info.dim
            @simd ivdep for p in 1:info.block_size
                xv = _widen(A, x[p, j])
                aa[p] += abs2(xv)
                ab[p] += _re_dot(xv, _widen(A, y[p, j]))
            end
        end
    else
        @inbounds for p in 1:info.block_size
            saa, sab = zero(A), zero(A)
            @simd for j in 1:info.dim
                xv = _widen(A, x[j, p])
                saa += abs2(xv)
                sab += _re_dot(xv, _widen(A, y[j, p]))
            end
            aa[p], ab[p] = saa, sab
        end
    end
    return nothing
end

_combine_pair_dots!(aa::Vector{A}, ab::Vector{A}, next::VecOrMat{E}, curr::VecOrMat{E},
                    prev::VecOrMat{E}, info::ProbeInfo) where {A,E<:Number} =
    _combine_pair_dots_kernel!(aa, ab, _as_block(next), _as_block(curr), _as_block(prev), info)

function _combine_pair_dots_kernel!(aa::Vector{A}, ab::Vector{A}, next::Matrix{E},
                                    curr::Matrix{E}, prev::Matrix{E},
                                    info::ProbeInfo) where {A,E<:Number}
    if info.layout === :rows
        fill!(aa, zero(A)); fill!(ab, zero(A))
        @inbounds for j in 1:info.dim
            @simd ivdep for p in 1:info.block_size
                n = 2 * next[p, j] - prev[p, j]
                next[p, j] = n
                nv = _widen(A, n)
                aa[p] += abs2(nv)
                ab[p] += _re_dot(nv, _widen(A, curr[p, j]))
            end
        end
    else
        @inbounds for p in 1:info.block_size
            saa, sab = zero(A), zero(A)
            @simd for j in 1:info.dim
                n = 2 * next[j, p] - prev[j, p]
                next[j, p] = n
                nv = _widen(A, n)
                saa += abs2(nv)
                sab += _re_dot(nv, _widen(A, curr[j, p]))
            end
            aa[p], ab[p] = saa, sab
        end
    end
    return nothing
end

# --- probe scheduling -------------------------------------------------------------------

function _check_moment_args(N::Int, R::Int, ntasks::Int)
    N >= 1 || throw(ArgumentError("Number of moments N must be >= 1, got $N."))
    R >= 1 || throw(ArgumentError("Number of probes R must be >= 1, got $R."))
    ntasks >= 1 || throw(ArgumentError("Number of KPM tasks must be >= 1, got $ntasks."))
    return nothing
end

"""
    _ProbeSource

Hands out probe blocks in probe order. [`_next_block!`](@ref) draws a block's probes while
holding the lock, so block `b` is always drawn right after block `b - 1` — the random stream is
the same however many tasks pull from it, and in whatever order they arrive.
"""
mutable struct _ProbeSource{RNG<:AbstractRNG,PU<:ProcessingUnit}
    const rng::RNG
    const pu::PU
    const info::ProbeInfo
    const R::Int
    const lock::ReentrantLock
    next::Int
end

_ProbeSource(rng::AbstractRNG, pu::ProcessingUnit, info::ProbeInfo, R::Int) =
    _ProbeSource(rng, pu, info, R, ReentrantLock(), 1)

"""
    _next_block!(src, block, scratch) -> (first, active)

Draw the next block of probes into `block`: probes `first : first + active - 1`. Returns
`(0, 0)` once all `R` probes are handed out.
"""
function _next_block!(src::_ProbeSource, block, scratch)
    lock(src.lock) do
        first = src.next
        first > src.R && return (0, 0)
        active = min(src.info.block_size, src.R - first + 1)
        src.next = first + active
        _draw_probe_block!(block, scratch, src.info, active, src.pu, src.rng)
        return (first, active)
    end
end

"""
    _moments_worker!(per_probe, f!, template, src) -> nothing

Pull probe blocks from `src` until it is empty, writing each probe's moments into its own
column of `per_probe`. One worker per task; each owns its buffers, and the columns it writes
belong to its probes alone.
"""
function _moments_worker!(per_probe::Matrix, f!, template, src::_ProbeSource)
    info = src.info
    work = ChebyshevWorkspace(template)
    scratch = similar(template, info.dim)
    μ = Matrix{eltype(per_probe)}(undef, size(per_probe, 1), info.block_size)
    while true
        first, active = _next_block!(src, work.prev, scratch)
        first == 0 && return nothing
        probe_moments!(μ, f!, work.prev, work, info)
        copyto!(view(per_probe, :, first:first + active - 1), view(μ, :, 1:active))
    end
end

"""
    kpm_moments(f!, N, R, template, pu; rng = Random.default_rng(), ntasks = 1) -> Vector

Estimate the first `N` Chebyshev moments using `R` random probes.

`template` is only a shape and element-type example:

- `Vector{E}(undef, dim)` for `f!(y, x) = H*x`
- `Matrix{E}(undef, dim, block)` for `f!(Y, X) = H*X`
- `Matrix{E}(undef, block, dim)` for `f!(Y, X) = X*H`

For matrix templates, each column or row is one probe. `R` is the total number
of probes, so a block size of `2` with `R == 100` performs `50` matrix
applications per recurrence step. Keep `block < dim`: the larger axis is read as the
Hilbert-space dimension, and a square template as columns.

`rng` seeds the probes. The estimate is stochastic, so two calls with the default generator
give different moments — pass a seeded generator when the result has to be reproducible. On a
[`GPU`](@ref) the device generator is used and `rng` is ignored (see [`_random_normal!`](@ref)).

`ntasks` blocks are processed concurrently, each on its own task with its own buffers, so
`f!` must be safe to call concurrently on distinct arrays when `ntasks > 1`.

# Reproducibility
Probe `r` is the `r`-th vector drawn from `rng`, whatever the block width or `ntasks`; each
probe's moments are kept apart and summed in probe order `1…R`. So `ntasks` never changes the
result, and the block width changes it only through `f!`'s own rounding on a wider block.
"""
function kpm_moments(
    f!,
    N::Int,
    R::Int,
    template::AbstractVecOrMat{E},
    pu::ProcessingUnit;
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Int = 1,
) where {E<:Number}
    _check_moment_args(N, R, ntasks)
    info = probe_info(template)
    S = real(E)

    per_probe = Matrix{S}(undef, N, R)
    src = _ProbeSource(rng, pu, info, R)
    nt = min(ntasks, cld(R, info.block_size))
    if nt == 1
        _moments_worker!(per_probe, f!, template, src)
    else
        @sync for _ in 1:nt
            Threads.@spawn _moments_worker!(per_probe, f!, template, src)
        end
    end

    moments = zeros(S, N)
    @inbounds for r in 1:R, k in 1:N
        moments[k] += per_probe[k, r]
    end
    moments .*= info.dim / R
    return moments
end
