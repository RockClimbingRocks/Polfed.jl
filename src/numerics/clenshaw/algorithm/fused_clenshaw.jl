# Fused Clenshaw evaluators: the same recurrence as `clenshaw_algorithm!`, restructured so a step
# is not "operator call, then a separate threaded combine". The generic path alternates two thread
# pools every step (the operator's own threads, then a Julia `@threads` region for the combine);
# waking the second pool on every one of the `order` steps can cost more than the step itself.
#
#   AxpbyClenshaw — the operator can accumulate (`Y ← α·A·X + β·Y`), so a step is that call plus
#                   one `Y += c·X`, both issued by the operator's own threading (e.g. MKL).
#   RangeClenshaw — the operator has a serial range form; ONE threaded region runs the whole
#                   recurrence, with a spin barrier between each chunk's product and its combine.

"""
    AxpbyClenshaw(family, coefficients, order, axpby!, add!)

Clenshaw evaluation of `S(A)·X` for an operator that can accumulate:

- `axpby!(Y, X, α, β)` sets `Y ← α·A·X + β·Y` (`β = 0` must not read `Y`),
- `add!(Y, c, X)` sets `Y ← Y + c·X`.

Each step overwrites the oldest recurrence buffer in place, so there is no separate combine pass
and every call goes through the operator's own threading. Callable as `e(Y, X, b)` like
[`ClenshawExpansion`](@ref); uses `b[2]`, `b[3]`. Agrees with it to rounding.
"""
struct AxpbyClenshaw{Family,C,A<:Function,D<:Function}
    coefficients::C
    order::Int
    axpby!::A
    add!::D
end

function AxpbyClenshaw(family::Symbol, coefficients::Union{Function,AbstractVector{<:Number}},
                       order::Int, axpby!::Function, add!::Function)
    is_supported_family(family) || throw(ArgumentError(
        "Unknown polynomial type :$family. Supported families: $(SUPPORTED_FAMILIES)"))
    coeff = _precompute_coefficients(coefficients, order)
    return AxpbyClenshaw{family,typeof(coeff),typeof(axpby!),typeof(add!)}(coeff, order, axpby!, add!)
end

function (e::AxpbyClenshaw{Fam})(Y::AbstractVecOrMat{<:Number}, X::AbstractVecOrMat{<:Number},
                                 b::AbstractVector{<:AbstractVecOrMat{<:Number}}) where {Fam}
    α, β, ϕ₀, ϕ₁ = recurrence_weights(Val(Fam))
    b2, b3 = b[2], b[3]                        # b_{k+1}, b_{k+2}
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))
    for k in e.order:-1:1
        e.axpby!(b3, b2, α(k), β(k + 1))       # b_k = α·A·b_{k+1} + β·b_{k+2} …
        e.add!(b3, e.coefficients(k), X)       # … + c_k·X, written over b_{k+2}
        b2, b3 = b3, b2
    end
    e.axpby!(Y, b2, ϕ₁, 0)                     # Y = c₀ϕ₀·X + ϕ₁·A·b₁ + β₁ϕ₀·b₂
    e.add!(Y, β(1) * ϕ₀, b3)
    e.add!(Y, e.coefficients(0) * ϕ₀, X)
    return Y
end

"""
    RangeClenshaw(family, coefficients, order, range!, units; alignment = 1,
                  ntasks = Threads.nthreads())

Clenshaw evaluation of `S(A)·X` over an operator given in SERIAL range form:
`range!(Y, X, lo, hi)` writes the part of `A·X` owned by work units `lo:hi`, distinct ranges
writing disjoint parts of `Y` and `1:units` writing all of it.

The whole recurrence runs in ONE `@threads :static` region: each task computes its unit chunk,
meets the others at a spin barrier, combines its element chunk, and meets them again — two
barriers per step instead of two region launches. Chunk boundaries are multiples of `alignment`.
Callable as `e(Y, X, b)` like [`ClenshawExpansion`](@ref), with the same arithmetic per element,
so the result is bit-identical to it (and to any `ntasks`) for the same operator.

!!! warning
    Opens a `:static` region, so it cannot run inside another `@threads :static` region.
"""
struct RangeClenshaw{Family,C,R<:Function}
    coefficients::C
    order::Int
    range!::R
    units::Int
    alignment::Int
    ntasks::Int
end

function RangeClenshaw(family::Symbol, coefficients::Union{Function,AbstractVector{<:Number}},
                       order::Int, range!::Function, units::Int; alignment::Int = 1,
                       ntasks::Int = Threads.nthreads(:default))
    is_supported_family(family) || throw(ArgumentError(
        "Unknown polynomial type :$family. Supported families: $(SUPPORTED_FAMILIES)"))
    units >= 1 && alignment >= 1 && ntasks >= 1 || throw(ArgumentError(
        "RangeClenshaw: units, alignment and ntasks must be >= 1, got $units, $alignment, $ntasks."))
    coeff = _precompute_coefficients(coefficients, order)
    return RangeClenshaw{family,typeof(coeff),typeof(range!)}(coeff, order, range!, units,
                                                               alignment, ntasks)
end

# `lo:hi` of chunk `t` of `1:n` split into `nt` chunks whose boundaries are multiples of `align`
# (the last may be short, trailing ones empty).
function _chunk_range(n::Int, align::Int, nt::Int, t::Int)
    per = cld(cld(n, align), nt) * align
    return (t - 1) * per + 1, min(t * per, n)
end

# Tasks for one application: never more than the threads (every task must be running for the
# barrier to open) and not so many that a chunk has fewer than ~64 units.
_range_tasks(e::RangeClenshaw) =
    max(1, min(e.ntasks, Threads.nthreads(:default), cld(e.units, max(64, e.alignment))))

function (e::RangeClenshaw)(Y::AbstractVecOrMat{<:Number}, X::AbstractVecOrMat{<:Number},
                            b::AbstractVector{<:AbstractVecOrMat{<:Number}})
    length(b) == 3 || throw(ArgumentError("workspace `b` must have length 3, got $(length(b))"))
    size(X) == size(Y) == size(b[1]) == size(b[2]) == size(b[3]) ||
        throw(DimensionMismatch("RangeClenshaw: Y, X and the workspace must share one shape."))
    nt = _range_tasks(e)
    n = length(Y)
    if nt == 1
        _range_clenshaw!(e, Y, X, b[1], b[2], b[3], 1, e.units, 1, n, nothing)
        return Y
    end
    barrier = _SpinBarrier(nt)
    Threads.@threads :static for t in 1:nt
        ulo, uhi = _chunk_range(e.units, e.alignment, nt, t)
        elo, ehi = _chunk_range(n, 1, nt, t)
        try
            _range_clenshaw!(e, Y, X, b[1], b[2], b[3], ulo, uhi, elo, ehi, barrier)
        catch
            _abort!(barrier)   # release the tasks waiting at a barrier this one will never reach
            rethrow()
        end
    end
    return Y
end

"""
    _range_clenshaw!(e, Y, X, b1, b2, b3, ulo, uhi, elo, ehi, barrier) -> nothing

One task's share of [`RangeClenshaw`](@ref): products over units `ulo:uhi`, combines over the
linear elements `elo:ehi`. The buffer rotation is local and identical in every task.
"""
function _range_clenshaw!(e::RangeClenshaw{Fam}, Y, X, b1, b2, b3, ulo::Int, uhi::Int,
                          elo::Int, ehi::Int, barrier) where {Fam}
    α, β, ϕ₀, ϕ₁ = recurrence_weights(Val(Fam))
    z = zero(eltype(b1))
    @inbounds for i in elo:ehi
        b1[i] = z; b2[i] = z; b3[i] = z
    end
    _sync!(barrier)
    for k in e.order:-1:1
        ulo <= uhi && e.range!(b1, b2, ulo, uhi)           # b1 ← A·b2, my units
        _sync!(barrier)
        c, a, bb = e.coefficients(k), α(k), β(k + 1)
        @inbounds @simd for i in elo:ehi                    # b1 ← c·X + α·b1 + β·b3, my elements
            b1[i] = c * X[i] + a * b1[i] + bb * b3[i]
        end
        _sync!(barrier)
        b1, b2, b3 = b3, b1, b2
    end
    ulo <= uhi && e.range!(Y, b2, ulo, uhi)                 # Y = c₀ϕ₀·X + ϕ₁·A·b₁ + β₁ϕ₀·b₂
    _sync!(barrier)
    c0, β1 = e.coefficients(0), β(1)
    @inbounds @simd for i in elo:ehi
        Y[i] = c0 * ϕ₀ * X[i] + ϕ₁ * Y[i] + β1 * ϕ₀ * b3[i]
    end
    return nothing
end

"""
    _SpinBarrier(n)

A reusable barrier for `n` tasks that are all running at once (one per thread, as in a
`:static` region). Waiting spins instead of sleeping, so passing it costs microseconds rather
than a thread wake-up. It reaches a GC safepoint while spinning, and [`_abort!`](@ref) releases
the waiters with a [`_BarrierAborted`](@ref) when a task fails.
"""
mutable struct _SpinBarrier
    const n::Int
    @atomic arrived::Int
    @atomic generation::Int
    @atomic aborted::Bool
end

_SpinBarrier(n::Int) = _SpinBarrier(n, 0, 0, false)

"""Thrown in the tasks waiting at a [`_SpinBarrier`](@ref) when another task failed."""
struct _BarrierAborted <: Exception end
Base.showerror(io::IO, ::_BarrierAborted) =
    print(io, "_BarrierAborted: another task of the threaded region failed (see its error)")

_sync!(::Nothing) = nothing

function _sync!(b::_SpinBarrier)
    gen = @atomic b.generation
    if (@atomic b.arrived += 1) == b.n
        @atomic b.arrived = 0            # reset BEFORE releasing, so the next round starts at 0
        @atomic b.generation = gen + 1
    else
        spins = 0
        while (@atomic b.generation) == gen
            (@atomic b.aborted) && throw(_BarrierAborted())
            spins += 1
            # Past a short spin, give the core away: when the OS puts two of these tasks on one
            # core, the one still working only runs if the waiting one yields.
            spins < _SPIN_LIMIT ? ccall(:jl_cpu_pause, Cvoid, ()) : _yield_core()
            GC.safepoint()
        end
    end
    return nothing
end

"""Pause iterations a [`_SpinBarrier`](@ref) waiter spins before yielding its core (~tens of µs)."""
const _SPIN_LIMIT = 2000

_yield_core() = Sys.iswindows() ? (ccall(:SwitchToThread, stdcall, Cint, ()); nothing) :
                                  (ccall(:sched_yield, Cint, ()); nothing)

_abort!(b::_SpinBarrier) = (@atomic b.aborted = true; nothing)
