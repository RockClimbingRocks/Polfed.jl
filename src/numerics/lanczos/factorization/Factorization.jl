"""
    KrylovFactorization{S<:Real, T<:Number, P<:ProcessingUnit}

Abstract supertype for Lanczos factorization state (`LanczosFactorization`,
`BlockLanczosFactorization`).
"""
abstract type KrylovFactorization{S<:Real, T<:Number, P<:ProcessingUnit} end

"""
    _breakdown_tol(::Type{T}) -> Real

Threshold below which a Lanczos coupling norm (`β`) signals **breakdown** — an
invariant Krylov subspace has been reached and the iteration must stop instead of
dividing by a vanishing norm. Assumes a roughly `O(1)`-scaled operator (POLFED
rescales the spectrum into `[-1, 1]`).
"""
@inline _breakdown_tol(::Type{T}) where {T<:Number} = eps(real(T))^(3//4)

include("factorizationtype/Lanczos.jl")
include("factorizationtype/BlockLanczos.jl")

include("factorizationsteps/calculate_overlap.jl")
include("factorizationsteps/mapping.jl")
include("factorizationsteps/orthogonalization.jl")
include("factorizationsteps/reorthogonalization.jl")
include("diagnostics.jl")

"""
    update!(fac::KrylovFactorization, v::AbstractArray) -> nothing

Append the new vector/block `v` to the factorization basis, shift the
`v_last`/`v_secondlast` workspace, and advance `krylovdim`. Mutates `fac`.
"""
function update!(fac::KrylovFactorization, v::AbstractArray)
    add!(fac.basis, v)
    fac.v_secondlast .= fac.v_last
    fac.v_last .= v
    fac.krylovdim += blocksize(fac.layout, v)   # #vectors added: bs per layout (1 for a vector)
end

"""
    lanczositer(iterator::LanczosIterator, basis::OrthonormalBasis, maxdim::Int, pu::ProcessingUnit) -> KrylovFactorization

Initialize the Lanczos factorization state from the starting seed in `iterator`.
"""
function lanczositer(iterator::LanczosIterator, basis::OrthonormalBasis, maxdim::Int, pu::ProcessingUnit)
    x0 = iterator.x0

    factorizationtype = isa(x0, AbstractVector) ? LanczosFactorization : BlockLanczosFactorization
    factorization = factorizationtype(maxdim, basis, x0, pu, iterator.layout)

    update!(factorization, x0)
    iterator.f!(factorization.r, x0)

    calculate_overlap!(factorization)

    # orthogonalize the residual against the seed
    α = getoverlap(factorization)
    if x0 isa AbstractVector
        factorization.r -= x0 * α                          # scalar α
    else
        project!(iterator.layout, factorization.r, x0, α)  # block: r ← r − x0·α (col) / α·x0 (row)
    end

    return factorization
end

"""
    lanczositer!(iterator::LanczosIterator, factorization::KrylovFactorization, iteration::Integer) -> Bool

Run one Lanczos iteration stage sequence (tridiagonalize → map → overlap →
orthogonalize → reorthogonalize). Returns `false` if tridiagonalization detected
breakdown (invariant subspace), short-circuiting the remaining stages; otherwise `true`.

Each stage is annotated with [`@stage`](@ref), which accumulates into whichever [`@stages`](@ref)
scope the caller opened (see [`lanczos_algorithm`](@ref)) — so the timings need no `walltimes`/
`cputimes` parameters here, and are simply not collected when a caller opens no scope.
"""
function lanczositer!(
    iterator::LanczosIterator,
    factorization::KrylovFactorization,
    iteration::Integer,
)
    ok = @stage :tridiag tridiagonalization!(factorization)
    ok || return false
    log_factorization_step_diagnostics!(factorization, :tridiagonalization, iteration)

    @stage :matvec mapping!(factorization, iterator.f!)
    log_factorization_step_diagnostics!(factorization, :mapping, iteration)

    @stage :overlap calculate_overlap!(factorization)
    log_factorization_step_diagnostics!(factorization, :calculate_overlap, iteration)

    @stage :orthog orthogonalization!(factorization)
    log_factorization_step_diagnostics!(factorization, :orthogonalization, iteration)

    @stage :reorthog reorthogonalization!(factorization, iterator.rot)
    log_factorization_step_diagnostics!(factorization, :reorthogonalization, iteration)
    return true
end
