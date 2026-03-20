
"""
    abstract type KrylovFactorization{S<:Real, T<:Number, P<:ProcessingUnit}
"""
abstract type KrylovFactorization{S<:Real, T<:Number, P<:ProcessingUnit} end


include("LanczosFactorization.jl")
include("BlockLanczosFactorization.jl")


include("FactorizationSteps/calculate_overlap.jl")
include("FactorizationSteps/mapping.jl")
include("FactorizationSteps/orthogonalization.jl")
include("FactorizationSteps/reorthogonalization.jl")
include("FactorizationSteps/tridiagonalization.jl")
include("diagnostics.jl")


"""
    update!(fac::KrylovFactorization, v::AbstractArray) -> nothing

Append new vector/block `v` into factorization basis state and advance
`krylovdim`. Mutates `fac` in-place.
"""
function update!(fac::KrylovFactorization, v::AbstractArray)
    # Step 1: Add vector/matrix to basis
    add!(fac.basis, v)

    # Step 2: Shift v_last → v_secondlast
    fac.v_secondlast .= fac.v_last

    # Step 3: Copy new vector/matrix into v_last
    fac.v_last .= v

    # Step 4: Increment Krylov dimension
    fac.krylovdim += ndims(v) == 1 ? 1 : size(v, 2)
end






"""
    lanczositer(iterator::LanczosIterator, basis::OrthonormalBasis, maxdim::Int, pu::ProcessingUnit) -> KrylovFactorization

Initialize Lanczos factorization state from the starting seed in `iterator`.
"""
function lanczositer(iterator::LanczosIterator, basis::OrthonormalBasis, maxdim::Int, pu::ProcessingUnit)
    # initialize without using eltype
    x0 = iterator.x0

    factorizationtype = isa(x0, AbstractVector) ? LanczosFactorization : BlockLanczosFactorization
    factorization = factorizationtype(maxdim, basis, x0, pu)

    update!(factorization, x0)
    iterator.f!(factorization.r, x0)
    
    calculate_overlap!(factorization)

    # orthogonalize it
    α = getoverlap(factorization)
    factorization.r -= x0 * α

    return factorization
end


"""
    lanczositer!(iterator::LanczosIterator, factorization::KrylovFactorization, walltimes, cputimes, iteration) -> nothing

Run one Lanczos iteration stage sequence and accumulate stage timings.
"""
function lanczositer!(
    iterator::LanczosIterator,
    factorization::KrylovFactorization,
    walltimes::Vector{<:Real},
    cputimes::Vector{<:Real},
    iteration::Integer,
)
    @addtime! walltimes cputimes 1 tridiagonalization!(factorization)
    log_factorization_step_diagnostics!(factorization, :tridiagonalization, iteration)

    @addtime! walltimes cputimes 2 mapping!(factorization, iterator.f!)
    log_factorization_step_diagnostics!(factorization, :mapping, iteration)

    @addtime! walltimes cputimes 3 calculate_overlap!(factorization)
    log_factorization_step_diagnostics!(factorization, :calculate_overlap, iteration)

    @addtime! walltimes cputimes 4 orthogonalization!(factorization)
    log_factorization_step_diagnostics!(factorization, :orthogonalization, iteration)

    @addtime! walltimes cputimes 5 reorthogonalization!(factorization, iterator.rot)
    log_factorization_step_diagnostics!(factorization, :reorthogonalization, iteration)
end
