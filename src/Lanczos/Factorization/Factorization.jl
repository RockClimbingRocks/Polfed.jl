
"""
    abstract type KrylovFactorization{E<:Real, P<:ProcessingUnit}
"""
abstract type KrylovFactorization{E<:Real, P<:ProcessingUnit} end


include("LanczosFactorization.jl")
include("BlockLanczosFactorization.jl")


include("FactorizationSteps/calculate_overlap.jl")
include("FactorizationSteps/mapping.jl")
include("FactorizationSteps/orthogonalization.jl")
include("FactorizationSteps/reorthogonalization.jl")
include("FactorizationSteps/tridiagonalization.jl")


function update!(fac::KrylovFactorization, v::AbstractArray)
    # Step 1: Add vector/matrix to basis
    add!(fac.basis, v)

    # Step 2: Shift v_last → v_secondlast
    fac.v_secondlast .= fac.v_last

    # Step 3: Copy new vector/matrix into v_last
    fac.v_last .= v

    # Step 4: Increment Krylov dimension
    fac.krylovdim += ndims(v) == 1 ? 1 : size(v, 2)

    # return fac
end






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


function lanczositer!(iterator::LanczosIterator, factorization::KrylovFactorization, walltimes::Vector{<:Real}, cputimes::Vector{<:Real})
    @addtime! walltimes cputimes 1 tridiagonalization!(factorization)
    @addtime! walltimes cputimes 2 mapping!(factorization, iterator.f!)
    @addtime! walltimes cputimes 3 calculate_overlap!(factorization)
    @addtime! walltimes cputimes 4 orthogonalization!(factorization)
    @addtime! walltimes cputimes 5 reorthogonalization!(factorization, iterator.rot)
end