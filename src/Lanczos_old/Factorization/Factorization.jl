
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






function lanczositer(iterator::LanczosIterator, basis::Basis, maxdim::Int, pu::ProcessingUnit)
    # initialize without using eltype
    x₀ = iterator.x₀

    factorizationtype = isa(x₀, AbstractVector) ? LanczosFactorization : BlockLanczosFactorization
    factorization = factorizationtype(maxdim, basis, x₀, pu)

    add!(factorization.basis, x₀); factorization.krylovdim += size(x₀,2)
    iterator.f!(factorization.r, x₀)
    
    calculate_overlap!(factorization)

    # orthogonalize it
    α = getoverlap(factorization)
    factorization.r -= x₀ * α

    

    return factorization
end


function lanczositer!(iterator::LanczosIterator, factorization::KrylovFactorization, times)
    times[1] += @elapsed tridiagonalization!(factorization)    
    times[2] += @elapsed mapping!(factorization, iterator.f!)
    times[3] += @elapsed calculate_overlap!(factorization)
    times[4] += @elapsed orthogonalization!(factorization)
    times[5] += @elapsed reorthogonalization!(factorization, iterator.rot, iterator.reorth)

end


# 0.000162199  0.0134559  0.0
# 0.0134559    0.737049   0.326419
# 0.0          0.326419   0.199545