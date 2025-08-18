const permutation = SVector{3,Int64}(2,3,1)


function dos_moments(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Real}, pu::ProcessingUnit)
    
    vecs = [pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3]
    μs = zeros(Float64, N)

    r = pu.Vector{E}(undef, hilbertspacedim)
    for _ in 1:R
        r .= pu.randn(hilbertspacedim)
        r .*= 1/norm(r)
        trace!(f!, r, μs, vecs)
    end

    μs .*= hilbertspacedim/R
    return μs
end



function trace!(f!::Function, α::AbstractVector{<:Number}, μs::AbstractVector{<:Number}, vecs::Vector{<:AbstractVecOrMat{<:Number}})
    # Dont change the order because it might depand on it! ( if α == β)

    vecs[1] .= α
    μs[1] += α⋅vecs[1]

    f!(vecs[2], vecs[1])
    μs[2] += α⋅vecs[2]

    for i in 2:length(μs)-1
        f!(vecs[3], vecs[2])
        @inbounds @. vecs[3] *= 2.
        @inbounds @. vecs[3] -= vecs[1]
        μs[i+1] += α⋅vecs[3];
        permute!(vecs, permutation);
    end
    
end
