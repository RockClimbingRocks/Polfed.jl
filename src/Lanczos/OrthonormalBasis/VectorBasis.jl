mutable struct VectorBasis{T,E} <: OrthonormalBasis
    basis::Vector{T}
    nvecs::Int
    blockdim::Int

    function VectorBasis{E}(maxdim::Int, x0::T, ::ProcessingUnit) where {T<:AbstractVecOrMat, E<:Real}
        basis = [similar(x0) for _ in 1:maxdim]
        blockdim = size(x0,20)
        nvecs = 0
        new{T,E}(basis, nvecs, blockdim)
    end
end

# mutable struct MatrixBasis2{T,E} <: OrthonormalBasis 
#     basis::AbstractMatrix{E}
#     nvecs::Int
#     blockdim::Int

#     function MatrixBasis{E}(maxdim::Int, x0::T, pu::ProcessingUnit) where {T<:AbstractVecOrMat, E<:Real}
#         hilbertspacedim = size(x0,1)
#         blockdim = size(x0,2)
#         basis = pu.Matrix{E}(undef, hilbertspacedim, maxdim)
#         nvecs = 0
#         new{T,E}(basis, nvecs, blockdim)
#     end
# end

function add!(basis::VectorBasis{T, E}, v::T) where {T,E}
    if basis.nvecs < length(basis.basis)
        basis.nvecs += 1
        basis.basis[basis.nvecs] .= v
    else
        # Shift all vectors to the left
        for i in 1:basis.nvecs-1
            basis.basis[i] .= basis.basis[i+1]
        end
        # Add the new vector to the last position
        basis.basis[basis.nvecs] .= v
    end
end

function all(basis::VectorBasis)
    if basis.nvecs == 0
        throw(ArgumentError("Basis is empty"))
    end
    return basis.basis[1:basis.nvecs]
end

function all_withoutlasttwo(basis::VectorBasis)
    blockdim = basis.blockdim
    if basis.nvecs < 2 * blockdim
        throw(ArgumentError("Not enough vectors in the basis"))
    end
    return basis.basis[1:basis.nvecs-2blockdim]
end
