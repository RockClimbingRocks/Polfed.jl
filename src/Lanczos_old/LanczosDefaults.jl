
# mutable struct LanczosConfig
#     rot::ReOrthTechnique,
#     reorth::ReOrthogonalizer, 
#     basistype::Type{<:OrthonormalBasis},
#     maxdim::Union{Int, Nothing}, 
#     tol::Real, 
#     eigentol::Union{Real, Nothing}, 
#     sorted::Bool,
#     mapvals::Union{Function, Nothing},

#     function LanczosConfig(rot::ReOrthTechnique, reorth::ReOrthogonalizer, basistype::Type{<:OrthonormalBasis}, maxdim::Int, tol::Real, eigentol::Real, sorted::Bool, mapvals::Function)
#         new(rot, reorth, basistype, maxdim, tol, eigentol, sorted, mapvals)
#     end

#     function LanczosConfig(;rot::ReOrthTechnique=LanczosDefaults.rot, 
#                             reorth::ReOrthogonalizer=LanczosDefaults.reorth, 
#                             basistype::Type{<:OrthonormalBasis}=LanczosDefaults.basistype, 
#                             maxdim:::Union{Int, Nothing}=LanczosDefaults.maxdim, 
#                             tol::Real=LanczosDefaults.tol, 
#                             eigentol:::Union{Real, Nothing}=LanczosDefaults.eigentol, 
#                             sorted::Bool=LanczosDefaults.sorted, 
#                             mapvals:::Union{Function, Nothing}=nothing)
                            
#         new(rot, reorth, basistype, maxdim, tol, eigentol, sorted, mapvals)
#     end

# end










# module LanczosDefaults
# using ..Lanczos

# # Lanczos defaults 
# const rot = Polfed.FullRO()
# const reorth = Polfed.MatrixGramSchmidt()
# const basistype = Polfed.MatrixBasis\
# const maxdim = 1000
# const tol = 1e-14
# const eigentol = 1e-8
# const sorted = true

# end