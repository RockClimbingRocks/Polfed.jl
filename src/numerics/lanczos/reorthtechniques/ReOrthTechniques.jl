"""
    ReOrthTechnique

Abstract strategy type controlling Lanczos reorthogonalization behavior.
"""
abstract type ReOrthTechnique end

"""
    FullRO()

Full reorthogonalization on every iteration.
"""
struct FullRO <: ReOrthTechnique end

"""
    PartialRO(skip::Int)

Partial reorthogonalization: reorthogonalize only on selected iterations,
controlled by `skip`.
"""
struct PartialRO <: ReOrthTechnique
    skip::Int
end

"""
    SelectiveRO()

Placeholder reorthogonalization strategy for selective schemes (not yet implemented).
"""
struct SelectiveRO <: ReOrthTechnique end
