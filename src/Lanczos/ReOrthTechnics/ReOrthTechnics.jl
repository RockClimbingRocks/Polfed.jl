
abstract type ReOrthTechnique end
struct FullRO <: ReOrthTechnique end
struct PartialRO <:ReOrthTechnique
    skip::Integer

    function PartialRO(skip::Integer)
        new(skip)
    end
end
struct SelectiveRO <:ReOrthTechnique end


# abstract type ReOrthogonalizer end
# struct ClassicalGramSchmidt <: ReOrthogonalizer end
# struct MatrixGramSchmidt <: ReOrthogonalizer end

# struct FullReorthogonalization <: Orthogonalizer end
# struct PartialReorthogonalization <: Orthogonalizer end

# """
#     ModifiedGramSchmidt2()

# # Represents the modified Gram Schmidt algorithm with a second reorthogonalization step
# # always taking place.
# # """
# struct ModifiedGramSchmidt2 <: Reorthogonalizer end

# Iterative reorthogonalization
# """
#     ClassicalGramSchmidtIR(η::Real = 1/sqrt(2))

# Represents the classical Gram Schmidt algorithm with iterative (i.e. zero or more)
# reorthogonalization until the norm of the vector after an orthogonalization step has not
# decreased by a factor smaller than `η` with respect to the norm before the step. The
# default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.
# """
# struct ClassicalGramSchmidtIR{S<:Real} <: Reorthogonalizer
#     η::S
# end
# ClassicalGramSchmidtIR() = ClassicalGramSchmidtIR(1 / sqrt(2)) # Daniel-Gragg-Kaufman-Stewart

# """
#     ModifiedGramSchmidtIR(η::Real = 1/sqrt(2))

# Represents the modified Gram Schmidt algorithm with iterative (i.e. zero or more)
# reorthogonalization until the norm of the vector after an orthogonalization step has not
# decreased by a factor smaller than `η` with respect to the norm before the step. The
# default value corresponds to the Daniel-Gragg-Kaufman-Stewart condition.
# """
# struct ModifiedGramSchmidtIR{S<:Real} <: Reorthogonalizer
#     η::S
# end
# ModifiedGramSchmidtIR() = ModifiedGramSchmidtIR(1 / sqrt(2)) # Daniel-Gragg-Kaufman-Stewart
