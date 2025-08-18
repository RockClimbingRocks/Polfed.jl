abstract type ProcessingUnit end

struct CPU <: ProcessingUnit
    mat::Type{<:Matrix}
    vec::Type{<:Vector}
    zeros::Function
    ones::Function

    function CPU()
        return new(Matrix, Vector, LinearAlgebra.zeros, LinearAlgebra.ones)
    end
end
struct GPU <: ProcessingUnit
    mat::Type{<:CuMatrix}
    vec::Type{<:CuVector}
    zeros::Function
    ones::Function

    function GPU()
        return new(CuMatrix, CuVector, CUDA.zeros, CUDA.ones)
    end
end