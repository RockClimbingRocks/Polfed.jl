abstract type ProcessingUnit end

struct CPU <: ProcessingUnit
    Matrix::Type{<:Matrix}
    Vector::Type{<:Vector}
    zeros::Function
    ones::Function

    function CPU()
        return new(Matrix, Vector, LinearAlgebra.zeros, LinearAlgebra.ones)
    end
end
struct GPU <: ProcessingUnit
    Matrix::Type{<:CuMatrix}
    Vector::Type{<:CuVector}
    zeros::Function
    ones::Function

    function GPU()
        return new(CuMatrix, CuVector, CUDA.zeros, CUDA.ones)
    end
end