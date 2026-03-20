"""
    ProcessingUnit

Abstract processing-unit descriptor for Lanczos internals.
"""
abstract type ProcessingUnit end

"""
    CPU() -> CPU

Construct CPU allocation/type adapter for Lanczos work arrays.
"""
struct CPU <: ProcessingUnit
    Matrix::Type{<:Matrix}
    Vector::Type{<:Vector}
    zeros::Function
    ones::Function

    """Construct CPU backend dispatch helpers."""
    function CPU()
        return new(Matrix, Vector, LinearAlgebra.zeros, LinearAlgebra.ones)
    end
end
"""
    GPU() -> GPU

Construct GPU allocation/type adapter for Lanczos work arrays.
When CUDA is unavailable, construction throws an informative error.
"""
struct GPU <: ProcessingUnit
    Matrix::Type{<:CuMatrix}
    Vector::Type{<:CuVector}
    zeros::Function
    ones::Function

    """Construct GPU backend dispatch helpers."""
    function GPU()
        CUDA_AVAILABLE || error("CUDA is not available; GPU processing unit cannot be constructed.")
        return new(CuMatrix, CuVector, CUDA.zeros, CUDA.ones)
    end
end
