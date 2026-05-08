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
    Matrix::Any
    Vector::Any
    zeros::Function
    ones::Function

    """Construct GPU backend dispatch helpers."""
    function GPU()
        cuda_available() || error("CUDA is not available; GPU processing unit cannot be constructed.")
        return new(gpu_matrix_type(), gpu_vector_type(), gpu_zeros, gpu_ones)
    end
end
