"""
    ProcessingUnit

Abstract processing-unit descriptor used to select CPU/GPU storage and
allocation helpers through dispatch.
"""
abstract type ProcessingUnit end

"""
    CPU() -> CPU

Construct a CPU processing-unit adapter containing matrix/vector types and
allocation/random helper functions.
"""
struct CPU <: ProcessingUnit
    Matrix::Type{<:Matrix}
    Vector::Type{<:Vector}
    zeros::Function
    ones::Function
    rand::Function
    randn::Function

    """Construct CPU backend dispatch helpers."""
    function CPU()
        return new(Matrix, Vector, LinearAlgebra.zeros, LinearAlgebra.ones, rand, randn)
    end
end
"""
    GPU() -> GPU

Construct a GPU processing-unit adapter containing CUDA matrix/vector types
and allocation/random helper functions.
When CUDA is unavailable, construction throws an informative error.
"""
struct GPU <: ProcessingUnit
    Matrix::Any
    Vector::Any
    zeros::Function
    ones::Function
    rand::Function
    randn::Function

    """Construct GPU backend dispatch helpers."""
    function GPU()
        cuda_available() || error("CUDA is not available; GPU processing unit cannot be constructed.")
        return new(gpu_matrix_type(), gpu_vector_type(), gpu_zeros, gpu_ones, gpu_rand, gpu_randn)
    end
end
