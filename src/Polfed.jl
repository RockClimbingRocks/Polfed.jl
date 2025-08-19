module Polfed

# using external packages 
using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf#, Base.Threads, Polynomials
# using CPUTime
using StaticArrays: SVector
using CUDA, CUDA.CUSPARSE

include("ClenshawMapping/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")
# import .Lanczos: lanczosmethod, FullRO, PartialRO, MatrixBasis, VectorBasis, ClassicalGramSchmidt, MatrixGramSchmidt, ConvergenceInfoOut, display_convergenceinfo, print_convergenceinfo
import .ClenshawMapping: Clenshaw, ClenshawKernel
import .Lanczos: lanczos, FactorizationReport, display_report, print_report, @addtime!, FullRO, PartialRO, ReOrthTechnique, MatrixBasis, HybridMatrixBasis, OrthonormalBasis

# struct FullRO end
# struct MatrixGramSchmidt end
# struct MatrixBasis end

include("polfed/polfed.jl")

export polfed
# export ...
end
