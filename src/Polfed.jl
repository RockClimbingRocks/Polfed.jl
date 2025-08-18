module Polfed

# using external packages 
using LinearAlgebra, UnPack, Base.Threads, QuadGK#, Base.Threads, Polynomials, Printf
using StaticArrays: SVector
using CUDA, CUDA.CUSPARSE

include("ClenshawMapping/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")
# import .Lanczos: lanczosmethod, FullRO, PartialRO, MatrixBasis, VectorBasis, ClassicalGramSchmidt, MatrixGramSchmidt, ConvergenceInfoOut, display_convergenceinfo, print_convergenceinfo
import .ClenshawMapping: Clenshaw, ClenshawKernel
import .Lanczos: lanczos, FactorizationReport, display_report, print_report, timeit



struct FullRO end
struct MatrixGramSchmidt end
struct MatrixBasis end


include("polfed/polfed.jl")

export polfed
# export ...
end


D=100

mat_ = randn(D,D)
mat = mat_ + mat_'
x0 = rand(D)
howmany = 2
target = 0.


Polfed.polfed(mat, x0, howmany, target)
