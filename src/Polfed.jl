module Polfed

using Distributed

using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf, SharedArrays, SparseArrays
# using KrylovKit
# @everywhere using LinearAlgebra
using StaticArrays: SVector
using CUDA, CUDA.CUSPARSE

include("common/common.jl")
include("ClenshawMapping/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")

import .Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!
import .ClenshawMapping: Clenshaw, ClenshawKernel
import .Lanczos: lanczos, FactorizationReport, display_factorization_report,
        FullRO, PartialRO, ReOrthTechnique,
        MatrixBasis, HybridMatrixBasis, OrthonormalBasis


const main_module_file = abspath(@__FILE__)


include("polfed/polfed.jl")

export polfed, display_report

end # module