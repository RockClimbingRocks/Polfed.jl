module Polfed

using Distributed

using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf, SharedArrays, SparseArrays
# using KrylovKit
# @everywhere using LinearAlgebra
using StaticArrays: SVector
using CUDA, CUDA.CUSPARSE

include("ClenshawMapping/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")

import .ClenshawMapping: Clenshaw, ClenshawKernel
import .Lanczos: lanczos, FactorizationReport, display_report, print_report,
                 @addtime!, FullRO, PartialRO, ReOrthTechnique,
                 MatrixBasis, HybridMatrixBasis, OrthonormalBasis

const main_module_file = abspath(@__FILE__)


include("polfed/polfed.jl")


export polfed

end # module