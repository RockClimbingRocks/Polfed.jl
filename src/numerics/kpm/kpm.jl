"Kernel polynomial method (KPM): direction-independent Chebyshev moments and DoS reconstruction."
module KPM

using LinearAlgebra, Random

import ..Clenshaw: ClenshawExpansion
import ...CPU, ...GPU, ...ProcessingUnit
import ...is_gpu_array   # device probe blocks take the batched dots (moments.jl)

include("kernels.jl")
include("probes.jl")
include("moments.jl")
include("density.jl")

export kpm_moments, kpm_density, get_kernel, kpm_coeffs

end # module KPM
