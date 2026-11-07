module Numerics

# Reusable, POLFED-agnostic spectral primitives, loaded dependency-first:
# Clenshaw → KPM (which uses Clenshaw) → Lanczos.

include("clenshaw/Clenshaw.jl")
include("kpm/kpm.jl")             # uses Clenshaw — include after it
include("lanczos/Lanczos.jl")

using .Clenshaw: ClenshawExpansion
using .KPM: kpm_moments, kpm_density, get_kernel
using .Lanczos: lanczos, lanczos_extrema, LanczosResult, has_converged,
                EigTarget, EndSpec, NumEigs, ToValue, Lowest, Highest, BothEnds,
                LowestEigs, HighestEigs,
                FullRO, PartialRO, MatrixBasis, VectorBasis,
                FactorizationReport

export ClenshawExpansion
export kpm_moments, kpm_density, get_kernel
export lanczos, lanczos_extrema, LanczosResult, has_converged
export EigTarget, NumEigs, ToValue, Lowest, Highest, BothEnds, LowestEigs, HighestEigs
end
