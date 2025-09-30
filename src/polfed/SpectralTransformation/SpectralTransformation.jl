
include("Bisection.jl")
include("GetBounds.jl")
include("GetOrderOfExpansion.jl")
include("ParallelizationClenshaw.jl")
include("ParallelizationMultiplication.jl")
include("DefineTransformation.jl")
# include("MultiplicationTypeClenshaw.jl")


function getspectraltransform!(
    dos::DoSConfigFull, 
    spectral_transform::SpectralTransformConfigFull, 
    fact::FactorizationConfigFull, 
    pu::ProcessingUnit
)
   
    getbounds!(spectral_transform, dos.ρ, :mean)
    getorderofexapnsion!(spectral_transform)

    definetransformation!(spectral_transform, fact.x0, pu)
end

