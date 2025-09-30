
include("Moments.jl")
include("Kernels.jl")


function getdos!(dosconfig::DoSConfigFull, spectral_transform::SpectralTransformConfigFull, lanczos::FactorizationConfigFull, pu::ProcessingUnit)
    @unpack N, R, kernel = dosconfig
    @unpack f!_rescaled = spectral_transform
    @unpack elmtype, x0 = lanczos

    hilbertspacedim = size(x0,1)
    moments = dos_moments(f!_rescaled, N, R, hilbertspacedim, elmtype, pu)
    gn = eval(kernel)
    coeffs = [gn(i,N) * moments[i+1]*(2 - ==(i,0)) for i in 0:N-1]

    # Tn = ChebyshevT(coeffs)
    Tn = Clenshaw(:Chebyshev, coeffs, N-1, (y,x)->*(y,x), 1, Float64)
    ρ_KPM(x) = Tn(x)/(π*√(1-x^2))
    
    # xs = LinRange(-0.99,0.99, 250)
    # println("---------")
    # println(ρ_KPM.(xs))
    # println(ρ_KPM(0.))
    # println("---------")
    
    dosconfig.ρ = ρ_KPM
end




