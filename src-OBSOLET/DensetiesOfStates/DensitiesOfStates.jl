
include("Moments.jl")
include("Kernels.jl")


"""
    getdos!(dosconfig::DoSConfigFull, mapping_plan::MappingPlan, lanczos::FactorizationConfigFull, pu::ProcessingUnit) -> nothing

Estimate density of states and store it in `dosconfig.ρ`.

The estimated DoS function is represented in rescaled coordinates (roughly
`[-1, 1]`) using a Chebyshev/KPM expansion with configured kernel smoothing.
"""
function getdos!(dosconfig::DoSConfigFull, mapping_plan::MappingPlan, lanczos::FactorizationConfigFull, pu::ProcessingUnit)
    @unpack N, R, kernel_fn = dosconfig
    @unpack f!_rescaled = mapping_plan
    @unpack elmtype, x0 = lanczos

    PolfedDefaults.polfed_log(
        PolfedDefaults.POLFED_INFO_LEVEL,
        "Computing density of states.",
        N=N,
        R=R,
        kernel=dosconfig.kernel,
    )

    hilbertspacedim = size(x0,1)
    moments = dos_moments(f!_rescaled, N, R, hilbertspacedim, elmtype, pu)
    coeffs = [kernel_fn(i, N) * moments[i+1]*(2 - ==(i,0)) for i in 0:N-1]

    # Tn = ChebyshevT(coeffs)
    Tdos = real(elmtype)
    Tn = Clenshaw(:Chebyshev, coeffs, N-1, (y,x)->*(y,x), 1, Tdos)
    ρ_KPM(x) = Tn(x)/(π*√(1-x^2))
    
    # xs = LinRange(-0.99,0.99, 250)
    # println("---------")
    # println(ρ_KPM.(xs))
    # println(ρ_KPM(0.))
    # println("---------")
    
    dosconfig.ρ = ρ_KPM
end
