

include("Structs/Structs.jl")
include("PolfedDefaults.jl")
include("polfed_algorithm.jl")
include("DensetiesOfStates/DensetiesOfStates.jl")
include("SpectralTransformation/SpectralTransformation.jl")



##ALSO ADD THE ENTRYPOINT FOR without X0!! MAYBE? 

function polfed(mat::AbstractMatrix{T}, x0::AbstractVector{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    spectral_transform      = SpectralTransformConfig(),
    lanczos                 = LanczosConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}

    f!(Y::AbstractVecOrMat, X::AbstractVecOrMat) = mul!(Y , mat, X)

    polfed(f!, x0, howmany, target; 
        produce_report      = produce_report,
        spectral_transform  = spectral_transform,
        lanczos             = lanczos,
        dos                 = dos
    )
end




function polfed(f!::Function, x0::AbstractVector{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    spectral_transform      = SpectralTransformConfig(),
    lanczos                 = LanczosConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}


    walltime = zeros(Float64, 1)
    cputime = zeros(Float64, 1)
    @addtime! walltime cputime 1 begin
        pu = isa(x0, CuArray) ? GPU() : CPU()

        spectral_transform_config = SpectralTransformConfigFull(spectral_transform, f!, howmany, target, pu)
        lanczos_config = LanczosConfigFull(lanczos, spectral_transform_config, x0, howmany)
        dos_config = DoSConfigFull(dos)


        vals, vecs, factorization_report = polfed_algorithm(spectral_transform_config, lanczos_config, dos_config, pu)

        nothing
    end
    
    
    spectral_transform_report = SpectralTransformReport(spectral_transform_config)
    benchmark_report = BenchmarkReport(factorization_report, walltime[1], cputime[1])
    report = Report(spectral_transform_report, factorization_report, benchmark_report)



    produce_report && (return (vals, vecs, report))
    return (vals, vecs)

end














