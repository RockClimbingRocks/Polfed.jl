

include("Structs/Structs.jl")
include("PolfedDefaults.jl")
include("polfed_algorithm.jl")
include("DensetiesOfStates/DensetiesOfStates.jl")
include("SpectralTransformation/SpectralTransformation.jl")
include("Optimization/optimization.jl")



##ALSO ADD THE ENTRYPOINT FOR without X0!! MAYBE? 

function polfed(mat::AbstractMatrix{T}, x0::AbstractVecOrMat{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    optimize_mapping::Bool  = PolfedDefaults.optimize_mapping,
    spectral_transform      = SpectralTransformConfig(),
    lanczos                 = LanczosConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}


    f! = (Y,X) -> mul!(Y, mat, X)
    optimize_mapping && (f! = optimize_spectral_transform(mat, spectral_transform))

    polfed(f!, x0, howmany, target; 
        produce_report      = produce_report,
        spectral_transform  = spectral_transform,
        lanczos             = lanczos,
        dos                 = dos
    )
end




function polfed(f!::Function, x0::AbstractVecOrMat{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    spectral_transform      = SpectralTransformConfig(),
    lanczos                 = LanczosConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}

    if spectral_transform.parallelization isa TwoLevelParallel
        nvecs = size(x0, 2)

        println("nvecs = $nvecs")

        set_workers(nvecs, spectral_transform.parallelization.nt_per_col)
    else
        println("im fucked.....")
    end



    walltime = zeros(Float64, 1)
    cputime = zeros(Float64, 1)
    @addtime! walltime cputime 1 begin
        pu = isa(x0, CuArray) ? GPU() : CPU()

        spectral_transform_config = SpectralTransformConfigFull(spectral_transform, f!, x0, howmany, target, pu)
        lanczos_config = LanczosConfigFull(lanczos, spectral_transform_config, x0, howmany)
        dos_config = DoSConfigFull(dos)

        vals, vecs, factorization_report = polfed_algorithm(spectral_transform_config, lanczos_config, dos_config, pu)

        nothing
    end

    spectral_transform_report = SpectralTransformReport(spectral_transform_config, factorization_report)
    benchmark_report = BenchmarkReport(factorization_report, walltime[1], cputime[1], x0, pu)
    report = Report(spectral_transform_report, factorization_report, benchmark_report)


    produce_report && (return (vals, vecs, report))
    return (vals, vecs)
end

















