
include("BenchmarkReport.jl")
include("SpectralTransformReport.jl")



mutable struct Report 
    spectral_transform::SpectralTransformReport
    factorization::FactorizationReport
    benchmark::BenchmarkReport

    function Report(spectral_transform, factorization, benchmark)
  
        new(
            spectral_transform,
            factorization,
            benchmark
        )
    end
end



function display_report(report::Report;
    include_spectral_transform::Bool=true,
    include_factorization::Bool=true,
    show_convergence_details=false, 
    include_benchmark::Bool=true,
)

    include_spectral_transform && display_report(report.spectral_transform)

    include_factorization && display_report(report.factorization; 
        show_convergence_details=show_convergence_details, 
        show_timings=false
    )

    include_benchmark && display_report(report.benchmark)
end