
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


"""
    display_report(report::Report; 
                   include_spectral_transform::Bool=true,
                   include_factorization::Bool=true,
                   show_convergence_details=false, 
                   include_benchmark::Bool=true)

Displays a comprehensive report for the given `Report` object. The function allows selective inclusion of different report sections via keyword arguments.

# Arguments
- `report::Report`: The report object to display.
- `include_spectral_transform::Bool=true`: If `true`, includes the spectral transform section in the report.
- `include_factorization::Bool=true`: If `true`, includes the factorization section in the report.
- `show_convergence_details=false`: If `true`, displays detailed convergence information in the factorization section.
- `include_benchmark::Bool=true`: If `true`, includes the benchmark section in the report.

# Notes
Each section is displayed by calling `display_report` on the corresponding subfield of the `Report` object. The `show_timings` option for the factorization section is always set to `false` in this function.
"""
function display_report(report::Report;
    include_spectral_transform::Bool=true,
    include_factorization::Bool=true,
    show_convergence_details=false, 
    include_benchmark::Bool=true,
)

    include_spectral_transform && display_spectral_report(report.spectral_transform)

    include_factorization && display_factorization_report(report.factorization; 
        show_convergence_details=show_convergence_details, 
        show_timings=false
    )

    include_benchmark && display_benchmark_report(report.benchmark)
end