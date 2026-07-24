
include("BenchmarkReport.jl")
include("SpectralTransformReport.jl")


"""
    struct Report

A container for aggregating the main results of a simulation or computation. One can deisplay it with [`display_report`](@ref) function.

# Fields
- [`SpectralTransformReport`](@ref) - Tells you about spectral transformation parameter like rescaled targeted interval, order of polynomial, number of matrix-vector multiplications etc.
- [`FactorizationReport`](@ref) - Provides insights into the factorization process, including convergence behavior, iteration counts and errors of obtained eigenpairs.
- [`BenchmarkReport`](@ref) - Summarizes performance such as total walltime and CPU time, as well as the times spend in different parts of the factorization algorithm.
"""
mutable struct Report 
    spectral_transform::SpectralTransformReport
    factorization::FactorizationReport
    benchmark::BenchmarkReport

    function Report(spectral_transform, factorization, benchmark)
        new(spectral_transform, factorization, benchmark)
    end
end


"""
    display_report(report::Report;
                   use_colors::Bool=true,
                   include_spectral_transform::Bool=true,
                   include_factorization::Bool=true,
                   show_convergence_details::Bool=false,
                   include_benchmark::Bool=true)

Displays a comprehensive report for a [`Polfed.PolfedCore.Report`](@ref) object.

# Keyword Arguments
- `use_colors::Bool=true`: Enable or disable ANSI color formatting.
- `include_spectral_transform::Bool=true`: Include the spectral transform section [`display_spectral_report`](@ref).
- `include_factorization::Bool=true`: Include the factorization section [`display_factorization_report`](@ref).
- `show_convergence_details::Bool=false`: If `true`, display detailed convergence information in the factorization report.
- `include_benchmark::Bool=true`: Include the benchmark section [`display_benchmark_report`](@ref).

# Notes
The `show_timings` flag for the factorization section is always set to `false` in this combined display function.
"""
function display_report(report::Report;
    use_colors::Bool=true,
    include_spectral_transform::Bool=true,
    include_factorization::Bool=true,
    show_convergence_details::Bool=false,
    include_benchmark::Bool=true,
)
    include_spectral_transform && display_spectral_report(report.spectral_transform, use_colors)

    include_factorization && display_factorization_report(report.factorization, use_colors; 
        show_convergence_details=show_convergence_details, 
        show_timings=false
    )

    include_benchmark && display_benchmark_report(report.benchmark, use_colors)
end
