

"""
    struct TimingReport

Stores walltime and CPU time statistics for a factorization and overall simulation.

# Fields
- `total_wt::Float64`: Total walltime (seconds) of the overall simulation.
- `total_ct::Float64`: Total CPU time (seconds) of the overall simulation.
- `fact_wt::Vector{Float64}`: Per-stage walltime data for the factorization.
- `fact_ct::Vector{Float64}`: Per-stage CPU time data for the factorization.
"""
mutable struct TimingReport
    total_wt::Float64
    total_ct::Float64
    fact_wt::AbstractVector{Float64}
    fact_ct::AbstractVector{Float64}
end


"""
    struct MemoryReport

Reports detailed memory usage for CPU and/or GPU computations.

# Fields
- `total_mem_used::Integer`: Total memory used during computation (bytes).
- `total_mem_reserved::Integer`: Total memory reserved by the allocator (bytes).
- `basis_mem_used::Integer`: Memory used by basis vectors (bytes).
- `basis_mem_reserved::Integer`: Memory reserved for basis vectors (bytes).
"""
mutable struct MemoryReport
    total_mem_used::Integer
    total_mem_reserved::Integer
    basis_mem_used::Integer
    basis_mem_reserved::Integer
end


"""
    struct BenchmarkReport

Aggregates performance statistics from a single POLFED run, such as total walltime and CPU time, as well as the times spend in different parts of the factorization algorithm. It serves as a high-level summary of the computational efficiency of a
[`polfed`](@ref).

# Fields
- `timings::TimingReport`: Stores walltime and CPU time statistics across the full simulation and individual factorization stages.
- `memory_cpu::Union{MemoryReport, Nothing}`: Memory usage report for CPU computations (or `nothing` if not applicable).
- `memory_gpu::Union{MemoryReport, Nothing}`: Memory usage report for GPU computations (or `nothing` if not applicable).

# Constructors
```julia
BenchmarkReport(
    fact_report::FactorizationReport,
    total_wt::Real,
    total_ct::Real,
    x0::AbstractVecOrMat{T},
    pu::ProcessingUnit
) where {T<:Real}

Constructs a new BenchmarkReport from a FactorizationReport, recording total walltime
and CPU time along with detailed memory usage estimates.
-   The TimingReport is created automatically from fact_report.walltimes and fact_report.cputimes.
-   The MemoryReport entries are generated based on the basistype in fact_report and the given processing unit pu (CPU() or GPU()).
-   If a HybridMatrixBasis is detected, both CPU and GPU memory usage are recorded.
"""
struct BenchmarkReport
    timings::TimingReport
    memory_cpu::Union{MemoryReport, Nothing}
    memory_gpu::Union{MemoryReport, Nothing}

    function BenchmarkReport(fact_report::FactorizationReport, total_wt::Real, total_ct::Real, x0::AbstractVecOrMat{T},pu::ProcessingUnit) where {T<:Real}
        # --- Timings ---
        timings = TimingReport(total_wt, total_ct, fact_report.walltimes, fact_report.cputimes)

        memory_cpu = nothing
        memory_gpu = nothing

        # helper for computing memory usage
        function mem_report(vecdim, ncols_reserved, ncols_needed, eltyp, device::Symbol)
            bytes_per_elem = sizeof(eltyp)
            basis_mem_reserved = vecdim * ncols_reserved * bytes_per_elem
            basis_mem_used     = vecdim * ncols_needed   * bytes_per_elem

            if device == :cpu
                total_mem_used     = Base.summarysize(fact_report)   # approximation
                total_mem_reserved = total_mem_used                  # no real notion of "reserved" on CPU
            else
                free, total = CUDA.memory_info()
                total_mem_used     = total - free
                total_mem_reserved = total
            end

            return MemoryReport(
                total_mem_used,
                total_mem_reserved,
                basis_mem_used,
                basis_mem_reserved,
            )
        end

        # Decide based on basis type
        basistype = fact_report.basistype
        vecdim = size(x0, 1) 
        eltyp = T

        if basistype <: HybridMatrixBasis
            # both CPU + GPU basis exist
            memory_cpu = mem_report(vecdim, fact_report.ncols_cpu_reserved, fact_report.ncols_cpu_needed, eltyp, :cpu)
            memory_gpu = mem_report(vecdim, fact_report.ncols_gpu_reserved, fact_report.ncols_gpu_needed, eltyp, :gpu)

        elseif basistype <: MatrixBasis
            # decide based on processing unit
            if pu isa CPU
                memory_cpu = mem_report(vecdim, fact_report.ncols_cpu_reserved, fact_report.ncols_cpu_needed, eltyp, :cpu)
            elseif pu isa GPU
                memory_gpu = mem_report(vecdim, fact_report.ncols_gpu_reserved, fact_report.ncols_gpu_needed, eltyp, :gpu)
            else
                error("Unsupported processing unit: $(fact_report.pu)")
            end
        else
            error("Unsupported basis type: $basistype")
        end

        new(timings, memory_cpu, memory_gpu)
    end
end



"""
    display_benchmark_report(report::BenchmarkReport;
                             show_timings::Bool=true,
                             show_memory::Bool=false,
                             use_colors::Bool=true)

Pretty-prints a [`BenchmarkReport`](@ref) with optional colorized output.

# Keyword Arguments
- `show_timings::Bool=true`: Display timing information (walltime and CPU time).
- `show_memory::Bool=false`: Display memory usage details.
- `use_colors::Bool=true`: Use ANSI colors in the printed output.

# Output
- Summarizes total walltime and CPU time.
- Shows distribution of time across factorization stages.
- Optionally displays memory usage for CPU and/or GPU computations.
"""
function display_benchmark_report(report::BenchmarkReport, use_colors::Bool;
        show_timings::Bool=true,
        show_memory::Bool=false)

    f = Formatter(use_colors)

    # --- Helper for time formatting ---
    function format_time(t)
        if t <= 60*2
            return (t, "seconds")
        elseif t <= 60*60*2
            return (t/60, "minutes")
        else
            return (t/3600, "hours")
        end
    end

    # --- Walltime ---
    fact_tot_wt = sum(report.timings.fact_wt)
    tot_wt      = report.timings.total_wt

    fact_tot_wt_u, fact_units_wt = format_time(fact_tot_wt)
    fact_total_wt = cyan(f, @sprintf("%.2f %s", fact_tot_wt_u, fact_units_wt))

    tot_wt_u, units_wt = format_time(tot_wt)
    tot_wt_str = cyan(f, @sprintf("%.2f %s", tot_wt_u, units_wt))

    percentages_wt = [t / fact_tot_wt * 100 for t in report.timings.fact_wt]
    percentages_grouped_wt = [percentages_wt[2], percentages_wt[5], percentages_wt[6],
                              percentages_wt[1] + percentages_wt[3] + percentages_wt[4]]
    fact_formatted_percentages_wt = join([cyan(f, @sprintf("%.2f%%", p)) for p in percentages_grouped_wt], ", ")

    # --- CPU time ---
    fact_tot_ct = sum(report.timings.fact_ct)
    tot_ct      = report.timings.total_ct

    fact_tot_ct_u, fact_units_ct = format_time(fact_tot_ct)
    fact_total_ct = cyan(f, @sprintf("%.2f %s", fact_tot_ct_u, fact_units_ct))

    tot_ct_u, units_ct = format_time(tot_ct)
    tot_ct_str = cyan(f, @sprintf("%.2f %s", tot_ct_u, units_ct))

    percentages_ct = [t / fact_tot_ct * 100 for t in report.timings.fact_ct]
    percentages_grouped_ct = [percentages_ct[2], percentages_ct[5], percentages_ct[6],
                              percentages_ct[1] + percentages_ct[3] + percentages_ct[4]]
    fact_formatted_percentages_ct = join([cyan(f, @sprintf("%.2f%%", p)) for p in percentages_grouped_ct], ", ")

    # --- Printing ---
    if show_timings
        println(blue(f, "Timings: ") *
            "Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)")
        println("- Total polfed run took: $tot_wt_str (walltime), $tot_ct_str (CPU time)")
        println("- Walltime of factorization took: $fact_total_wt ($fact_formatted_percentages_wt)")
        println("- CPU time of factorization took: $fact_total_ct ($fact_formatted_percentages_ct)")
    end 

    if show_memory && !isnothing(report.memory_cpu)
        println(blue(f, "CPU memory usage:"))
        println("- Total memory used: $(report.memory_cpu.total_mem_used) GB, reserved: $(report.memory_cpu.total_mem_reserved) GB")
        println("- Basis memory used: $(report.memory_cpu.basis_mem_used) GB, reserved: $(report.memory_cpu.basis_mem_reserved) GB")
    end

    if show_memory && !isnothing(report.memory_gpu)
        println(blue(f, "GPU memory usage:"))
        println("- Total memory used: $(report.memory_gpu.total_mem_used/1e9) GB, reserved: $(report.memory_gpu.total_mem_reserved/1e9) GB")
        println("- Basis memory used: $(report.memory_gpu.basis_mem_used/1e9) GB, reserved: $(report.memory_gpu.basis_mem_reserved/1e9) GB")
    end
end

