
mutable struct TimingReport
    total_wt::Float64
    total_ct::Float64
    fact_wt::AbstractVector{Float64}
    fact_ct::AbstractVector{Float64}

    function TimingReport(total_wt::Float64, total_ct::Float64, fact_wt::AbstractVector{Float64}, fact_ct::AbstractVector{Float64})

        new(
            total_wt,
            total_ct,
            fact_wt,
            fact_ct
        )
        
    end
end

mutable struct MemoryReport
    total_mem_used::Integer
    total_mem_reserved::Integer
    basis_mem_used::Integer
    basis_mem_reserved::Integer

    function MemoryReport(
        total_mem_used::Integer,
        total_mem_reserved::Integer,
        basis_mem_used::Integer,
        basis_mem_reserved::Integer,
    )
        new(
            total_mem_used,
            total_mem_reserved,
            basis_mem_used,
            basis_mem_reserved,
        )
    end
end


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
    display_benchmark_report(report::BenchmarkReport; show_timings=true, show_memory=false, use_colors=true)

Pretty-prints a Benchmark Report with optional ANSI colors.
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