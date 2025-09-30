
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


function display_benchmark_report(report::BenchmarkReport; show_timings::Bool=true, show_memory::Bool=false)

    fact_tot_wt = sum(report.timings.fact_wt) 
    tot_wt = report.timings.total_wt

    fact_tot_wt_u, fact_units_wt =   fact_tot_wt <= 60*2 ? (fact_tot_wt, "seconds") :
                        fact_tot_wt <= 60*60*2 ? (fact_tot_wt/60, "minutes") : (fact_tot_wt/60/60, "hours")
    percentages_wt = [t / fact_tot_wt * 100 for t in report.timings.fact_wt]
    percentages_ordered_grouped_wt = [percentages_wt[2], percentages_wt[5], percentages_wt[6], percentages_wt[1]+percentages_wt[3]+percentages_wt[4]]
    fact_formatted_percentages_wt = join(["\e[1;36m" * @sprintf("%.2f", p) * "%\e[0m" for p in percentages_ordered_grouped_wt], ", ")
    fact_total_wt            = @sprintf("\e[1;36m %.2f %s \e[0m", fact_tot_wt_u, fact_units_wt)

    tot_wt_u, units_wt =   tot_wt <= 60*2 ? (tot_wt, "seconds") :
                        tot_wt <= 60*60*2 ? (tot_wt/60, "minutes") : (tot_wt/60/60, "hours")
    tot_wt_u            = @sprintf("\e[1;36m %.2f %s \e[0m", tot_wt_u, units_wt)


    fact_tot_ct = sum(report.timings.fact_ct) 
    tot_ct = report.timings.total_ct

    fact_tot_ct_u, fact_units_ct =   fact_tot_ct <= 60*2 ? (fact_tot_ct, "seconds") :
                        fact_tot_ct <= 60*60*2 ? (fact_tot_ct/60, "minutes") : (fact_tot_ct/60/60, "hours")
    percentages_ct = [t / fact_tot_ct * 100 for t in report.timings.fact_ct]
    percentages_ordered_grouped_ct = [percentages_ct[2], percentages_ct[5], percentages_ct[6], percentages_ct[1]+percentages_ct[3]+percentages_ct[4]]
    fact_formatted_percentages_ct = join(["\e[1;36m" * @sprintf("%.2f", p) * "%\e[0m" for p in percentages_ordered_grouped_ct], ", ")
    fact_total_ct            = @sprintf("\e[1;36m %.2f %s \e[0m", fact_tot_ct_u, fact_units_ct)

    tot_ct_u, units_ct =   tot_ct <= 60*2 ? (tot_ct, "seconds") :
                        tot_ct <= 60*60*2 ? (tot_ct/60, "minutes") : (tot_ct/60/60, "hours")

    tot_ct_u            = @sprintf("\e[1;36m %.2f %s \e[0m", tot_ct_u, units_ct)

    if show_timings
        println(  "\e[1;34mTimings:\e[0m Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)")
        println("- Total polfed run took: $(tot_wt_u) (walltime), $(tot_ct_u) (CPU time)")
        println("- Walltime of factorization took: $fact_total_wt ($fact_formatted_percentages_wt)")
        println("- CPU time of factorization took: $fact_total_ct ($fact_formatted_percentages_ct)")
    end 
    if show_memory && !isnothing(report.memory_cpu)
        println(  "\e[1;34mCPU memory usage:\e[0m")
        println("- Total memory used: $(report.memory_cpu.total_mem_used) GB, reserved: $(report.memory_cpu.total_mem_reserved) GB")
        println("- Basis memory used: $(report.memory_cpu.basis_mem_used) GB, reserved: $(report.memory_cpu.basis_mem_reserved) GB")
    end
    if show_memory && !isnothing(report.memory_gpu)
        println(  "\e[1;34mGPU memory usage:\e[0m")
        println("- Total memory used: $(report.memory_gpu.total_mem_used/1e9) GB, reserved: $(report.memory_gpu.total_mem_reserved/1e9) GB")
        println("- Basis memory used: $(report.memory_gpu.basis_mem_used/1e9) GB, reserved: $(report.memory_gpu.basis_mem_reserved/1e9) GB")
    end


end
