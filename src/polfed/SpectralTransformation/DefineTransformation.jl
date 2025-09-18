
function definetransformation!(
    spectral_transform::SpectralTransformConfigFull,
    x0::AbstractVecOrMat{E},
    pu::ProcessingUnit
) where {E<:Real}

    @unpack coefficients, polynomialtype, normalization, target,
            parallelization, order, f!_rescaled = spectral_transform
    hilbertspacedim = size(x0,1)

    transform = Clenshaw(polynomialtype, n -> coefficients(target,n),
                         order, f!_rescaled, hilbertspacedim, E)
    norm_ = 1/transform(target)*normalization
    coefficients_normalized(n::Int) = coefficients(target,n) * norm_

    clenshawtransform = define_clenshawtransformation(
        spectral_transform, coefficients_normalized, hilbertspacedim, E
    )
    b_storage = get_b_storage(parallelization, pu, x0)



    f!_transformed = (Y::AbstractVecOrMat{<:Real}, X::AbstractVecOrMat{<:Real}) -> begin

        clenshaw(clenshawtransform, Y, X, b_storage, pu, parallelization)
        nothing
    end

    spectral_transform.f!_transformed = f!_transformed
end




function define_clenshawtransformation(
    spectral_transform::SpectralTransformConfigFull, 
    coefficients_normalized::Function,
    hilbertspacedim::Int, 
    E::Type
)

    is_clenshw_reocurence_set   = !isnothing(spectral_transform.clenshaw_recurrence)
    is_clenshw_finalsum_set     = !isnothing(spectral_transform.clenshaw_finalsum)

    return is_clenshw_reocurence_set && is_clenshw_finalsum_set ?
        ClenshawKernel(
            coefficients_normalized, 
            spectral_transform.order, 
            spectral_transform.polynomialtype, 
            spectral_transform.clenshaw_recurrence, 
            spectral_transform.clenshaw_finalsum, 
            hilbertspacedim, 
            E
        ) :
        Clenshaw(
            spectral_transform.polynomialtype, 
            coefficients_normalized, 
            spectral_transform.order, 
            spectral_transform.f!_rescaled, 
            hilbertspacedim, 
            E
        )
end






"""
    set_workers(requested_workers::Int, threads_per_worker::Int)

Configures the distributed environment to have an exact number of workers,
each with a specific number of threads.

It will remove existing workers if they do not match the desired thread count
and create new ones. This ensures a clean and predictable environment.
"""
function set_workers(requested_workers::Int, threads_per_worker::Int)
    nproc = nprocs()
    nwork = nworkers()
    println("Info: Current environment has $nproc process(es) and $nwork worker(s).")
    println("Info: Adding $(requested_workers) new worker(s), each with $(threads_per_worker) threads.")


    println("Workers: ", workers())
    println("Process: ", procs())
    addprocs(requested_workers; exeflags=["--project", "--threads=$threads_per_worker"])
    println("Workers: ", workers())
    println("Process: ", procs())


    # main_module_file = joinpath(dirname(Base.active_project()), "src/Polfed.jl")
    main_module_file = "/home/rokpintar/projects/Polfed/src/Polfed.jl"

    println("Info: Loading module source code from `$(main_module_file)` on all workers...")
    @everywhere workers() include($main_module_file)
    println("Info: All workers are set up.")

end