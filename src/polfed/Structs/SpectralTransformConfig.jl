
mutable struct SpectralTransformConfig
    coefficients::Function
    normalization::Real
    cutoff::Real
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    order::Union{Int, Nothing}
    order_safety_factor::Real
    parallelization::Parallelization
    f!_rescaled::Union{Nothing, Function}   
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    overestimate_iters::Real

    function SpectralTransformConfig(; 
        coefficients::Function                          =   PolfedDefaults.coefficients, 
        normalization::Real                             =   PolfedDefaults.normalization, 
        cutoff::Real                                    =   PolfedDefaults.cutoff, 
        left::Union{Real, Nothing}                     =   nothing,
        right::Union{Real, Nothing}                    =   nothing,
        order::Union{Integer, Nothing}                  =   nothing,
        order_safety_factor::Real                       =   PolfedDefaults.order_safety_factor,
        parallelization::Parallelization                =   PolfedDefaults.parallelization,
        f!_rescaled::Union{Nothing, Function}           =   nothing,
        clenshaw_recurrence::Union{Nothing, Function}   =   nothing,
        clenshaw_finalsum::Union{Nothing, Function}     =   nothing,
        overestimate_iters::Real                        =   PolfedDefaults.overestimate_iters,
    )
        new(coefficients, normalization, cutoff, left, right, order, order_safety_factor, parallelization, f!_rescaled, clenshaw_recurrence, clenshaw_finalsum, overestimate_iters)
    end
end



mutable struct SpectralTransformConfigFull
    coefficients::Function
    normalization::Real
    polynomialtype::Symbol
    cutoff::Real
    target::Union{Real, Nothing}
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    howmany::Integer
    order::Union{Integer, Nothing}
    order_safety_factor::Real
    parallelization::Parallelization
    f!::Function
    f!_rescaled::Union{Function,Nothing}
    f!_transformed::Union{Function,Nothing}
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    overestimate_iters::Real 


    function SpectralTransformConfigFull(
        cfg::SpectralTransformConfig,
        f!::Function,
        howmany::Integer,
        target::Real,
        pu::ProcessingUnit,
    )
        parallelization = nothing
        if isa(cfg.parallelization, Nothing)
            isa(pu, CPU) && (parallelization = MulColsParallel())
            isa(pu, GPU) && (parallelization = NoParallel())
        else
            parallelization = cfg.parallelization
        end

        f!_rescaled = nothing 
        if isa(cfg.f!_rescaled, Nothing)


            
            println("THIS IS NOT OK YET!!!!")
            Emin = lanczos(f!, x0, 1; which=:smallest, maxdim=100)
            Emax = lanczos(f!, x0, 1; which=:largest,  maxdim=100)
            # Emax = +400.
            # Emin = -400.

            a = (Emax-Emin)/2
            b = (Emax+Emin)/2

            f!_rescaled_fun(Y::AbstractVecOrMat,X::AbstractVecOrMat) = begin
                f!(Y,X) 
                @. Y *= 1/a
                @. Y -= (b/a)*X
            end

            f!_rescaled = f!_rescaled_fun
        end


        new(
            cfg.coefficients,
            cfg.normalization,
            PolfedDefaults.polynomialtype,
            cfg.cutoff,
            target,
            cfg.left,
            cfg.right,
            howmany,
            cfg.order,
            cfg.order_safety_factor,
            cfg.parallelization,
            f!,
            f!_rescaled,
            nothing,
            cfg.clenshaw_recurrence,
            cfg.clenshaw_finalsum,
            cfg.overestimate_iters,
        )
    end
end



