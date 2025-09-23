
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




