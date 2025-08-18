
get_bounds_fun = Dict{Symbol, Vector{Function}}(
    :mean => [(λ̃,δ) -> λ̃ - δ , (λ̃,δ) -> λ̃ + δ ],
    :left => [(λ̃,δ) -> λ̃     , (λ̃,δ) -> λ̃ + 2δ],
    :right=> [(λ̃,δ) -> λ̃ - 2δ, (λ̃,δ) -> λ̃     ],
)


function getbounds!(
    spectral_transform::SpectralTransformConfigFull,  
    ρ::Function, 
    target_position::Symbol
)

    are_bounds_set = !isnothing(spectral_transform.left) && !isnothing(spectral_transform.right)
    is_target_set = !isnothing(spectral_transform.target)
    is_order_set = !isnothing(spectral_transform.order)
    !is_target_set && (spectral_transform.target=findmaximaldensity(ρ))


    if are_bounds_set && is_order_set
        throw(error("Both (interval and order of expension) are specified! U can only specify one of them or none!")) 
    elseif are_bounds_set
        nothing
    elseif is_order_set
        getbounds_from_K!(spectral_transform, target_position)
    else
        getbounds_from_dos!(spectral_transform, ρ, target_position)
    end
end


function findmaximaldensity(ρ::Function)
    x = LinRange(-0.99,0.99,5000)
    
    _, i  = findmax(xi->ρ(xi), x)


    println("targeting maximal densety of states at energy λ=", x[i])
    return x[i]
end

function getbounds_from_dos!(spectraltransformconfig::SpectralTransformConfigFull, 
    ρ::Function, 
    target_position::Symbol
)
    @unpack target, howmany = spectraltransformconfig
    â, b̂ = get_bounds_fun[target_position]


    bisection_fun(δ) = begin
        a = max(â(target,δ), -1.)
        b = min(b̂(target,δ),  1.)

        howmany_dos, _ = quadgk(ρ, a, b; atol=1e-12, rtol=1e-12)
        howmany_dos >= howmany && (return -1)
        howmany_dos <  howmany && (return +1)
    end

    δmin = 0.
    δmax = 1.
    δ = bisection(bisection_fun, δmin, δmax; tol=1e-8)

    spectraltransformconfig.left = max(â(target,δ), -1.)
    spectraltransformconfig.right = min(b̂(target,δ),  1.)

end


function getbounds_from_K!(
    spectraltransformconfig::SpectralTransformConfigFull, 
    target_position::Symbol
)
    @unpack order, polynomialtype, normalization, coefficients, cutoff, target = spectraltransformconfig
    â, b̂ = get_bounds_fun[target_position]

    bisection_fun(δ) = begin
        a = max(â(target,δ), -1.)
        b = min(b̂(target,δ),  1.)
        # p = Clenshaw(polynomialtype, coefficients, polynomialorder, (y,x)->*(y,x), 1, Float64)
        transform = Clenshaw(polynomialtype, n -> coefficients(target,n), order, (y,x)->*(y,x), 1, Float64)
        normalization_ = transform(target) 
        p(x::Real) = transform(x)/normalization_ * normalization

        p_a = p(a)
        p_b = p(b)

        max(p_a, p_b) >= cutoff && (return -1)
        max(p_a, p_b) <  cutoff && (return +1)
    end

    δmin = 0.
    δmax = 1.
    δ = bisection(bisection_fun, δmin, δmax; tol=1e-8)

    spectraltransformconfig.left = max(â(target,δ), -1.)
    spectraltransformconfig.right = min(b̂(target,δ), 1.)
end



# bisection_fun(order) = begin
#     order = floor(Int, order) 
#     transform = Clenshaw(type, n -> coefficients(λ,n), order, (y,x)->*(y,x), 1, Float64)
#     normalization_ = transform(λ) 
#     p(x::Real) = transform(x)/normalization_ * normalization

#     p_a = p(a) 
#     p_b = p(b)

#     max(p_a, p_b) >= cutoff && (return -1)
#     max(p_a, p_b) <  cutoff && (return +1)
# end
