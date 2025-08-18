
function analytical_orderofexapnsion(λ::Real, δ::Real)
    return 2.65499*√(1-λ^2)/δ
end



function getorderofexapnsion!(spectral_transform::SpectralTransformConfigFull)
    is_order_set = !isnothing(spectral_transform.order)

    !is_order_set && (getorderofexapnsion_in_interval!(spectral_transform))
end


#HERE i need to corrrect it because i leav a and b as thy should be, but i decres K as i want!.... than a and b go furher away 
function getorderofexapnsion_in_interval!(spectral_transform::SpectralTransformConfigFull)
    @unpack coefficients, polynomialtype, normalization, cutoff, a, b, order_safety_factor = spectral_transform
    target = (a+b)/2
    halfwidth = (b-a)/2

    # coeffs = map(n -> coefficients(λ,n,Kmax), 0:Kmax)
    # coeffs_ = @view(coeffs[1:K+1])
    # transform = ClenshawMapping(coeffs_, polynomialtype, (y,x)->*(y,x), 1, Float64)
    bisection_fun(order) = begin
        order = floor(Int, order) 
        transform = Clenshaw(polynomialtype, n -> coefficients(target,n), order, (y,x)->*(y,x), 1, Float64)
        normalization_ = transform(target) 
        p(x::Real) = transform(x)/normalization_ * normalization
    
        p_a = p(a) 
        p_b = p(b)

        max(p_a, p_b) >= cutoff && (return -1)
        max(p_a, p_b) <  cutoff && (return +1)
    end

    Kmax = ceil(Int, analytical_orderofexapnsion(target, halfwidth)*2)
    Kmin = 10
    K = bisection(bisection_fun, Kmin, Kmax; tol=1)

    spectral_transform.order = floor(Int64, K*order_safety_factor)

    println("Before resetting: left = $left, right = $right")
    (order_safety_factor < 1.) && (getbounds_from_K!(spectral_transform, :mean); println("After resetting: left = $(spectral_transform.left), right = $(spectral_transform.right)"))

end
