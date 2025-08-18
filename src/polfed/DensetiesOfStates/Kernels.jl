

# mutable struct Kernel
#     type::Symbol
#     params::Dict{String,<:Any}
#     ĝ::Function

#     function Kernel(type::Symbol; params::Dict{String,<:Any}=Dict{String,Any}())     
#         use_defoult_params = length(params)==0
#         ĝ(n, Ñ) = use_defoult_params ? eval(type)(n, Ñ) : eval(type)(n, Ñ; parmas)

#         new(type, params, ĝ)
#     end
# end





function Jackson(n::Int,N::Int)
    return ((N-n+1)*cos(π*n/(N+1)) + sin(π*n/(N+1))*cot(π/(N+1)))/(N+1)
end

function Lorentz(n::Int,N::Int; λ::Real=1.)
    return sinh(λ*(1-n/N))/sinh(λ)
end

function Fejer(n::Int,N::Int)
    return 1-n/N
end

function LanczosK(n::Int,N::Int; M::Int=3)
    if n==0
        return 1
    else
        return (sin(π*n/N)/(π*n/N))^M
    end
end

function WangZunger(n::Int,N::Int; α::Real=1., β::Real=1.)
    return exp(-(α*n/N)^β)
end

function Dirichlet(n::Int,N::Int)
    return 1
end

