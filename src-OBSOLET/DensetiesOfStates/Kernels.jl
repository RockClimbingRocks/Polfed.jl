

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





"""
    Jackson(n::Int, N::Int) -> Real
    Lorentz(n::Int, N::Int; λ::Real=1) -> Real
    Fejer(n::Int, N::Int) -> Real
    LanczosK(n::Int, N::Int; M::Int=3) -> Real
    WangZunger(n::Int, N::Int; α::Real=1, β::Real=1) -> Real
    Dirichlet(n::Int, N::Int) -> Real

Kernel damping profiles for KPM (Chebyshev moment) expansions.

Each function returns the damping weight applied to moment index `n` for an
expansion of order `N`.
"""
function Jackson(n::Int,N::Int)
    return ((N-n+1)*cos(π*n/(N+1)) + sin(π*n/(N+1))*cot(π/(N+1)))/(N+1)
end

function Lorentz(n::Int,N::Int; λ::Real=1.)
    return sinh(λ*(1-n/N))/sinh(λ)
end

"""Fejer kernel damping weight for moment index `n` and order `N`."""
function Fejer(n::Int,N::Int)
    return 1-n/N
end

"""Lanczos sigma-kernel damping weight with exponent `M`."""
function LanczosK(n::Int,N::Int; M::Int=3)
    if n==0
        return 1
    else
        return (sin(π*n/N)/(π*n/N))^M
    end
end

"""Wang-Zunger exponential damping weight."""
function WangZunger(n::Int,N::Int; α::Real=1., β::Real=1.)
    return exp(-(α*n/N)^β)
end

"""Dirichlet kernel (no damping)."""
function Dirichlet(n::Int,N::Int)
    return 1
end

const KERNELS = Dict{Symbol,Function}(
    :Jackson    => Jackson,
    :Lorentz    => Lorentz,
    :Fejer      => Fejer,
    :LanczosK   => LanczosK,
    :WangZunger => WangZunger,
    :Dirichlet  => Dirichlet,
)

"""
    get_kernel(kernel::Symbol) -> Function

Return the kernel weighting function registered under `kernel`.

# Throws
- `ArgumentError`: If the kernel symbol is unknown.
"""
@inline function get_kernel(kernel::Symbol)
    fn = get(KERNELS, kernel, nothing)
    fn === nothing && throw(ArgumentError("Unknown kernel: $(kernel). Available: $(collect(keys(KERNELS)))"))
    return fn
end
