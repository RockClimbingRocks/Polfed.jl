"""
    Jackson(n, N)        -> Real
    Lorentz(n, N; lambda=1) -> Real
    Fejer(n, N)          -> Real
    LanczosK(n, N; M=3)  -> Real
    WangZunger(n, N; alpha=1, beta=1) -> Real
    Dirichlet(n, N)      -> Real

Kernel damping profiles for KPM Chebyshev expansions.

Each function returns the damping weight for moment index `n` in an expansion with
`N` moments. `Jackson` is the usual smooth default; `Dirichlet` applies no damping.
"""
function Jackson(n::Int, N::Int)
    return ((N - n + 1) * cos(pi * n / (N + 1)) +
            sin(pi * n / (N + 1)) * cot(pi / (N + 1))) / (N + 1)
end

"""Lorentz kernel damping weight with resolution parameter `lambda`."""
function Lorentz(n::Int, N::Int; lambda::Real = 1.0)
    return sinh(lambda * (1 - n / N)) / sinh(lambda)
end

"""Fejer kernel damping weight."""
Fejer(n::Int, N::Int) = 1 - n / N

"""Lanczos sigma-kernel damping weight."""
function LanczosK(n::Int, N::Int; M::Int = 3)
    n == 0 && return 1.0
    x = pi * n / N
    return (sin(x) / x)^M
end

"""Wang-Zunger exponential damping weight."""
WangZunger(n::Int, N::Int; alpha::Real = 1.0, beta::Real = 1.0) =
    exp(-(alpha * n / N)^beta)

"""Dirichlet kernel: no damping."""
Dirichlet(::Int, ::Int) = 1.0

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

Return the damping function registered under `kernel`.
"""
function get_kernel(kernel::Symbol)
    fn = get(KERNELS, kernel, nothing)
    fn === nothing && throw(ArgumentError(
        "Unknown KPM kernel :$kernel. Available kernels: $(sort!(collect(keys(KERNELS))))."))
    return fn
end
