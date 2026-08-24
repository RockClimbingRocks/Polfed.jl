

using BenchmarkTools, LinearAlgebra, SparseArrays, StaticArrays, Random, StaticArrays

using UniPolfed
const OPT = UniPolfed.Optimization

include("clenshaw/Clenshaw.jl")
include("../qrem.jl")            # qrem_hamiltonian


function _threaded_combine!(b1, X, b3, c, α, β)
    n = length(b1)
    n == 0 && return b1

    nt = min(Base.Threads.nthreads(:default), n)
    chunk = cld(n, nt)

    if nt == 1
        @inbounds @simd for i in eachindex(b1)
            b1[i] = c * X[i] + α * b1[i] + β * b3[i]
        end
    else
        Base.Threads.@threads :static for t in 1:nt
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, n)

            @inbounds @simd for i in lo:hi
                b1[i] = c * X[i] + α * b1[i] + β * b3[i]
            end
        end
    end

    return b1
end


L=16
bs = 4

rep = 1000

mat = qrem_hamiltonian(L; Γ=1.0, rng=MersenneTwister(42))


# ── spin_flip/packed mapping kernel ───────────────────────────────────────────
# `classify` turns the matrix into the value-grouped MappingStructure the kernels
# consume; `build_kernel` specializes the closure on the block size, so `bs` goes
# in as a `Val`. The result is the ROW-layout matvec `f!(Y, X) -> nothing`, with
# `Y`/`X` dense `bs × D` `Matrix`es (the packed kernel takes raw pointers, so a
# view or a lazy transpose is not acceptable).
kernel = OPT.MappingKernel(:spin_flip, :packed, W = 16, G = 4)

D = size(mat, 1)
s = OPT.classify(mat)
f! = OPT.build_kernel(kernel, s, Val(bs))


X = rand(Float64, bs, D)
Y = rand(Float64, bs, D)

f!(Y, X)
@btime f!(Y, X)


f!(Y, X)
time = @elapsed begin
    for i in 1:rep
        f!(Y, X)
    end
end

println("time per iter: ", round(time / rep * 1e3, digits=3), " ms")
println("time per iter per col: ", round(time / rep / bs * 1e3, digits=3), " ms")

println("-"^50)
println("-"^50)


b = [X,Y]

f!(Y, X)
time = @elapsed begin
    for i in 1:rep
        f!(b[1], b[2])
    end
end

println("time per iter: ", round(time / rep * 1e3, digits=3), " ms")
println("time per iter per col: ", round(time / rep / bs * 1e3, digits=3), " ms")


println("-"^50)
println("-"^50)

pvec = SVector{2, Int}(2,1)

b = [X,Y]

f!(Y, X)
time = @elapsed begin
    for i in 1:rep
        f!(b[1], b[2])
        permute!(b, pvec)
    end
end

println("time per iter: ", round(time / rep * 1e3, digits=3), " ms")
println("time per iter per col: ", round(time / rep / bs * 1e3, digits=3), " ms")


println("-"^50)
println("-"^50)

pvec = SVector{2, Int}(2,1)

b = [X,Y]

f!(Y, X)
time = @elapsed begin
    for i in 1:2:rep
        f!(b[1], b[2])
        
        b[2] .= b[1]
        f!(b[1], b[2])
    end
end

println("time per iter: ", round(time / rep * 1e3, digits=3), " ms")
println("time per iter per col: ", round(time / rep / bs * 1e3, digits=3), " ms")


println("-"^50)