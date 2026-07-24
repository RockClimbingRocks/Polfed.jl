


using Test
using LinearAlgebra, SparseArrays, Random
using UniPolfed: run_patched_polfed_window_pipeline



function _qrem(L::Int; hx = 0.7, T = Float64)
    D = 2^L
    diagv = randn(T, D)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, T(hx))
        end
    end
    return sparse(I, J, V, D, D)
end



@testset "run_patched_polfed_window_pipeline - TMP" begin

    L = 10; D = 2^L
    H = _qrem(L)
    # vals_exact, vecs_exact = @time eigen(Symmetric(Matrix(H)))
    vals_exact = @time sort(eigvals(Symmetric(Matrix(H))))
    
    
    f!(Y,X) = mul!(Y, H, X)
    x0 = rand(D); x0 = x0 / norm(x0)
    howmany_per_patch = 50
    iv = (-0.95, 0.995)
    target = (:rescaled, iv)

    vals, vecs, report = @time run_patched_polfed_window_pipeline(f!, x0, howmany_per_patch, target)

    display(report)
    is_exact = findall(e-> Base.isbetween(iv[1], e,  iv[2]), vals_exact)

    # println("vals       = ", round.(vals,digits=5))
    # println("vals exact = ", round.(vals_exact[is_exact], digits=5))
    

    i_start = findfirst(isapprox(vals[1]; atol=1e-10), vals_exact)
    vals_exact_in = vals_exact[i_start:i_start+length(vals)-1]

    @test isapprox(vals_exact_in[1], vals[1]; atol=1e-10)
    @test all(isapprox.(vals_exact_in, vals; atol=1e-10))

end

