# Boundary-condition robustness: unlike QREM's σˣ (single-site, boundary-agnostic), a coupled
# model's off-diagonal structure changes with the boundary condition. The σˣσˣ chain is the clean
# Stage-1-compatible case (regular, single value): periodic has L bonds → L neighbours/row, open
# has L-1. This checks the general reads kernels against mul! for BOTH boundaries. (The shuf flip
# kernel is σˣ single-bit only, so it is not exercised here.)

"""
    xx_chain(L; boundary=:periodic, J=1.0, T=Float64) -> SparseMatrixCSC

`diag(ε) + J·Σ σᵢˣσᵢ₊₁ˣ` on `L` spins. Each bond flips two adjacent bits, so every basis state
has exactly (#bonds) off-diagonal neighbours — `L` for `:periodic`, `L-1` for `:open` — all with
value `J` (regular, single-value → Stage-1 compatible).
"""
function xx_chain(L::Int; boundary::Symbol = :periodic, J = 1.0, T = Float64)
    D = 2^L
    diagv = randn(T, D)
    bonds = Tuple{Int,Int}[]                       # 0-based site pairs
    for i in 0:L-2
        push!(bonds, (i, i + 1))
    end
    boundary === :periodic && L > 2 && push!(bonds, (L - 1, 0))
    I = Int[]; Jc = Int[]; V = T[]
    for s in 0:D-1
        push!(I, s + 1); push!(Jc, s + 1); push!(V, diagv[s + 1])
        for (a, b) in bonds
            t = xor(s, (1 << a) | (1 << b))
            push!(I, s + 1); push!(Jc, t + 1); push!(V, T(J))
        end
    end
    return sparse(I, Jc, V, D, D)
end

@testset "boundary conditions (σˣσˣ chain)" begin
    L = 4
    D = 2^L
    for boundary in (:periodic, :open)
        expected_L = boundary === :periodic ? L : L - 1
        for T in (Float64, ComplexF64)
            H = xx_chain(L; boundary = boundary, T = T)
            s = value_grouping(H)
            @test s.L == expected_L                # neighbour count tracks the boundary condition
            @test s.hx ≈ T(1.0)
            for bs in (1, 2, 4), G in (1, 2)
                Xr = randn(T, bs, D)
                Yref = mul_row_ref(H, Xr)
                for id in (:csr_simd, :ell32_svec, :csr64_svec)
                    m! = build_row_mapping(id, s, Val(bs); G = G)
                    Yr = zeros(T, bs, D)
                    m!(Yr, Xr)
                    @test isapprox(Yr, Yref; rtol = 1e-10)
                end
            end
        end
    end
end
