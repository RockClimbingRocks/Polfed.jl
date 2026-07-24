@testset "reads kernels vs mul!" begin
    L = 4
    D = 2^L
    for T in (Float64, ComplexF64)
        H = qrem(L; T = T)
        s = value_grouping(H)
        for bs in (1, 2, 4), G in (1, 2, 4)
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
