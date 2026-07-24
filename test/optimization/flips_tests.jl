@testset "shuf flip kernel vs mul!" begin
    L = 4
    D = 2^L
    H = qrem(L)                       # Float64 (shuf is real-only)
    s = value_grouping(H)

    # (bs, W, G): bs, W powers of two, bs ≤ W, D % (W÷bs) == 0, L ≥ log2(W÷bs)
    combos = [(1, 4, 1), (1, 4, 2), (1, 8, 1), (1, 8, 4),
              (2, 4, 1), (2, 4, 2), (2, 8, 2),
              (4, 4, 1), (4, 8, 1)]
    for (bs, W, G) in combos
        Xr = randn(Float64, bs, D)
        Yref = mul_row_ref(H, Xr)
        m! = build_row_mapping(:shuf, s, Val(bs); W = W, G = G)
        Yr = zeros(Float64, bs, D)
        m!(Yr, Xr)
        @test isapprox(Yr, Yref; rtol = 1e-10)
    end
end
