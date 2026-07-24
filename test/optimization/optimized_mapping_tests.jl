import UniPolfed.Optimization: build_optimized_mapping, rescale, resolve_mapping_kernel

@testset "optimized mapping builder" begin
    L = 4
    D = 2^L
    H = qrem(L)                       # single off-diagonal value, regular structure (from fixture)
    Hd = Matrix(H)
    Emin, Emax = -12.0, 12.0
    a = (Emax - Emin) / 2             # 12
    b = (Emax + Emin) / 2             # 0

    # row reference: Yr[k,:] = X[k,:] · A  (row operator X·A)
    rowop(X, A) = X * A

    @testset "rescale structure" begin
        s = value_grouping(H)
        sr = rescale(s, a, b)
        @test sr.hx ≈ s.hx / a
        @test sr.diags ≈ (s.diags .- b) ./ a
        @test sr.nb32 == s.nb32       # sparsity pattern unchanged by rescaling
        @test sr.L == s.L && sr.dim == s.dim
    end

    @testset "f!_rescaled = H̃·X (bs×D transform kernel), forced, bs=$bs" for bs in (1, 2, 4)
        opt = build_optimized_mapping(H, :ell32_svec, Val(bs), a, b)
        @test opt.id == :ell32_svec

        # f!_rescaled feeds the Clenshaw filter — a bs×D block, rescaled: X·H̃ = (X·H − bX)/a.
        # (This is the only operator built here; the one-time map-back is a generic mul! in the
        #  POLFED matrix entrypoint, not a value-grouped kernel — see build_optimized_mapping.)
        X = randn(bs, D)
        Y = zeros(bs, D); opt.f!_rescaled(Y, X)
        @test isapprox(Y, (rowop(X, Hd) .- b .* X) ./ a; rtol = 1e-10)
    end

    @testset ":auto picks a reads kernel and is correct" begin
        opt = build_optimized_mapping(H, :auto, Val(2), a, b; secs = 0.02)
        @test opt.id in (:csr_simd, :ell32_svec, :csr64_svec)
        X = randn(2, D); Y = zeros(2, D); opt.f!_rescaled(Y, X)
        @test isapprox(Y, (rowop(X, Hd) .- b .* X) ./ a; rtol = 1e-10)
    end

    @testset "NamedTuple pins kernel + params" begin
        opt = build_optimized_mapping(H, (kernel = :csr64_svec, G = 2), Val(2), a, b)
        @test opt.id == :csr64_svec
        @test opt.G == 2
        @test resolve_mapping_kernel((kernel = :shuf, W = 8, G = 4), H, Val(1)) == (:shuf, 4, 8)
    end

    @testset "MappingConfig accepts the widened optimize_mapping" begin
        @test UniPolfed.MappingConfig(optimize_mapping = false).optimize_mapping === false
        @test UniPolfed.MappingConfig(optimize_mapping = :auto).optimize_mapping === :auto
        @test UniPolfed.MappingConfig(optimize_mapping = :ell32_svec).optimize_mapping === :ell32_svec
        @test UniPolfed.MappingConfig(optimize_mapping = (kernel = :shuf, G = 4)).optimize_mapping ==
              (kernel = :shuf, G = 4)
    end
end
