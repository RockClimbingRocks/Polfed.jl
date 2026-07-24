using Test
using UniPolfed

K = UniPolfed.Numerics.KPM

_dos_test_cheb(x, k) = cos(k * acos(x))

@testset "DoS" begin
    @testset "DoSConfig keyword overrides" begin
        cfg = UniPolfed.DoSConfig(N = 100, R = 50, kernel = :Dirichlet)
        @test cfg.N == 100
        @test cfg.R == 50
        @test cfg.kernel == :Dirichlet
    end

    @testset "calculate_dos returns moments, coefficients, and density" begin
        c = 0.3
        dim = 16
        cfg = UniPolfed.DoSConfig(N = 64, R = 8, kernel = :Jackson)
        sys = UniPolfed.SysInfo(zeros(Float64, dim), UniPolfed.CPU())
        f!(y, x) = (y .= c .* x)

        res = UniPolfed.calculate_dos(cfg, f!, sys)

        exact = [dim * _dos_test_cheb(c, k) for k in 0:cfg.N-1]
        @test res isa UniPolfed.DoSResult
        @test res.moments ≈ exact rtol = 1e-10
        @test res.coefficients ≈ K.kpm_coeffs(res.moments, cfg.kernel)

        ρ_ref = K.kpm_density(res.moments; kernel = cfg.kernel)
        @test res.ρ(0.1) ≈ ρ_ref(0.1)
        @test res.ρ(-0.4) ≈ ρ_ref(-0.4)
    end

    @testset "calculate_dos uses SysInfo probe shape" begin
        c = 0.4
        dim = 10
        block = 2
        cfg = UniPolfed.DoSConfig(N = 8, R = 5)
        sys = UniPolfed.SysInfo(zeros(Float64, dim, block), UniPolfed.CPU())
        shapes = Set{Tuple{Int,Int}}()
        f!(Y, X) = (push!(shapes, size(X)); Y .= c .* X)

        res = UniPolfed.calculate_dos(cfg, f!, sys)

        exact = [dim * _dos_test_cheb(c, k) for k in 0:cfg.N-1]
        @test res.moments ≈ exact
        @test shapes == Set([(dim, block)])
    end

    @testset "kernel changes coefficients and density but not moments" begin
        c = 0.25
        dim = 12
        sys = UniPolfed.SysInfo(zeros(Float64, dim), UniPolfed.CPU())
        f!(y, x) = (y .= c .* x)

        res_j = UniPolfed.calculate_dos(UniPolfed.DoSConfig(N = 64, R = 8, kernel = :Jackson), f!, sys)
        res_d = UniPolfed.calculate_dos(UniPolfed.DoSConfig(N = 64, R = 8, kernel = :Dirichlet), f!, sys)

        @test res_j.moments ≈ res_d.moments
        @test !isapprox(res_j.coefficients, res_d.coefficients)
        @test !isapprox(res_j.ρ(c), res_d.ρ(c))
    end
end
