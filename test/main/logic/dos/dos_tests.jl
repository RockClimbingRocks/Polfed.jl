using Test
using UniPolfed
using Random
using LinearAlgebra

K = UniPolfed.Numerics.KPM

_dos_test_cheb(x, k) = cos(k * acos(x))

@testset "DoS" begin
    @testset "DoSConfig keyword overrides" begin
        cfg = UniPolfed.DoSConfig(N = 100, R = 50, kernel = :Dirichlet)
        @test cfg.N == 100
        @test cfg.R == 50
        @test cfg.kernel == :Dirichlet
    end

    @testset "a seeded rng pins the stochastic probes" begin
        # The whole point of the field: without it two runs on the SAME operator resolve
        # slightly different windows, which is invisible until a target lands somewhere else.
        # `H = c·I` is zero-variance, so a random operator is needed for the probes to matter.
        dim = 64
        Random.seed!(20250101)
        A = Symmetric(randn(dim, dim) ./ (4 * sqrt(dim)))
        f!(y, x) = mul!(y, A, x)
        sys = UniPolfed.SysInfo(zeros(Float64, dim), UniPolfed.CPU())

        seeded(seed) = UniPolfed.calculate_dos(
            UniPolfed.DoSConfig(N = 32, R = 4, rng = MersenneTwister(seed)), f!, sys).moments

        @test seeded(7) == seeded(7)                    # bit-identical, not just ≈
        @test seeded(7) != seeded(8)                    # a different seed really does differ

        unseeded() = UniPolfed.calculate_dos(UniPolfed.DoSConfig(N = 32, R = 4), f!, sys).moments
        @test unseeded() != unseeded()                  # the default stays stochastic

        # The generator type is a parameter, so the config stays concretely typed.
        @test UniPolfed.DoSConfig(rng = MersenneTwister(1)) isa
              UniPolfed.DoSConfig{MersenneTwister}
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
