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

    @testset "calculate_dos probes ONE vector, whatever the block size" begin
        # The operator must see a single Krylov vector, never the `dim × block` Krylov block:
        # that is what makes the estimated density independent of the block size, and it is why
        # the DoS is handed the generic `dos_operator` rather than the whole-block transform.
        c = 0.4
        dim = 10
        cfg = UniPolfed.DoSConfig(N = 8, R = 5)
        exact = [dim * _dos_test_cheb(c, k) for k in 0:cfg.N-1]

        for block in (1, 2, 4)
            sys = UniPolfed.SysInfo(zeros(Float64, dim, block), UniPolfed.CPU())
            shapes = Set{Tuple}()
            f!(Y, X) = (push!(shapes, size(X)); Y .= c .* X)

            res = UniPolfed.calculate_dos(cfg, f!, sys)

            @test res.moments ≈ exact
            @test shapes == Set([(dim,)])   # a length-dim vector, never (dim, block)
        end
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

    @testset "the DoS does not depend on the Krylov block size" begin
        # The density is a property of H alone, so a seeded run must give the SAME answer at
        # every block size. It did not: the probe template was `sys.x0`, so `randn!` filled a
        # `dim × bs` block in one draw — a different random stream from `bs` single-vector
        # draws — and one seed produced a different density, interval and eigenpair set per
        # block size. The DoS now probes one vector at a time (`_dos_probe_template`).
        H = UniPolfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.7, 0.5; rng = MersenneTwister(7))
        D = size(H, 1)
        seeded() = UniPolfed.DoSConfig(rng = MersenneTwister(2024))

        reference = nothing
        for bs in (1, 2, 4, 8)
            x0 = bs == 1 ? normalize!(rand(MersenneTwister(5), D)) :
                           Matrix(qr(rand(MersenneTwister(5), D, bs)).Q)
            vals = collect(polfed(H, x0, 12, :maxdos; verbosity = 0, dosc = seeded()).vals)
            if isnothing(reference)
                reference = vals
                @test !isempty(reference)
            else
                @test length(vals) == length(reference)
                @test vals ≈ reference atol = 1e-10
            end
        end

        # …and the same call twice with the same seed is bit-for-bit reproducible.
        x0 = Matrix(qr(rand(MersenneTwister(5), D, 4)).Q)
        v1 = polfed(H, x0, 12, :maxdos; verbosity = 0, dosc = seeded()).vals
        v2 = polfed(H, x0, 12, :maxdos; verbosity = 0, dosc = seeded()).vals
        @test length(v1) == length(v2)
        @test v1 == v2

        # The optimized kernel path shares the same generic DoS operator, so it agrees too.
        v3 = polfed(H, x0, 12, :maxdos; verbosity = 0, dosc = seeded(),
                    mc = UniPolfed.MappingConfig(optimize_mapping = true)).vals
        @test length(v3) == length(reference)
        @test v3 ≈ reference atol = 1e-10
    end

    @testset "the probe template is one Krylov vector, shaped by the layout" begin
        L = UniPolfed.Numerics.Lanczos
        # column layout: a length-dim vector, the shape `mul!(y, H, x)` takes
        sysc = UniPolfed.SysInfo(rand(16, 4), UniPolfed.CPU(), L.ColumnLayout())
        @test size(UniPolfed._dos_probe_template(sysc)) == (16,)
        # row layout: a 1 × dim matrix, the shape `mul!(Y, X, H)` takes
        sysr = UniPolfed.SysInfo(rand(4, 16), UniPolfed.CPU(), L.RowLayout())
        @test size(UniPolfed._dos_probe_template(sysr)) == (1, 16)
        # a vector x0 is already one Krylov vector
        sysv = UniPolfed.SysInfo(rand(16), UniPolfed.CPU(), L.ColumnLayout())
        @test size(UniPolfed._dos_probe_template(sysv)) == (16,)
    end
end
