using Test
using Polfed
using Random
using LinearAlgebra

K = Polfed.Numerics.KPM

_dos_test_cheb(x, k) = cos(k * acos(x))

@testset "DoS" begin
    @testset "DoSConfig keyword overrides" begin
        cfg = Polfed.DoSConfig(N = 100, R = 50, kernel = :Dirichlet)
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
        sys = Polfed.SysInfo(zeros(Float64, dim), Polfed.CPU())

        seeded(seed) = Polfed.calculate_dos(
            Polfed.DoSConfig(N = 32, R = 4, rng = MersenneTwister(seed)), f!, sys).moments

        @test seeded(7) == seeded(7)                    # bit-identical, not just ≈
        @test seeded(7) != seeded(8)                    # a different seed really does differ

        unseeded() = Polfed.calculate_dos(Polfed.DoSConfig(N = 32, R = 4), f!, sys).moments
        @test unseeded() != unseeded()                  # the default stays stochastic

        # The generator type is a parameter, so the config stays concretely typed.
        @test Polfed.DoSConfig(rng = MersenneTwister(1)) isa
              Polfed.DoSConfig{MersenneTwister}
    end

    @testset "calculate_dos returns moments, coefficients, and density" begin
        c = 0.3
        dim = 16
        cfg = Polfed.DoSConfig(N = 64, R = 8, kernel = :Jackson)
        sys = Polfed.SysInfo(zeros(Float64, dim), Polfed.CPU())
        f!(y, x) = (y .= c .* x)

        res = Polfed.calculate_dos(cfg, f!, sys)

        exact = [dim * _dos_test_cheb(c, k) for k in 0:cfg.N-1]
        @test res isa Polfed.DoSResult
        @test res.moments ≈ exact rtol = 1e-10
        @test res.coefficients ≈ K.kpm_coeffs(res.moments, cfg.kernel)

        ρ_ref = K.kpm_density(res.moments; kernel = cfg.kernel)
        @test res.ρ(0.1) ≈ ρ_ref(0.1)
        @test res.ρ(-0.4) ≈ ρ_ref(-0.4)
    end

    @testset "calculate_dos probes ONE vector by default, whatever the Krylov block size" begin
        # The operator never sees the `dim × block` Krylov block — the DoS width is its own, and
        # on the operator entrypoints it defaults to a single vector (`f!` is only promised that).
        c = 0.4
        dim = 10
        cfg = Polfed.DoSConfig(N = 8, R = 5)
        exact = [dim * _dos_test_cheb(c, k) for k in 0:cfg.N-1]

        for block in (1, 2, 4)
            sys = Polfed.SysInfo(zeros(Float64, dim, block), Polfed.CPU())
            shapes = Set{Tuple}()
            f!(Y, X) = (push!(shapes, size(X)); Y .= c .* X)

            res = Polfed.calculate_dos(cfg, f!, sys)

            @test res.moments ≈ exact
            @test shapes == Set([(dim,)])   # a length-dim vector, never (dim, block)
        end
    end

    @testset "kernel changes coefficients and density but not moments" begin
        c = 0.25
        dim = 12
        sys = Polfed.SysInfo(zeros(Float64, dim), Polfed.CPU())
        f!(y, x) = (y .= c .* x)

        res_j = Polfed.calculate_dos(Polfed.DoSConfig(N = 64, R = 8, kernel = :Jackson), f!, sys)
        res_d = Polfed.calculate_dos(Polfed.DoSConfig(N = 64, R = 8, kernel = :Dirichlet), f!, sys)

        @test res_j.moments ≈ res_d.moments
        @test !isapprox(res_j.coefficients, res_d.coefficients)
        @test !isapprox(res_j.ρ(c), res_d.ρ(c))
    end

    @testset "the DoS does not depend on the Krylov block size" begin
        # The density is a property of H alone, so a seeded run must give the SAME answer at
        # every block size. It did not: the probe template was `sys.x0`, so `randn!` filled a
        # `dim × bs` block in one draw — a different random stream from `bs` single-vector
        # draws — and one seed produced a different density, interval and eigenpair set per
        # block size. The DoS now draws its probes one at a time into its own block width.
        H = Polfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.7, 0.5; rng = MersenneTwister(7))
        D = size(H, 1)
        seeded() = Polfed.DoSConfig(rng = MersenneTwister(2024))

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
                    mc = Polfed.MappingConfig(optimize_mapping = true)).vals
        @test length(v3) == length(reference)
        @test v3 ≈ reference atol = 1e-10
    end

    @testset "the probe template is the DoS block, shaped by the layout" begin
        L = Polfed.Numerics.Lanczos
        # column layout: dim × block, a plain vector for one probe (what `mul!(y, H, x)` takes)
        sysc = Polfed.SysInfo(rand(16, 4), Polfed.CPU(), L.ColumnLayout())
        @test size(Polfed._dos_probe_template(sysc, 1)) == (16,)
        @test size(Polfed._dos_probe_template(sysc, 3)) == (16, 3)
        # row layout: block × dim (what `mul!(Y, X, H)` takes)
        sysr = Polfed.SysInfo(rand(4, 16), Polfed.CPU(), L.RowLayout())
        @test size(Polfed._dos_probe_template(sysr, 1)) == (1, 16)
        @test size(Polfed._dos_probe_template(sysr, 8)) == (8, 16)
        # the width is the DoS's own, not the Krylov block's
        sysv = Polfed.SysInfo(rand(16), Polfed.CPU(), L.ColumnLayout())
        @test size(Polfed._dos_probe_template(sysv, 5)) == (16, 5)
    end

    @testset "block_size and tasks: overrides, validation, auto resolution" begin
        cfg = Polfed.DoSConfig(block_size = 4, tasks = 3)
        @test (cfg.block_size, cfg.tasks) == (4, 3)
        @test_throws ArgumentError Polfed.DoSConfig(block_size = 0)
        @test_throws ArgumentError Polfed.DoSConfig(tasks = 0)

        # matrix entrypoints own the operator: auto → the DoS default width, all threads
        owned = Polfed._owned_operator_dos(Polfed.DoSConfig(N = 7, R = 9, kernel = :Dirichlet))
        @test (owned.N, owned.R, owned.kernel) == (7, 9, :Dirichlet)
        @test owned.block_size == Polfed.Defaults.polfed.dos_block_size
        @test owned.tasks == Threads.nthreads()
        kept = Polfed._owned_operator_dos(cfg)
        @test (kept.block_size, kept.tasks) == (4, 3)

        # what calculate_dos runs with: auto → 1 (operator entrypoints); width clamped to R and
        # below dim; one probe and one task on a GPU
        sys = Polfed.SysInfo(zeros(10), Polfed.CPU())
        @test Polfed._dos_parallelism(Polfed.DoSConfig(), sys) == (1, 1)
        @test Polfed._dos_parallelism(Polfed.DoSConfig(block_size = 8, tasks = 2), sys) == (8, 2)
        @test Polfed._dos_parallelism(Polfed.DoSConfig(block_size = 16), sys) == (9, 1)
        @test Polfed._dos_parallelism(Polfed.DoSConfig(R = 3, block_size = 8), sys) == (3, 1)
    end

    @testset "tasks and block width leave the density unchanged" begin
        H = Polfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.7, 0.5; rng = MersenneTwister(7))
        D = size(H, 1)
        a = opnorm(Matrix(H), 2) * 1.01
        f!(Y, X) = (mul!(Y, X, H); Y ./= a; nothing)
        sys = Polfed.SysInfo(zeros(4, D), Polfed.CPU(), Polfed.Numerics.Lanczos.RowLayout())
        μ(bs, nt) = Polfed.calculate_dos(
            Polfed.DoSConfig(N = 40, R = 21, rng = MersenneTwister(1), block_size = bs, tasks = nt),
            f!, sys).moments

        @test μ(8, 1) == μ(8, 3)             # task count: bit-identical
        @test μ(8, 1) ≈ μ(1, 1) rtol = 1e-12 # width: same probes, the product rounds per width
    end

    @testset "calculate_dos restores the BLAS thread count" begin
        before = LinearAlgebra.BLAS.get_num_threads()
        sys = Polfed.SysInfo(zeros(Float64, 12), Polfed.CPU())
        Polfed.calculate_dos(Polfed.DoSConfig(N = 8, R = 3), (y, x) -> (y .= 0.2 .* x), sys)
        @test LinearAlgebra.BLAS.get_num_threads() == before
    end
end
