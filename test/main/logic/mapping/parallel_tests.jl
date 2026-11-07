# Parallelization strategy resolution.
#
# The rule is stated in terms of what each strategy CAN do, not in terms of arithmetic:
#   VectorWise runs one Clenshaw per Krylov vector, so it pays off only when every thread gets
#   exactly one whole vector, and at `block_size == 1` it is serial.
#   BlockWise splits inside one recurrence step and always has a valid decomposition — worst
#   case a single group, the pure basis-state split — so it works at any thread count.
# Everything below follows from those two facts. In particular divisibility never decides the
# STRATEGY; it only affects how many groups BlockWise resolves internally.
#
# `optimize_mapping` and `nthreads` are BOTH passed explicitly throughout. `nthreads` because a
# test that reads `Threads.nthreads()` asserts something different on every machine it runs on;
# `optimize_mapping` because it sits before `nthreads` in the signature, so omitting it silently
# binds a thread count to the mapping flag and the thread count then defaults to the host's.

using Test
using Polfed
using LinearAlgebra, SparseArrays, Random
import Polfed: resolve_parallelization, warn_parallelization, with_verbosity,
                  _vector_slice, MappingConfig, polfed,
                  BlockWise, VectorWise, CPU, ColumnLayout, RowLayout

@testset "resolve_parallelization" begin

    # Two layers, tested separately because they can mask each other:
    #   the RULE decides from (block_size, optimize_mapping, nthreads);
    #   the DOWNGRADE then refuses VectorWise above one thread, because the Clenshaw combine
    #   opens a `@threads :static` region inside VectorWise's own and Julia rejects that.
    # Testing the two together would let a broken rule hide behind the downgrade.

    @testset "an explicit strategy is honoured where it can run" begin
        @test resolve_parallelization(BlockWise(), CPU(), 8, true, 8) isa BlockWise
        @test resolve_parallelization(BlockWise(groups = 4), CPU(), 8, true, 8).groups == 4
        @test resolve_parallelization(VectorWise(), CPU(), 2, true, 1) isa VectorWise
    end

    @testset "an explicit VectorWise is downgraded above one thread, not left to crash" begin
        # Silently reaching "`@threads :static` cannot be used concurrently or nested" from
        # inside a task is worse than being told the strategy was swapped.
        for nt in (2, 8, 32)
            @test resolve_parallelization(VectorWise(), CPU(), 32, true, nt) isa BlockWise
        end
    end

    @testset "block size 1 takes BlockWise — VectorWise would be serial" begin
        for nt in (1, 2, 8, 32)
            @test resolve_parallelization(nothing, CPU(), 1, true, nt) isa BlockWise
        end
    end

    @testset "an unoptimized mapping takes BlockWise — generic mul! brings its own threads" begin
        # Checked at `block_size == nthreads`, the one shape that would otherwise resolve
        # VectorWise, so it is the flag deciding here and not the arithmetic.
        for nt in (1, 2, 8, 32)
            @test resolve_parallelization(nothing, CPU(), nt, false, nt) isa BlockWise
        end
    end

    @testset "block size /= threads takes BlockWise" begin
        # VectorWise wants one whole vector per thread and nothing else; any other shape either
        # idles threads or hands a buffer between cores.
        @test resolve_parallelization(nothing, CPU(), 4, true, 8) isa BlockWise    # divides
        @test resolve_parallelization(nothing, CPU(), 8, true, 32) isa BlockWise   # divides
        @test resolve_parallelization(nothing, CPU(), 4, true, 6) isa BlockWise    # does NOT divide
        @test resolve_parallelization(nothing, CPU(), 4, true, 7) isa BlockWise    # coprime
        @test resolve_parallelization(nothing, CPU(), 32, true, 1) isa BlockWise   # fewer threads
    end

    @testset "auto never returns a strategy that cannot run" begin
        # The property that matters most: whatever the shape, an automatic choice must be
        # runnable. Above one thread that means never VectorWise — including at
        # `block_size == nthreads`, where the rule selects it and the downgrade takes it away
        # again. That pairing is why auto cannot hand out VectorWise at all today.
        for bs in (1, 2, 3, 4, 8, 16, 32), nt in (2, 3, 4, 8, 16, 32)
            @test resolve_parallelization(nothing, CPU(), bs, true, nt) isa BlockWise
        end
    end

    @testset "divisibility does not change the answer" begin
        # The rule turns on `block_size == nthreads` exactly, never on whether one divides the
        # other — a divisor and a coprime thread count land in the same place.
        @test resolve_parallelization(nothing, CPU(), 6, true, 3) isa BlockWise    # divides
        @test resolve_parallelization(nothing, CPU(), 6, true, 12) isa BlockWise   # divides
        for nt in 7:24
            @test resolve_parallelization(nothing, CPU(), 6, true, nt) isa BlockWise
        end
    end

    # No GPU testset: `GPU()` throws "CUDA is not available" on a host without a device, so the
    # `isa(pu, GPU) -> BlockWise` branch is not constructible here. It is exercised by the gated
    # GPU entrypoint tests on the aarch64/CUDA build.
end

@testset "warn_parallelization" begin

    @testset "counts Krylov vectors through the layout, not a fixed axis" begin
        # A RowLayout block is `bs x D`. Counting vectors with `size(x0, 2)` would read D — the
        # Hilbert dimension — as the vector count and warn on every run of every row-layout
        # problem. Both calls below describe the SAME problem (4 vectors, dimension 64).
        col = zeros(64, 4)
        row = zeros(4, 64)
        with_verbosity(1) do
            @test warn_parallelization(col, VectorWise(), CPU(), ColumnLayout()) === nothing
            @test warn_parallelization(row, VectorWise(), CPU(), RowLayout()) === nothing
        end
    end

    @testset "warns only when threads outnumber vectors" begin
        # More vectors than threads is the INTENDED VectorWise case (each thread takes several
        # whole vectors, nothing handed between cores) and must stay silent. Auto-resolution
        # never produces the reverse, so a warning means the strategy was set by hand.
        many = zeros(64, 128)     # 128 vectors, far more than any plausible nthreads
        with_verbosity(1) do
            @test_logs warn_parallelization(many, VectorWise(), CPU(), ColumnLayout())
        end
    end

    @testset "silent at verbosity 0" begin
        one_vec = zeros(64, 1)
        with_verbosity(0) do
            @test_logs warn_parallelization(one_vec, VectorWise(), CPU(), ColumnLayout())
        end
    end
end

@testset "BlockWise groups validation" begin
    # A user-facing knob, so the error belongs at the call site that wrote it, not deep inside
    # the decomposition that eventually reads it.
    @test BlockWise(groups = 4).groups == 4
    @test BlockWise().groups === nothing
    @test_throws ArgumentError BlockWise(groups = 0)
    @test_throws ArgumentError BlockWise(groups = -3)
end


@testset "_vector_slice is shaped for the layout's operator" begin
    # Not just "the right numbers" — the right SHAPE. Column operators take vectors; row
    # operators take a `bs x D` block, and a one-vector block is `1 x D`, still 2-D. The `i:i`
    # that keeps the axis is the whole point: `view(A, i, :)` drops it and has no `mul!` method
    # against a D x D operator, which is how VectorWise on a row block used to die inside a
    # task with an unreadable error instead of failing at the call.
    col = reshape(1.0:12.0, 4, 3) |> collect          # ColumnLayout: D=4, bs=3
    row = permutedims(col)                            # RowLayout:    bs=3, D=4

    sc = _vector_slice(ColumnLayout(), col, 2)
    @test sc isa AbstractVector && size(sc) == (4,)

    sr = _vector_slice(RowLayout(), row, 2)
    @test sr isa AbstractMatrix && size(sr) == (1, 4)

    # same Krylov vector either way, whatever the container shape
    @test vec(sr) == sc

    # a bare vector is already a single Krylov vector
    @test _vector_slice(RowLayout(), col[:, 1], 1) == col[:, 1]
end

@testset "BlockWise and VectorWise agree end to end" begin
    # Deliberately the OPERATOR entrypoint, not the matrix one: the matrix entrypoints coerce
    # VectorWise to BlockWise (see `_row_path_strategy`), so running this there would compare
    # BlockWise against itself and assert nothing.
    #
    # This is also single-threaded-only in practice — above one thread VectorWise hits the
    # concurrent `@threads :static` in the Clenshaw combine. The suite runs at one thread, so the
    # test is meaningful here and would need the recurrence made serial to extend.
    #
    # Ground truth is `eigvals`, not one strategy against the other, so a shared error cannot
    # hide: both must land on the true spectrum.
    L = 8; D = 2^L
    rng = MersenneTwister(3)
    Ii = Int[]; Jj = Int[]; Vv = Float64[]
    for i in 0:(D - 1)
        push!(Ii, i + 1); push!(Jj, i + 1); push!(Vv, randn(rng))
        for m in 0:(L - 1)
            push!(Ii, i + 1); push!(Jj, (i ⊻ (1 << m)) + 1); push!(Vv, 0.5)
        end
    end
    H = sparse(Ii, Jj, Vv, D, D)
    exact = sort(eigvals(Symmetric(Matrix(H))))
    target = exact[cld(D, 2)]
    x0 = rand(MersenneTwister(11), D, 4)

    # The DoS MUST be seeded, and identically for both runs. Its KPM probes are stochastic and
    # `DoSConfig` defaults to `Random.default_rng()`, so two unseeded calls estimate slightly
    # different densities, size slightly different target intervals, and can legitimately return
    # windows shifted by one eigenpair — which is not a difference between the STRATEGIES and
    # made this comparison intermittently fail. With the probe fixed, the two agree exactly.
    f!(Y, X) = mul!(Y, H, X)
    dosc = () -> DoSConfig(rng = MersenneTwister(2024))
    rb = polfed(f!, x0, 4, target; verbosity = 0, dosc = dosc(),
                mc = MappingConfig(parallel_strategy = BlockWise()))
    rv = polfed(f!, x0, 4, target; verbosity = 0, dosc = dosc(),
                mc = MappingConfig(parallel_strategy = VectorWise()))
    @test rb.report.mapping.parallel == "BlockWise"
    # Above one thread `resolve_parallelization` downgrades VectorWise (the Clenshaw combine
    # nests a `:static` region inside it), so the strategies genuinely differ only at nt = 1.
    @test rv.report.mapping.parallel == (Threads.nthreads() > 1 ? "BlockWise" : "VectorWise")

    # Counts must MATCH, not hit a fixed number: POLFED returns what converged within the
    # budget, so pinning it to `howmany` would make this a convergence test on a fixture chosen
    # for something else. What the strategies must agree on is the answer, not the budget.
    @test length(rb.vals) == length(rv.vals)
    @test !isempty(rb.vals)
    # Ground truth is `eigvals`, so a shared error cannot hide behind the two agreeing.
    for r in (rb, rv), v in r.vals
        @test minimum(abs.(exact .- v)) < 1e-10
    end
    @test sort(rb.vals) ≈ sort(rv.vals) atol = 1e-9
end
