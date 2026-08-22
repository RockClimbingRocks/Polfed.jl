# Parallelization strategy resolution.
#
# The rule is stated in terms of what each strategy CAN do, not in terms of arithmetic:
#   VectorWise runs one Clenshaw per Krylov vector, so it can never occupy more than
#   `block_size` threads, and at `block_size == 1` it is serial.
#   BlockWise splits inside one recurrence step and always has a valid decomposition — worst
#   case a single group, the pure basis-state split — so it works at any thread count.
# Everything below follows from those two facts. In particular divisibility never decides the
# STRATEGY; it only affects how many groups BlockWise resolves internally.
#
# `nthreads` is passed explicitly throughout: a test that reads `Threads.nthreads()` asserts
# something different on every machine it runs on.

using Test
using UniPolfed
import UniPolfed: resolve_parallelization, warn_parallelization, with_verbosity,
                  _matrix_path_strategy, BlockWise, VectorWise, CPU, ColumnLayout, RowLayout

@testset "resolve_parallelization" begin

    @testset "an explicit strategy is returned untouched" begin
        # Including the combinations auto-resolution would never produce, which is the point:
        # the user overrides the rule, the rule does not override the user.
        @test resolve_parallelization(BlockWise(), CPU(), 8, 8) isa BlockWise
        @test resolve_parallelization(VectorWise(), CPU(), 2, 32) isa VectorWise
        @test resolve_parallelization(BlockWise(groups = 4), CPU(), 8, 8).groups == 4
    end

    @testset "block size 1 takes BlockWise — VectorWise would be serial" begin
        for nt in (1, 2, 8, 32)
            @test resolve_parallelization(nothing, CPU(), 1, nt) isa BlockWise
        end
    end

    @testset "threads <= block size takes VectorWise — every thread gets a whole vector" begin
        @test resolve_parallelization(nothing, CPU(), 8, 8) isa VectorWise    # one each
        @test resolve_parallelization(nothing, CPU(), 16, 4) isa VectorWise   # four each
        @test resolve_parallelization(nothing, CPU(), 32, 1) isa VectorWise   # single thread
    end

    @testset "threads > block size takes BlockWise — VectorWise would idle threads" begin
        @test resolve_parallelization(nothing, CPU(), 4, 8) isa BlockWise     # divides
        @test resolve_parallelization(nothing, CPU(), 8, 32) isa BlockWise    # divides
        @test resolve_parallelization(nothing, CPU(), 4, 6) isa BlockWise     # does NOT divide
        @test resolve_parallelization(nothing, CPU(), 4, 7) isa BlockWise     # coprime
    end

    @testset "divisibility does not change the answer" begin
        # The pairs below straddle every divisibility relation at a fixed block size; the
        # strategy depends only on which side of `block_size` the thread count falls.
        bs = 6
        for nt in 1:bs
            @test resolve_parallelization(nothing, CPU(), bs, nt) isa VectorWise
        end
        for nt in (bs + 1):(4 * bs)
            @test resolve_parallelization(nothing, CPU(), bs, nt) isa BlockWise
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

@testset "_matrix_path_strategy" begin
    # The matrix entrypoints build their transform at a fixed Val(bs), so VectorWise — which
    # hands the operator 1xD slices — cannot run against the OPTIMIZED kernel. It is fine
    # against the generic mul! row operator, which takes strided views.
    #
    # The point of the function is that an explicit request is never silently dropped: it is
    # honoured, or refused with the reason. Before this, `mc.parallel_strategy` was overwritten
    # unconditionally and a user asking for VectorWise got BlockWise with no indication.
    @test _matrix_path_strategy(nothing, true) isa BlockWise
    @test _matrix_path_strategy(nothing, false) isa BlockWise
    @test _matrix_path_strategy(BlockWise(), true) isa BlockWise
    @test _matrix_path_strategy(BlockWise(groups = 2), true).groups == 2
    @test _matrix_path_strategy(VectorWise(), false) isa VectorWise
    @test_throws ArgumentError _matrix_path_strategy(VectorWise(), true)
    @test_throws ArgumentError _matrix_path_strategy(VectorWise(), :universal)
end
