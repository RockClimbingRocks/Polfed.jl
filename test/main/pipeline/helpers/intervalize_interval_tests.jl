using Logging
using Test
using UniPolfed: Common, DoSResult, Interval, integrate_chebyshev_weighted,
              intervalize_interval, intervalize_interval_overlapping

function _pipeline_uniform_dos(total_states::Real = 1000)
    coefficients = [Float64(total_states)]
    density = x -> Float64(total_states) / (π * sqrt(1 - x^2))
    return DoSResult(copy(coefficients), coefficients, density)
end

_pipeline_state_count(dos, interval) =
    integrate_chebyshev_weighted(
        dos.coefficients, interval.left, interval.right,
    ) / π

@testset "interior intervalization" begin
    dos = _pipeline_uniform_dos()
    request = Interval(-0.6, 0.6)
    howmany = 100
    patches = intervalize_interval(dos, request, howmany)

    @test first(patches).left == request.left
    @test last(patches).right == request.right
    @test all(patches[i].right == patches[i + 1].left for i in 1:length(patches)-1)
    counts = _pipeline_state_count.(Ref(dos), patches)
    @test all(count -> isapprox(count, howmany; rtol = 1e-5), counts[1:end-1])
    @test 0 < last(counts) <= howmany * (1 + 1e-5)

    overlap = intervalize_interval_overlapping(dos, request, howmany, 20)
    @test overlap.unpadded == patches
    @test all(
        overlap.padded[i].right > overlap.padded[i + 1].left
        for i in 1:length(overlap.padded)-1
    )
    @test intervalize_interval_overlapping(
        dos, request, howmany, 0,
    ).padded == patches

    @test_throws ArgumentError intervalize_interval(dos, request, 0)
    @test_throws ArgumentError intervalize_interval_overlapping(
        dos, request, howmany, -1,
    )
end

@testset "undersized final patch is expected and warning-free" begin
    dos = _pipeline_uniform_dos(115)
    request = Interval(-1.0, 1.0)
    logger = Test.TestLogger()
    saved_verbosity = Common.verbosity[]
    Common.verbosity[] = Common.POLFED_WARN_LEVEL
    try
        patches = with_logger(logger) do
            intervalize_interval(dos, request, 50)
        end
        counts = _pipeline_state_count.(Ref(dos), patches)
        @test length(patches) == 3
        @test all(isapprox(count, 50; rtol = 1e-5) for count in counts[1:2])
        @test isapprox(counts[3], 15; rtol = 1e-5)
        @test isapprox(sum(counts), 115; rtol = 1e-5)
    finally
        Common.verbosity[] = saved_verbosity
    end
    @test !any(
        record -> occursin("widest window around the target", string(record.message)),
        logger.logs,
    )
end
