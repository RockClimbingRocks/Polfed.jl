using Test
using UniPolfed

@testset "with_openmp_blocktime" begin
    @test UniPolfed.with_openmp_blocktime(() -> :value, 0) === :value
    if UniPolfed._iomp_available()
        before = UniPolfed._kmp_get_blocktime()
        @test UniPolfed.with_openmp_blocktime(UniPolfed._kmp_get_blocktime, 0) == 0
        @test UniPolfed._kmp_get_blocktime() == before
        @test_throws ErrorException UniPolfed.with_openmp_blocktime(() -> error("boom"), 0)
        @test UniPolfed._kmp_get_blocktime() == before
    end
end
