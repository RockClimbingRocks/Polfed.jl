using Test
using Polfed

@testset "with_openmp_blocktime" begin
    @test Polfed.with_openmp_blocktime(() -> :value, 0) === :value
    if Polfed._iomp_available()
        before = Polfed._kmp_get_blocktime()
        @test Polfed.with_openmp_blocktime(Polfed._kmp_get_blocktime, 0) == 0
        @test Polfed._kmp_get_blocktime() == before
        @test_throws ErrorException Polfed.with_openmp_blocktime(() -> error("boom"), 0)
        @test Polfed._kmp_get_blocktime() == before
    end
end
