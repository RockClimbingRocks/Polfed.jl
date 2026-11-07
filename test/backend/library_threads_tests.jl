using Test
using Polfed
using LinearAlgebra

@testset "with_serial_libraries" begin
    before_blas = BLAS.get_num_threads()
    before_mkl = Polfed._mkl_max_threads()

    @testset "pins BLAS and MKL to one thread inside, returns f's value" begin
        inside = Polfed.with_serial_libraries() do
            (BLAS.get_num_threads(), Polfed._mkl_max_threads(), :value)
        end
        @test inside[1] == 1
        @test inside[2] == (before_mkl == 0 ? 0 : 1)   # 0 = MKL unavailable on this platform
        @test inside[3] === :value
        @test BLAS.get_num_threads() == before_blas
        @test Polfed._mkl_max_threads() == before_mkl
    end

    @testset "restores both counts when f throws" begin
        @test_throws ErrorException Polfed.with_serial_libraries(() -> error("boom"))
        @test BLAS.get_num_threads() == before_blas
        @test Polfed._mkl_max_threads() == before_mkl
    end

    @testset "covers tasks spawned inside" begin
        # The MKL count is process-wide, so a task on another thread sees it too.
        seen = Polfed.with_serial_libraries() do
            fetch(Threads.@spawn (BLAS.get_num_threads(), Polfed._mkl_max_threads()))
        end
        @test seen[1] == 1
        @test seen[2] == (before_mkl == 0 ? 0 : 1)
    end
end
