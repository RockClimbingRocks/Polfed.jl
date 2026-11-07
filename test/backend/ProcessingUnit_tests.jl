using Test
using Polfed

@testset "ProcessingUnit" begin
    @testset "CPU() carries the unified field set and they work" begin
        cpu = Polfed.CPU()
        @test cpu isa Polfed.ProcessingUnit
        @test cpu.Matrix === Matrix
        @test cpu.Vector === Vector
        @test cpu.zeros(2) == [0.0, 0.0]
        @test cpu.ones(3)  == ones(3)
        @test cpu.rand(4)  isa Vector{Float64}
        @test length(cpu.rand(4)) == 4
        @test cpu.randn(4) isa Vector{Float64}
    end

    @testset "GPU() errors when CUDA is not loaded" begin
        @test Polfed.cuda_available() == false
        @test_throws ErrorException Polfed.GPU()
    end

    @testset "type hierarchy" begin
        @test Polfed.CPU <: Polfed.ProcessingUnit
        @test Polfed.GPU <: Polfed.ProcessingUnit
    end
end
