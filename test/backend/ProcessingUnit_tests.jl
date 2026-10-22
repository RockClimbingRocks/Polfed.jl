using Test
using UniPolfed

@testset "ProcessingUnit" begin
    @testset "CPU() carries the unified field set and they work" begin
        cpu = UniPolfed.CPU()
        @test cpu isa UniPolfed.ProcessingUnit
        @test cpu.Matrix === Matrix
        @test cpu.Vector === Vector
        @test cpu.zeros(2) == [0.0, 0.0]
        @test cpu.ones(3)  == ones(3)
        @test cpu.rand(4)  isa Vector{Float64}
        @test length(cpu.rand(4)) == 4
        @test cpu.randn(4) isa Vector{Float64}
    end

    @testset "GPU() errors when CUDA is not loaded" begin
        @test UniPolfed.cuda_available() == false
        @test_throws ErrorException UniPolfed.GPU()
    end

    @testset "type hierarchy" begin
        @test UniPolfed.CPU <: UniPolfed.ProcessingUnit
        @test UniPolfed.GPU <: UniPolfed.ProcessingUnit
    end
end
