using Polfed
using Test

@testset "Polfed.jl" begin
    include("test_qsun.jl")
    include("optimization.jl")
    include("test_polfed.jl")
end
