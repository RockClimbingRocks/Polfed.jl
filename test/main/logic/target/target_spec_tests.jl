using Test
using Polfed



@testset "target" begin    

    @testset "Assign a struct that isa TargetSpec" begin
        targets = [
            2.2, 
            4, 
            (:unrescaled, 0.5), 
            (:rescaled, 0.5),
            :maxdos,
            :middle,
            (:offset, 0.5),
            (:offset, -0.5),
            (:rescaled, 1.0),
            (:rescaled, -1.0),
            (:offset, 1.0),
            (:offset, -1.0),
        ]

        for target in targets
            target_spec = Polfed.normalize_target_spec(target)
            @test target_spec isa Polfed.TargetSpec
        end
    end



    @testset "Assign a struct that is not TargetSpec" begin
        targets = [
            :hobo,
            (:mucho_erectus, 0.5),
            "middle",
            (:rescaled, "0.5"),
            (:unrescaled, "0.5"),
            (:offset, 1.5),
            (:offset, -1.5),
            (:rescaled, 1.5),
            (:rescaled, -1.5),
            2+4im,
            (:offset, 2+4im),
            (:rescaled, 2+4im),
            (:unrescaled, 2+4im),
        ]

        for target in targets
            @test_throws ArgumentError Polfed.normalize_target_spec(target)
        end

    end



end

