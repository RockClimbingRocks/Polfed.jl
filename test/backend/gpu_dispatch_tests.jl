using Test
using Polfed

# These lock the no-CUDA fallback contract that the CUDA extension relies on.
# GPU execution is covered separately on CUDA-capable hosts.
@testset "gpu_dispatch (no CUDA loaded)" begin
    @testset "detection predicates are false on host data" begin
        @test Polfed.cuda_available() == false
        @test Polfed.is_gpu_array([1.0, 2.0]) == false
        @test Polfed.is_gpu_array(42) == false
        @test Polfed.is_gpu_sparse_matrix([1.0 2.0; 3.0 4.0]) == false
    end

    @testset "GPU accessors throw when the backend is unset" begin
        @test_throws ErrorException Polfed.gpu_array([1.0])
        @test_throws ErrorException Polfed.gpu_matrix([1.0])
        @test_throws ErrorException Polfed.gpu_vector([1.0])
        @test_throws ErrorException Polfed.gpu_matrix_type()
        @test_throws ErrorException Polfed.gpu_vector_type()
        @test_throws ErrorException Polfed.gpu_zeros(2)
    end

    @testset "soft accessors no-op without CUDA" begin
        @test Polfed.gpu_functional()  == false
        @test Polfed.gpu_device_count() == 0
        @test Polfed.gpu_synchronize() === nothing
        @test Polfed.gpu_reclaim()     === nothing
        @test Polfed.gpu_allowscalar(() -> 42) == 42   # no CUDA → just calls f
    end
end
