using Test
using UniPolfed

# These lock the no-CUDA fallback contract that the CUDA extension relies on.
# (The GPU code paths themselves cannot run on the CPU host; see RESTRUCTURE.md.)
@testset "gpu_dispatch (no CUDA loaded)" begin
    @testset "detection predicates are false on host data" begin
        @test UniPolfed.cuda_available() == false
        @test UniPolfed.is_gpu_array([1.0, 2.0]) == false
        @test UniPolfed.is_gpu_array(42) == false
        @test UniPolfed.is_gpu_sparse_matrix([1.0 2.0; 3.0 4.0]) == false
    end

    @testset "GPU accessors throw when the backend is unset" begin
        @test_throws ErrorException UniPolfed.gpu_array([1.0])
        @test_throws ErrorException UniPolfed.gpu_matrix([1.0])
        @test_throws ErrorException UniPolfed.gpu_vector([1.0])
        @test_throws ErrorException UniPolfed.gpu_matrix_type()
        @test_throws ErrorException UniPolfed.gpu_vector_type()
        @test_throws ErrorException UniPolfed.gpu_zeros(2)
    end

    @testset "soft accessors no-op without CUDA" begin
        @test UniPolfed.gpu_functional()  == false
        @test UniPolfed.gpu_device_count() == 0
        @test UniPolfed.gpu_synchronize() === nothing
        @test UniPolfed.gpu_reclaim()     === nothing
        @test UniPolfed.gpu_allowscalar(() -> 42) == 42   # no CUDA → just calls f
    end
end
