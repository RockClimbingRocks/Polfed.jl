module Optimization

using Test
using SparseArrays
using LinearAlgebra
include("../src/Polfed.jl")
using Polfed: optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!, NoParallel, MulColsParallel, TwoLevelParallel
using Polfed.PolfedCore: get_diags_and_offdiagonals_by_value

# Helper function to create a test matrix.
# This matrix has two distinct off-diagonal values (-2.0 and 0.5)
# to test both the single and multiple off-diagonal cases.
function create_test_matrix()
    H = spzeros(5, 5)
    H[1, 1] = 1.0; H[2, 2] = 2.0; H[3, 3] = 3.0; H[4, 4] = 4.0; H[5, 5] = 5.0
    
    # Off-diagonal value 1: -2.0
    H[1, 2] = H[2, 1] = -2.0
    H[2, 4] = H[4, 2] = -2.0
    H[4, 5] = H[5, 4] = -2.0

    # Off-diagonal value 2: 0.5
    H[1, 3] = H[3, 1] = 0.5
    H[2, 5] = H[5, 2] = 0.5
    
    return H
end

@testset "All Optimization Tests" begin

    H = create_test_matrix()
    diagonals, offdiagonals = get_diags_and_offdiagonals_by_value(H)

    @testset "1. Data Structure Generation" begin
        @test diagonals == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test length(offdiagonals) == 2 # Should find two groups of values (-2.0 and 0.5)
        
        # Sort by value to ensure consistent test order
        sort!(offdiagonals, by = x -> x[1])

        val1, flat1, starts1 = offdiagonals[1] # Should be -2.0
        val2, flat2, starts2 = offdiagonals[2] # Should be 0.5

        # println("Offdiagonal 1: val=$val1, flat=$flat1, starts=$starts1")
        # println("Offdiagonal 2: val=$val2, flat=$flat2, starts=$starts2")

        @test val1 == -2.0
        @test val2 == 0.5

        # Manually verify the flattened structure for -2.0
        @test flat1 == [2, 1, 4, 2, 5, 4]
        @test starts1 == [1, 2, 4, 4, 6]
        
        # Manually verify the flattened structure for 0.5
        @test flat2 == [3, 5, 1, 2]
        @test starts2 == [1, 2, 3, 4, 4]
    end

    @testset "2. Optimized Mapping" begin
        # --- Test Data ---
        X_vec = Float64[1, 2, 3, 4, 5]
        X_mat = hcat(X_vec, 2 * X_vec)
        
        # --- Ground Truth ---
        Y_expected_vec = H * X_vec
        Y_expected_mat = H * X_mat

        # --- Serial Execution ---
        map_s! = optimized_mapping!(diagonals, offdiagonals, NoParallel())
        Y_s_vec = similar(X_vec)
        Y_s_mat = similar(X_mat)
        map_s!(Y_s_vec, X_vec)
        map_s!(Y_s_mat, X_mat)
        
        @test Y_s_vec ≈ Y_expected_vec
        @test Y_s_mat ≈ Y_expected_mat

        # --- Parallel Execution ---
        map_p! = optimized_mapping!(diagonals, offdiagonals, TwoLevelParallel(1))
        Y_p_vec = similar(X_vec)
        Y_p_mat = similar(X_mat)
        map_p!(Y_p_vec, X_vec)
        map_p!(Y_p_mat, X_mat)

        @test Y_p_vec ≈ Y_expected_vec
        @test Y_p_mat ≈ Y_expected_mat

        # --- Single-process threaded block execution ---
        map_mt! = optimized_mapping!(diagonals, offdiagonals, MulColsParallel(2))
        Y_mt_vec = similar(X_vec)
        Y_mt_mat = similar(X_mat)
        map_mt!(Y_mt_vec, X_vec)
        map_mt!(Y_mt_mat, X_mat)

        @test Y_mt_vec ≈ Y_expected_vec
        @test Y_mt_mat ≈ Y_expected_mat

        # --- Ensure Serial and Parallel match ---
        @test Y_s_vec ≈ Y_p_vec
    end

    @testset "3. Clenshaw Recurrence Relation" begin
        # --- Test Data ---
        c = 0.5
        X = Float64[1, 2, 3, 4, 5]
        b2 = Float64[5, 4, 3, 2, 1]
        b3 = Float64[1, 1, 1, 1, 1]
        
        # --- Ground Truth ---
        # Formula: b1 = c*X + 2*(H*b2) - b3
        b1_expected = c .* X + 2 * (H * b2) - b3

        # --- Serial Execution ---
        crr_s! = optimized_clenshaw_recurrence_relation!(diagonals, offdiagonals, NoParallel())
        b1_s = similar(X)
        crr_s!(b1_s, b2, b3, c, 1, X)
        
        @test b1_s ≈ b1_expected

        # --- Parallel Execution ---
        crr_p! = optimized_clenshaw_recurrence_relation!(diagonals, offdiagonals, TwoLevelParallel(1))
        b1_p = similar(X)
        crr_p!(b1_p, b2, b3, c, 1, X)

        @test b1_p ≈ b1_expected

        # --- Single-process threaded block execution ---
        X_mat = hcat(X, 2 * X)
        b2_mat = hcat(b2, 2 * b2)
        b3_mat = hcat(b3, 3 * b3)
        b1_expected_mat = c .* X_mat + 2 * (H * b2_mat) - b3_mat

        crr_mt! = optimized_clenshaw_recurrence_relation!(diagonals, offdiagonals, MulColsParallel(2))
        b1_mt = similar(X_mat)
        crr_mt!(b1_mt, b2_mat, b3_mat, c, 1, X_mat)

        @test b1_mt ≈ b1_expected_mat
    end

    @testset "4. Clenshaw Final Sum" begin
        # --- Test Data ---
        c = 0.5
        X = Float64[1, 2, 3, 4, 5]
        b1 = Float64[5, 4, 3, 2, 1]
        b2 = Float64[1, 1, 1, 1, 1]
        
        # --- Ground Truth ---
        # Formula: Y = c*X + H*b1 - b2
        Y_expected = c .* X + (H * b1) - b2

        # --- Serial Execution ---
        cfs_s! = optimized_clenshaw_final_sum!(diagonals, offdiagonals, NoParallel())
        Y_s = similar(X)
        # Note: Closure definition is (b1, b2, c, Y, X)
        cfs_s!(b1, b2, c, Y_s, X) 
        
        @test Y_s ≈ Y_expected

        # --- Parallel Execution ---
        cfs_p! = optimized_clenshaw_final_sum!(diagonals, offdiagonals, TwoLevelParallel(1))
        Y_p = similar(X)
        cfs_p!(b1, b2, c, Y_p, X)

        @test Y_p ≈ Y_expected

        # --- Single-process threaded block execution ---
        X_mat = hcat(X, 2 * X)
        b1_mat = hcat(b1, 2 * b1)
        b2_mat = hcat(b2, 3 * b2)
        Y_expected_mat = c .* X_mat + (H * b1_mat) - b2_mat

        cfs_mt! = optimized_clenshaw_final_sum!(diagonals, offdiagonals, MulColsParallel(2))
        Y_mt = similar(X_mat)
        cfs_mt!(b1_mat, b2_mat, c, Y_mt, X_mat)

        @test Y_mt ≈ Y_expected_mat
    end
    
    @testset "5. Test Single Off-diagonal Case" begin
        # Create a matrix with only ONE off-diagonal value
        H_single = spzeros(3,3)
        H_single[1,1]=1.0; H_single[2,2]=2.0; H_single[3,3]=3.0
        H_single[1,2]=H_single[2,1]=-0.5
        H_single[2,3]=H_single[3,2]=-0.5
        
        d_single, o_single_vec = get_diags_and_offdiagonals_by_value(H_single)
        
        @test length(o_single_vec) == 1
        o_single_tuple = o_single_vec[1] # This is now a Tuple, not a Vector{Tuple}
        
        X = [1.0, 2.0, 3.0]
        Y_expected = H_single * X
        
        # Test serial mapping with the tuple directly
        map_s! = optimized_mapping!(d_single, o_single_tuple, NoParallel())
        Y_s = similar(X)
        map_s!(Y_s, X)
        @test Y_s ≈ Y_expected

        # Test parallel mapping with the tuple directly
        map_p! = optimized_mapping!(d_single, o_single_tuple, TwoLevelParallel(1))
        Y_p = similar(X)
        map_p!(Y_p, X)
        @test Y_p ≈ Y_expected

        map_mt! = optimized_mapping!(d_single, o_single_tuple, MulColsParallel(2))
        Y_mt = similar(X)
        map_mt!(Y_mt, X)
        @test Y_mt ≈ Y_expected
    end
end
end
