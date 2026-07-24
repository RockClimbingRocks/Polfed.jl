using Test
using UniPolfed: patch_results_sequential!, patch_results_owned!, _find_left_overlap, _find_right_overlap
using UniPolfed: RowLayout, ColumnLayout, Interval


vectors(vals) = reshape(copy(vals), 1, length(vals))


@testset "_find_left_overlap" begin
    @testset "single-value overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vals_i = [5.0, 6.0, 7.0, 8.0, 9.0]

        @test _find_left_overlap(vals, vals_i, 1e-4) == [6]
    end

    @testset "two-value overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vals_i = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        @test _find_left_overlap(vals, vals_i, 1e-4) == [5, 6]
    end

    @testset "two values, one good and one bad" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vals_i = [4.0, 5.5, 6.0, 7.0, 8.0, 9.0]

        @test_throws ErrorException _find_left_overlap(vals, vals_i, 1e-4)
    end

    @testset "no overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vals_i = [6.0, 7.0, 8.0, 9.0]

        @test_throws ErrorException _find_left_overlap(vals, vals_i, 1e-4)
    end
end


@testset "_find_right_overlap" begin
    @testset "single-value overlap" begin
        vals = [5.0, 6.0, 7.0, 8.0, 9.0]
        vals_i = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        @test _find_right_overlap(vals, vals_i, 1e-4) == [1]
    end

    @testset "two-value overlap" begin
        vals = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        vals_i = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        @test _find_right_overlap(vals, vals_i, 1e-4) == [1, 2]
    end

    @testset "two values, one good and one bad" begin
        vals = [4.5, 5.0, 6.0, 7.0, 8.0, 9.0]
        vals_i = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        @test_throws ErrorException _find_right_overlap(vals, vals_i, 1e-4)
    end

    @testset "no overlap" begin
        vals = [5.0, 6.0, 7.0, 8.0, 9.0]
        vals_i = [0.0, 1.0, 2.0, 3.0, 4.0]

        @test_throws ErrorException _find_right_overlap(vals, vals_i, 1e-4)
    end
end


@testset "patch_results_sequential!" begin


    @testset "single-value overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vecs = [[value] for value in vals]
        vals_i = [5.0, 6.0, 7.0, 8.0, 9.0]
        vecs_i = vectors(vals_i)


        vals_expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        vecs_expected = [[value] for value in vals_expected]

        patch_results_sequential!(vals_i, vecs_i, vals, vecs, 1e-4)

        @test vals == vals_expected
        @test vecs == vecs_expected
    end


    @testset "two-value overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vecs = [[value] for value in vals]
        vals_i = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        vecs_i = vectors(vals_i)


        vals_expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        vecs_expected = [[value] for value in vals_expected]

        patch_results_sequential!(vals_i, vecs_i, vals, vecs, 1e-4)

        @test vals == vals_expected
        @test vecs == vecs_expected
    end


    @testset "two-value overlap - one good - one bad" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vecs = [[value] for value in vals]
        vals_i = [4.0, 5.5, 6.0, 7.0, 8.0, 9.0]
        vecs_i = vectors(vals_i)


        @test_throws ErrorException patch_results_sequential!(vals_i, vecs_i, vals, vecs, 1e-4)
    end


    @testset "no overlap" begin
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        vecs = [[value] for value in vals]
        vals_i = [6.0, 7.0, 8.0, 9.0]
        vecs_i = vectors(vals_i)


        @test_throws ErrorException patch_results_sequential!(vals_i, vecs_i, vals, vecs, 1e-4)
    end



end


@testset "patch_results_owned!" begin

    @testset "keeps only the eigenpairs inside the ownership interval (half-open)" begin
        # Converged patch spans well past its ownership on both sides (brackets + overlap band);
        # only the values in [own.left, own.right) are kept.
        vals_i = [-0.5, 0.0, 0.1, 0.2, 0.3, 0.9]
        vecs_i = vectors(vals_i)
        vals = Float64[]; vecs = Vector{Vector{Float64}}()

        patch_results_owned!(vals_i, vecs_i, Interval(0.0, 0.3), vals, vecs, ColumnLayout())
        @test vals == [0.0, 0.1, 0.2]              # 0.3 excluded (half-open right), -0.5/0.9 outside
        @test vecs == [[0.0], [0.1], [0.2]]        # eigenvectors follow, as views
    end

    @testset "tiling patches produce a gap-free, duplicate-free run" begin
        # Two patches whose ownership intervals share the exact endpoint 0.3. The shared eigenvalue
        # 0.3 (converged by BOTH) must be counted exactly once — by the right patch (half-open left).
        vals = Float64[]; vecs = Vector{Vector{Float64}}()

        left_i  = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.35]     # patch owning [0.0, 0.3)
        patch_results_owned!(left_i, vectors(left_i), Interval(0.0, 0.3), vals, vecs, ColumnLayout())
        right_i = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7]      # patch owning [0.3, 0.6], last → closed right
        patch_results_owned!(right_i, vectors(right_i), Interval(0.3, 0.6), vals, vecs, ColumnLayout();
                             include_right = true)

        @test vals == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]   # contiguous, 0.3 once, no gaps
        @test issorted(vals)
        @test allunique(vals)
    end

    @testset "include_left leaves the lower endpoint open (edge/interior seam)" begin
        # A value exactly on the lower boundary is left to a neighbour (e.g. an edge peel that
        # already owns it) when include_left=false; kept when true.
        vals_i = [0.0, 0.1, 0.2]
        for (inc, expected) in ((true, [0.0, 0.1, 0.2]), (false, [0.1, 0.2]))
            vals = Float64[]; vecs = Vector{Vector{Float64}}()
            patch_results_owned!(vals_i, vectors(vals_i), Interval(0.0, 0.3), vals, vecs,
                                 ColumnLayout(); include_left = inc)
            @test vals == expected
        end
    end

    @testset "include_right keeps the closed right endpoint (last patch only)" begin
        vals_i = [0.1, 0.2, 0.3]
        for (inc, expected) in ((false, [0.1, 0.2]), (true, [0.1, 0.2, 0.3]))
            vals = Float64[]; vecs = Vector{Vector{Float64}}()
            patch_results_owned!(vals_i, vectors(vals_i), Interval(0.0, 0.3), vals, vecs,
                                 ColumnLayout(); include_right = inc)
            @test vals == expected
        end
    end

    @testset "row layout keeps eigenvectors as row views" begin
        vals_i = [0.0, 0.1, 0.2, 0.9]
        vecs_i = reshape(copy(vals_i), length(vals_i), 1)   # row layout: one row per eigenvector
        vals = Float64[]; vecs = Vector{typeof(view(vecs_i, 1, :))}()

        patch_results_owned!(vals_i, vecs_i, Interval(0.0, 0.3), vals, vecs, RowLayout())
        @test vals == [0.0, 0.1, 0.2]
        @test vecs[1] === view(vecs_i, 1, :)                # stored as a view, not a copy
    end

    @testset "mismatched eigenvector count is rejected" begin
        vals_i = [0.1, 0.2, 0.3]
        vecs_i = vectors([0.1, 0.2])                        # one short
        @test_throws DimensionMismatch patch_results_owned!(
            vals_i, vecs_i, Interval(0.0, 0.3), Float64[], Vector{Vector{Float64}}(), ColumnLayout())
    end
end


# @testset "bad overlap does not modify the result" begin
#     vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
#     vecs = AbstractVector{Float64}[[value] for value in vals]
#     vals_i = [4.0, 5.5, 6.0, 7.0]

#     @test_throws ErrorException patch_results_sequential!(
#         vals_i,
#         vectors(vals_i),
#         vals,
#         vecs,
#         1e-4,
#     )
#     @test vals == collect(0.0:5.0)
#     @test first.(vecs) == vals
# end


# @testset "row-layout eigenvectors are also views" begin
#     vals = [0.0, 1.0, 2.0]
#     vecs = AbstractVector{Float64}[[value] for value in vals]
#     vals_i = [2.0, 3.0, 4.0]
#     vecs_i = reshape(copy(vals_i), length(vals_i), 1)

#     patch_results_sequential!(vals_i, vecs_i, vals, vecs, 1e-4, RowLayout())

#     @test vals == collect(0.0:4.0)
#     @test first.(vecs) == vals
#     @test vecs[4] === view(vecs_i, 2, :)
# end
