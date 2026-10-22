using LinearAlgebra
using Test
using Polfed: ColumnLayout, RowLayout, _DenseResultBuffer, _WindowResultBuffer,
              _store_pair!, _store_interval!, _finalize_result!

@testset "result buffers" begin
    source = Matrix{Float64}(I, 4, 4)
    column = ColumnLayout()

    dense = _DenseResultBuffer(
        Vector{Float64}(undef, 4), Matrix{Float64}(undef, 4, 4),
        0, column, nothing, 1e-8,
    )
    _store_pair!(dense, -0.5, source, 1, column)
    _store_pair!(dense, -0.5 + 1e-10, source, 1, column)
    _store_pair!(dense, -0.5, source, 2, column)
    _store_pair!(dense, 0.2, source, 3, column)
    vals, vecs = _finalize_result!(dense)
    @test vals ≈ [-0.5, -0.5, 0.2]
    @test size(vecs) == (4, 3)

    target = _DenseResultBuffer(
        Vector{Float64}(undef, 2), Matrix{Float64}(undef, 4, 2),
        0, column, 0.2, 1e-8,
    )
    for (i, value) in enumerate((-0.9, 0.1, 0.3))
        _store_pair!(target, value, source, i, column)
    end
    target_vals, target_vecs = _finalize_result!(target)
    @test target_vals ≈ [0.1, 0.3]
    @test size(target_vecs) == (4, 2)

    row = RowLayout()
    rows = _DenseResultBuffer(
        Vector{Float64}(undef, 2), Matrix{Float64}(undef, 2, 4),
        0, row, nothing, 1e-8,
    )
    _store_pair!(rows, 0.3, permutedims(source), 3, row)
    _store_pair!(rows, -0.5, permutedims(source), 1, row)
    row_vals, row_vecs = _finalize_result!(rows)
    @test row_vals ≈ [-0.5, 0.3]
    # Returned eigenvectors are ALWAYS columns (dim × n), whatever layout the buffer used
    # internally — a row-layout buffer is transposed on the way out.
    @test size(row_vecs) == (4, 2)
    @test size(row_vecs) == size(target_vecs)
    # and the i-th column is the i-th eigenvector, matching the column-layout path
    @test row_vecs[:, 2] ≈ target_vecs[:, 2]

    window = _WindowResultBuffer(
        Float64[], Vector{Vector{Float64}}(), column, 1e-8,
    )
    _store_interval!(window, [-0.5, 0.2], source, 1:2, -0.6, 0.1, column)
    window_vals, window_vecs = _finalize_result!(window)
    @test window_vals == [-0.5]
    @test length(window_vecs) == 1
    @test window_vecs[1] == source[:, 1]
end
