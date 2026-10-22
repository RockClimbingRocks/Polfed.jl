using LinearAlgebra
using Test
using Polfed: MappingConfig, RowLayout, full_polfed, patched_polfed

struct _EntrypointProbeOperator
    origin::Symbol
end

struct _EntrypointProbeMatrix <: AbstractMatrix{Float64}
    dim::Int
end

Base.size(matrix::_EntrypointProbeMatrix) = (matrix.dim, matrix.dim)
Base.getindex(::_EntrypointProbeMatrix, i::Int, j::Int) = Float64(i == j)

const _ENTRYPOINT_ROW_SEED = reshape(collect(1.0:4.0), 1, 4)

function Polfed._prepare_matrix_row_pipeline(
    ::_EntrypointProbeMatrix,
    ::AbstractVecOrMat,
    mc::MappingConfig,
    dosc::Polfed.DoSConfig,
)
    return _EntrypointProbeOperator(:matrix), _ENTRYPOINT_ROW_SEED, mc, dosc
end

function Polfed.run_patched_polfed_target_pipeline(
    operator::_EntrypointProbeOperator,
    x0::AbstractVecOrMat,
    howmany::Integer,
    howmany_per_patch::Integer,
    target;
    kwargs...,
)
    return (;
        mode = :target, operator.origin, x0, howmany, howmany_per_patch, target,
        options = (; kwargs...),
    )
end

function Polfed.run_patched_polfed_window_pipeline(
    operator::_EntrypointProbeOperator,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer,
    target;
    kwargs...,
)
    return (;
        mode = :window, operator.origin, x0, howmany_per_patch, target,
        options = (; kwargs...),
    )
end

function Polfed.run_full_polfed_pipeline(
    operator::_EntrypointProbeOperator,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer;
    kwargs...,
)
    return (;
        mode = :full, operator.origin, x0, howmany_per_patch,
        options = (; kwargs...),
    )
end

@testset "patched/full entrypoint routing" begin
    operator = _EntrypointProbeOperator(:operator)
    matrix = _EntrypointProbeMatrix(4)
    x0 = ones(4)
    window = (:rescaled, (-0.5, 0.5))

    target_operator = patched_polfed(
        operator, x0, 6, 3, :middle; tol = 2e-8, edge_distance = 0.17,
    )
    @test target_operator.mode == :target
    @test target_operator.origin == :operator
    @test target_operator.howmany == 6
    @test target_operator.howmany_per_patch == 3
    @test target_operator.options.tol == 2e-8
    @test target_operator.options.edge_distance == 0.17
    @test !hasproperty(target_operator.options, :layout)

    window_operator = patched_polfed(operator, x0, 3, window)
    @test window_operator.mode == :window
    @test window_operator.origin == :operator

    target_matrix = patched_polfed(matrix, x0, 6, 3, :middle)
    @test target_matrix.mode == :target
    @test target_matrix.origin == :matrix
    @test target_matrix.x0 === _ENTRYPOINT_ROW_SEED
    @test target_matrix.options.layout isa RowLayout

    window_matrix = patched_polfed(matrix, x0, 3, window)
    @test window_matrix.mode == :window
    @test window_matrix.origin == :matrix
    @test window_matrix.options.layout isa RowLayout

    full_operator = full_polfed(operator, x0, 3; retry_growth = 1.7)
    @test full_operator.mode == :full
    @test full_operator.origin == :operator
    @test full_operator.options.retry_growth == 1.7
    @test !hasproperty(full_operator.options, :layout)

    full_matrix = full_polfed(matrix, x0, 3)
    @test full_matrix.mode == :full
    @test full_matrix.origin == :matrix
    @test full_matrix.x0 === _ENTRYPOINT_ROW_SEED
    @test full_matrix.options.layout isa RowLayout
end

struct _NeverCalledOperator end
(::_NeverCalledOperator)(args...) = error("wrong target mode reached the operator")

@testset "patched target mode is validated before operator work" begin
    operator = _NeverCalledOperator()
    @test_throws ArgumentError patched_polfed(operator, ones(4), 2, :middle)
    @test_throws ArgumentError patched_polfed(
        operator, ones(4), 2, 2, (:rescaled, (-0.5, 0.5)),
    )
end
