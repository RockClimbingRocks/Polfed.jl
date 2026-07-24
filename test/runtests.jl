using UniPolfed
using Test

@testset "UniPolfed.jl" begin

    @testset "Infrastructure" begin
        @testset "Common" begin
            include("common/Formatter_tests.jl")
            include("common/logging_tests.jl")
            include("common/stage_timer_tests.jl")
            @testset "Utils" begin
                include("common/utils/bisection_tests.jl")
                include("common/utils/chebyshev_integration_tests.jl")
            end
        end
        @testset "Backend" begin
            include("backend/ProcessingUnit_tests.jl")
            include("backend/gpu_dispatch_tests.jl")
        end
    end

    @testset "Numerics" begin
        @testset "Clenshaw" begin
            include("numerics/clenshaw/algorithm/recurrence_relations_tests.jl")
            include("numerics/clenshaw/algorithm/final_sum_tests.jl")
            include("numerics/clenshaw/algorithm/clenshaw_algorithm_tests.jl")
            include("numerics/clenshaw/ClenshawExpansion_tests.jl")
        end
        @testset "KPM" begin
            include("numerics/kpm/kernels_tests.jl")
            include("numerics/kpm/moments_tests.jl")
            include("numerics/kpm/density_tests.jl")
        end
        @testset "Lanczos" begin
            include("numerics/lanczos/Layout_tests.jl")
            @testset "OrthonormalBasis" begin
                include("numerics/lanczos/orthonormalbasis/OrthonormalBasis_tests.jl")
                include("numerics/lanczos/orthonormalbasis/MatrixBasis_tests.jl")
                include("numerics/lanczos/orthonormalbasis/VectorBasis_tests.jl")
            end
            @testset "Factorization" begin
                include("numerics/lanczos/factorization/factorizationsteps/orthogonalization_tests.jl")
                include("numerics/lanczos/factorization/factorizationsteps/reorthogonalization_tests.jl")
                include("numerics/lanczos/factorization/diagnostics_tests.jl")
                include("numerics/lanczos/factorization/Factorization_tests.jl")
            end
            @testset "Convergence" begin
                include("numerics/lanczos/convergence/EigTarget_tests.jl")
                include("numerics/lanczos/convergence/end_spec_tests.jl")
                include("numerics/lanczos/convergence/converged_run_tests.jl")
                include("numerics/lanczos/convergence/calc_ritz_residuals_tests.jl")
                include("numerics/lanczos/convergence/calc_eigen_residuals_tests.jl")
            end
            include("numerics/lanczos/lanczos_tests.jl")
            include("numerics/lanczos/row_layout_tests.jl")
            include("numerics/lanczos/LanczosResult_tests.jl")
            include("numerics/lanczos/FactorizationReport_tests.jl")
        end
    end

    @testset "Models" begin
        @testset "QSun" begin
            include("models/qsun/spin_operators_tests.jl")
            include("models/qsun/random_utils_tests.jl")
            include("models/qsun/hilbert_space_tests.jl")
            include("models/qsun/quantum_sun_tests.jl")
            include("models/qsun/quantum_sun_u1_conserving_tests.jl")
            include("models/qsun/QSun_tests.jl")
        end
        include("models/spin_half_chains_tests.jl")
        include("models/Models_tests.jl")
    end

    @testset "Optimization" begin
        include("optimization/qrem_fixture.jl")          # qrem(L) + mul_row_ref helpers
        include("optimization/value_grouping_tests.jl")
        include("optimization/reads_tests.jl")
        include("optimization/flips_tests.jl")
        include("optimization/boundary_tests.jl")
        include("optimization/benchmark_tests.jl")
        include("optimization/optimized_mapping_tests.jl")
    end

    @testset "main" begin
        @testset "logic" begin
            @testset "dos" begin
                include("main/logic/dos/dos_tests.jl")
            end

            @testset "target" begin
                include("main/logic/target/target_spec_tests.jl")
                include("main/logic/target/resolve_target_tests.jl")
                include("main/logic/target/resolve_window_tests.jl")
            end

            @testset "interval" begin
                include("main/logic/interval/resolve_interval_tests.jl")
            end

            @testset "filter" begin
                include("main/logic/filter/order_bisection_tests.jl")
                include("main/logic/filter/polynomial_order_tests.jl")
                include("main/logic/filter/edge_side_tests.jl")
                include("main/logic/filter/resolve_filter_tests.jl")
                include("main/logic/filter/passband_tests.jl")
                @testset "filters" begin
                    @testset "delta_up" begin
                        include("main/logic/filter/filters/delta_up/delta_up_tests.jl")
                        include("main/logic/filter/filters/delta_up/coefficients_tests.jl")
                        include("main/logic/filter/filters/delta_up/build_tests.jl")
                        include("main/logic/filter/filters/delta_up/validate_tests.jl")
                    end
                    @testset "delta_down" begin
                        include("main/logic/filter/filters/delta_down/delta_down_tests.jl")
                        include("main/logic/filter/filters/delta_down/coefficients_tests.jl")
                        include("main/logic/filter/filters/delta_down/build_tests.jl")
                        include("main/logic/filter/filters/delta_down/validate_tests.jl")
                    end
                    @testset "double_delta" begin
                        include("main/logic/filter/filters/double_delta/double_delta_tests.jl")
                        include("main/logic/filter/filters/double_delta/coefficients_tests.jl")
                        include("main/logic/filter/filters/double_delta/build_tests.jl")
                        include("main/logic/filter/filters/double_delta/validate_tests.jl")
                    end
                    @testset "edge" begin
                        include("main/logic/filter/filters/edge/edge_tests.jl")
                    end
                end
            end

            @testset "transform" begin
                @testset "clenshaw" begin
                    include("main/logic/transform/clenshaw/build_expansion_tests.jl")
                end
                @testset "parallel" begin
                    include("main/logic/transform/parallel/b_storage_tests.jl")
                end
                include("main/logic/transform/build_transform_tests.jl")
            end

            @testset "report" begin
                include("main/logic/report/report_fixture.jl")
                include("main/logic/report/MappingReport_tests.jl")
                include("main/logic/report/SpectralTransformReport_tests.jl")
                include("main/logic/report/PolfedResult_tests.jl")
                include("main/logic/report/report_tests.jl")
                include("main/logic/report/PatchedReport_tests.jl")
            end
        end

        @testset "pipeline" begin
            @testset "helpers" begin
                include("main/pipeline/helpers/intervalize_interval_tests.jl")
                include("main/pipeline/helpers/result_buffer_tests.jl")
            end
            include("main/pipeline/pipeline_tests.jl")
        end

        @testset "entrypoint" begin
            include("main/entrypoint/matrix_entrypoint_tests.jl")
            include("main/entrypoint/window_entrypoint_tests.jl")
            include("main/entrypoint/edge_entrypoint_tests.jl")
            include("main/entrypoint/patched_full_entrypoint_tests.jl")
        end
    end
end
