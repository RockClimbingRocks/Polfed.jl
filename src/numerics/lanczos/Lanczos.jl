"Lanczos and block-Lanczos eigensolvers for Hermitian/symmetric operators (extremal, both-ends, and shift-targeted)."
module Lanczos

using LinearAlgebra, Printf, Random

# Backend GPU-dispatch helpers + processing-unit types live at the UniPolfed top level.
import ...is_gpu_array, ...is_gpu_sparse_matrix
import ...gpu_matrix, ...gpu_randn
import ...gpu_synchronize, ...gpu_reclaim, ...gpu_allowscalar
import ...CPU, ...GPU, ...ProcessingUnit

import ...Common: Formatter, fmt, bold, cyan, blue, green, red, yellow
import ...Common: StageTimer, @stage, @stages, stage_names, total_wall, total_cpu
import ...Common: POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL, POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL,
                  POLFED_DEEP_DEBUG_LEVEL
import ...Common: verbosity, should_log, polfed_log
import ...Common: render_detail_table

include("Layout.jl")
include("reorthtechniques/ReOrthTechniques.jl")
include("orthonormalbasis/OrthonormalBasis.jl")
include("LanczosIterator.jl")
include("factorization/Factorization.jl")
include("convergence/Convergence.jl")
include("LanczosDefaults.jl")     # after Convergence — DEFAULT_NEXT_CHECK references ConvergenceState/ConvergenceCriterion
include("FactorizationReport.jl")
include("LanczosResult.jl")
include("lanczos_algorithm.jl")
include("lanczos_method.jl")
include("lanczos.jl")

export lanczos, lanczos_extrema
export EigTarget, EndSpec, NumEigs, ToValue, Window, Lowest, Highest, BothEnds
export LowestEigs, HighestEigs
export LanczosResult, has_converged
export Layout, ColumnLayout, RowLayout
export ReOrthTechnique, FullRO, PartialRO, SelectiveRO
export OrthonormalBasis, MatrixBasis, VectorBasis
export FactorizationReport, display_factorization_report, print_factorization_report
export CPU, GPU

end # module Lanczos
