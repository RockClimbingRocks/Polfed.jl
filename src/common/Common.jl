module Common

using Printf: @sprintf

include("Formatter.jl")
include("stage_timer.jl")
include("logging.jl")
include("utils/bisection.jl")
include("utils/chebyshev_integration.jl")
include("utils/validate_leakage.jl")

export Formatter, fmt, bold, cyan, blue, green, red, yellow
export StageTimer, @stage, @stages, stage_names, total_wall, total_cpu
export POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL, POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL,
       POLFED_DEEP_DEBUG_LEVEL
export verbosity, should_log, polfed_log, with_verbosity
export bisection, integrate_chebyshev_weighted, validate_no_leakage_in_interval_numerically
end
