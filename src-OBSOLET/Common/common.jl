
module Common

using CPUTime

include("formater.jl")
include("addtime.jl")
include("logging.jl")

export Formatter, fmt, bold, cyan, blue, green, red, yellow
export @addtime!
export POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL, POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL
export verbosity, should_log, polfed_log
end
