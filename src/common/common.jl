
module Common

using CPUTime

include("formater.jl")
include("addtime.jl")

export Formatter, fmt, bold, cyan, blue, green, red, yellow
export @addtime!
end