import Logging: @info, @warn

const POLFED_SILENT_LEVEL = 0
const POLFED_WARN_LEVEL   = 1
const POLFED_INFO_LEVEL   = 2
const POLFED_DEBUG_LEVEL  = 3

const verbosity = Ref(POLFED_WARN_LEVEL)

"""
    should_log(level::Int) -> Bool

Return `true` if current verbosity allows logging at `level`.
"""
@inline should_log(level::Int) = verbosity[] >= level

"""
    polfed_log(level::Int, msg; kwargs...) -> nothing

Emit a structured POLFED log message gated by [`verbosity`](@ref).

Log routing:
- debug level: `@info` with `[DEBUG]` prefix,
- info level: `@info`,
- warn level: `@warn`.
"""
function polfed_log(level::Int, msg; kwargs...)
    should_log(level) || return nothing
    if level >= POLFED_DEBUG_LEVEL
        @info "[DEBUG] $(msg)" kwargs...
    elseif level >= POLFED_INFO_LEVEL
        @info msg kwargs...
    elseif level >= POLFED_WARN_LEVEL
        @warn msg kwargs...
    end
    return nothing
end
