import Logging: @info, @warn

"""
    POLFED_SILENT_LEVEL::Int = 0

Disable POLFED logging output.
"""
const POLFED_SILENT_LEVEL = 0

"""
    POLFED_WARN_LEVEL::Int = 1

Emit warning-level POLFED logs.
"""
const POLFED_WARN_LEVEL   = 1

"""
    POLFED_INFO_LEVEL::Int = 2

Emit informational and warning POLFED logs.
"""
const POLFED_INFO_LEVEL   = 2

"""
    POLFED_DEBUG_LEVEL::Int = 3

Emit debug, info, and warning POLFED logs.
"""
const POLFED_DEBUG_LEVEL  = 3

"""
    verbosity::Base.RefValue{Int}

Runtime POLFED logging level. Typical values:
- `0` (`POLFED_SILENT_LEVEL`)
- `1` (`POLFED_WARN_LEVEL`)
- `2` (`POLFED_INFO_LEVEL`)
- `3` (`POLFED_DEBUG_LEVEL`)
"""
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
