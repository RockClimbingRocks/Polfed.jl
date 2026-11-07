# `@info` / `@warn` are Base exports (Base.CoreLogging), so no `using Logging` is needed here.
# The `Logging` stdlib is a TEST-only dependency of this package.

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
    POLFED_DEEP_DEBUG_LEVEL::Int = 4

Emit deep-debug logs on top of everything else: per-iteration / per-check tracing whose
volume scales with the run (one line per convergence check, per Clenshaw build, per
bound probe). Useful for diagnosing *where inside a stage* a run misbehaves; too noisy
to leave on for production runs.
"""
const POLFED_DEEP_DEBUG_LEVEL = 4

"""
    verbosity::Base.RefValue{Int}

Runtime POLFED logging level. Typical values:
- `0` (`POLFED_SILENT_LEVEL`) — no logging
- `1` (`POLFED_WARN_LEVEL`) — WARN only (potential issues / suboptimal setup)
- `2` (`POLFED_INFO_LEVEL`) — INFO + WARN (major run phases and setup diagnostics)
- `3` (`POLFED_DEBUG_LEVEL`) — DEBUG + INFO + WARN (detailed tracing)
- `4` (`POLFED_DEEP_DEBUG_LEVEL`) — DEEP + DEBUG + INFO + WARN (per-iteration tracing)

Set it at runtime, e.g. `Polfed.Common.verbosity[] = 2`.
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

Log routing (the level is the message's OWN level, not the current verbosity):
- deep-debug level: `@info` with `[DEEP]` prefix,
- debug level: `@info` with `[DEBUG]` prefix,
- info level: `@info`,
- warn level: `@warn`.

Debug and deep-debug route through `@info` rather than `@debug` deliberately: `@debug` is
compiled out below Julia's own `Logging.Debug` threshold, which would make POLFED's
`verbosity[]` silently ineffective unless the user ALSO raised the global logger's level.
"""
function polfed_log(level::Int, msg; kwargs...)
    should_log(level) || return nothing
    if level >= POLFED_DEEP_DEBUG_LEVEL
        @info "[DEEP] $(msg)" kwargs...
    elseif level >= POLFED_DEBUG_LEVEL
        @info "[DEBUG] $(msg)" kwargs...
    elseif level >= POLFED_INFO_LEVEL
        @info msg kwargs...
    elseif level >= POLFED_WARN_LEVEL
        @warn msg kwargs...
    end
    return nothing
end

"""
    with_verbosity(f, level::Int)

Run `f()` with [`verbosity`](@ref) temporarily set to `level`, restoring the previous level
afterwards — also when `f` throws. This is the scoped alternative to assigning the global
`verbosity[]`, and what the entrypoints' `verbosity` keyword uses internally.

```julia
with_verbosity(POLFED_DEBUG_LEVEL) do
    polfed(H, x0, 50, :maxdos)
end
```
"""
function with_verbosity(f, level::Int)
    previous = verbosity[]
    verbosity[] = level
    try
        return f()
    finally
        verbosity[] = previous
    end
end
