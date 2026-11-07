# Stage timing: name-addressed, self-accumulating, ambient.
#
# Replaced the index-addressed `@addtime!(wallarr, cpuarr, idx, expr)` macro (since deleted), whose
# three costs were (a) a magic index that had to stay in sync with a separate names array, (b) the
# timed call being swallowed by the macro (`@addtime! wt ct 4 (result = lanczos(...))`), and (c) two
# timing arrays threaded as parameters through hot-loop function boundaries just to accumulate into.
#
# Here a stage is addressed by NAME and accumulates itself: the first `@stage :matvec …` creates the
# entry, every later one adds to it (wall, CPU, and a hit count), so a loop body needs no special
# casing between its first and its thousandth iteration. `@stage` RETURNS the expression's value, so
# the natural `result = @stage :lanczos lanczos(…)` reads normally.
#
# The timer is AMBIENT (task-local): `@stages` opens a scope, and any `@stage` running inside that
# dynamic extent — however deep the call stack — records into it. That is what lets `lanczositer!`
# drop its `walltimes`/`cputimes` parameters entirely. Nesting is preserved: a `@stage` hit while
# another is open becomes its CHILD, so timings form a tree rather than two disconnected flat arrays.
#
# Opt-in by the caller: with no `@stages` scope open, `@stage` is a pass-through that just evaluates
# the expression. Library code can therefore be annotated unconditionally and costs nothing when the
# caller does not ask for timings.
#
# TASKS: `task_local_storage` is per-task and is NOT inherited by spawned tasks, so a `@stage` inside
# a `Threads.@spawn`/`@distributed` region records nothing. Annotate OUTSIDE the parallel region (or
# pass a timer explicitly with the 3-argument `@stage timer :name expr` form).

"""
    StageTimer()

A node in a tree of accumulated stage timings. The value returned by [`@stages`](@ref) is the root;
each [`@stage`](@ref) name encountered inside becomes a child, and a name hit repeatedly accumulates
(wall time, CPU time, and hit count) into the same child.

Children keep **insertion order** — the order the algorithm first ran them — so a rendered table
reads in execution order rather than alphabetically.

# Fields
- `name::Symbol`: this node's stage name (`:root` for a timer created by `StageTimer()`).
- `wall::Float64`, `cpu::Float64`: accumulated seconds.
- `n::Int`: how many times this stage was entered.
- `order::Vector{Symbol}`, `children::Dict{Symbol,StageTimer}`: nested stages.

# Accessors
`t[:name]` (wall seconds), `t[:name, :cpu]`, `t[:name, :n]`, [`stage_names`](@ref),
[`total_wall`](@ref), [`total_cpu`](@ref).
"""
mutable struct StageTimer
    name::Symbol
    wall::Float64
    cpu::Float64
    n::Int
    order::Vector{Symbol}
    children::Dict{Symbol,StageTimer}
end

StageTimer(name::Symbol = :root) = StageTimer(name, 0.0, 0.0, 0, Symbol[], Dict{Symbol,StageTimer}())

# ── ambient (task-local) current node ─────────────────────────────────────────

const _STAGE_TLS_KEY = :polfed_stage_timer

"""Current ambient stage node, or `nothing` when no [`@stages`](@ref) scope is open."""
_stage_current() = get(task_local_storage(), _STAGE_TLS_KEY, nothing)

"""Install `node` as the ambient current node (`nothing` clears it)."""
function _stage_set_current!(node)
    if node === nothing
        delete!(task_local_storage(), _STAGE_TLS_KEY)
    else
        task_local_storage()[_STAGE_TLS_KEY] = node
    end
    return node
end

"""
    _stage_child!(parent::StageTimer, name::Symbol) -> StageTimer

The child of `parent` named `name`, created (and appended to `parent.order`) on first use — this is
the "if the field is already there, just add to it" lookup behind [`@stage`](@ref).
"""
function _stage_child!(parent::StageTimer, name::Symbol)
    child = get(parent.children, name, nothing)
    if child === nothing
        child = StageTimer(name)
        parent.children[name] = child
        push!(parent.order, name)
    end
    return child
end

"""Fold one measurement into `node` (accumulating, never replacing)."""
function _stage_record!(node::StageTimer, wall::Float64, cpu::Float64)
    node.wall += wall
    node.cpu  += cpu
    node.n    += 1
    return node
end

# ── macros ────────────────────────────────────────────────────────────────────

"""
    @stages expr -> StageTimer

Open a stage-timing scope for the dynamic extent of `expr` and return the root [`StageTimer`](@ref).
Every [`@stage`](@ref) executed inside — including inside functions called from `expr` — records
into it, so hot-loop helpers need no timing parameters.

`expr` runs inside a `try`/`finally` (so the ambient scope unwinds even on a throw), and `try`
**introduces a local scope**. A variable you need after the block must therefore be declared
*before* it, exactly as you would around a `try`:

```julia
result = nothing                      # declared OUTSIDE …
timings = @stages while true
    result = step!(...)               # … so this rebinds the outer `result`
    done && break
end
@show timings[:matvec] result
```

A variable first assigned *inside* the block is local to it and not visible afterwards.

Scopes nest: an inner `@stages` shadows the outer one for its extent, and the previous ambient node
is restored afterwards even if `expr` throws.
"""
macro stages(expr)
    quote
        local __root = StageTimer()
        local __prev = _stage_current()
        _stage_set_current!(__root)
        try
            $(esc(expr))
        finally
            _stage_set_current!(__prev)
        end
        __root
    end
end

"""
    @stage name expr        -> value of expr   (records into the ambient scope)
    @stage timer name expr  -> value of expr   (records into `timer` explicitly)

Time `expr` and accumulate it under `name`, returning **`expr`'s value** — so the assignment stays
outside the macro and the call reads normally:

```julia
result = @stage :lanczos lanczos(f!_transformed, x0, eig_target; maxdim = maxdim)
```

Repeat hits add: the first creates the entry, each later one accumulates wall time, CPU time, and
the hit count. A `@stage` entered while another is open nests as its child.

With **no [`@stages`](@ref) scope open** the ambient form is a pass-through — `expr` is evaluated
untimed — so annotated library code costs nothing until a caller asks for timings. The 3-argument
form always records, and is the escape hatch where an ambient timer cannot reach (e.g. inside a
spawned task).

Timing is restored correctly if `expr` throws.
"""
macro stage(name, expr)
    quote
        local __parent = _stage_current()
        if __parent === nothing
            $(esc(expr))                      # no scope open → pass through, untimed
        else
            _stage_run!(__parent, $(esc(name))) do
                $(esc(expr))
            end
        end
    end
end

macro stage(timer, name, expr)
    quote
        _stage_run!($(esc(timer)), $(esc(name))) do
            $(esc(expr))
        end
    end
end

"""
    _stage_run!(f, parent::StageTimer, name::Symbol)

Run `f()` as the stage `name` under `parent`: make the child ambient for the call, measure wall and
CPU time, accumulate, restore the previous ambient node, and return `f()`'s value. Shared by both
[`@stage`](@ref) forms.
"""
function _stage_run!(f, parent::StageTimer, name::Symbol)
    node = _stage_child!(parent, name)
    prev = _stage_current()
    _stage_set_current!(node)
    t0 = time_ns()
    c0 = _cpu_time_ns()
    try
        return f()
    finally
        _stage_record!(node, _elapsed(time_ns(), t0, 1e-9), _elapsed(_cpu_time_ns(), c0, 1e-9))
        _stage_set_current!(prev)
    end
end

"""
    _elapsed(stop, start, scale) -> Float64

`(stop - start) * scale` as a **non-negative** Float64.

Both clocks return **unsigned** counters, so a plain `stop - start` wraps to ~1.8e19 (`2^64`) the
moment `stop < start` — a one-tick inversion renders as 18 million years. The guard is an unsigned
**comparison**, not a widening cast: `time_ns()` counts nanoseconds since boot and routinely
exceeds `typemax(Int64)`, so `Int64(stop)` would throw `InexactError` on a long-running machine.
"""
@inline _elapsed(stop::Integer, start::Integer, scale::Float64) =
    stop >= start ? (stop - start) * scale : 0.0

"""
    _cpu_time_ns() -> UInt64

Process CPU time (user + system, summed over threads) in nanoseconds, for the CPU column of a
[`StageTimer`](@ref).

Reads `CLOCK_PROCESS_CPUTIME_ID` via `clock_gettime` directly rather than using `CPUTime.jl`'s
`CPUtime_us()`, which is **not safe to build a report on**: it `Libc.malloc`s an *uninitialized*
`uv_rusage_t`, ignores `uv_getrusage`'s return code, then reads `tv_sec`/`tv_usec` at hardcoded
offsets and multiplies by 10⁶. When the call leaves the buffer untouched it returns stale heap
bytes scaled by a million — observed here as a steady `1.3578360341175732e13`-second reading for
`:reorthog` (~430 000 years), byte-identical across runs because `malloc` recycles the address.
`clock_gettime` writes into a `Ref` we own, and its status is checked.

Returns `0` when the clock is unavailable (non-Unix, or a failing call). Deltas then floor to `0`
via [`_elapsed`](@ref), so CPU time reads as zero rather than as fiction.
"""
function _cpu_time_ns end

@static if Sys.isunix()
    const _CLOCK_PROCESS_CPUTIME_ID = Cint(2)   # POSIX; per-process CPU clock

    function _cpu_time_ns()
        ts = Ref((Clong(0), Clong(0)))          # struct timespec { tv_sec; tv_nsec; }
        status = ccall(:clock_gettime, Cint, (Cint, Ref{Tuple{Clong,Clong}}),
                       _CLOCK_PROCESS_CPUTIME_ID, ts)
        status == 0 || return UInt64(0)
        sec, nsec = ts[]
        (sec < 0 || nsec < 0) && return UInt64(0)
        return UInt64(sec) * UInt64(1_000_000_000) + UInt64(nsec)
    end
else
    _cpu_time_ns() = UInt64(0)                  # no process-CPU clock: report 0, never garbage
end

# ── accessors ─────────────────────────────────────────────────────────────────

"""    stage_names(t::StageTimer) -> Vector{Symbol} — child stages in insertion (execution) order."""
stage_names(t::StageTimer) = t.order

"""    haskey(t::StageTimer, name) -> Bool — whether the stage was ever entered."""
Base.haskey(t::StageTimer, name::Symbol) = haskey(t.children, name)

"""    t[:name] -> Float64 — accumulated WALL seconds for stage `name` (`0.0` if never entered)."""
Base.getindex(t::StageTimer, name::Symbol) =
    haskey(t.children, name) ? t.children[name].wall : 0.0

"""
    t[:name, field] -> Float64 | Int

Accumulated `:wall` / `:cpu` seconds, the hit count `:n`, or the child node itself (`:node`) for
stage `name`. Missing stages read as zero, so report code needs no `haskey` guards.
"""
function Base.getindex(t::StageTimer, name::Symbol, field::Symbol)
    child = get(t.children, name, nothing)
    field === :node && return child
    child === nothing && return field === :n ? 0 : 0.0
    field === :wall && return child.wall
    field === :cpu  && return child.cpu
    field === :n    && return child.n
    throw(ArgumentError("StageTimer: unknown field :$field (expected :wall, :cpu, :n, or :node)"))
end

"""    total_wall(t::StageTimer) -> Float64 — summed wall seconds over `t`'s direct child stages."""
total_wall(t::StageTimer) = sum(c -> c.wall, values(t.children); init = 0.0)

"""    total_cpu(t::StageTimer) -> Float64 — summed CPU seconds over `t`'s direct child stages."""
total_cpu(t::StageTimer) = sum(c -> c.cpu, values(t.children); init = 0.0)

# ── display ───────────────────────────────────────────────────────────────────

Base.show(io::IO, t::StageTimer) =
    print(io, "StageTimer(", t.name, ", ", length(t.order), " stages, ",
          @sprintf("%.3g s wall", total_wall(t)), ")")

function Base.show(io::IO, ::MIME"text/plain", t::StageTimer)
    println(io, "StageTimer — ", @sprintf("%.4g s wall / %.4g s cpu", total_wall(t), total_cpu(t)))
    _show_stages(io, t, total_wall(t), "")
    return nothing
end

function _show_stages(io::IO, t::StageTimer, total::Float64, indent::String)
    for (i, name) in enumerate(t.order)
        c    = t.children[name]
        last = i == length(t.order)
        pct  = total > 0 ? c.wall / total * 100 : 0.0
        println(io, indent, last ? "└─ " : "├─ ", rpad(string(name), 12), " ",
                @sprintf("%8.4g s", c.wall), @sprintf("  %5.1f%%", pct),
                c.n > 1 ? @sprintf("  (×%d)", c.n) : "")
        isempty(c.order) || _show_stages(io, c, total, indent * (last ? "   " : "│  "))
    end
    return nothing
end
