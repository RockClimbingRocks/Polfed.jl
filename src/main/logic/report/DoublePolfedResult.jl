# DoublePolfedResult: the value `double_polfed` returns. It is deliberately THIN — two ordinary
# `PolfedResult`s plus the decision that produced them. The wrapper exists for one reason: on the
# `:double` path both windows come out of ONE Lanczos run, so both sub-results carry the SAME
# `Report`. Handing back a bare tuple would invite reading that run's cost twice; the wrapper is
# where the run structure is stated once, in the header, before the two reports are shown.

"""
    DoublePolfedResult

Result of [`double_polfed`](@ref): the eigenpairs of the two requested windows, as two ordinary
[`PolfedResult`](@ref)s, plus the strategy that produced them.

The two sub-results are returned in the **caller's argument order** — `first` corresponds to
`target1`, `second` to `target2` — regardless of which window sits lower in the spectrum
(internally the windows are always ordered low→high, because the double filter requires it).

Iterate or destructure it to get the two results, each an ordinary [`PolfedResult`](@ref) that
in turn destructures as `(vals, vecs, report)`:

```julia
low, high = double_polfed(H, x0, (-0.4, -0.3), (0.3, 0.4))

vals, vecs, report = low          # each window unpacks like any other POLFED run
v1 = eigenvector(low, 1)          # layout-aware per-eigenvector access
```

Indexing (`r[1]`, `r[2]`) does the same; `length(r)` is always `2`. Displaying it prints the
decision header followed by both sub-reports.

# Fields
- `first::R1`, `second::R2`: the two [`PolfedResult`](@ref)s, in the caller's argument order.
- `strategy::Symbol`: `:double` (one joint [`DoubleDeltaFilter`](@ref) run served both windows)
  or `:separate` (two independent single-filter runs).
- `reason::Symbol`: why that strategy was taken — `:ratio`, `:forced`, `:edge`, `:double_failed`
  (see [`DoubleStrategy`](@ref)).
- `order1::Int`, `order2::Int`: the single-window polynomial orders `K₁`, `K₂` of the LOWER and
  HIGHER window — the decision inputs. Both are computed even on the `:double` path (they are
  what the decision is made on), so the two paths stay comparable in a benchmark.
- `order_double::Int`: the joint filter's order `K_d` paid per matvec on the `:double` path, `0`
  on the `:separate` path.
- `ratio::Float64`: `min(K₁,K₂)/max(K₁,K₂)`, compared against
  [`Defaults.double_polfed.double_ratio`](@ref).

# Notes
On the `:double` path the two sub-results share one [`Report`](@ref): its convergence counts,
`H·v` cost and timings describe the single joint run, not one window's share of it. The shared
`:bounds`/`:dos` stages (resolved once for both windows) likewise appear in both reports on the
`:separate` path.
"""
struct DoublePolfedResult{R1<:PolfedResult, R2<:PolfedResult}
    first::R1
    second::R2
    strategy::Symbol
    reason::Symbol
    order1::Int
    order2::Int
    order_double::Int
    ratio::Float64
end

Base.length(::DoublePolfedResult) = 2
Base.firstindex(::DoublePolfedResult) = 1
Base.lastindex(::DoublePolfedResult) = 2

"""    getindex(r::DoublePolfedResult, i) -> PolfedResult — window `i` (`1` or `2`, caller's order)."""
function Base.getindex(r::DoublePolfedResult, i::Integer)
    i == 1 && return r.first
    i == 2 && return r.second
    throw(BoundsError(r, i))
end

function Base.iterate(r::DoublePolfedResult, i::Int=1)
    i > 2 && return nothing
    return (r[i], i + 1)
end

"""
    _double_strategy_line(r::DoublePolfedResult) -> String

One-line summary of the strategy decision: the mode and the orders it was made on — the numbers a
benchmark wants to see without opening either sub-report.
"""
function _double_strategy_line(r::DoublePolfedResult)
    if r.strategy === :double
        return @sprintf("one joint DoubleDeltaFilter run  (K_d = %d, from K₁ = %d, K₂ = %d)",
                        r.order_double, r.order1, r.order2)
    end
    return @sprintf("two separate single-filter runs  (K₁ = %d, K₂ = %d)", r.order1, r.order2)
end

"""
    show(io::IO, ::MIME"text/plain", r::DoublePolfedResult)

Print the `double POLFED result` header — strategy, decision numbers, and how many eigenpairs
each window returned — followed by both sub-reports. Honors the same IO properties as
[`Report`](@ref) (`:color`, `:convergence_details`).
"""
function Base.show(io::IO, ::MIME"text/plain", r::DoublePolfedResult)
    f = Formatter(get(io, :color, false))
    println(io, blue(f, "double POLFED result"))
    println(io, blue(f, "═"^60))
    println(io, "  strategy : ", bold(f, string(r.strategy)), "  — ", _double_strategy_line(r))
    println(io, @sprintf("  decision : ratio = %.3f  (threshold %.3f, reason :%s)",
                         r.ratio, Defaults.double_polfed.double_ratio, r.reason))
    println(io, "  windows  : ", length(r.first.vals), " + ", length(r.second.vals), " eigenpairs")
    r.strategy === :double && println(io, "  note     : both windows come from ONE Lanczos run; " *
                                          "the two reports below describe that same run.")
    println(io)
    println(io, blue(f, "── window 1 " * "─"^49))
    show(io, MIME"text/plain"(), r.first)
    println(io)
    println(io, blue(f, "── window 2 " * "─"^49))
    show(io, MIME"text/plain"(), r.second)
    return nothing
end

Base.show(io::IO, r::DoublePolfedResult) =
    print(io, "DoublePolfedResult(", length(r.first.vals), " + ", length(r.second.vals),
              " eigenpairs, :", r.strategy, ")")
