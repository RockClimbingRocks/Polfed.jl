# Benchmarking: measure the applicable kernels on THIS matrix and machine, sweeping their
# parameters, and hand back something you can pass straight back into `optimize_mapping`.
#
# Sweeping lives here rather than in user code: the set of parameters is a property of the kernel
# (the registry knows it), so the caller should not have to write the loop.
#
# One apply is the WHOLE operator on a `bs x D` block, so its cost scales with `bs` and raw
# per-apply seconds are not comparable across block sizes. Everything that ranks — the sort, the
# table, `best` — therefore works in seconds PER COLUMN (`seconds / bs`); the per-apply number is
# still shown, because that is what a Clenshaw step actually pays.

"""
    MappingBench

One measured configuration: the kernel, the block size, and the median time per apply.

`seconds` is the time for one apply of the full operator on a `bs x D` block. Divide by `bs`
([`per_column`](@ref)) to compare configurations measured at different block sizes.
"""
struct MappingBench
    kernel::MappingKernel
    bs::Int
    seconds::Float64
end

"""
    per_column(r::MappingBench) -> Float64

Seconds per Krylov column, `r.seconds / r.bs` — the block-size-independent cost, and the quantity
every ranking in this file uses.
"""
per_column(r::MappingBench) = r.seconds / r.bs

"""
    MappingBenchmark

The result of [`benchmark_mapping`](@ref): the measured rows, sorted fastest per column first,
plus the `mul!` baseline per block size and the name of the backend that served it.

Behaves as a read-only vector of [`MappingBench`](@ref) (`length`, indexing, iteration), so
anything that worked on the raw vector still works. It exists so that DISPLAYING the result shows
the tables rather than dumping every measured struct — the same `show`-based convention the
POLFED reports follow.

# Fields
- `rows::Vector{MappingBench}`: every measurement, sorted by [`per_column`](@ref).
- `refs::Dict{Int,Float64}`: `mul!` seconds per apply, keyed by block size.
- `ref_label::String`: which package served that `mul!` — see [`_mul_backend`](@ref).
"""
struct MappingBenchmark
    rows::Vector{MappingBench}
    refs::Dict{Int,Float64}
    ref_label::String
end

Base.length(b::MappingBenchmark)          = length(b.rows)
Base.isempty(b::MappingBenchmark)         = isempty(b.rows)
Base.getindex(b::MappingBenchmark, i)     = b.rows[i]
Base.firstindex(b::MappingBenchmark)      = firstindex(b.rows)
Base.lastindex(b::MappingBenchmark)       = lastindex(b.rows)
Base.iterate(b::MappingBenchmark, st...)  = iterate(b.rows, st...)
Base.eltype(::Type{MappingBenchmark})     = MappingBench

"""    Base.show(io, b::MappingBenchmark) — the one-line form, e.g. inside a container."""
function Base.show(io::IO, b::MappingBenchmark)
    print(io, "MappingBenchmark(", length(b.rows), " configurations, bs = ",
          sort!(unique(r.bs for r in b.rows)), ")")
    return nothing
end

"""    Base.show(io, ::MIME"text/plain", b::MappingBenchmark) — the full per-block-size tables."""
Base.show(io::IO, ::MIME"text/plain", b::MappingBenchmark) =
    _show_bench(io, b.rows, b.refs, b.ref_label)

"""
    _time_apply(f!, Y, X; reps) -> Float64 — median seconds per apply over `reps` runs."""
function _time_apply(f!, Y, X; reps::Int = 5)
    f!(Y, X)                                   # warm up / compile
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f!(Y, X)
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return sort!(ts)[cld(length(ts), 2)]
end

"""
    benchmark_mapping(H; bs=(1,4,8), sweep=…, reps=5, verbose=true) -> MappingBenchmark

Measure every kernel that applies to `H`, sweeping each kernel's own parameters, and return the
results sorted fastest first — fastest **per column**, so block sizes rank against each other
fairly (see [`per_column`](@ref)).

Only kernels the registry says can run on `H` are tried, and each is swept over the parameters
*it* declares — so nothing has to be filtered by hand. A generic `mul!` baseline is timed
alongside for reference, once per block size.

The result DISPLAYS as one table per block size, each row ending in the kernel's copy-paste form
(`MappingKernel(:spin_flip, :packed, W = 16, G = 4)`) — paste it straight into `optimize_mapping` — a configuration only competes with the others
measured at its own `bs`, and each section carries its own `mul!` reference. Displaying is all it
takes (`bench` at the REPL, `display(bench)` in a script); nothing is printed as a side effect
except the progress lines, so a caller that only wants [`best`](@ref) gets no output.

This is the intended way to pick tuning parameters. Defaults recorded in a kernel are one
machine's measurement; `W` (vector width) and `G` (accumulators in flight) both trade against
the register file, so their optimum moves with the ISA, the block size, the element type and
the operator. Measure, then pin what you found:

```julia
bench = benchmark_mapping(H; bs = (4,))      # the block size you will actually run
display(bench)                               # the table (automatic at the REPL)
polfed(H, x0, howmany, target; mc = MappingConfig(optimize_mapping = best(bench)))
```

# Arguments
- `bs`: Krylov block sizes to try. Restrict this to the block size of your `x0` when the point is
  to pin a kernel — a parameter optimum moves with `bs`.
- `sweep`: candidate values per parameter NAME. Each kernel is swept over the cross product of
  the entries matching the parameters it declares; names it does not declare are ignored, so
  one `sweep` serves every kernel — `G` reaches the `spin_flip` kernels, `W` both patterns'
  packed kernels, `tile` the `hopping` ones. Combinations a kernel refuses (an unrepresentable
  `bs`/`W` pairing, say) are skipped rather than failing the run.
- `reps`: timed repetitions per configuration; the median is kept. The default is low because the
  sweep is wide (the default grid is ~20 configurations per block size, per pattern); raise it
  when re-measuring a shortlist, where the differences are small.
- `verbose`: print a live progress line as each block size finishes. The tables are NOT a side
  effect of the run — they are how the returned [`MappingBenchmark`](@ref) displays.
"""
function benchmark_mapping(H::AbstractMatrix{T};
                           bs = (1, 4, 8),
                           sweep::NamedTuple = (G = (1, 2, 4, 8, 16), W = (0, 8, 16),
                                               tile = (0, 16, 64)),
                           reps::Int = 5, verbose::Bool = true) where {T}
    s = classify(H)
    D = s.dim
    out = MappingBench[]
    refs = Dict{Int,Float64}()                   # mul! baseline, per block size
    ref_label = "mul! (reference)"               # overwritten with the backend that serves it

    for b in bs
        X = rand(T, b, D)
        Y = similar(X)
        Xc = permutedims(X)                      # column form for the mul! reference
        Yc = similar(Xc)
        ref = _time_apply((Yr, Xr) -> mul!(Yc, H, Xc), Y, X; reps = reps)
        refs[Int(b)] = ref
        ref_label = _mul_backend(Yc, H, Xc)
        measured = 0

        for info in REGISTRY
            can_apply(Val(info.pattern), Val(info.variant), s) || continue
            for pg in _param_grid(info, sweep)
                k = MappingKernel(info.pattern, info.variant; pg...)
                # a kernel may refuse a combination (e.g. W < bs); that is not a run failure
                f! = try
                    build_kernel(k, s, Val(b))
                catch err
                    err isa ArgumentError || rethrow()
                    continue
                end
                push!(out, MappingBench(k, b, _time_apply(f!, Y, X; reps = reps)))
                measured += 1
            end
        end
        verbose && @printf(" bs = %-3d %3d configurations   mul! reference: %8.3f ms/apply\n",
                           b, measured, ref * 1e3)
    end

    sort!(out; by = per_column)
    return MappingBenchmark(out, refs, ref_label)
end

"""
    _param_grid(info::KernelInfo, sweep::NamedTuple) -> Vector{NamedTuple}

Every combination of the parameters `info` declares, restricted to the candidate values in
`sweep`. A kernel with no parameters yields one empty combination, so the caller needs no
special case.
"""
function _param_grid(info::KernelInfo, sweep::NamedTuple)
    names = filter(n -> haskey(sweep, n), keys(info.params))
    isempty(names) && return [NamedTuple()]
    return [NamedTuple{names}(c)
            for c in Iterators.product((getfield(sweep, n) for n in names)...)]
end

"""
    _mul_backend(Yc, H, Xc) -> String

Which package actually serves the `mul!` reference, e.g. `mul! [MKLSparse]`. The 5-argument form
is the one inspected: `mul!(C, A, B)` falls through `LinearAlgebra` to `mul!(C, A, B, true, false)`
in every case, so the 3-argument method says nothing about the backend.

This is a label, not a claim about how the reference is threaded — see [`_thread_context`](@ref).
"""
function _mul_backend(Yc, H, Xc)
    m = try
        which(mul!, Tuple{typeof(Yc), typeof(H), typeof(Xc), Bool, Bool})
    catch
        return "mul! (reference)"
    end
    return string("mul! [", nameof(parentmodule(m)), "]")
end

"""
    _thread_context() -> String

The threading the numbers were taken under. The row kernels are threaded by Julia; MKL's sparse
`mul!` is threaded by MKL itself and follows `MKL_NUM_THREADS`, NOT `Threads.nthreads()` — so the
reference and the kernels can be running on different numbers of cores, and the header says which.
"""
_thread_context() = string("Julia threads: ", Threads.nthreads(),
                           "   BLAS threads: ", BLAS.get_num_threads(),
                           "   MKL_NUM_THREADS: ", get(ENV, "MKL_NUM_THREADS", "unset"))

"""    _kernel_col(k::MappingKernel) -> String — the `spin_flip/packed` column entry."""
_kernel_col(k::MappingKernel) = string(k.pattern, "/", k.variant)

"""    _params_col(k::MappingKernel) -> String — the `W=16, G=4` column entry."""
_params_col(k::MappingKernel) =
    isempty(k.params) ? "—" : join(("$name=$v" for (name, v) in pairs(k.params)), ", ")

"""
    _show_bench(io, rows, refs)

One table per block size, fastest first, columns padded to the widest entry actually present so
everything lines up. `refs` maps a block size to its `mul!` baseline and `ref_label` names the backend that served it;
a block size missing from `refs` simply prints no reference row.

The kernel spec is split into a name column and a parameter column — the two vary independently,
and a single `MappingKernel(...)` string is both long and mostly punctuation.
"""
function _show_bench(io::IO, rows::Vector{MappingBench},
                     refs::AbstractDict{Int,Float64} = Dict{Int,Float64}(),
                     ref_label::AbstractString = "mul! (reference)")
    isempty(rows) && return nothing

    println(io, "\n ", _thread_context())
    wk = maximum(length(_kernel_col(r.kernel)) for r in rows)
    wk = max(wk, length("kernel"), length(ref_label))
    wp = maximum(length(_params_col(r.kernel)) for r in rows)
    wp = max(wp, length("params"))
    wt = 12                                        # "1234.567 ms"
    ws = maximum(length(sprint(show, r.kernel)) for r in rows)
    ws = max(ws, length("optimize_mapping ="))

    for b in sort!(unique(r.bs for r in rows))
        println(io)
        println(io, " block size bs = ", b)
        println(io, " ", rpad("kernel", wk), "  ", rpad("params", wp), "  ",
                lpad("per column", wt), "   ", "optimize_mapping =")
        println(io, " ", "─"^(wk + wp + wt + ws + 7))
        for r in rows
            r.bs == b || continue
            _bench_row(io, _kernel_col(r.kernel), _params_col(r.kernel), r.seconds, b,
                       wk, wp, wt, sprint(show, r.kernel))
        end
        haskey(refs, b) && _bench_row(io, ref_label, "—", refs[b], b, wk, wp, wt, "—")
    end
    return nothing
end

"""
    _bench_row(io, name, params, seconds, bs, wk, wp, wt, spec)

One padded table line. `spec` is the last column and is NOT padded: it is the copy-paste form of
the kernel, so nothing may follow it on the line.
"""
function _bench_row(io::IO, name::AbstractString, params::AbstractString,
                    seconds::Float64, bs::Int, wk::Int, wp::Int, wt::Int,
                    spec::AbstractString)
    println(io, " ", rpad(name, wk), "  ", rpad(params, wp), "  ",
            lpad(@sprintf("%.3f ms", seconds / bs * 1e3), wt), "   ", spec)
    return nothing
end

"""
    best(bench; bs = nothing) -> MappingKernel

The fastest measured kernel — fastest **per column** ([`per_column`](@ref)), so a run that swept
several block sizes does not simply return whatever was measured at the smallest one (one apply on
a `bs x D` block costs `bs` times as much work, so raw per-apply seconds always favour `bs = 1`).

`bs` restricts the comparison to one block size — the one you will actually run, since a
parameter optimum moves with the block size:

```julia
bench = benchmark_mapping(H)               # sweep several block sizes
k     = best(bench; bs = 4)                # pin the winner AT bs = 4
polfed(H, x0, howmany, target; mc = MappingConfig(optimize_mapping = k))
```

The block size is not part of what is returned: `optimize_mapping` takes a kernel, and the block
size comes from your `x0`.

# Throws
- `ArgumentError` on an empty result, or when `bs` names a block size that was not measured —
  rather than returning something arbitrary.
"""
function best(bench::MappingBenchmark; bs::Union{Nothing,Integer} = nothing)
    return best(bench.rows; bs = bs)
end

function best(bench::Vector{MappingBench}; bs::Union{Nothing,Integer} = nothing)
    isempty(bench) && throw(ArgumentError("best: no benchmark results."))
    isnothing(bs) && return argmin(per_column, bench).kernel
    rows = filter(r -> r.bs == bs, bench)
    isempty(rows) && throw(ArgumentError(
        "best: no results at block size $bs; measured block sizes: " *
        "$(sort!(unique(r.bs for r in bench)))."))
    return argmin(per_column, rows).kernel
end
