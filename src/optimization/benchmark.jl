# benchmark_mappings: compare the registered kernels × params × block sizes on THIS machine.
# Runs at the current launch thread count (thread count is fixed per process — for a thread sweep
# relaunch `julia -t k`). Every kernel is checked against a `mul!(Y,H,X)` reference, which doubles
# as the `:mul` baseline row. No BenchmarkTools dependency — a simple budgeted min-time loop.

"""
    MappingBench

Result of [`benchmark_mappings`](@ref).

# Fields
- `dim::Int`, `threads::Int`, `eltype::DataType`, `L::Int`: run metadata.
- `table::Vector{NamedTuple}`: one row per `(kernel, G, W, bs)` — `ms_per_col`, `gbps`, `relerr`,
  `ok` (correctness), and `applicable` (`false` → shown as `x`).
- `best::Dict{Int,NamedTuple}`: fastest applicable kernel per block size.
- `fastest::NamedTuple`: the single fastest `(kernel, …, bs, ms_per_col)` overall.

Pretty-prints as a `kernel/params × bs` table of per-column milliseconds.
"""
struct MappingBench
    dim::Int
    threads::Int
    eltype::DataType
    L::Int
    table::Vector{NamedTuple}
    best::Dict{Int,NamedTuple}
    fastest::NamedTuple
end

"""
    _time_min(f, secs) -> Float64

Minimum wall time (seconds) of `f()` over repeated calls within the budget `secs` (≥ 3 reps).
`f` is warmed up once first (compilation / cache) before timing.
"""
function _time_min(f, secs::Real)
    f()
    tmin = Inf
    t0 = time()
    reps = 0
    while true
        t = @elapsed f()
        tmin = min(tmin, t)
        reps += 1
        (time() - t0 ≥ secs && reps ≥ 3) && break
        reps ≥ 1_000_000 && break
    end
    return tmin
end

_relerr(A, B) = norm(vec(A) .- vec(B)) / max(norm(vec(B)), floatmin(Float64))

"""
    _naive_row_ref!(Y, X, s::MappingStructure) -> nothing

Trusted row-layout reference: `Y[b,i] = diags[i]·X[b,i] + hx·Σ_c X[b, nb(i,c)]`, computed
directly from the structure with a plain scalar loop (no transpose, no `mul!`). Used to check
the fast kernels in [`benchmark_mappings`](@ref).
"""
function _naive_row_ref!(Y, X, s::MappingStructure)
    bs = size(X, 1)
    @inbounds for i in 1:s.dim
        d = s.diags[i]
        for b in 1:bs
            Y[b, i] = d * X[b, i]
        end
        base = (i - 1) * s.L
        for c in 1:s.L
            j = Int(s.nb32[base + c])
            for b in 1:bs
                Y[b, i] += s.hx * X[b, j]
            end
        end
    end
    return nothing
end

"""
    benchmark_mappings(H; bs_list=(1,2,4,8), G_list=(1,2,4,8), W_list=(1,2,4,8), secs=1.0) -> MappingBench

Benchmark every applicable row mapping over the given block sizes and `G`/`W` parameter sweeps,
each verified against `mul!(Y,H,X)`. See [`MappingBench`](@ref). `secs` is the per-cell time
budget (total ≈ `secs × #cells`). `W_list` is used only by the flip kernels (later stage).

Kernels whose structure does not fit `H` (Stage 1: the flip kernels) are recorded as
non-`applicable` and shown as `x`.
"""
function benchmark_mappings(H::AbstractMatrix; bs_list = (1, 2, 4, 8), G_list = (1, 2, 4, 8),
                            W_list = (1, 2, 4, 8), secs::Real = 1.0)
    s = value_grouping(H)
    T = eltype(H)
    D = s.dim
    L = s.L
    rows = NamedTuple[]

    ampbytes = (L + 1) * D * sizeof(T)              # rough amplitude traffic per column
    gbps(t) = ampbytes / t / 1e9
    percol(t, bs) = t * 1e3 / bs

    for bs in bs_list
        Xr = randn(T, bs, D)
        Yref = zeros(T, bs, D)
        _naive_row_ref!(Yref, Xr, s)                # trusted row reference (no transpose)

        # :mul baseline — plain column-layout sparse matvec on independent data
        Xc = randn(T, D, bs)
        Yc = similar(Xc)
        tmul = _time_min(() -> mul!(Yc, H, Xc), secs)
        push!(rows, (kernel = :mul, G = 0, W = 0, bs = bs,
                     ms_per_col = percol(tmul, bs), gbps = gbps(tmul), relerr = 0.0,
                     ok = true, applicable = true))

        Yr = zeros(T, bs, D)
        # csr_simd (no G)
        m! = build_row_mapping(:csr_simd, s, Val(bs))
        fill!(Yr, 0); m!(Yr, Xr); err = _relerr(Yr, Yref)
        t = _time_min(() -> m!(Yr, Xr), secs)
        push!(rows, (kernel = :csr_simd, G = 0, W = 0, bs = bs,
                     ms_per_col = percol(t, bs), gbps = gbps(t), relerr = err,
                     ok = err < 1e-8, applicable = true))

        # ell32_svec / csr64_svec over G
        for id in (:ell32_svec, :csr64_svec), G in G_list
            m! = build_row_mapping(id, s, Val(bs); G = G)
            fill!(Yr, 0); m!(Yr, Xr); err = _relerr(Yr, Yref)
            t = _time_min(() -> m!(Yr, Xr), secs)
            push!(rows, (kernel = id, G = G, W = 0, bs = bs,
                         ms_per_col = percol(t, bs), gbps = gbps(t), relerr = err,
                         ok = err < 1e-8, applicable = true))
        end

        # flip kernels — not applicable to a bare matrix in Stage 1 (shown as x)
        push!(rows, (kernel = :shuf, G = 0, W = 0, bs = bs,
                     ms_per_col = NaN, gbps = NaN, relerr = NaN,
                     ok = false, applicable = false))
    end

    best = Dict{Int,NamedTuple}()
    for bs in bs_list
        cand = filter(r -> r.bs == bs && r.applicable && r.kernel !== :mul && r.ok, rows)
        isempty(cand) || (best[bs] = argmin(r -> r.ms_per_col, cand))
    end
    applic = filter(r -> r.applicable && r.kernel !== :mul && r.ok, rows)
    fastest = isempty(applic) ? (kernel = :none, bs = 0, ms_per_col = NaN) : argmin(r -> r.ms_per_col, applic)

    return MappingBench(D, Threads.nthreads(), T, L, rows, best, fastest)
end

# ── pretty printing: kernel/params × bs table of per-column ms ─────────────────
_rowlabel(k::Symbol, G::Int) = k === :mul ? "mul! (ref)" :
                               k === :csr_simd ? "csr_simd" :
                               string(k, "  G=", G)

function Base.show(io::IO, ::MIME"text/plain", r::MappingBench)
    bss = sort(unique(getfield.(r.table, :bs)))
    # ordered unique (kernel, G) row keys in registry order
    order = [(:mul, 0), (:csr_simd, 0)]
    for id in (:csr64_svec, :ell32_svec), G in sort(unique(t.G for t in r.table if t.kernel === id))
        push!(order, (id, G))
    end
    cell = Dict((t.kernel, t.G, t.bs) => t for t in r.table)

    println(io, "timings — mapping per column [ms], D=$(r.dim), threads=$(r.threads), $(r.eltype)")
    println(io)
    print(io, rpad(" kernel / params", 18), "|")
    for bs in bss
        print(io, lpad("bs=$bs", 8))
    end
    println(io)
    println(io, " ", "─"^17, "┼", "─"^(8 * length(bss)))
    for (k, G) in order
        print(io, " ", rpad(_rowlabel(k, G), 16), "|")
        for bs in bss
            t = get(cell, (k, G, bs), nothing)
            print(io, lpad(t === nothing ? "-" : (isnan(t.ms_per_col) ? "x" :
                           (t.ok ? _fmt(t.ms_per_col) : "ERR")), 8))
        end
        println(io)
    end
    # flips section (Stage 1: not applicable → x)
    if any(t -> t.kernel === :shuf, r.table)
        println(io, " ─ flips (need matching off-diagonal positions) ", "─"^max(0, 8 * length(bss) - 22))
        print(io, " ", rpad("shuf", 16), "|")
        for _ in bss
            print(io, lpad("x", 8))
        end
        println(io, "   (σˣ flip mapping; not built in Stage 1)")
    end
    println(io)
    print(io, " best per bs: ")
    for bs in bss
        haskey(r.best, bs) && print(io, " bs=$bs → $(_rowlabel(r.best[bs].kernel, r.best[bs].G)) ($(_fmt(r.best[bs].ms_per_col)))  ")
    end
    println(io)
    # correctness footer
    bad = filter(t -> t.applicable && !t.ok, r.table)
    isempty(bad) || println(io, " ⚠ ", length(bad), " kernel(s) failed the correctness check!")
    return nothing
end

_fmt(x) = @sprintf("%.3f", x)
