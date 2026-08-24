

# ── stage slices ────────────────────
include("mapping/mapping.jl")
include("dos/dos.jl")
include("target/target.jl")
include("interval/interval.jl")
include("filter/filter.jl")
include("transform/transform.jl")
include("report/report.jl")
include("trace/trace.jl")      # cross-slice log lines (run header, pre-Lanczos summary, outcome)