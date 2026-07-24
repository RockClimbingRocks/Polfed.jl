# Report rendering & behavior: shared helpers, the four sections, color on/off, the optional
# convergence-history table, and GPU CPU-time suppression. Uses the shared fixture.

import UniPolfed: _underscore, _fmt_secs, _safe_pct, Report, render_timings, Formatter
using UniPolfed.Common: stage_names, total_wall

@testset "Report rendering" begin
    @testset "formatting helpers" begin
        @test _underscore(471_800) == "471_800"
        @test _underscore(5) == "5"
        @test _underscore(1_000) == "1_000"
        @test _fmt_secs(30)[2] == "seconds"
        @test _fmt_secs(200)[2] == "minutes"
        @test _fmt_secs(10_000)[2] == "hours"
        @test _safe_pct(1, 4) == 25.0
        @test _safe_pct(2, 0) == 0.0                         # guards divide-by-zero
    end

    @testset "color is driven by the IO :color property" begin
        plain   = _render(RES; color = false)
        colored = _render(RES; color = true)
        @test !occursin("\e[", plain)                        # file sink → no ANSI
        @test occursin("\e[", colored)                       # TTY sink → ANSI
    end

    @testset "all four sections and their key lines are present" begin
        s = _render(RES)
        @test occursin("POLFED result", s)
        for section in ("Mapping", "Spectral Transformation", "Convergence", "Timings")
            @test occursin(section, s)
        end
        # spectral-transform headline cost + the two convergence gates, each with its own
        # tolerance and worst residual, + the Krylov budget usage line.
        @test occursin("mat-vec muls", s)
        @test occursin("Number of converged eigenpairs", s)
        @test occursin("Ritz", s) && occursin("Eigen", s)
        @test occursin("max res.", s)
        @test occursin("Iterations needed", s)
        @test occursin("Returned set", s)
    end

    @testset "timings: outer breakdown + Lanczos decomposition, wall percentages" begin
        s = _render(RES)
        # The outer split is deliberately COARSE — Lanczos vs everything that set it up. Assert on
        # that LINE, not the whole report: "filter"/"interval" are legitimate words in the
        # Transformation section. The per-stage detail stays on `report.timer` for whoever wants it.
        dist = only(filter(l -> occursin("Walltime distribution", l), split(s, '\n')))
        @test occursin("Lanczos", dist) && occursin("preparation", dist)
        for noisy in ("bounds", "interval", "filter", "transform", "DoS")
            @test !occursin(noisy, dist)                         # not at report altitude
        end
        @test count(==('·'), dist) == 1                          # exactly two entries
        for label in ("matvec", "reorthog.", "tridiag.", "conv.check") # Lanczos-internal
            @test occursin(label, s)
        end
        @test occursin("%", s)
        @test occursin("walltime", s) && occursin("CPU time", s)   # CPU run shows both
    end

    @testset "the pipeline records every POLFED stage, in execution order" begin
        # Guards against a stage silently never being annotated — the pre-StageTimer code allocated
        # four slots but only ever filled `Lanczos`, so the report read `bounds 0% · DoS 0%` on
        # every run, and the printed total under-reported the wall time it never measured.
        t = RES.report.timer
        @test stage_names(t) == [:bounds, :dos, :interval, :filter, :transform, :lanczos]
        @test all(s -> t[s] > 0.0, stage_names(t))
        @test total_wall(t) ≈ sum(t[s] for s in stage_names(t))
        # `resolve_target` + `resolve_interval` share the :interval name, so it accumulates BOTH
        # calls into one entry (matching the window pipeline, where `resolve_window` is one call).
        @test t[:interval, :n] == 2
        @test all(s -> t[s, :n] == 1, (:bounds, :dos, :filter, :transform, :lanczos))
        # Lanczos dominates: it is the bottleneck the whole design targets.
        @test t[:lanczos] > sum(t[s] for s in (:bounds, :dos, :interval, :filter, :transform))
    end

    @testset "convergence-history table only when :convergence_details is set" begin
        @test !occursin("Convergence check ran", _render(RES; details = false))
        withtbl = _render(RES; details = true)
        @test occursin("Convergence check ran", withtbl)
        @test occursin("Checking", withtbl) && occursin("Krylov dim.", withtbl)
    end

    @testset "GPU runs omit CPU time" begin
        # Simulate a GPU factorization (basis columns land on the GPU) without a GPU host.
        frg = deepcopy(RES.report.factorization)
        frg.ncols_gpu_needed   = frg.ncols_cpu_needed
        frg.ncols_gpu_reserved = frg.ncols_cpu_reserved
        frg.ncols_cpu_needed   = 0
        frg.ncols_cpu_reserved = 0
        rg = Report(RES.report.mapping, RES.report.transform, frg, RES.report.timer)
        s = sprint(io -> render_timings(io, Formatter(false), rg))
        @test occursin("walltime", s)
        @test !occursin("CPU time", s)                       # suppressed on GPU
    end
end
