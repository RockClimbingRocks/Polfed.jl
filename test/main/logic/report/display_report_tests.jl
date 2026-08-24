# The `display_report` print verb: a keyword-argument front end over the `show(io,
# MIME"text/plain"(), …)` renderers. The rendering itself is covered by report_tests.jl /
# PatchedReport_tests.jl — what is tested here is the wrapper's own behavior: that each
# keyword lands on the IO property it stands for, that both call forms agree, and that the
# accepted argument set is the report/result union and nothing else. Uses the shared fixture.

import UniPolfed: display_report, _auto_color, ReportLike, Report, PolfedResult

@testset "display_report" begin
    # Render through the verb; `sprint` gives a non-TTY sink, so `color` defaults to false.
    _via_verb(r; kwargs...) = sprint(io -> display_report(io, r; kwargs...))

    @testset "matches the show it wraps, plus a trailing newline" begin
        @test _via_verb(RES) == _render(RES) * "\n"
        @test _via_verb(RES.report) == _render(RES.report) * "\n"
    end

    @testset "keywords drive the two IO properties" begin
        @test _via_verb(RES; color = true) == _render(RES; color = true) * "\n"
        @test occursin("\e[", _via_verb(RES; color = true))
        # A non-TTY sink is the default-off case for color: no escape sequences unasked.
        @test !occursin("\e[", _via_verb(RES))
        @test !_auto_color(IOBuffer())

        detailed = _via_verb(RES.report; convergence_details = true)
        @test occursin("Convergence check ran", detailed)
        @test !occursin("Convergence check ran", _via_verb(RES.report))
        @test length(detailed) > length(_via_verb(RES.report))    # the table is appended
    end

    @testset "io positional and io keyword agree" begin
        buf_pos = IOBuffer(); display_report(buf_pos, RES)
        buf_kw  = IOBuffer(); display_report(RES; io = buf_kw)
        @test String(take!(buf_pos)) == String(take!(buf_kw))
    end

    @testset "accepts the report/result union only" begin
        @test RES isa ReportLike && RES.report isa ReportLike
        @test_throws MethodError display_report(42)
        @test_throws MethodError display_report(IOBuffer(), "not a report")
    end
end
