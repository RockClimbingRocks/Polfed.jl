using Test
using UniPolfed.Common: plain_table, render_detail_table, set_table_renderer!, reset_table_renderer!

# Ground truth for a table renderer is its exact glyph layout, so these assert on the rendered
# string. The PrettyTables backend is exercised too: the package extension must produce a table
# with the same shape and the same cells, since the two are meant to be interchangeable.

"""Render through `f` into a buffer and return the resulting string."""
_capture(f, args...; kwargs...) = (io = IOBuffer(); f(io, args...; kwargs...); String(take!(io)))

const HDR  = ["Checking", "Krylov dim.", "Converged", "Residual"]
const ROWS = ["1" "10" "0" "1.30e-02"
              "2" "20" "3" "4.40e-05"]

@testset "table" begin

    @testset "plain_table layout" begin
        s = _capture(plain_table, HDR, ROWS)
        lines = split(rstrip(s), '\n')

        # 2 rules + header + separator + 2 data rows
        @test length(lines) == 6
        @test startswith(lines[1], "╭") && endswith(lines[1], "╮")
        @test startswith(lines[3], "├") && endswith(lines[3], "┤")
        @test startswith(lines[end], "╰") && endswith(lines[end], "╯")

        # every line is the same display width — the property that makes a table a table
        @test length(unique(textwidth.(lines))) == 1

        # header text and all cells survive
        @test all(h -> occursin(h, lines[2]), HDR)
        @test occursin("1.30e-02", lines[4])
        @test occursin("4.40e-05", lines[5])
    end

    @testset "alignment" begin
        # all-numeric column right-aligns, mixed column left-aligns
        rows = ["1" "edge_lowest"; "217" "interior"]
        s = _capture(plain_table, ["K", "kind"], rows)
        lines = split(rstrip(s), '\n')
        @test occursin("│   1 │ edge_lowest │", lines[4])
        @test occursin("│ 217 │ interior    │", lines[5])
    end

    @testset "width driven by the widest cell, header included" begin
        # header wider than any cell
        s = _capture(plain_table, ["averylongheader"], reshape(["x"], 1, 1))
        @test occursin("averylongheader", s)
        @test length(unique(textwidth.(split(rstrip(s), '\n')))) == 1

        # cell wider than the header
        s = _capture(plain_table, ["h"], reshape(["averylongcell"], 1, 1))
        @test occursin("averylongcell", s)
        @test length(unique(textwidth.(split(rstrip(s), '\n')))) == 1
    end

    @testset "edge cases" begin
        # no rows renders nothing at all, not an empty frame
        @test _capture(plain_table, HDR, Matrix{String}(undef, 0, 4)) == ""

        # non-string cells are stringified
        s = _capture(plain_table, ["a", "b"], [1 2.5])
        @test occursin("1", s) && occursin("2.5", s)

        # header/column count must agree
        @test_throws DimensionMismatch _capture(plain_table, ["only-one"], ROWS)
    end

    @testset "bold_header" begin
        plainrun = _capture(plain_table, HDR, ROWS; bold_header = false)
        bolded   = _capture(plain_table, HDR, ROWS; bold_header = true)
        @test !occursin("\e[1m", plainrun)
        @test occursin("\e[1m", bolded)
        # the escape must not disturb the layout: strip it and the two agree exactly
        @test replace(bolded, "\e[1m" => "", "\e[0m" => "") == plainrun
    end

    @testset "renderer backend swap" begin
        reset_table_renderer!()
        builtin = _capture(render_detail_table, HDR, ROWS)
        @test builtin == _capture(plain_table, HDR, ROWS)

        # a custom backend takes over, and the reset restores the built-in one
        set_table_renderer!((io, h, r; bold_header = false) -> print(io, "STUB"))
        @test _capture(render_detail_table, HDR, ROWS) == "STUB"
        reset_table_renderer!()
        @test _capture(render_detail_table, HDR, ROWS) == builtin
    end

    @testset "PrettyTables extension" begin
        reset_table_renderer!()
        builtin = _capture(render_detail_table, HDR, ROWS)

        @eval using PrettyTables
        @test Base.get_extension(UniPolfed, :UniPolfedPrettyTablesExt) !== nothing

        pretty = _capture(render_detail_table, HDR, ROWS)
        @test pretty != ""

        # The two backends are interchangeable: same row count, same width-uniformity, and
        # every header and cell present. (Glyph-for-glyph equality is NOT required — that
        # would just re-encode PrettyTables' internal style choices.)
        plines, blines = split(rstrip(pretty), '\n'), split(rstrip(builtin), '\n')
        @test length(plines) == length(blines)
        @test length(unique(textwidth.(plines))) == 1
        @test all(h -> occursin(h, pretty), HDR)
        @test all(c -> occursin(c, pretty), ROWS)

        # the extension honours the empty-table contract too
        @test _capture(render_detail_table, HDR, Matrix{String}(undef, 0, 4)) == ""

        reset_table_renderer!()
        @test _capture(render_detail_table, HDR, ROWS) == builtin
    end
end
