using Test
using Polfed

C = Polfed.Common

@testset "Formatter" begin
    on  = C.Formatter(true)
    off = C.Formatter(false)

    @testset "fmt applies / suppresses ANSI codes" begin
        @test C.fmt(on, "x"; code="1") == "\e[1mx\e[0m"
        @test C.fmt(off, "x"; code="1") == "x"      # color disabled → plain
        @test C.fmt(on, "x"; code="") == "x"        # empty code → plain even with color
        @test C.fmt(on, "x") == "x"                 # default code is empty
    end

    @testset "named style wrappers carry the right codes" begin
        @test C.bold(on, "x")   == "\e[1mx\e[0m"
        @test C.cyan(on, "x")   == "\e[1;36mx\e[0m"
        @test C.blue(on, "x")   == "\e[1;34mx\e[0m"
        @test C.green(on, "x")  == "\e[1;32mx\e[0m"
        @test C.red(on, "x")    == "\e[1;31mx\e[0m"
        @test C.yellow(on, "x") == "\e[1;33mx\e[0m"
    end

    @testset "wrappers respect color=false" begin
        @test C.bold(off, "x")   == "x"
        @test C.green(off, "x")  == "x"
        @test C.yellow(off, "x") == "x"
    end
end
