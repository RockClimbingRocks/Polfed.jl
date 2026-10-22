using Test
using Polfed
using Polfed.Numerics.Lanczos
using Polfed.Common: stage_names, total_wall, total_cpu
using LinearAlgebra
using Random

LZ = Polfed.Numerics.Lanczos

Random.seed!(424242)

_fr_make_sym(λ) = (n = length(λ); Q = Matrix(qr(randn(n, n)).Q); Matrix(Hermitian(Q * Diagonal(λ) * Q')))
_fr_seed(n)  = Float64[1 / (i + 0.5) - 0.3cos(1.3i) + 0.2 for i in 1:n]
_fr_block(n) = Float64[1 / (i + j + 0.5) - 0.2cos(1.3i + 0.7j) for i in 1:n, j in 1:2]

@testset "FactorizationReport" begin
    λ = sort(vcat([-10.0, -8.0, -6.0, 6.0, 8.0, 10.0], range(-1.0, 1.0; length=34)))
    n = length(λ); A = _fr_make_sym(λ); md = n

    @testset "scalar / MatrixBasis field values" begin
        r = lanczos(A, _fr_seed(n), LowestEigs(3); maxdim=md)
        rep = r.report
        @test rep.factorizationtype == "LanczosFactorization"
        @test rep.blocksize == 1
        @test rep.basistype <: MatrixBasis
        @test rep.howmany == 3
        @test rep.converged_tol == 3
        @test rep.converged_eigentol == 3
        @test rep.itersreserved == md             # maxiter = maxdim ÷ blocksize
        @test rep.itersneeded <= rep.itersreserved
        @test rep.ncols_cpu_reserved == md        # CPU run → all columns on CPU
        @test rep.ncols_gpu_reserved == 0
        @test rep.ncols_cpu_needed <= rep.ncols_cpu_reserved
        # Stage timings are name-addressed and self-accumulating over iterations: the 5
        # factorization stages + the convergence check.
        @test Set(stage_names(rep.timer)) ==
              Set([:tridiag, :matvec, :overlap, :orthog, :reorthog, :conv_check])
        @test all(s -> rep.timer[s] >= 0.0, stage_names(rep.timer))
        @test total_wall(rep.timer) > 0.0
        # Every in-loop stage is hit once per loop iteration, so their counts agree…
        @test all(s -> rep.timer[s, :n] == rep.timer[:matvec, :n],
                  (:tridiag, :overlap, :orthog, :reorthog, :conv_check))
        # …and that is one fewer than `itersneeded`: the initializing iteration runs before the loop.
        @test rep.timer[:matvec, :n] == rep.itersneeded - 1
        # check_convergence! is CALLED every iteration but only performs a check when scheduled.
        @test rep.timer[:conv_check, :n] >= rep.numofchecks
    end

    @testset "block / VectorBasis field values" begin
        r = lanczos(A, _fr_block(n), LowestEigs(2); basistype=VectorBasis, maxdim=md)
        rep = r.report
        @test rep.factorizationtype == "BlockLanczosFactorization"
        @test rep.blocksize == 2
        @test rep.basistype <: VectorBasis
        @test rep.howmany == 2
        @test rep.converged_eigentol == 2
        @test rep.ncols_cpu_reserved == md        # cld(md, 2) entries × 2 columns = md
        @test rep.ncols_gpu_reserved == 0
    end

    @testset "value-target howmany is the dynamic converged count" begin
        r = lanczos(A, _fr_seed(n), Lowest(ToValue(-5.0)); maxdim=md)   # → -10, -8, -6
        @test r.report.howmany == 3
        @test r.report.howmany == r.converged
    end

    @testset "display runs (smoke test, colors off)" begin
        r = lanczos(A, _fr_seed(n), LowestEigs(3); maxdim=md)
        ran = redirect_stdout(devnull) do
            LZ.display_factorization_report(r.report, false; show_timings=true, show_convergence_details=true)
            LZ.print_factorization_report(r.report)
            true
        end
        @test ran
    end
end
