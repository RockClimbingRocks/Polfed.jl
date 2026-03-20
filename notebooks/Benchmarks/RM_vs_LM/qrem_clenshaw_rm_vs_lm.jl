using LinearAlgebra
using Random
using SparseArrays
using BenchmarkTools
using Printf, UnPack

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed

const QREM_ROOT = normpath(joinpath(@__DIR__, "..", "QREM"))
include(joinpath(QREM_ROOT, "QREM.jl"))

const VecOrMat = Union{AbstractVector, AbstractMatrix}

parse_or_default(::Type{T}, idx::Int, default::T) where {T} =
    length(ARGS) >= idx ? parse(T, ARGS[idx]) : default

function make_qrem(L::Int, hx::Float64, spin::Float64, avgs::Int)
    params = Dict{Symbol, Any}(
        :model_name => "qrem",
        :L => L,
        :hx => hx,
        :spin => spin,
        :avgs => avgs,
        :runname => "rm_vs_lm_bench",
    )
    return construct_model(params)
end

function lanczos_extrema(f!::Function, x0::AbstractVector{<:Real})
    Emin = first(collect(Polfed.Lanczos.lanczos(f!, x0, 1; which=:SR, maxdim=1000)[1]))
    Emax = last(collect(Polfed.Lanczos.lanczos(f!, x0, 1; which=:LR, maxdim=1000)[1]))
    return Emin, Emax
end

function rescale_qrem(qrem::QREM, a::Float64, b::Float64)
    return QREM(
        qrem.name,
        qrem.L,
        qrem.hx / a,
        qrem.spin,
        (qrem.diags .- b) ./ a,
    )
end

@inline cheb_dirac_coeff(target::Float64, n::Int) =
    (2 - (n == 0)) * cos(n * acos(target))

function clenshaw_scalar(coeff::Function, order::Int, x::Float64)
    bkp1 = 0.0
    bkp2 = 0.0
    @inbounds for k in order:-1:1
        bk = coeff(k) + 2 * x * bkp1 - bkp2
        bkp2, bkp1 = bkp1, bk
    end
    return coeff(0) + x * bkp1 - bkp2
end

function make_normalized_coefficients(target_rescaled::Float64, order::Int; normalization::Float64 = 1.0)
    coeff_raw(n::Int) = cheb_dirac_coeff(target_rescaled, n)
    value_at_target = clenshaw_scalar(coeff_raw, order, target_rescaled)
    scale = normalization / value_at_target
    coeff_normalized(n::Int) = coeff_raw(n) * scale
    return coeff_normalized, scale, value_at_target
end

function mapleft_with_qrem!(
    Y::AbstractVector,
    X::AbstractVector,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
)
    # QREM Hamiltonian is real-symmetric: x' * H = (H * x)'.
    mapvec_with_qrem!(Y, X, diags, offdiagonals)
    return Y
end

function mapleft_with_qrem!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
)
    @assert size(Y) == size(X)
    @assert size(X, 2) == length(diags)

    for row in axes(X, 1)
        mapleft_with_qrem!(view(Y, row, :), view(X, row, :), diags, offdiagonals)
    end
    return Y
end

function clenshaw_right_mul!(
    Y::VecOrMat,
    X::VecOrMat,
    H::SparseMatrixCSC,
    order::Int,
    coeff::Function,
    b1::VecOrMat,
    b2::VecOrMat,
    b3::VecOrMat,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        mul!(bb1, H, bb2)
        @. bb1 = coeff(k) * X + 2 * bb1 - bb3
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    mul!(Y, H, bb2)
    @. Y = coeff(0) * X + Y - bb3
    return Y
end

function clenshaw_left_mul!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    H::SparseMatrixCSC,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        mul!(bb1, bb2, H)
        @. bb1 = coeff(k) * X + 2 * bb1 - bb3
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    mul!(Y, bb2, H)
    @. Y = coeff(0) * X + Y - bb3
    return Y
end

function clenshaw_right_mul_threaded!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    H::SparseMatrixCSC,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        Threads.@threads for col in axes(X, 2)
            xcol = view(X, :, col)
            ycol = view(bb1, :, col)
            b2col = view(bb2, :, col)
            b3col = view(bb3, :, col)
            mul!(ycol, H, b2col)
            @. ycol = coeff(k) * xcol + 2 * ycol - b3col
        end
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    Threads.@threads for col in axes(X, 2)
        xcol = view(X, :, col)
        ycol = view(Y, :, col)
        b2col = view(bb2, :, col)
        b3col = view(bb3, :, col)
        mul!(ycol, H, b2col)
        @. ycol = coeff(0) * xcol + ycol - b3col
    end
    return Y
end

function clenshaw_left_mul_threaded!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    H::SparseMatrixCSC,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    HT = transpose(H)
    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        Threads.@threads for row in axes(X, 1)
            xrow = view(X, row, :)
            yrow = view(bb1, row, :)
            b2row = view(bb2, row, :)
            b3row = view(bb3, row, :)
            mul!(yrow, HT, b2row)
            @. yrow = coeff(k) * xrow + 2 * yrow - b3row
        end
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    Threads.@threads for row in axes(X, 1)
        xrow = view(X, row, :)
        yrow = view(Y, row, :)
        b2row = view(bb2, row, :)
        b3row = view(bb3, row, :)
        mul!(yrow, HT, b2row)
        @. yrow = coeff(0) * xrow + yrow - b3row
    end
    return Y
end

function clenshaw_right_custom!(
    Y::VecOrMat,
    X::VecOrMat,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
    order::Int,
    coeff::Function,
    b1::VecOrMat,
    b2::VecOrMat,
    b3::VecOrMat,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        mapvec_with_qrem!(bb1, bb2, diags, offdiagonals)
        @. bb1 = coeff(k) * X + 2 * bb1 - bb3
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    mapvec_with_qrem!(Y, bb2, diags, offdiagonals)
    @. Y = coeff(0) * X + Y - bb3
    return Y
end

function clenshaw_right_custom_threaded!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        Threads.@threads for col in axes(X, 2)
            xcol = view(X, :, col)
            ycol = view(bb1, :, col)
            b2col = view(bb2, :, col)
            b3col = view(bb3, :, col)
            mapvec_with_qrem!(ycol, b2col, diags, offdiagonals)
            @. ycol = coeff(k) * xcol + 2 * ycol - b3col
        end
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    Threads.@threads for col in axes(X, 2)
        xcol = view(X, :, col)
        ycol = view(Y, :, col)
        b2col = view(bb2, :, col)
        b3col = view(bb3, :, col)
        mapvec_with_qrem!(ycol, b2col, diags, offdiagonals)
        @. ycol = coeff(0) * xcol + ycol - b3col
    end
    return Y
end

function clenshaw_left_custom!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        mapleft_with_qrem!(bb1, bb2, diags, offdiagonals)
        @. bb1 = coeff(k) * X + 2 * bb1 - bb3
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    mapleft_with_qrem!(Y, bb2, diags, offdiagonals)
    @. Y = coeff(0) * X + Y - bb3
    return Y
end

function clenshaw_left_custom_threaded!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
    order::Int,
    coeff::Function,
    b1::AbstractMatrix,
    b2::AbstractMatrix,
    b3::AbstractMatrix,
)
    fill!(b1, zero(eltype(b1)))
    fill!(b2, zero(eltype(b2)))
    fill!(b3, zero(eltype(b3)))

    bb1, bb2, bb3 = b1, b2, b3
    @inbounds for k in order:-1:1
        Threads.@threads for row in axes(X, 1)
            xrow = view(X, row, :)
            yrow = view(bb1, row, :)
            b2row = view(bb2, row, :)
            b3row = view(bb3, row, :)
            mapleft_with_qrem!(yrow, b2row, diags, offdiagonals)
            @. yrow = coeff(k) * xrow + 2 * yrow - b3row
        end
        bb1, bb2, bb3 = bb3, bb1, bb2
    end

    Threads.@threads for row in axes(X, 1)
        xrow = view(X, row, :)
        yrow = view(Y, row, :)
        b2row = view(bb2, row, :)
        b3row = view(bb3, row, :)
        mapleft_with_qrem!(yrow, b2row, diags, offdiagonals)
        @. yrow = coeff(0) * xrow + yrow - b3row
    end
    return Y
end

function normalize_columns!(X::AbstractMatrix)
    for col in axes(X, 2)
        n = norm(view(X, :, col))
        n == 0 && continue
        @views X[:, col] ./= n
    end
    return X
end

function relative_error(A, B)
    denom = max(norm(B), eps(Float64))
    return norm(A - B) / denom
end

function trial_median_ns(trial::BenchmarkTools.Trial)
    ts = sort!(copy(trial.times))
    return ts[cld(length(ts), 2)]
end

function print_trial(label::AbstractString, trial::BenchmarkTools.Trial)
    ts = sort!(copy(trial.times))
    min_ns = ts[1]
    med_ns = ts[cld(length(ts), 2)]
    mean_ns = sum(ts) / length(ts)
    min_entry = minimum(trial)
    @printf(
        "%-36s min=%9.3f ms  med=%9.3f ms  mean=%9.3f ms  alloc=%8.2f KiB  allocs=%d\n",
        label,
        min_ns / 1e6,
        med_ns / 1e6,
        mean_ns / 1e6,
        min_entry.memory / 1024,
        min_entry.allocs,
    )
end

function run_bench(label::AbstractString, f::Function; samples::Int, evals::Int)
    GC.gc()
    f() # warm-up compilation
    GC.gc()
    trial = @benchmark $f() samples=samples evals=evals
    print_trial(label, trial)
    return trial
end

function run_shape_bench(
    shape_label::String,
    V::VecOrMat,
    Vt::AbstractMatrix,
    H::SparseMatrixCSC,
    diags::AbstractVector{Float64},
    offdiags::AbstractVector,
    order::Int,
    coeff::Function;
    samples::Int,
    evals::Int,
    threaded::Bool = false,
)
    println("\n=== ", shape_label, " ===")
    println("right input size : ", size(V))
    println("left input size  : ", size(Vt))

    Yr_mul = similar(V)
    Yr_custom = similar(V)
    Yl_mul = similar(Vt)
    Yl_custom = similar(Vt)

    b1_rm, b2_rm, b3_rm = similar(V), similar(V), similar(V)
    b1_rc, b2_rc, b3_rc = similar(V), similar(V), similar(V)
    b1_lm, b2_lm, b3_lm = similar(Vt), similar(Vt), similar(Vt)
    b1_lc, b2_lc, b3_lc = similar(Vt), similar(Vt), similar(Vt)

    f_right_mul! = if threaded
        () -> clenshaw_right_mul_threaded!(Yr_mul, V, H, order, coeff, b1_rm, b2_rm, b3_rm)
    else
        () -> clenshaw_right_mul!(Yr_mul, V, H, order, coeff, b1_rm, b2_rm, b3_rm)
    end

    f_right_custom! = if threaded
        () -> clenshaw_right_custom_threaded!(Yr_custom, V, diags, offdiags, order, coeff, b1_rc, b2_rc, b3_rc)
    else
        () -> clenshaw_right_custom!(Yr_custom, V, diags, offdiags, order, coeff, b1_rc, b2_rc, b3_rc)
    end

    f_left_mul! = if threaded
        () -> clenshaw_left_mul_threaded!(Yl_mul, Vt, H, order, coeff, b1_lm, b2_lm, b3_lm)
    else
        () -> clenshaw_left_mul!(Yl_mul, Vt, H, order, coeff, b1_lm, b2_lm, b3_lm)
    end

    f_left_custom! = if threaded
        () -> clenshaw_left_custom_threaded!(Yl_custom, Vt, diags, offdiags, order, coeff, b1_lc, b2_lc, b3_lc)
    else
        () -> clenshaw_left_custom!(Yl_custom, Vt, diags, offdiags, order, coeff, b1_lc, b2_lc, b3_lc)
    end

    f_right_mul!()
    f_right_custom!()
    f_left_mul!()
    f_left_custom!()

    err_right = relative_error(Yr_custom, Yr_mul)
    err_left = relative_error(Yl_custom, Yl_mul)
    @printf("relative error right (custom vs mul!) = %.3e\n", err_right)
    @printf("relative error left  (custom vs mul!) = %.3e\n", err_left)

    trial_right_mul = run_bench("right mul!      P(H) * V", f_right_mul!; samples=samples, evals=evals)
    trial_right_custom = run_bench("right custom    P(H) * V", f_right_custom!; samples=samples, evals=evals)
    trial_left_mul = run_bench("left mul!       Vt * P(H)", f_left_mul!; samples=samples, evals=evals)
    trial_left_custom = run_bench("left custom     Vt * P(H)", f_left_custom!; samples=samples, evals=evals)

    speedup_right = trial_median_ns(trial_right_mul) / trial_median_ns(trial_right_custom)
    speedup_left = trial_median_ns(trial_left_mul) / trial_median_ns(trial_left_custom)
    @printf("median speedup custom vs mul! (right) = %.3fx\n", speedup_right)
    @printf("median speedup custom vs mul! (left)  = %.3fx\n", speedup_left)
end

function main()
    L = parse_or_default(Int, 1, 12)

    hx = 1.0
    spin = 0.5
    avgs = 0
    order = 250
    target_rescaled = 0.0
    samples = 25
    evals = 1
    s_list = [1, 2, 4, 8, 16]

    println("Building QREM model...")
    qrem_raw = make_qrem(L, hx, spin, avgs)

    D = qrem_raw.hilbertspacedim
    Random.seed!(1234)
    x0 = randn(Float64, D)
    x0 ./= norm(x0)

    map_raw = map_vec(qrem_raw; a=1.0, b=0.0, pu="cpu")
    Emin, Emax = lanczos_extrema(map_raw, x0)
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    qrem_rescaled = rescale_qrem(qrem_raw, a, b)

    H = construct_matrix(qrem_rescaled; pu="cpu")
    diags = qrem_rescaled.diags
    offdiags = get_offdiagonals_by_value(qrem_rescaled)
    D == size(H, 1) || error("Inconsistent Hilbert space dimension.")

    coeff, scale, value_at_target = make_normalized_coefficients(target_rescaled, order)

    println("L = ", L)
    println("D = ", D)
    println("order = ", order)
    println("s list = ", s_list)
    println("threads = ", Threads.nthreads())
    println("target_rescaled = ", target_rescaled)
    println("Lanczos extrema: Emin = ", Emin, ", Emax = ", Emax)
    println("rescaling from Lanczos: a = ", a, ", b = ", b)
    println("benchmark samples = ", samples, ", evals = ", evals)
    println("coeff normalization scale = ", scale)
    println("raw P(target) before normalization = ", value_at_target)

    for s in s_list
        s <= D || continue

        V = randn(Float64, D, s)
        normalize_columns!(V)
        Vt = copy(permutedims(V))

        run_shape_bench(
            "Block input (threaded, s=$s, right: Dxs, left: sxD)",
            V,
            Vt,
            H,
            diags,
            offdiags,
            order,
            coeff;
            samples=samples,
            evals=evals,
            threaded=true,
        )
    end
end

main()
