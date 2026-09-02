using UniPolfed
using LinearAlgebra: diag
using Polyester: @batch
using Printf
using Random
using SIMD: Vec, shufflevector, vload, vstore
using StaticArrays: SVector
using Statistics: median

include("../qrem.jl")

const Opt = UniPolfed.Optimization

# Standalone experiment: fuse the affine Clenshaw update
#
#     out = c * seed + alpha * (Htilde * input) + beta * older
#
# into the final store of the QREM spin_flip/packed SIMD kernel. Nothing in src/ is
# modified by this benchmark.

struct PackedQREMData{T,M,MASKS,W,BS,G}
    diags::Vector{T}
    values::SVector{M,T}
end

function PackedQREMData(fs::Opt.FlipStructure{T}, ::Val{W}, ::Val{BS}, ::Val{G}) where {T,W,BS,G}
    M = length(fs.values)
    masks = Tuple(fs.masks)
    values = SVector{M,T}(fs.values)
    return PackedQREMData{T,M,masks,W,BS,G}(fs.diags, values)
end

@inline function packed_partition(D::Int, W::Int, bs::Int, G::Int)
    P = W ÷ bs
    align = P * G
    nt = Threads.nthreads() == 1 ? 1 : max(1, min(Threads.nthreads(), fld(D, align)))
    chunk = align * cld(cld(D, nt), align)
    return nt, chunk
end

@generated function fused_packed_block!(
    p_out::Ptr{T},
    p_input::Ptr{T},
    p_diags::Ptr{T},
    p_seed::Ptr{T},
    p_older::Ptr{T},
    values::SVector{M,T},
    cv::Vec{W,T},
    av::Vec{W,T},
    bv::Vec{W,T},
    ::Val{MODE},
    p0::Int,
    ::Val{MASKS},
    ::Val{W},
    ::Val{bs},
    ::Val{G},
) where {T,M,MASKS,W,bs,G,MODE}
    P = W ÷ bs
    sz = sizeof(T)
    init = Expr[]
    update = Expr[]
    store = Expr[]

    for g in 0:(G - 1)
        input_vec = Symbol(:input_vec_, g)
        accum = Symbol(:accum_, g)
        diag_vec = Symbol(:diag_vec_, g)
        seed_vec = Symbol(:seed_vec_, g)
        older_vec = Symbol(:older_vec_, g)
        result_vec = Symbol(:result_vec_, g)
        elem_offset = g * W
        column_offset = g * P

        push!(init, :(
            $input_vec = vload(
                Vec{$W,T},
                p_input + (p0 * $bs + $elem_offset) * $sz,
            )
        ))

        if bs == 1
            push!(init, :(
                $diag_vec = vload(Vec{$W,T}, p_diags + (p0 + $column_offset) * $sz)
            ))
        else
            push!(init, :(
                $diag_vec = Vec{$W,T}($(Expr(
                    :tuple,
                    [:(unsafe_load(
                        p_diags,
                        p0 + $(column_offset + (lane - 1) ÷ bs) + 1,
                    )) for lane in 1:W]...,
                )))
            ))
        end

        push!(init, :($accum = $diag_vec * $input_vec))

        result_expression = if MODE === :plain
            :($result_vec = (cv * $seed_vec + av * $accum) + bv * $older_vec)
        elseif MODE === :fma
            :($result_vec = muladd(cv, $seed_vec,
                                   muladd(av, $accum, bv * $older_vec)))
        elseif MODE === :cheb_plain_recurrence || MODE === :qrem_cheb_plain_recurrence
            :($result_vec = (cv * $seed_vec + ($accum + $accum)) - $older_vec)
        elseif MODE === :cheb_fma_recurrence
            :($result_vec = muladd(cv, $seed_vec, ($accum + $accum) - $older_vec))
        elseif MODE === :cheb_plain_final || MODE === :qrem_cheb_plain_final
            :($result_vec = (cv * $seed_vec + $accum) - $older_vec)
        elseif MODE === :cheb_fma_final
            :($result_vec = muladd(cv, $seed_vec, $accum - $older_vec))
        else
            error("unknown fused arithmetic mode: $MODE")
        end

        # Seed and older are contiguous in exactly the same row-layout pack as the output.
        # Load them only after the Hamiltonian sum, to avoid extending their live ranges.
        push!(store, quote
            $seed_vec = vload(
                Vec{$W,T},
                p_seed + (p0 * $bs + $elem_offset) * $sz,
            )
            $older_vec = vload(
                Vec{$W,T},
                p_older + (p0 * $bs + $elem_offset) * $sz,
            )
            $result_expression
            vstore(
                $result_vec,
                p_out + (p0 * $bs + $elem_offset) * $sz,
            )
        end)
    end

    for k in 1:M
        mask = MASKS[k]
        value_index = startswith(String(MODE), "qrem_") ? 1 : k
        mlo = mask & (P - 1)
        mhi = mask & ~(P - 1)
        permutation = ntuple(lane -> xor(lane - 1, mlo * bs), W)

        for g in 0:(G - 1)
            input_vec = Symbol(:input_vec_, g)
            accum = Symbol(:accum_, g)
            column_offset = g * P

            source = if mhi == 0
                mlo == 0 ? input_vec : :(shufflevector($input_vec, Val($permutation)))
            else
                loaded = :(
                    vload(
                        Vec{$W,T},
                        p_input + (xor(p0 + $column_offset, $mhi) * $bs) * $sz,
                    )
                )
                mlo == 0 ? loaded : :(shufflevector($loaded, Val($permutation)))
            end

            push!(update, :(
                $accum = muladd(Vec{$W,T}(@inbounds(values[$value_index])), $source, $accum)
            ))
        end
    end

    return quote
        $(init...)
        $(update...)
        $(store...)
        return nothing
    end
end

@inline function fused_packed_range!(
    p_out::Ptr{T},
    p_input::Ptr{T},
    p_diags::Ptr{T},
    p_seed::Ptr{T},
    p_older::Ptr{T},
    values::SVector{M,T},
    cv::Vec{W,T},
    av::Vec{W,T},
    bv::Vec{W,T},
    mode::Val{MODE},
    p0::Int,
    hi::Int,
    ::Val{MASKS},
    ::Val{W},
    ::Val{bs},
    ::Val{G},
) where {T,M,MASKS,W,bs,G,MODE}
    P = W ÷ bs

    while p0 + G * P <= hi
        fused_packed_block!(
            p_out,
            p_input,
            p_diags,
            p_seed,
            p_older,
            values,
            cv,
            av,
            bv,
            mode,
            p0,
            Val(MASKS),
            Val(W),
            Val(bs),
            Val(G),
        )
        p0 += G * P
    end

    while p0 + P <= hi
        fused_packed_block!(
            p_out,
            p_input,
            p_diags,
            p_seed,
            p_older,
            values,
            cv,
            av,
            bv,
            mode,
            p0,
            Val(MASKS),
            Val(W),
            Val(bs),
            Val(1),
        )
        p0 += P
    end

    return nothing
end

@inline function packed_map_base!(
    out::Matrix{T},
    input::Matrix{T},
    data::PackedQREMData{T,M,MASKS,W,BS,G},
) where {T,M,MASKS,W,BS,G}
    Opt._packed_apply!(
        out,
        input,
        data.diags,
        data.values,
        Val(MASKS),
        Val(W),
        Val(BS),
        Val(G),
    )
    return out
end

function packed_map_poly!(
    out::Matrix{T},
    input::Matrix{T},
    data::PackedQREMData{T,M,MASKS,W,BS,G},
) where {T,M,MASKS,W,BS,G}
    D = size(input, 2)
    nt, chunk = packed_partition(D, W, BS, G)
    diags = data.diags

    GC.@preserve out input diags begin
        p_out = pointer(out)
        p_input = pointer(input)
        p_diags = pointer(diags)

        if nt == 1
            Opt._packed_range!(
                p_out,
                p_input,
                p_diags,
                data.values,
                0,
                D,
                Val(MASKS),
                Val(W),
                Val(BS),
                Val(G),
            )
        else
            @batch per=thread for t in 1:nt
                Opt._packed_range!(
                    p_out,
                    p_input,
                    p_diags,
                    data.values,
                    (t - 1) * chunk,
                    min(t * chunk, D),
                    Val(MASKS),
                    Val(W),
                    Val(BS),
                    Val(G),
                )
            end
        end
    end

    return out
end

@inline fused_packed_base!(out, input, older, seed, c, alpha, beta, data) =
    fused_packed_base_mode!(out, input, older, seed, c, alpha, beta, Val(:plain), data)

function fused_packed_base_mode!(
    out::Matrix{T},
    input::Matrix{T},
    older::Matrix{T},
    seed::Matrix{T},
    c,
    alpha,
    beta,
    mode::Val{MODE},
    data::PackedQREMData{T,M,MASKS,W,BS,G},
) where {T,M,MASKS,W,BS,G,MODE}
    D = size(input, 2)
    nt, chunk = packed_partition(D, W, BS, G)
    cv = Vec{W,T}(T(c))
    av = Vec{W,T}(T(alpha))
    bv = Vec{W,T}(T(beta))
    diags = data.diags

    GC.@preserve out input older seed diags begin
        p_out = pointer(out)
        p_input = pointer(input)
        p_diags = pointer(diags)
        p_seed = pointer(seed)
        p_older = pointer(older)

        if nt == 1
            fused_packed_range!(
                p_out,
                p_input,
                p_diags,
                p_seed,
                p_older,
                data.values,
                cv,
                av,
                bv,
                mode,
                0,
                D,
                Val(MASKS),
                Val(W),
                Val(BS),
                Val(G),
            )
        else
            Threads.@threads :static for t in 1:nt
                fused_packed_range!(
                    p_out,
                    p_input,
                    p_diags,
                    p_seed,
                    p_older,
                    data.values,
                    cv,
                    av,
                    bv,
                    mode,
                    (t - 1) * chunk,
                    min(t * chunk, D),
                    Val(MASKS),
                    Val(W),
                    Val(BS),
                    Val(G),
                )
            end
        end
    end

    return out
end

@inline fused_packed_poly!(out, input, older, seed, c, alpha, beta, data) =
    fused_packed_poly_mode!(out, input, older, seed, c, alpha, beta, Val(:plain), data)

function fused_packed_poly_mode!(
    out::Matrix{T},
    input::Matrix{T},
    older::Matrix{T},
    seed::Matrix{T},
    c,
    alpha,
    beta,
    mode::Val{MODE},
    data::PackedQREMData{T,M,MASKS,W,BS,G},
) where {T,M,MASKS,W,BS,G,MODE}
    D = size(input, 2)
    nt, chunk = packed_partition(D, W, BS, G)
    cv = Vec{W,T}(T(c))
    av = Vec{W,T}(T(alpha))
    bv = Vec{W,T}(T(beta))
    diags = data.diags

    GC.@preserve out input older seed diags begin
        p_out = pointer(out)
        p_input = pointer(input)
        p_diags = pointer(diags)
        p_seed = pointer(seed)
        p_older = pointer(older)

        if nt == 1
            fused_packed_range!(
                p_out,
                p_input,
                p_diags,
                p_seed,
                p_older,
                data.values,
                cv,
                av,
                bv,
                mode,
                0,
                D,
                Val(MASKS),
                Val(W),
                Val(BS),
                Val(G),
            )
        else
            @batch per=thread for t in 1:nt
                fused_packed_range!(
                    p_out,
                    p_input,
                    p_diags,
                    p_seed,
                    p_older,
                    data.values,
                    cv,
                    av,
                    bv,
                    mode,
                    (t - 1) * chunk,
                    min(t * chunk, D),
                    Val(MASKS),
                    Val(W),
                    Val(BS),
                    Val(G),
                )
            end
        end
    end

    return out
end

@inline function combine_range!(out, seed, older, c, alpha, beta, lo::Int, hi::Int)
    @inbounds @simd for i in lo:hi
        out[i] = (c * seed[i] + alpha * out[i]) + beta * older[i]
    end
    return nothing
end

function combine_base!(out::Matrix{T}, seed::Matrix{T}, older::Matrix{T}, c, alpha, beta) where {T}
    n = length(out)
    nt = min(Threads.nthreads(), n)
    chunk = cld(n, nt)
    cT, alphaT, betaT = T(c), T(alpha), T(beta)

    if nt == 1
        combine_range!(out, seed, older, cT, alphaT, betaT, 1, n)
    else
        Threads.@threads :static for t in 1:nt
            combine_range!(
                out,
                seed,
                older,
                cT,
                alphaT,
                betaT,
                (t - 1) * chunk + 1,
                min(t * chunk, n),
            )
        end
    end

    return out
end


function combine_poly!(out::Matrix{T}, seed::Matrix{T}, older::Matrix{T}, c, alpha, beta) where {T}
    cT, alphaT, betaT = T(c), T(alpha), T(beta)
    @batch per=thread for i in eachindex(out)
        @inbounds out[i] = (cT * seed[i] + alphaT * out[i]) + betaT * older[i]
    end
    return out
end

@inline function step_separate_base!(out, input, older, seed, c, alpha, beta, data)
    packed_map_base!(out, input, data)
    combine_base!(out, seed, older, c, alpha, beta)
    return out
end

@inline function step_separate_poly!(out, input, older, seed, c, alpha, beta, data)
    packed_map_poly!(out, input, data)
    combine_poly!(out, seed, older, c, alpha, beta)
    return out
end

@inline step_fused_base!(out, input, older, seed, c, alpha, beta, data) =
    fused_packed_base!(out, input, older, seed, c, alpha, beta, data)

@inline step_fused_poly!(out, input, older, seed, c, alpha, beta, data) =
    fused_packed_poly!(out, input, older, seed, c, alpha, beta, data)

struct FusedStep{SCHEDULER,MODE} end

@inline function (::FusedStep{:base,MODE})(out, input, older, seed, c, alpha, beta, data) where {MODE}
    return fused_packed_base_mode!(
        out, input, older, seed, c, alpha, beta, Val(MODE), data,
    )
end

@inline function (::FusedStep{:polyester,MODE})(out, input, older, seed, c, alpha, beta, data) where {MODE}
    return fused_packed_poly_mode!(
        out, input, older, seed, c, alpha, beta, Val(MODE), data,
    )
end

@inline make_fused_step(scheduler::Symbol, mode::Symbol) = FusedStep{scheduler,mode}()

@inline workspace_like(seed) = (similar(seed), similar(seed), similar(seed))

function clenshaw_split!(recurrence!, final!, coefficients, order::Int, workspace, seed, output, data)
    fill!(workspace[1], zero(eltype(seed)))
    fill!(workspace[2], zero(eltype(seed)))
    fill!(workspace[3], zero(eltype(seed)))

    b1, b2, b3 = workspace
    @inbounds for k in order:-1:1
        recurrence!(b1, b2, b3, seed, coefficients[k + 1], 2, -1, data)
        b1, b2, b3 = b3, b1, b2
    end

    return final!(output, b2, b3, seed, coefficients[1], 1, -1, data)
end

@inline clenshaw!(step!, coefficients, order::Int, workspace, seed, output, data) =
    clenshaw_split!(step!, step!, coefficients, order, workspace, seed, output, data)

@inline function max_relative_error(reference, candidate)
    denominator = max(maximum(abs, reference), eps(eltype(reference)))
    return maximum(abs, reference .- candidate) / denominator
end

function check_close!(label, reference, candidate; rtol=2e-11, atol=2e-11)
    abs_error = maximum(abs, reference .- candidate)
    rel_error = max_relative_error(reference, candidate)
    @printf("  %-22s max abs = %.3e   max rel = %.3e\n", label, abs_error, rel_error)
    isapprox(reference, candidate; rtol=rtol, atol=atol) ||
        error("$label is numerically different (max abs error $abs_error, max relative error $rel_error)")
    return nothing
end

function time_samples(f, samples::Int; warmups::Int=2)
    for _ in 1:warmups
        f()
    end

    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        start = time_ns()
        f()
        times[sample] = (time_ns() - start) * 1e-9
    end
    return times
end

function time_paired(first, second, samples::Int; warmups::Int=2)
    for _ in 1:warmups
        first()
        second()
    end

    first_times = Vector{Float64}(undef, samples)
    second_times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        if isodd(sample)
            start = time_ns()
            first()
            first_times[sample] = (time_ns() - start) * 1e-9

            start = time_ns()
            second()
            second_times[sample] = (time_ns() - start) * 1e-9
        else
            start = time_ns()
            second()
            second_times[sample] = (time_ns() - start) * 1e-9

            start = time_ns()
            first()
            first_times[sample] = (time_ns() - start) * 1e-9
        end
    end
    return first_times, second_times
end

function summarize_times(label, times; scale=1e3, unit="ms", divisor=1)
    scaled = times .* (scale / divisor)
    sorted = sort(scaled)
    lo = sorted[max(1, cld(length(sorted), 4))]
    hi = sorted[min(length(sorted), cld(3 * length(sorted), 4))]
    @printf(
        "  %-22s median %9.3f %s   min %9.3f %s   IQR [%9.3f, %9.3f]\n",
        label,
        median(scaled),
        unit,
        minimum(scaled),
        unit,
        lo,
        hi,
    )
    return median(scaled)
end

function main()
    L = parse(Int, get(ENV, "QREM_L", "16"))
    bs = parse(Int, get(ENV, "BLOCK_SIZE", "4"))
    order = parse(Int, get(ENV, "CLENSHAW_ORDER", "880"))
    requested_W = parse(Int, get(ENV, "PACK_WIDTH", "0"))
    G = parse(Int, get(ENV, "PACK_GROUPS", "4"))
    samples = parse(Int, get(ENV, "BENCH_SAMPLES", "9"))
    full_samples = parse(Int, get(ENV, "FULL_SAMPLES", "7"))
    step_repeats = parse(Int, get(ENV, "STEP_REPEATS", "100"))
    scheduler = Symbol(lowercase(get(ENV, "BENCH_SCHEDULER", "base")))
    arithmetic = Symbol(lowercase(get(ENV, "FUSED_ARITHMETIC", "cheb_plain")))

    ispow2(bs) || error("BLOCK_SIZE must be a power of two")
    order >= 0 || error("CLENSHAW_ORDER must be nonnegative")
    (requested_W == 0 || ispow2(requested_W)) ||
        error("PACK_WIDTH must be zero (automatic) or a positive power of two")
    G >= 1 || error("PACK_GROUPS must be positive")
    samples >= 3 || error("BENCH_SAMPLES must be at least 3")
    full_samples >= 3 || error("FULL_SAMPLES must be at least 3")
    step_repeats >= 1 || error("STEP_REPEATS must be positive")
    scheduler in (:base, :polyester) ||
        error("BENCH_SCHEDULER must be either 'base' or 'polyester'")
    arithmetic in (:plain, :fma, :cheb_plain, :cheb_fma, :qrem_cheb_plain) ||
        error("FUSED_ARITHMETIC must be plain, fma, cheb_plain, cheb_fma, or qrem_cheb_plain")

    rng = MersenneTwister(42)
    gamma = 1.0
    H = qrem_hamiltonian(L; Γ=gamma, rng=rng)
    D = size(H, 1)

    diagonal = diag(H)
    energy_min = minimum(diagonal) - L * abs(gamma)
    energy_max = maximum(diagonal) + L * abs(gamma)
    scale = (energy_max - energy_min) / 2
    shift = (energy_max + energy_min) / 2

    fs = Opt.FlipStructure(
        collect(diagonal),
        [1 << bit for bit in 0:(L - 1)],
        fill(gamma, L),
    )
    rescaled = Opt.rescale(fs, scale, shift)
    W = requested_W == 0 ? Opt._packed_lane_width(Float64) : requested_W
    bs <= W || error("block size $bs exceeds packed SIMD width $W")
    data = PackedQREMData(rescaled, Val(W), Val(bs), Val(G))

    seed = randn(rng, bs, D)
    input = randn(rng, bs, D)
    older = randn(rng, bs, D)
    coefficients = randn(rng, order + 1) ./ sqrt(order + 1)
    c = 0.37
    is_chebyshev_specialized = occursin("cheb_", String(arithmetic))
    alpha, beta = is_chebyshev_specialized ? (2.0, -1.0) : (1.91, -0.83)

    recurrence_mode, final_mode = if arithmetic === :plain
        (:plain, :plain)
    elseif arithmetic === :fma
        (:fma, :fma)
    elseif arithmetic === :cheb_plain
        (:cheb_plain_recurrence, :cheb_plain_final)
    elseif arithmetic === :cheb_fma
        (:cheb_fma_recurrence, :cheb_fma_final)
    else
        all(==(rescaled.values[1]), rescaled.values) ||
            error("qrem_cheb_plain requires one uniform spin-flip amplitude")
        (:qrem_cheb_plain_recurrence, :qrem_cheb_plain_final)
    end
    fused_recurrence! = make_fused_step(scheduler, recurrence_mode)
    fused_final! = make_fused_step(scheduler, final_mode)

    println("Fused packed-QREM Clenshaw benchmark")
    println("  Julia: $(VERSION)")
    println("  QREM: L=$L, D=$D, block size=$bs")
    println("  packed kernel: W=$W, G=$G")
    println("  Julia threads: $(Threads.nthreads())")
    println("  scheduler process: $scheduler")
    println("  fused arithmetic: $arithmetic")
    println("  Clenshaw order: $order")
    println("  samples: step=$samples x $step_repeats repeats, full=$full_samples")

    # Build every Base-threaded reference before the first Polyester batch. Once Polyester's
    # sticky workers have run, they can contaminate later Base timing even after they park.
    single_reference = similar(seed)
    step_separate_base!(single_reference, input, older, seed, c, alpha, beta, data)

    short_coefficients = randn(rng, 9)
    short_references = [similar(seed) for _ in 0:8]
    reference_workspace = workspace_like(seed)
    for short_order in 0:8
        clenshaw!(
            step_separate_base!,
            short_coefficients,
            short_order,
            reference_workspace,
            seed,
            short_references[short_order + 1],
            data,
        )
    end

    full_reference = similar(seed)
    clenshaw!(
        step_separate_base!,
        coefficients,
        order,
        reference_workspace,
        seed,
        full_reference,
        data,
    )

    println("\nCorrectness: $scheduler")
    candidate = similar(seed)
    candidate_workspace = workspace_like(seed)
    if scheduler === :base
        fused_recurrence!(candidate, input, older, seed, c, alpha, beta, data)
        check_close!("single fused Base", single_reference, candidate)
        for short_order in 0:8
            clenshaw_split!(
                fused_recurrence!,
                fused_final!,
                short_coefficients,
                short_order,
                candidate_workspace,
                seed,
                candidate,
                data,
            )
            isapprox(short_references[short_order + 1], candidate; rtol=2e-11, atol=2e-11) ||
                error("fused Base correctness failed at order $short_order")
        end
        println("  fused Base orders 0:8 passed")
        clenshaw_split!(
            fused_recurrence!, fused_final!, coefficients, order,
            candidate_workspace, seed, candidate, data,
        )
        check_close!("full fused Base", full_reference, candidate; rtol=1e-10, atol=1e-10)
    else
        step_separate_poly!(candidate, input, older, seed, c, alpha, beta, data)
        check_close!("single separate Poly", single_reference, candidate)
        fused_recurrence!(candidate, input, older, seed, c, alpha, beta, data)
        check_close!("single fused Poly", single_reference, candidate)
        for short_order in 0:8
            clenshaw!(
                step_separate_poly!,
                short_coefficients,
                short_order,
                candidate_workspace,
                seed,
                candidate,
                data,
            )
            isapprox(short_references[short_order + 1], candidate; rtol=2e-11, atol=2e-11) ||
                error("separate Polyester correctness failed at order $short_order")
            clenshaw_split!(
                fused_recurrence!,
                fused_final!,
                short_coefficients,
                short_order,
                candidate_workspace,
                seed,
                candidate,
                data,
            )
            isapprox(short_references[short_order + 1], candidate; rtol=2e-11, atol=2e-11) ||
                error("fused Polyester correctness failed at order $short_order")
        end
        println("  both Polyester paths, orders 0:8 passed")
        clenshaw!(
            step_separate_poly!, coefficients, order, candidate_workspace, seed, candidate, data,
        )
        check_close!("full separate Poly", full_reference, candidate; rtol=1e-10, atol=1e-10)
        clenshaw_split!(
            fused_recurrence!, fused_final!, coefficients, order,
            candidate_workspace, seed, candidate, data,
        )
        check_close!("full fused Poly", full_reference, candidate; rtol=1e-10, atol=1e-10)
    end

    separate_step_out = similar(seed)
    fused_step_out = similar(seed)
    separate_workspace = workspace_like(seed)
    fused_workspace = workspace_like(seed)
    separate_output = similar(seed)
    fused_output = similar(seed)

    if scheduler === :base
        repeat_separate_step = () -> begin
            for _ in 1:step_repeats
                step_separate_base!(separate_step_out, input, older, seed, c, alpha, beta, data)
            end
            nothing
        end
        repeat_fused_step = () -> begin
            for _ in 1:step_repeats
                fused_recurrence!(fused_step_out, input, older, seed, c, alpha, beta, data)
            end
            nothing
        end
        run_separate = () -> clenshaw!(
            step_separate_base!, coefficients, order, separate_workspace, seed, separate_output, data,
        )
        run_fused = () -> clenshaw_split!(
            fused_recurrence!, fused_final!, coefficients, order,
            fused_workspace, seed, fused_output, data,
        )
        separate_label, fused_label = "separate Base+Base", "fused Base"
    else
        repeat_separate_step = () -> begin
            for _ in 1:step_repeats
                step_separate_poly!(separate_step_out, input, older, seed, c, alpha, beta, data)
            end
            nothing
        end
        repeat_fused_step = () -> begin
            for _ in 1:step_repeats
                fused_recurrence!(fused_step_out, input, older, seed, c, alpha, beta, data)
            end
            nothing
        end
        run_separate = () -> clenshaw!(
            step_separate_poly!, coefficients, order, separate_workspace, seed, separate_output, data,
        )
        run_fused = () -> clenshaw_split!(
            fused_recurrence!, fused_final!, coefficients, order,
            fused_workspace, seed, fused_output, data,
        )
        separate_label, fused_label = "separate Poly+Poly", "fused Polyester"
    end

    println("\nOne complete recurrence (time per step)")
    separate_step_times, fused_step_times = time_paired(
        repeat_separate_step, repeat_fused_step, samples,
    )
    separate_step_us = summarize_times(
        separate_label,
        separate_step_times;
        scale=1e6,
        unit="us",
        divisor=step_repeats,
    )
    fused_step_us = summarize_times(
        fused_label,
        fused_step_times;
        scale=1e6,
        unit="us",
        divisor=step_repeats,
    )
    @printf("  fusion speedup      %.3fx\n", separate_step_us / fused_step_us)

    println("\nComplete Clenshaw application")
    separate_full_times, fused_full_times = time_paired(
        run_separate, run_fused, full_samples; warmups=1,
    )
    separate_full_ms = summarize_times(separate_label, separate_full_times)
    fused_full_ms = summarize_times(fused_label, fused_full_times)
    @printf("  fusion speedup      %.3fx\n", separate_full_ms / fused_full_ms)
    fastest_label = fused_full_ms < separate_full_ms ? fused_label : separate_label
    fastest_ms = min(fused_full_ms, separate_full_ms)
    @printf("  fastest             %s (%.3f ms)\n", fastest_label, fastest_ms)

    bytes_saved_per_step = 2 * sizeof(Float64) * length(seed)
    @printf(
        "\nFusing removes %.3f MiB of intermediate b1 traffic per recurrence (%.3f GiB over order %d).\n",
        bytes_saved_per_step / 2.0^20,
        bytes_saved_per_step * order / 2.0^30,
        order,
    )

    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
