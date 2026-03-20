using Distributed
using CUDA
using Random
using LinearAlgebra

"""
Split `1:nrows` into `nparts` contiguous, near-equal ranges.
"""
function split_rows(nrows::Int, nparts::Int)
    base = div(nrows, nparts)
    extra = rem(nrows, nparts)

    ranges = Vector{UnitRange{Int}}(undef, nparts)
    start_row = 1
    for p in 1:nparts
        len = base + (p <= extra ? 1 : 0)
        stop_row = start_row + len - 1
        ranges[p] = start_row:stop_row
        start_row = stop_row + 1
    end
    return ranges
end

@everywhere begin
    using CUDA

    """
    QREM/TFIM-style mapping kernel on a row-sharded output vector.

    `Y_local` stores rows `[row_start, row_stop]` of the global output.
    `X_global` and `diag_global` are currently replicated on each GPU.
    """
    function tfim_map_kernel_sharded!(
        Y_local::CuDeviceVector{T},
        X_global::CuDeviceVector{T},
        Leff::Int,
        diag_global::CuDeviceVector{T},
        hx::T,
        row_start::Int,
        basis_length::Int,
    ) where {T<:AbstractFloat}
        local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

        @inbounds if local_i <= length(Y_local)
            i = row_start + local_i - 1

            d = diag_global[i]
            x = X_global[i]
            offdiag_val = zero(T)

            @simd for l in 0:Leff
                newstate = (i - 1) ⊻ (1 << l)
                row = newstate + 1
                offdiag_val += X_global[row]
            end

            Y_local[local_i] = d * x + offdiag_val * hx
        end

        return nothing
    end

    """
    Run the mapping for one contiguous row chunk on the worker's active GPU.

    Also allocates a local zero matrix with `ncols` columns (distributed memory
    layout); the mapped vector is written into column 1.
    """
    function map_chunk_on_gpu(
        diag_h::Vector{T},
        x_h::Vector{T},
        row_start::Int,
        row_stop::Int,
        Leff::Int,
        hx::T,
        ncols::Int,
    ) where {T<:AbstractFloat}
        nlocal = row_stop - row_start + 1

        diag_d = CuArray(diag_h)
        x_d = CuArray(x_h)

        y_local = CUDA.zeros(T, nlocal)
        y_block = CUDA.zeros(T, nlocal, ncols)

        threads = 256
        blocks = cld(nlocal, threads)

        CUDA.@sync @cuda threads=threads blocks=blocks tfim_map_kernel_sharded!(
            y_local,
            x_d,
            Leff,
            diag_d,
            hx,
            row_start,
            length(x_h),
        )

        @views y_block[:, 1] .= y_local

        return Array(y_local)
    end
end

function main(; L::Int=12, ncols::Int=4000, hx::Float32=1f0, seed::Int=1234)
    basis_length = 1 << L
    Leff = L - 1

    devs = collect(CUDA.devices())
    display(devs)
    ngpu = length(devs)
    ngpu == 0 && error("No CUDA devices detected.")

    println("Detected $ngpu GPU(s).")
    println("L=$L, basis_length=$basis_length, ncols=$ncols")

    Random.seed!(seed)
    diag = randn(Float32, basis_length)
    x = randn(Float32, basis_length)
    x ./= norm(x)

    # Full output requested by you: size (2^L, 4000)
    Y = zeros(Float32, basis_length, ncols)

    project_path = Base.active_project()
    exeflags = isnothing(project_path) ? String[] : ["--project=$(project_path)"]
    workers_gpu = addprocs(ngpu; exeflags=exeflags)

    try
        @everywhere workers_gpu using CUDA

        # Pin one worker to one GPU.
        for (gpu_idx, w) in enumerate(workers_gpu)
            remotecall_wait(w, gpu_idx) do idx
                devs_local = collect(CUDA.devices())
                CUDA.device!(devs_local[idx])
                return nothing
            end
        end

        row_ranges = split_rows(basis_length, ngpu)
        chunks = Vector{Tuple{UnitRange{Int}, Vector{Float32}}}(undef, ngpu)

        @sync for (k, w) in enumerate(workers_gpu)
            rr = row_ranges[k]
            @async begin
                yk = remotecall_fetch(
                    map_chunk_on_gpu,
                    w,
                    diag,
                    x,
                    first(rr),
                    last(rr),
                    Leff,
                    hx,
                    ncols,
                )
                chunks[k] = (rr, yk)
            end
        end

        for (rr, yk) in chunks
            @views Y[rr, 1] .= yk
        end

        println("Done.")
        println("size(Y) = $(size(Y))")
        println("norm(Y[:, 1]) = $(norm(view(Y, :, 1)))")
        println("first 8 values of Y[:,1] = $(Y[1:min(8, end), 1])")

        return Y, x, diag
    finally
        rmprocs(workers_gpu)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
    ncols = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4000
    hx = length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 1f0

    main(; L=L, ncols=ncols, hx=hx)
end
