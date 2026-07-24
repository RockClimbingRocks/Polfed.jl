
@inline use_threads_in_loop(::NoParallel) = false
@inline use_threads_in_loop(parallel_strategy::MulColsParallel) = parallel_strategy.nt_per_col > 1
@inline use_threads_in_loop(::TwoLevelParallel) = true



# function make_loop(use_threads::Bool)
#     return @inline (range, body) -> begin
#         if use_threads
#             Threads.@threads for i in range
#                 body(i)
#             end
#         else
#             for i in range
#                 body(i)
#             end
#         end
#     end
# end

"""
    make_loop(use_threads::Bool) -> Function

Return a loop executor closure `(range, f) -> ...` used by optimized CPU
kernels.

- If `use_threads=true`, executes with `Threads.@threads`.
- Otherwise, executes serially.
"""
function make_loop(use_threads::Bool)
    if use_threads
        return (range, f) -> (Threads.@threads for i in range; f(i); end)
    else
        return (range, f) -> (for i in range; f(i); end)
    end
end

"""
    threaded_column_chunks!(body, nrows, ncols, nt_per_col) -> nothing

Run `body(start_row, end_row, col)` across a single team of Julia threads by
splitting each column into `nt_per_col` row chunks.
"""
function threaded_column_chunks!(body::Function, nrows::Int, ncols::Int, nt_per_col::Int)
    (nrows == 0 || ncols == 0) && return nothing

    nchunks = max(nt_per_col, 1)
    chunk_size = cld(nrows, nchunks)

    Threads.@threads for task in 1:(ncols * nchunks)
        col = (task - 1) ÷ nchunks + 1
        chunk = (task - 1) % nchunks + 1

        start_row = (chunk - 1) * chunk_size + 1
        end_row = min(chunk * chunk_size, nrows)
        start_row <= nrows || continue

        body(start_row, end_row, col)
    end

    return nothing
end
