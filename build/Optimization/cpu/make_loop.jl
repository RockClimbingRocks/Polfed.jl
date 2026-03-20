
const UseThreadsInLoop = Dict(
    NoParallel        => false,
    MulColsParallel   => false,
    TwoLevelParallel  => true,
)



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
