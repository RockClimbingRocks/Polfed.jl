
const UseThreadsInLoop = Dict(
    NoParallel        => false,
    MulColsParallel   => false,
    TwoLevelParallel  => true,
)



function make_loop(use_threads::Bool)
    return (range, body) -> begin
        if use_threads
            Threads.@threads for i in range
                body(i)
            end
        else
            for i in range
                body(i)
            end
        end
    end
end