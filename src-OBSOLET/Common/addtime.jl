
"""
    @addtime!(wallarr, cpuarr, idx, expr)

Execute `expr` and accumulate elapsed wall/CPU time into `wallarr[idx]` and
`cpuarr[idx]`.

Designed for repeated stage timing inside iterative algorithms.
"""
macro addtime!(wallarr, cpuarr, idx, expr)
    quote
        walltime = @elapsed begin 
            cputime = @CPUelapsed $(esc(expr)) 
        end
        $(esc(wallarr))[$(esc(idx))] += walltime
        $(esc(cpuarr))[$(esc(idx))] += cputime
    end
end
