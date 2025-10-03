
macro addtime!(wallarr, cpuarr, idx, expr)
    quote
        walltime = @elapsed begin 
            cputime = @CPUelapsed $(esc(expr)) 
        end
        $(esc(wallarr))[$(esc(idx))] += walltime
        $(esc(cpuarr))[$(esc(idx))] += cputime
    end
end