# Scoped Intel OpenMP settings — the runtime behind MKL's threads.

import MKL_jll

"""
    with_openmp_blocktime(f, ms::Int)

Run `f()` with the Intel OpenMP blocktime — how long MKL's idle worker threads keep spinning after
a parallel region before they sleep — set to `ms` on the calling thread, restoring it afterwards
(also on error). A no-op when that runtime is not available.

For phases that alternate MKL calls with Julia-threaded work: at the default 200 ms, MKL's idle
workers keep spinning on the cores the Julia threads need next.

The value applies to parallel regions the calling thread starts; a task that migrates to another
thread mid-phase runs there with that thread's setting.
"""
function with_openmp_blocktime(f::F, ms::Int) where {F}
    _iomp_available() || return f()
    previous = _kmp_get_blocktime()
    _kmp_set_blocktime(ms)
    try
        return f()
    finally
        _kmp_set_blocktime(Int(previous))
    end
end

_kmp_get_blocktime() = ccall((:kmp_get_blocktime, MKL_jll.IntelOpenMP_jll.libiomp), Cint, ())
_kmp_set_blocktime(ms::Int) =
    ccall((:kmp_set_blocktime, MKL_jll.IntelOpenMP_jll.libiomp), Cvoid, (Cint,), ms)

_iomp_available() = MKL_jll.is_available() && MKL_jll.IntelOpenMP_jll.is_available()
