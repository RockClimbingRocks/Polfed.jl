# Scoped thread counts for the threaded math libraries (dense BLAS, MKL incl. MKL Sparse).
# Included at the package top level with the rest of `backend/`, so both the POLFED stages
# and `Numerics` can reach it.

import MKL_jll

"""
    with_serial_libraries(f)

Run `f()` with dense BLAS and MKL (including MKL Sparse) pinned to one thread, restoring both
counts afterwards (also on error).

For stages made of many small single-vector library calls (DoS, bounds probe), which get their
parallelism from Julia tasks instead. Serial calls also round identically at every thread count.
The MKL count is process-wide, so tasks spawned inside `f` are covered on any thread.
"""
function with_serial_libraries(f::F) where {F}
    blas = BLAS.get_num_threads()
    mkl = _mkl_max_threads()
    BLAS.set_num_threads(1)
    _mkl_set_threads!(1)
    try
        return f()
    finally
        _mkl_set_threads!(mkl)
        BLAS.set_num_threads(blas)
    end
end

"""MKL's current thread count, or `0` when MKL is not available on this platform."""
_mkl_max_threads() =
    MKL_jll.is_available() ? Int(@ccall MKL_jll.libmkl_rt.MKL_Get_Max_Threads()::Cint) : 0

"""Set MKL's process-wide thread count; a no-op for `n < 1` or without MKL."""
function _mkl_set_threads!(n::Int)
    (n >= 1 && MKL_jll.is_available()) || return nothing
    @ccall MKL_jll.libmkl_rt.MKL_Set_Num_Threads(n::Cint)::Cvoid
    return nothing
end
