# The GPU gate shared by every GPU test file: loads CUDA.jl and records whether a working device is
# present. Included once, lazily, by the first GPU test file — AFTER `backend/gpu_dispatch_tests.jl`,
# which asserts that CUDA.jl is NOT yet loaded.
#
# Gated, not conditional-by-convention: CUDA.jl is a weak dependency and most machines running the
# suite have no device, so the GPU files must load and report cleanly without one. The gate tells
# the two "no GPU" cases apart (CUDA.jl not loadable vs loadable but no working device), because
# they mean different things when a run is unexpectedly skipped. `Pkg.test()` is the first case:
# its sandbox cannot load a weak dependency; the GPU tests need an environment with CUDA.jl in it.

const _GPU_SKIP = Ref("")

const GPU_OK = try
    @eval using CUDA
    if CUDA.functional()
        true
    else
        _GPU_SKIP[] = "CUDA.jl loaded but no functional device"
        false
    end
catch
    _GPU_SKIP[] = "CUDA.jl not loadable in this environment"
    false
end

"""
    gpu_tests_not_run(what)

Record that a GPU test file registered NOTHING. Julia's `@test_skip` would report them as
"Broken", which conventionally means "a test known to fail" — the wrong claim for hardware that is
simply not present. Printed rather than logged: runtests.jl raises the ambient logger to Error to
hide expected POLFED warnings, and a silently skipped suite is precisely what must NOT be hidden.
"""
gpu_tests_not_run(what) =
    printstyled(stderr, "$what NOT RUN — no CUDA device (", _GPU_SKIP[], ")\n"; color = :yellow)
