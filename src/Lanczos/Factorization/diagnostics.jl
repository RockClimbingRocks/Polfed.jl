@inline _isfinite_number(x::Real) = isfinite(x)
@inline _isfinite_number(x::Complex) = isfinite(real(x)) && isfinite(imag(x))

"""
    has_nonfinite(x::AbstractArray) -> Bool

Return `true` if any element of `x` is `NaN` or `Inf`.
"""
@inline function has_nonfinite(x::AbstractArray)
    if is_gpu_array(x)
        return CUDA.@allowscalar any(v -> !_isfinite_number(v), x)
    end

    @inbounds for v in x
        _isfinite_number(v) || return true
    end
    return false
end

@inline krylov_symmetry_defect(::LanczosFactorization) = 0.0

"""
    krylov_symmetry_defect(fac) -> Real

Return a normalized symmetry defect metric for the projected Krylov matrix.
For scalar Lanczos this is exactly `0`.
"""
@inline function krylov_symmetry_defect(fac::BlockLanczosFactorization)
    fac.krylovdim == 0 && return 0.0
    H = view(fac.mat, 1:fac.krylovdim, 1:fac.krylovdim)
    denom = max(norm(H), eps(real(eltype(H))))
    return norm(H - H') / denom
end

"""
    log_factorization_step_diagnostics!(factorization::KrylovFactorization, step::Symbol, iteration::Integer) -> nothing

Emit warning/debug diagnostics about numerical issues in factorization state.
"""
@inline function log_factorization_step_diagnostics!(
    factorization::KrylovFactorization,
    step::Symbol,
    iteration::Integer,
)
    if should_log(POLFED_WARN_LEVEL) && has_nonfinite(factorization.r)
        polfed_log(
            POLFED_WARN_LEVEL,
            "Non-finite values detected in Lanczos workspace.",
            iteration=iteration,
            step=String(step),
            krylovdim=factorization.krylovdim,
        )
    end

    if should_log(POLFED_DEBUG_LEVEL)
        symmetry_defect = krylov_symmetry_defect(factorization)
        threshold = sqrt(eps(real(eltype(factorization.r))))
        if symmetry_defect > threshold
            polfed_log(
                POLFED_DEBUG_LEVEL,
                "Lanczos projected matrix lost symmetry beyond tolerance.",
                iteration=iteration,
                step=String(step),
                krylovdim=factorization.krylovdim,
                symmetry_defect=symmetry_defect,
                threshold=threshold,
            )
        end
    end
    return nothing
end
