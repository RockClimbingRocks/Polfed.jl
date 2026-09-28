"""
    _filtered_lanczos(transformed, x0, eig_target; kwargs...) -> LanczosResult

[`lanczos`](@ref) on the filtered operator with MKL's idle threads sleeping at once
([`with_openmp_blocktime`](@ref)`(…, 0)`), so the BLAS calls between filter applications leave no
spinning threads on the cores the next application needs. Every filtered Lanczos goes through here.
"""
_filtered_lanczos(transformed, x0, eig_target; kwargs...) =
    with_openmp_blocktime(0) do
        lanczos(transformed, x0, eig_target; kwargs...)
    end
