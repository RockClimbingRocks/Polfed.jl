function clenshaw_cpu(qrem::QREM; a::Float64=1.0, b::Float64=0.0)
    qrem_rescaled = QREM(
        "qrem",
        qrem.L,
        qrem.hx / a,
        qrem.spin,
        (qrem.diags .- b) ./ a
    )

    offdiags = get_offdiagonals_by_value(qrem_rescaled)

    crr = @inline (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, X::AbstractVecOrMat) -> begin
        mapvec_with_qrem!(b1, b2, qrem_rescaled.diags, offdiags)
        @. b1 = c * X + 2 * b1 - b3
    end

    cfs = @inline (b1::AbstractVecOrMat, b2::AbstractVecOrMat, c::Real, Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        mapvec_with_qrem!(Y, b1, qrem_rescaled.diags, offdiags)
        @. Y = c * X + Y - b2
    end

    return crr, cfs
end
