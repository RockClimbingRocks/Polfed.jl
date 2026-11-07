"""
    calculate_overlap!(factorization::KrylovFactorization) -> nothing

Compute and store the overlap/projection coefficients for the current iteration.
"""
@inline calculate_overlap!(factorization::KrylovFactorization) = calcoverlap!(factorization)
