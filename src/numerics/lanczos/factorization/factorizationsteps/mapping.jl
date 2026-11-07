"""
    mapping!(factorization::KrylovFactorization, f!::Function) -> nothing

Apply the operator callback `f!` to the current basis state, writing the result
into `factorization.r`.
"""
@inline mapping!(factorization::KrylovFactorization, f!) = f!(factorization.r, factorization.v_last)
