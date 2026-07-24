
"""
    mapping!(factorization::KrylovFactorization, f!::Function) -> nothing

Apply operator callback `f!` to current basis state and write result to
`factorization.r`.
"""
@inline function mapping!(factorization::KrylovFactorization, f!::Function)

    f!(factorization.r, factorization.v_last) 
end

