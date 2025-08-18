
@inline function mapping!(factorization::KrylovFactorization, f!::Function)
    vec = last(factorization.basis)

    f!(factorization.r, vec) 
end


