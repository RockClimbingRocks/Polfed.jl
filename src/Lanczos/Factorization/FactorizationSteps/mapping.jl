
@inline function mapping!(factorization::KrylovFactorization, f!::Function)

    f!(factorization.r, factorization.v_last) 
end


