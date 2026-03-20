

"""
    next_checking!(convergenceinfo::ConvergenceInfo, k::Int64, factorization_or_blocksize) -> nothing

Update `convergenceinfo.nextcheck` (iteration index for next expensive
convergence evaluation) using current residual scale.
"""
function next_checking!(convergenceinfo::ConvergenceInfo, k::Int64, ::LanczosFactorization)
    next_checking!(convergenceinfo, k, 1)
end


function next_checking!(convergenceinfo::ConvergenceInfo, k::Int64, fact::BlockLanczosFactorization)
    next_checking!(convergenceinfo, k, fact.blocksize)
end





"""Blocksize-resolved scheduling kernel for `next_checking!`."""
function next_checking!(convergenceinfo::ConvergenceInfo, k::Int64, blocksize::Int64)
    ε_calc = convergenceinfo.residual
    ε = convergenceinfo.tol

    # blocksize_factor1 = max(ceil(8/s), 1)
    # blocksize_factor2 = max(ceil(8/s), 1)

    if  ε_calc > 1e-2
        convergenceinfo.nextcheck = k+20;

    elseif  ε_calc >1e-3
        convergenceinfo.nextcheck = k+10;
    else 

        # Δk = ε > 1e-16 ? ceil(Int64, abs(log10(ε)-log10(ε_calc))) : 1
        # Int(abs(log10(ε) - log10(ε_calc))÷1) + 1
        convergenceinfo.nextcheck = k + 5
    end

end
