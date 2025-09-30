

mutable struct FactorizationConfig
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}


    function FactorizationConfig(;
        rot=PolfedDefaults.rot, 
        basistype::Type{<:OrthonormalBasis}=PolfedDefaults.basistype,
        which::Symbol=PolfedDefaults.which,
        tol::Real=PolfedDefaults.tol, 
        eigentol::Union{Real, Nothing}=PolfedDefaults.eigentol, 
    )
        new(rot, basistype, which, tol, eigentol)
    end

end



mutable struct FactorizationConfigFull
    x0::AbstractVecOrMat
    elmtype::Type
    maxdim::Integer
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}


    function FactorizationConfigFull(
        fact::FactorizationConfig,
        spectral_transform::SpectralTransformConfigFull,
        x0::AbstractVecOrMat{T},
        howmany::Integer,
    ) where {T<:Real}
    
        blocksize = size(x0,2)
        
        new(
            x0,
            T,
            PolfedDefaults.expectedkrylovdim(howmany, blocksize, spectral_transform.overestimate_iters),
            fact.rot,
            fact.basistype,
            fact.which,
            fact.tol,
            fact.eigentol,
        )
    end

end

  