

mutable struct LanczosConfig
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}


    function LanczosConfig(;
        rot=PolfedDefaults.rot, 
        basistype::Type{<:OrthonormalBasis}=PolfedDefaults.basistype,
        which::Symbol=PolfedDefaults.which,
        tol::Real=PolfedDefaults.tol, 
        eigentol::Union{Real, Nothing}=PolfedDefaults.eigentol, 
    )
        new(rot, basistype, which, tol, eigentol)
    end

end



mutable struct LanczosConfigFull
    x0::AbstractVecOrMat
    elmtype::Type
    maxdim::Integer
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}


    function LanczosConfigFull(
        lanczos::LanczosConfig,
        spectral_transform::SpectralTransformConfigFull,
        x0::AbstractVecOrMat{T},
        howmany::Integer,
    ) where {T<:Real}
    
        blocksize = size(x0,2)
        
        new(
            x0,
            T,
            PolfedDefaults.expectedkrylovdim(howmany, blocksize, spectral_transform.overestimate_iters),
            lanczos.rot,
            lanczos.basistype,
            lanczos.which,
            lanczos.tol,
            lanczos.eigentol,
        )
    end

end

  