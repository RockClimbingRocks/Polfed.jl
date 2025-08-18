

mutable struct LanczosConfig
    rot#::ReOrthTechnique
    tol::Real
    eigentol::Union{Real,Nothing}


    function LanczosConfig(;
                           rot=PolfedDefaults.rot, 
                           tol::Real=PolfedDefaults.tol, 
                           eigentol::Union{Real, Nothing}=PolfedDefaults.eigentol, 
    )
        new(rot, tol, eigentol)
    end

end



mutable struct LanczosConfigFull
    x0::AbstractVecOrMat
    elmtype::Type
    maxdim::Integer
    rot#::ReOrthTechnique
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
            lanczos.tol,
            lanczos.eigentol,
        )
    end

end

  