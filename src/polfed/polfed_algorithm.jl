function polfed_algorithm(
    spectral_transform::SpectralTransformConfigFull,
    lanczos_config::LanczosConfigFull,
    dos::DoSConfigFull,
    pu::ProcessingUnit,
    produce_report::Bool
)

    getdos!(dos, spectral_transform, lanczos_config, pu)
    getspectraltransform!(dos, spectral_transform, lanczos_config, pu)



    vals, vecs, fact_report = lanczosmethod(
        spectral_transform.f!_transformed, lanczos_config.x0, spectral_transform.howmany;
        rot         = lanczos_config.rot,
        basistype   = lanczos_config.basistype, 
        maxdim      = lanczos_config.maxdim, 
        tol         = lanczos_config.tol, 
        eigentol    = lanczos_config.eigentol, 
        which       = lanczos_config.which,
        mapvals     = f!
    ) 

    return vals, vecs, fact_report
end