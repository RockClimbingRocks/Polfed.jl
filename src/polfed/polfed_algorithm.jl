function polfed_algorithm(
    spectral_transform::SpectralTransformConfigFull,
    fact_config::FactorizationConfigFull,
    dos::DoSConfigFull,
    pu::ProcessingUnit,
)
    getdos!(dos, spectral_transform, fact_config, pu)
    getspectraltransform!(dos, spectral_transform, fact_config, pu)

    vals, vecs, fact_report = lanczos(
        spectral_transform.f!_transformed, fact_config.x0, spectral_transform.howmany;
        rot         = fact_config.rot,
        basistype   = fact_config.basistype, 
        maxdim      = fact_config.maxdim, 
        tol         = fact_config.tol, 
        eigentol    = fact_config.eigentol, 
        which       = fact_config.which,
        mapvals     = spectral_transform.f!
    ) 

    return vals, vecs, fact_report
end