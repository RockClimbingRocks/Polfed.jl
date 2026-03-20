
# # Documentation for Polfed.jl
# This section provides an overview of the main `polfed` function and its associated configuration structs. It serves as a reference for users looking to understand the core functionality of the Polfed.jl package. The `POLFED` algorithm is meant for solving eigenvalue problems of type

# ```math
# H |\psi \rangle = E | \psi \rangle,
# ```

# and is based on Krylov factorization techniques. And just like this other Krylov factorization technique, `POLFED` also only needs the mapping. That is why polfd has two entry points, either with the mapping provided as a sparse matrix or as a function that performs the matrix-vector product, see [`polfed`](@ref).
# ```@docs
# Polfed.polfed
# ```
# As one can see, the `polfed` function has several keyword arguments that allow users to customize the behavior of the algorithm. These keyword arguments are instances of configuration structs that encapsulate various settings for different aspects of the algorithm, such as spectral transformation, factorization, and density of states estimation. The following sections provide detailed documentation for each of these configuration structs. Here is the example how one can pass in custom configuration structs into the `polfed` function:

# ```julia
# using Polfed 
# using Polfed.QSun: quantum_sun_hamiltonian
# using LinearAlgebra

# mat = quantum_sun_hamiltonian(12, 2; sparse=true) # define Hamiltonian matrix
# v0 = rand(size(mat, 1)); v0 ./= norm(v0) #
# howmany = 100
# target = 0.0

# spec_transf_cfg = SpectralTransformConfig(
#     normalization=10., # Norm. factor for the spec. transf. P_K(target) = normalization (default: 1.), 
#     cutoff=1.7, # Cutoff value (default: 0.17)
#     order_safety_factor=0.95, # Safety factor for lowering the polynomial order (default: 0.97)
# )

# fact_cfg = FactorizationConfig(
#     tol=1e-15, # Tolerance for convergence (default: 1e-14) 
#     eigentol= 1e-10, # Tolerance for eigenvalue convergence (default: 1e-9)
#     which=:largest, # Which eigenvalues to target (default: :largest)
# )

# dos_cfg = DoSConfig(
#     N=200, # Number of Chebyshev moments used in the DoS calculation.  (default: 250)
#     R=300, # Number of random vectors used to stochastically estimate the DoS. (default: 300)
# )

# vals, vecs, report = polfed(mat, v0, howmany, target; 
#     produce_report      = true,
#     optimize_mapping    = false,
#     spectral_transform  = spec_transf_cfg,
#     fact                = fact_cfg,
#     dos                 = dos_cfg,
# )
# ```

#  From the above example we can see how to change and pass different configuration structs into the `polfed` function. Each configuration struct is documented in detail below. Here we only randomly changed some of the default values, but one can change any of the fields listed in the documentation of each struct. Below we will provide few more examples of how to use these configuration structs in practice, to achive some goal, e.g. use different coefficents for the spectral transformation. 



# ## Spectral Transformation Configuration

# The `SpectralTransformConfig` struct allows users to customize the polynomial spectral transformation used in the POLFED algorithm and more. There are few main fields modifications that one can do with this struct:
# - Change the spectral transformation ([`see example`](@ref change_the_polynomial_spectral_transformation)). By defoult polfed uses Chebyshev polynomial spectral transformation of [Diracs delta function](https://en.wikipedia.org/wiki/Dirac_delta_function), and coefficents ``c_n`` are given by the projection of Diracs delta function onto Chebyshev polynomials ``c_n^{\lambda} = (2 - \delta_{n,0}) \cdot \cos(n\cdot\arccos(\lambda))``
# ```math
# P_{\lambda}^K(H) = \frac{\text{normalization}}{p(\lambda)} p(H), \quad \quad p(H) = \sum_{n=0}^{K} c_n^{\lambda} T_n(H).
# ```
# - Specify the order of polynomial `K` or  energy interval to target [`left`, `right`].
# - Change the maximal number of iterations with the `overestimate_iters` field, to ensure convergence (see example [`Not all eigenvectors have converged`](@ref not_all_eigenvectors_have_converged)).
# - Change the polynomial order safety factor with the `order_safety_factor` field, to ensure that all requested eigenvalues are captured (see example [`Not all eigenvectors have converged`](@ref not_all_eigenvectors_have_converged)).
# - Handle the parallelization strategy used for matrix-vector or matrix-matrix multiplications (see example [`Change the parallelization strategy`](@ref change_the_parallelization_strategy)).
# - Pass in rescaled mapping and optimized Clenshaw functions to reduce unnecessary memory access.

# Below we have the documentation for the `SpectralTransformConfig` struct and some of its usage examples.
# ```@docs
# Polfed.SpectralTransformConfig
# ```

# ### [Change the polynomial spectral transformation](@id change_the_polynomial_spectral_transformation)

# Lets now look at a few examples of how to use all of these. Lets say that we would like to try some new polynomial coefficients for the spectral transformation 

# ```julia
# using Polfed #hide
# using QuadGK 

# function chebyshev_coefficients_integral(f, N)
#     c = zeros(Float64, N+1)
#     for n in 0:N
#         integrand(x) = f(x) * cos(n * acos(x)) / sqrt(1 - x^2)
#         integral, _ = quadgk(integrand, -1, 1, rtol=1e-10)
#         c[n+1] = ((2 - (n == 0 ? 1 : 0)) / π) * integral
#     end
#     return c
# end


# mat = quantum_sun_hamiltonian(12, 2; sparse=true)
# target = 0.0
# howmany = 100
# v0 = rand(size(mat, 1)); v0 ./= norm(v0)


# coefs_funf(x; μ=target, σ=0.0001) = exp(-((x - μ)^2) /(2σ^2)) / (σ * sqrt(2π))
# coefs = chebyshev_coefficients_integral(coefs_funf, 500)

# spec_transf_cfg = SpectralTransformConfig(
#     coefficients = (λ, n) -> coefs[n+1], # Custom polynomial coefficients
#     normalization=1., # Lets keep the norm 1.0
#     cutoff=0.3, # Cutoff value (default: 0.17)
#     order_safety_factor=0.95, # Safety factor for lowering the polynomial order (default: 0.97)\
#     overestimate_iters=1.5, # Increase the number of iterations to ensure convergence
# )

# vals, vecs, report = polfed(mat, v0, howmany, target; 
#     produce_report      = true,
#     spectral_transform  = spec_transf_cfg,
# )
# display_report(report)
# ```

# Here we defined custom polynomial coefficients by projecting a Gaussian function onto Chebyshev polynomials. Obviously this can not be better alternative to Diracs delta function, since limit ``\simga\to 0.`` would be required to recover Diracs delta function. However, this example serves to illustrate how one can define and use custom polynomial coefficients in the spectral transformation. When one constructs a mapping it needs to check that the polynomial transformation with these coefficients actually peaks within the targeted energy interval and that outside the target interval the transformation is below the `cutoff` value, one can plot the above function and checks that.  Also when studying different spectral transformations they can have different convergence properties, that is why is good to set `overestimate_iters` to a higher value to ensure convergence.






# ### [Not all eigenvectors have converged](@id not_all_eigenvectors_have_converged)
# Going to the next example, let us consider the case where not all eigenpairs have converged, e.g. ``88`` out of ``100``. One can then check the report at the end of the run (with [`display_report`](@ref)), for more detaild report one can also set `show_convergence_details=true` in the `display_report` function to see how many Ritz values were converged at each iteration. It looks something like this:

using Polfed #hide
include("XXZ.jl")  #hide
L = 14; Nup = L÷2; Δ = 1.0 #hide
mat = construct_XXZ_matrix(L, Δ, Nup) #hide
target = 0.0; howmany = 100 #hide
v0 = randn(size(mat,1)); v0 ./= norm(v0) #hide

spec_transf_cfg = SpectralTransformConfig( #hide
    overestimate_iters=1.15, #hide
) #hide


vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true, optimize_mapping=true, spectral_transform=spec_transf_cfg) #hide
display_report(report)

# There one can see that not all eigenpairs are converged, and under [`Polfed.Lanczos.FactorizationReport`](@ref) one can see that all iterations were used. This is usually a clear sign that one should increase [`Polfed.PolfedDefaults.overestimate_iters`](@ref) factor.
spec_transf_cfg = SpectralTransformConfig(
    overestimate_iters=1.3, # Increase the number of iterations to ensure convergence
)
# Assuming that the problem persists even after increasing the maximal number of iterations. We can also display more detailed convergence report (with `show_convergence_details=true`) to see how many Ritz values were converged at each iteration.

display_report(report; show_convergence_details=true)

# Now we have more insight what is going on, if number of converged Ritz values stagnates well just a bit below the requested number of eigenvalues (last few convergence checkings have the same number of converged eigenpairs that is very close to the requested number), it is likely that polynomial order is too high and not all requested eigenvalues are captured within the filter. In that case one should reduce the [`Polfed.PolfedDefaults.order_safety_factor`](@ref) a bit more to ensure that all requested eigenvalues are captured.
spec_transf_cfg = SpectralTransformConfig(
    overestimate_iters=1.3, # Increase the number of iterations to ensure convergence
    order_safety_factor=0.925, # Reduce the polynomial order a bit more to capture all requested eigenvalues
)



# ### [Change the parallelization strategy](@id change_the_parallelization_strategy)
# Another useful feature of the `SpectralTransformConfig` struct is the ability to change the parallelization strategy used for matrix-vector or matrix-matrix multiplications. By default, Polfed uses [`MulColsParallel`](@ref) strategy (for CPU) and [`NoParallel`](@ref) (for GPU). Say that one does not want to use `MulColsParallel()` strategy any more, but wants to parallelize the code with `BLAS` threads instead. 
    
spec_transf_cfg = SpectralTransformConfig(
    parallelization = NoParallel(), 
)
BLAS.set_num_threads(Base.Threads.nthreads()) # Set BLAS threads to the number of Julia threads


vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true, spectral_transform=spec_transf_cfg)
display_report(report)

# When systems become very large, say of order $\sim 10^6$ or more, one The matrix mulitplication will become slower and slower, and increasing block size od `v0` is not efficent eny more because $\text{howmany}/\text{block_size} \eg 100$, and polfed will not be performent any more. That is why we offer [`TwoLevelParallel`](@ref) strategy, where first parallelization is done over the different vectors of the block, and second parallelization is done inside the mapping of one vector. Because [`TwoLevelParallel`](@ref) strategy creates new processes, user needs to specify how many threads per process should we provide. 
nt_per_process = 2
spec_transf_cfg = SpectralTransformConfig(
    parallelization = TwoLevelParallel(nt_per_process), 
)



# ## Factorization Configuration
# Within the `FactorizationConfig` struct, users can customize various aspects of the `Lanczos`/`Block Lanczos` factorization process used in the POLFED algorithm. This includes settings such as reorthogonalization technique, basis type, convergence tolerances, and more. Below is the documentation for the `FactorizationConfig` struct along with some usage examples. Below you can see some examples how to:
# - How to pick type of Krylov factorization. 
# - Adjust the convergence tolerances with the `tolerance` field and eigen vectors tolerance with `eigtol` ([`see example`](@ref change_tolerance_precision_of_the_factorization)).
# - How to change part of the spectrum to sort (e.g. `:smallest`, `largest`, `:smallest_by_magnitude`, `:largest_by_magnitude`), this is hendled with `which` field ([`see example`](@ref target_different_parts_of_the_spectrum)), that is later passed on to eigen sorter-er.


# ```@docs
# Polfed.FactorizationConfig
# ```

# ### [Set type of Krylov factorization](@id set_type_of_krylov_factorization)


# ### [Change tolerance/precision of the factorization](@id change_tolerance_precision_of_the_factorization)


# ### [Target different parts of the spectrum](@id target_different_parts_of_the_spectrum)





# ## Denseties of States Configuration
# ```@docs
# Polfed.DoSConfig
# ```
