
# # [Knowing your parallelization (POLFED for babys)](@id Knowing_your_parallelization)

# Parallelization of the spectral transform is of the utmost importance, since it is the most time-consuming part of the code. Although this may not be evident previous (relativly small) examples, for larger system sizes the spectral transformation (or mapping) can account for up to $99\%$ of the total runtime. This is because the polynomial expansion order $K$ grows exponentially with system size, and so does the number of matrix–vector multiplications. In the previous example we observed how parallelization is handled under the hood: each thread maps one column. One might ask why not simply parallelize the mapping of a single vector using prebuilt BLAS or MKL routines? The answer is that parallelization of sparse matrix multiplication with a dense vector (or matrix) is not well optimized, partly because of the variety of possible sparse matrix formats (try it by yourself). By using the default [`MulColsParallel`](@ref) strategy, one achieves almost perfect scaling of the spectral transformation.

# However, as the Lanczos block size grows, the factorization becomes less efficient (see the corresponding plot). At some point, it is not worth further increasing the block size. For this reason, we also allow parallelization across individual columns, so that each column can use multiple threads (e.g., two threads per column). This two-level parallelization is enabled with [`TwoLevelParallel`](@ref), where the argument specifies the number of threads per column.

# ```julia
# nt_per_col = 2
# parallel_strategy = TwoLevelParallel(nt_per_col)
# spec_trans = SpectralTransformConfig(;parallelization=parallel_strategy)
# f!(Y,X) = mul!(Y, mat, X) # Here it can be some custom mapping function
# vals, vecs, report = polfed(f!, v0, howmany, target; 
#     produce_report=true,
#     spectral_transform=spec_trans
# )
# ```

# This is another freedom of parallelization that we offer to the user, almost without any additional work! But sometimes there are better/smarter ways of doing things, this is one of the reasons why we also offer no parallelization strategy `NoParallel`. Here one can do its own parallelization, but in order to write a parallelized mapping one needs to pass it as an input parameter, that is why there are two different entrypoints for [`polfed`](@ref) function.

# First with matrix, the most straightforward, as we seen before, and the second with mapping function, as we will see later. 
# Another thing you need to keep in mind is that before performing the mapping we set number of `BLAS` and `MKL` threads equal to 1, so if for some reason you want to use these threads in the mapping you need to wrap them inside the mapping function. 

# To recap details about parallelization, if all of the parallelization is handled by the user, it is imporatant to set parallelization stratagy to [`NoParallel`](@ref), for example, when using GPUs, stratagy is automatically set to [`NoParallel`](@ref), sence everything is multiplied as once. Otherwise automatic parallelization across different columns of the mapping will be performed. It is beneffical to set number of threads the same as block size to gain the optimal performance, that way in the regime where $\text{howmany}/\text{blocksize} \geq 100$, it should scale almost perfectly. Same goes if the input parameter is a function, but it gives u additional freedom of possible two level parallelization [`TwoLevelParallel`](@ref). Using two level parallelization is not allways beneficial, becuase to ensure that there is no over threading and that user can really get as many threads as he asked for, we needed to create a seperate processes whit the exact number of threads as desired by the user. These processes are constructed with Julias [`Distributed.jl`](https://docs.julialang.org/en/v1/stdlib/Distributed/) package, these porcesses then need some time to be spawned, therefor it is not benefical to use them for small hilber space dimensions, usually one needs hilbrert space dimension $>100\_000$ for two level parallelization to be benificial, ofcurs it is also depanded on the number oif requested eigenpairs because it affects the order of polynomial and number of iterations required for convergence.

# Note that instead of passing in matrix, you should construct personallized, highly-optimized mapping function specific for the model at hand, and do parallelization across rows at your own, or even whole parallelization. For CPU's the default parallelization strategy is [`MulColsParallel`](@ref) with one thread per column. Everything else is parallelized by setting number of `BLAS` threads to be equat to number of threads. 