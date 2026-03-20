#
# # [Constructing optimized mapping (POLFED for teenagers)](@id Constructing_Optimized_Mapping)

# ## Constructing optimized mapping
# Now that we master the parallelization workflow of polfed method, we exploiting the fact that POLFED, like Lanczos, is again only depandend on the mapping! Therfore mapping can replace the hamiltonian (construct our own mapping for the studied disordered XXZ model, almost like big boys). We can take advantage of having all of the offdiagonal elements at a constant offdiagonal value, this way we can reduce the memory acces and make mapping faster, sience our code for mapping of a vector/matrix is highly memory bound.
# It is ussually beneficial to seperate diagonal and offdiagonal part, because mapping of diagonal elements is simple and fast because of the coallesacal memory acces. Laverageing this things we can construct quite general mapping for all hamiltonians with constant off diagonal element.

using Polfed #hide
include("XXZ.jl")  #hide



function get_diags_and_offdiagonals_single_value(mat::AbstractMatrix{T}; tol=1e-13, round_digits=14) where {T<:Real}
    dim = size(mat, 1)
    diagonals = [round(mat[i, i]; digits=round_digits) for i in 1:dim]
    flat = Int[]
    starts = Int[]
    idx = 1
    offdiag_val::Union{Nothing, T} = nothing

    for i in 1:dim
        push!(starts, idx)
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue

            v_rounded = round(v; digits=round_digits)
            if offdiag_val === nothing
                offdiag_val = v_rounded
            elseif abs(v_rounded - offdiag_val) > tol
                error("Matrix has multiple distinct off-diagonal values (found $v_rounded and $offdiag_val).")
            end
            push!(flat, j)
        end
        idx += length(flat) - starts[end] + 1
    end

    offdiag_val === nothing && error("No off-diagonal elements found above tolerance.")
    return diagonals, offdiag_val, flat, starts
end

function mapvec_with_xxz!(
    diags::Vector{Float64},
    offdiag_val::Float64,
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
)
    return (Y,X) -> begin
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], offdiag_val * sum_val)
        end
    end
end
nothing #hide 

# Here `diags` is a vector of diagonal matrix elements, `offdiags_flatten` is a flattened vector of indices of offdiagonal elements for each row, and `start_indices` indicates where each row's offdiagonal indices start in the flattened vector. Because `Julia`'s `mul!()` function is highly optimized, not every custom function can be as efficent, that is why one needs to always benchmark the performance. We should benchmark both mappings to see what is the speedup, if there even is one.

# ## Benchmarking the optimized mapping
using BenchmarkTools

function benchmark_mappings(L)
    println("Benchmarking mapping for L=$L")
    Nup = L÷2; Δ = 0.55
    mat = construct_XXZ_matrix(L, Δ, Nup)
    diagonals, offdiag_val, flat, starts = get_diags_and_offdiagonals_single_value(mat)
    mapv! = mapvec_with_xxz!(diagonals, offdiag_val, flat, starts)
    X = rand(size(mat,1)); Y = similar(X)
    @btime mul!($Y, $mat, $X)
    @btime $mapv!($Y, $X)
end
benchmark_mappings(18)

# Mapping is now factor of 1.427 times faster. Sometimes in possible to even further optimize the mapping, especially for models without $U(1)$ symmetry class (quantum Ising model, QREM or similar), where one can exploid the fact that state and its index position in the vector are trivially connceted (differenece of $1$, due to julias one-base indexing). To avoid unnecesary slowdowns, it is important to benchmark the constructed mappings. Now we can aspect approximatly $40\%$ faster code! Lets benchmark how does the time scale with $L$

benchmark_mappings(20)
benchmark_mappings(22)
benchmark_mappings(24)

# For smaller system sizes ($L=18$) there is approximate factor of $1.4$ speedup, where as for larger system sizes factor $2.5$ up to $3.$ sems to persist. 


# ## Using optimized mapping in [`polfed`](@ref)
# For a bit system size (e.g. $L=20$ and above), when spectral transfotrmation becomes more expensive there should be a clear speedup. Now that we have constructed optimized mapping, we can pass it to the polfed function as a mapping function, instead of passing the matrix itself. Remember that [`polfed`](@ref) has two entrypoints, one with matrix and one with mapping function, here we will use the latter one.

# !!! warning "Mapping function requirements"
#     The mapping function must have the signature `f!(Y, X)`, where `Y` is the output vector/matrix and `X` is the input vector/matrix. It should perform the operation `Y .= A * X`, where `A` is the implicit matrix represented by the mapping. The function must handle both single vectors and matrices (for block Lanczos) correctly. 


# Now lets see how does polfed perform with our optimized mapping
L = 14 ; Nup = L÷2; Δ = 0.55
mat = construct_XXZ_matrix(L, Δ, Nup)
diagonals, offdiag_val, flat, starts = get_diags_and_offdiagonals_single_value(mat)
mapv! = mapvec_with_xxz!(diagonals, offdiag_val, flat, starts)
v0 = rand(size(mat,1)); v0 ./= norm(v0) # v0_ = rand(size(mat,1), 4); v0 = Matrix(qr(v0_).Q)
howmany = 100; target = 0.0


vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true)
display_report(report)


