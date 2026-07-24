
"""
    EigSorter(mode::Symbol)

Sort policy for eigenvalues.

Supported modes:
- `:SR`: smallest real first
- `:LR`: largest real first
- `:LM`: largest magnitude first
- `:SM`: smallest magnitude first
"""
struct EigSorter
    mode::Symbol  # :SR, :LR, :LM
end

# function sortvals(vals::AbstractVector, sorter::EigSorter)
#     if sorter.mode == :SR
#         idxs = sortperm(vals)  # Sort in ascending order
#     elseif sorter.mode == :LR
#         idxs = sortperm(vals, rev=true)  # Sort in descending order
#     elseif sorter.mode == :LM
#         idxs = sortperm(abs.(vals), rev=true)  # Sort by absolute value
#     else
#         error("Unknown sorting mode: $(sorter.mode)")
#     end

#     if isa(idxs, CuArray)
#         idxs = Vector(idxs)
#     end
#     return idxs
# end



"""
    sortvals(vals::AbstractVector, sorter::EigSorter) -> Vector{Int}

Return permutation indices that sort `vals` according to `sorter.mode`.
"""
function sortvals(vals::AbstractVector, sorter::EigSorter)
    v = is_gpu_array(vals) ? collect(vals) : vals  # bring to CPU if needed

    if sorter.mode == :SR
        idxs = sortperm(v)
    elseif sorter.mode == :LR
        idxs = sortperm(v, rev=true)
    elseif sorter.mode == :LM
        idxs = sortperm(abs.(v), rev=true)
    elseif sorter.mode == :SM
        idxs = sortperm(abs.(v), rev=false)
    else
        error("Unknown sorting mode: $(sorter.mode)")
    end

    return idxs
end
