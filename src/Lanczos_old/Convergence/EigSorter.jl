
struct EigSorter
    mode::Symbol  # :smallest, :largest, :largest_by_amplitude
end

# function sortvals(vals::AbstractVector, sorter::EigSorter)
#     if sorter.mode == :smallest
#         idxs = sortperm(vals)  # Sort in ascending order
#     elseif sorter.mode == :largest
#         idxs = sortperm(vals, rev=true)  # Sort in descending order
#     elseif sorter.mode == :largest_by_amplitude
#         idxs = sortperm(abs.(vals), rev=true)  # Sort by absolute value
#     else
#         error("Unknown sorting mode: $(sorter.mode)")
#     end

#     if isa(idxs, CuArray)
#         idxs = Vector(idxs)
#     end
#     return idxs
# end



function sortvals(vals::AbstractVector, sorter::EigSorter)
    v = isa(vals, CuArray) ? collect(vals) : vals  # bring to CPU if needed

    if sorter.mode == :smallest
        idxs = sortperm(v)
    elseif sorter.mode == :largest
        idxs = sortperm(v, rev=true)
    elseif sorter.mode == :largest_by_amplitude
        idxs = sortperm(abs.(v), rev=true)
    else
        error("Unknown sorting mode: $(sorter.mode)")
    end

    return idxs
end