



function filterize_intervals_primitive(intervals::AbstractVector{Interval})
    filters = Vector{AbstractFilter}(undef, length(intervals))
    for (i, interval) in enumerate(intervals)
        filters[i] = resolve_filter(interval)
    end

    return filters
end




