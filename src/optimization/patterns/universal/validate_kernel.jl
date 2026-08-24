# universal — validation, in two parts.
#
#   can_apply   — BEFORE building: may this pattern's kernels run on this structure at all?
#   verify_kernel — AFTER building: does the built closure actually agree with plain `mul!`?
#
# The second is the one that matters in practice. A row kernel walks index tables by hand; a
# mistake there does not crash, it silently returns a slightly wrong matvec, and POLFED would
# happily converge to the wrong eigenpairs. Checking one block against `mul!` costs two matvecs
# against the thousands a run performs, so it is done on every build rather than left to tests.

"""
    can_apply(::Val{:universal}, s::MappingStructure) -> Bool

`true` — the universal kernels read each value group's own per-row extents, so they impose no
precondition on the structure: any number of distinct off-diagonal values, any row lengths.

A pattern with real requirements returns `false` here, and [`why_not`](@ref) explains it.
"""
can_apply(::Val{:universal}, ::MappingStructure) = true

"""
    why_not(::Val{:universal}, s::MappingStructure) -> String

Never reached — [`can_apply`](@ref) is unconditionally `true` for this pattern. Present so the
pattern satisfies the same interface as one that can decline.
"""
why_not(::Val{:universal}, ::MappingStructure) = "the universal kernels have no precondition"

"""
    can_apply(::Val{:universal}, ::Val{:row_bucketed}, s::MappingStructure) -> Bool

`row_bucketed` needs exactly ONE value group. Bucketing a multi-value structure would have to key
on each row's per-group length TUPLE rather than a single count, which is combinatorial in the
number of groups — so those structures are refused here and take `row_read` instead.
"""
can_apply(::Val{:universal}, ::Val{:row_bucketed}, s::MappingStructure) = n_values(s) == 1

"""    why_not(::Val{:universal}, ::Val{:row_bucketed}, s) -> String"""
why_not(::Val{:universal}, ::Val{:row_bucketed}, s::MappingStructure) =
    "row_bucketed needs a single off-diagonal value; this matrix has $(n_values(s))"

"""
    auto_score(::Val{:universal}, ::Val{:row_bucketed}, bs::Int) -> Int

Auto-selected only at `bs = 1`, where the compile-time trip count is worth ~1.9x over `row_read`.
At `bs >= 4` the `SVector{bs}` load already saturates the load ports and the row permutation's
scattered writes make it ~18% SLOWER, so it scores negative and is never picked automatically —
it remains available by name.
"""
auto_score(::Val{:universal}, ::Val{:row_bucketed}, bs::Int) = bs == 1 ? 10 : -1
