# Build the optimized ROW-layout operators for a structured Hamiltonian: the rescaled
# `f!_rescaled` (H̃ = (H-b)/a, fed to the Clenshaw filter) and the unscaled `f!` (H, used as
# the Lanczos map-back `mapvals`). This is the piece the matrix `polfed(H, …)` entrypoint calls:
# it value-groups H, resolves which kernel/params to use, and returns row operators. It does NOT
# estimate Emin/Emax or run the pipeline — the caller supplies the rescaling `(a, b)`.

"""
    rescale(s::MappingStructure, a::Real, b::Real) -> MappingStructure

The value-grouped structure of the **rescaled** operator `H̃ = (H - b)/a`: the diagonal becomes
`(diag - b)/a` and the single off-diagonal value becomes `hx/a`. The neighbour arrays are
unchanged (the sparsity pattern is invariant under rescaling).
"""
function rescale(s::MappingStructure{T}, a::Real, b::Real) where {T}
    diags_r = convert(Vector{T}, (s.diags .- b) ./ a)
    hx_r = convert(T, s.hx / a)
    return MappingStructure{T}(s.dim, s.L, hx_r, diags_r, s.nb32, s.flat, s.starts)
end

"""
    resolve_mapping_kernel(choice, H, ::Val{bs}; secs=0.5) -> (id::Symbol, G::Int, W::Int)

Resolve the `optimize_mapping` request into a concrete kernel id and tuning params:

- `true` / `:auto` → benchmark the applicable kernels at block size `bs` and pick the fastest;
- a kernel `Symbol` (`:ell32_svec`, `:csr_simd`, `:csr64_svec`, `:shuf`) → that kernel, default
  `G`/`W`;
- a `NamedTuple` `(kernel=…, G=…, W=…)` → those pinned values.

# Throws
- `ArgumentError`: on an unrecognized `choice` (or a NamedTuple missing `:kernel`).
"""
function resolve_mapping_kernel(choice, H::AbstractMatrix, ::Val{bs}; secs::Real = 0.5) where {bs}
    if choice === true || choice === :auto
        report = benchmark_mappings(H; bs_list = (bs,), G_list = (1, 2, 4, 8), secs = secs)
        haskey(report.best, bs) ||
            error("optimize_mapping = :auto — no applicable kernel passed the benchmark at bs=$bs.")
        best = report.best[bs]
        return (best.kernel, best.G, 8)
    elseif choice isa Symbol
        return (choice, 4, 8)                        # default G; W used only by :shuf
    elseif choice isa NamedTuple
        haskey(choice, :kernel) ||
            throw(ArgumentError("optimize_mapping NamedTuple must have a :kernel field, got $choice."))
        return (choice.kernel, get(choice, :G, 4), get(choice, :W, 8))
    else
        throw(ArgumentError(
            "optimize_mapping must be false, true/:auto, a kernel Symbol, or a " *
            "(kernel=…, G=…, W=…) NamedTuple; got $(repr(choice))."))
    end
end

"""
    build_optimized_mapping(H, choice, ::Val{bs}, a::Real, b::Real; secs=0.5)
        -> (; f!_rescaled, id, G, W)

Build the fast row-layout **transform** operator for `H` under the rescaling `H̃ = (H - b)/a`:

- `f!_rescaled(Y, X)`: the fast row kernel of `H̃` over `bs × D` **blocks** — the operator the
  Clenshaw filter (and thus every Lanczos matvec) runs on, so it is specialized to the Krylov
  block size `bs`. This is the POLFED bottleneck, and the only place the value-grouped kernel
  earns its keep;
- `id`, `G`, `W`: the resolved kernel and its params.

Only the rescaled transform operator is returned. The Lanczos map-back `mapvals` (applied a
handful of times per convergence check, to a *single* strided `1 × D` eigenvector) is **not**
built here: a value-grouped SIMD kernel neither pays off nor even loads from a strided row view
(the `:shuf` kernel needs contiguous memory). The matrix entrypoint uses a generic `mul!`-based
row map-back instead (see `_build_matrix_rowops`).

`choice` is the `optimize_mapping` request (see [`resolve_mapping_kernel`](@ref)); `H` must be a
structured Hamiltonian [`value_grouping`](@ref) accepts (single off-diagonal value, regular row
length).
"""
function build_optimized_mapping(H::AbstractMatrix, choice, ::Val{bs}, a::Real, b::Real;
                                 secs::Real = 0.5) where {bs}
    s = value_grouping(H)
    id, G, W = resolve_mapping_kernel(choice, H, Val(bs); secs = secs)
    s_rescaled = rescale(s, a, b)
    f!_rescaled = build_row_mapping(id, s_rescaled, Val(bs); G = G, W = W)   # bs×D block (transform)
    return (; f!_rescaled, id, G, W)
end
