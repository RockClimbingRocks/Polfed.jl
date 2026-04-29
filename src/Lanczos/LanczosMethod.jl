
function cleanup_after_lanczos!(pu::ProcessingUnit)
    if pu isa GPU
        CUDA.synchronize()
        CUDA.reclaim()
    end

    GC.gc(true)
    return nothing
end

function permute_columns_inplace!(A::AbstractMatrix, p, n::Integer)
    n <= 1 && return A

    p_cpu = collect(p)
    visited = falses(n)
    tmp = similar(view(A, :, 1))

    @inbounds for start in 1:n
        visited[start] && continue

        src = p_cpu[start]
        if src == start
            visited[start] = true
            continue
        end

        copyto!(tmp, view(A, :, start))

        dest = start
        while true
            src = p_cpu[dest]
            visited[dest] = true

            if src == start
                copyto!(view(A, :, dest), tmp)
                break
            else
                copyto!(view(A, :, dest), view(A, :, src))
                dest = src
            end
        end
    end

    return A
end


"""
    lanczos_method(iterator::LanczosIterator, convergenceinfo::ConvergenceInfo, basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit)

Allocate outputs, execute Lanczos iterations, and return converged eigenpairs
sorted in ascending-real order.
"""
function lanczos_method(
    iterator::LanczosIterator{T,ROT}, 
    convergenceinfo::ConvergenceInfo, 
    basistype::Type{<:OrthonormalBasis}, 
    pu::ProcessingUnit
) where {T<:AbstractVecOrMat,ROT<:ReOrthTechnique}
    hilbertspacedim = size(iterator.x0, 1)
    howmany = convergenceinfo.howmany

    vec_eltype = eltype(iterator.x0)
    val_eltype = real(vec_eltype)

    vals = pu.Vector{val_eltype}(undef, howmany)
    vecs = pu.Matrix{vec_eltype}(undef, (hilbertspacedim, howmany))


    factorization_report = lanczos_method!(vals, vecs, iterator, convergenceinfo, basistype, pu)
    cleanup_after_lanczos!(pu)

    converged = min(convergenceinfo.converged, convergenceinfo.eigenconverged)
    vals_   = view(vals, 1:converged)
    idxs    = sortvals(vals_, EigSorter(:SR))
    
    vals_converged_ordered = pu.Vector(view(vals,   idxs))
    permute_columns_inplace!(vecs, idxs, converged)
    vecs_converged_ordered = converged == size(vecs, 2) ? vecs : pu.Matrix(view(vecs, :, 1:converged))

    return vals_converged_ordered, vecs_converged_ordered, factorization_report
end



"""
    lanczos_method!(vals, vecs, iterator, convergenceinfo, basistype, pu) -> FactorizationReport

Low-level in-place Lanczos entry point using preallocated `vals` and `vecs`.
"""
function lanczos_method!(
    vals::AbstractVector, 
    vecs::AbstractMatrix, 
    iterator::LanczosIterator, 
    convergenceinfo::ConvergenceInfo, 
    basistype::Type{<:OrthonormalBasis}, 
    pu::ProcessingUnit,
)
    # Initialize the Krylov basis
    krylovbasis = createbasis(convergenceinfo.maxdim, iterator.x0, basistype, pu)


    # println("Starting Lanczos method with $(convergenceinfo.howmany) eigenvalues and vectors...")

    # println("LanczosIterator fields and values:")
    # for field in fieldnames(typeof(iterator))
    #     println("  $(field): ", getfield(iterator, field))
    # end

    # println("ConvergenceInfo fields and values:")
    # for field in fieldnames(typeof(convergenceinfo))
    #     println("  $(field): ", getfield(convergenceinfo, field))
    # end
    # Run the Lanczos algorithm
    lanczos_algorithm!(vals, vecs, iterator, convergenceinfo, krylovbasis, pu)
end
