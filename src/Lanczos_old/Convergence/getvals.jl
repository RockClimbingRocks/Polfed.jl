
function calculate_eigenvalues!(
    factorization::KrylovFactorization, 
    convergenceinfo::ConvergenceInfo, 
    valsmap!::Function, 
    vecs::AbstractMatrix{E}, 
    vals::AbstractVector{E}
) where {E<:Real}

    Hvec = factorization.pu.vec{E}(undef, size(vecs,1))

    converged = convergenceinfo.converged
    vals_ = view(vals, 1:converged)
    vecs_ = view(vecs, :, 1:converged)
    vals_cpu = zeros(E, converged)

    for i in 1:converged
        vec = view(vecs_, :, i)
        valsmap!(Hvec, vec)
        vals_cpu[i] = dot(vec, Hvec)
    end

    # vals_ .= CuVector(vals_cpu)
    vals_[1:end] .= factorization.pu.vec{E}(vals_cpu)
    return nothing
end


# function calculate_eigenvalues!(
#     factorization::KrylovFactorization,
#     convergenceinfo::ConvergenceInfo,
#     valsmap!::Function,
#     vecs::CuMatrix,  # GPU matrix of eigenvectors
#     vals::CuVector   # GPU vector of eigenvalues to be computed
# )
#     # Number of converged eigenvectors
#     converged = convergenceinfo.converged

#     # Views into the relevant part of the GPU eigenvectors and values
#     vecs_gpu = view(vecs, :, 1:converged)

#     # Transfer to CPU for safe processing
#     vecs_host = Array(vecs_gpu)                          # (size(vecs,1), converged)
#     vals_host = Vector{eltype(vals)}(undef, converged)   # Vector to hold computed eigenvalues
#     Hvec_host = similar(vecs_host[:, 1])                 # Temporary buffer for H * vec

#     # Loop over each eigenvector and compute ⟨ψ|H|ψ⟩
#     for i in 1:converged
#         vec = view(vecs_host, :, i)
#         valsmap!(Hvec_host, vec)  # Apply the Hamiltonian map
#         vals_host[i] = dot(vec, Hvec_host)
#     end

#     # Copy results back to GPU
#     vals[1:converged] .= vals_host

#     return nothing
# end






# function calculate_eigenvalues!(factorization::BlockLanczosFactorization, valsmap!::Function, vecs::Vector{<:AbstractVector}, vals::AbstractVector)

#     Hvec = factorization.pu.vec{eltype(vecs)}(undef, size(vecs,1))

#     map!(vec -> 
#         begin
#             valsmap!(Hvec, vec) 
#             return vec' * Hvec
#         end, 
#         vals, vecs
#     )
# end


